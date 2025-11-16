from beartype.typing import * # type: ignore[reportWildcardImportFromLibrary]

import torch
from torch.utils.checkpoint import _get_autocast_kwargs
from torch.utils._pytree import tree_unflatten, tree_flatten, TreeSpec

from .batch import Batch
from .profile import annotate
from .scheduler import ModelExecutePlan
from .transfer import async_h2d, async_d2h, upload_layer, download_layer, free_layer

if TYPE_CHECKING:
    from .device import DeviceManager
    from .RoundPipe import RoundPipe
else:
    from typing_extensions import TypeAliasType
    DeviceManager = TypeAliasType('DeviceManager', 'RoundPipe.device.DeviceManager')
    RoundPipe = TypeAliasType('RoundPipe', 'RoundPipe.RoundPipe')

class RoundPipeRunContext:
    model: 'RoundPipe'
    execute_plan: 'ModelExecutePlan'
    enable_grad: bool
    microbatch_id: int
    num_microbatches: int
    preserve_rng_state: bool
    device_autocast_kwargs: dict
    cpu_autocast_kwargs: dict
    flatten_inputs: List[List[Any]]
    flatten_specs: List[Optional['TreeSpec']]
    device_rng_states: List[Optional[torch.Tensor]]
    cpu_rng_states: List[Optional[torch.Tensor]]
    input_backward_events: Sequence[torch.cuda.Event]
    output_backward_events: Sequence[torch.cuda.Event]
    grad_states: List[Any]

    def __init__(self, model: 'RoundPipe',
                 execute_plan: 'ModelExecutePlan',
                 enable_grad: bool,
                 microbatch_id: int,
                 num_microbatches: int,
                 preserve_rng_state: bool) -> None:
        self.model = model
        self.execute_plan = execute_plan
        self.enable_grad = enable_grad
        self.microbatch_id = microbatch_id
        self.num_microbatches = num_microbatches
        self.preserve_rng_state = preserve_rng_state
        if model.use_fp16:
            self.device_autocast_kwargs = {
                'enabled': True,
                'dtype': torch.float16,
                'cache_enabled': False
            }
            self.cpu_autocast_kwargs = {
                'enabled': True,
                'dtype': torch.float16,
                'cache_enabled': False
            }
        else:
            self.device_autocast_kwargs, self.cpu_autocast_kwargs = _get_autocast_kwargs('cuda') # type: ignore[reportAttributeAccessIssue]
            # RoundPipe manages its own autocast cache
            self.device_autocast_kwargs['cache_enabled'] = False
            self.cpu_autocast_kwargs['cache_enabled'] = False

        self.flatten_inputs = [[] for _ in range(model.num_layers)]
        self.flatten_specs = [None for _ in range(model.num_layers)]
        self.device_rng_states = [None for _ in range(model.num_layers)]
        self.cpu_rng_states = [None for _ in range(model.num_layers)]

    def save_input(self, layer_id: int, batch: 'Batch', device: 'DeviceManager') -> None:
        if not (self.enable_grad and self.execute_plan.backward_need_input(layer_id)):
            return

        self.flatten_inputs[layer_id] = batch.flatten_states[self.microbatch_id]
        self.flatten_specs[layer_id] = batch.flatten_specs[self.microbatch_id]

        device.downstream.wait_stream(device.compute_stream)
        with torch.cuda.stream(device.downstream):
            for idx, item in enumerate(self.flatten_inputs[layer_id]):
                if isinstance(item, torch.Tensor) and item.device != torch.device('cpu'):
                    item.record_stream(device.downstream)
                    self.flatten_inputs[layer_id][idx] = item.to('cpu', non_blocking=True)

        if self.preserve_rng_state:
            with device.device:
                self.device_rng_states[layer_id] = torch.cuda.get_rng_state()
            self.cpu_rng_states[layer_id] = torch.get_rng_state()

    def restore_rng_state(self, layer_id: int, device: 'DeviceManager') -> None:
        if not self.preserve_rng_state:
            return

        device_states: torch.Tensor = self.device_rng_states[layer_id] # type: ignore[reportAssignmentType]
        cpu_states: torch.Tensor = self.cpu_rng_states[layer_id] # type: ignore[reportAssignmentType]
        assert device_states is not None or cpu_states is not None, "RNG states were not saved properly."

        with device.device:
            torch.cuda.set_rng_state(device_states)
        torch.set_rng_state(cpu_states)

@torch.no_grad()
def run_forward(layer_ids: range, batch: 'Batch',
                context: RoundPipeRunContext, device: 'DeviceManager') -> None:
    model = context.model
    batch_idx = context.microbatch_id
    grad_context = torch.enable_grad() if context.enable_grad else torch.no_grad()
    if batch_idx == 0:
        for layer_id in layer_ids:
            upload_layer(model.layers[layer_id], device.upstream, device.compute_stream, False)
    for layer_id in layer_ids:
        context.save_input(layer_id, batch, device)
        if layer_id == layer_ids[0]:
            batch.flatten_states[batch_idx] = async_h2d(
                device.compute_stream, device.upstream, batch.forward_events[batch_idx], batch.flatten_states[batch_idx], context.enable_grad
            )
            device.upstream.wait_stream(device.compute_stream)
        hidden_state = tree_unflatten(batch.flatten_states[batch_idx], batch.flatten_specs[batch_idx])
        with grad_context, \
             torch.cuda.stream(device.compute_stream), \
             torch.autocast('cuda', **context.device_autocast_kwargs), \
             torch.autocast('cpu', **context.cpu_autocast_kwargs), \
             annotate(f'{model.name}L[{layer_id}]B[{batch_idx}]Fwd'), \
             model.model_timer.time_forward(layer_id, device.compute_stream):
            try:
                if layer_id == 0:
                    args, kwargs = hidden_state
                    hidden_state = model.layers[layer_id].forward(*args, **kwargs)
                else:
                    hidden_state = model.layers[layer_id].forward(hidden_state)
            except Exception as e:
                e.args = e.args + (f'The above error occurred in {model.name} layer {layer_id} during forward pass.',)
                raise
        batch.flatten_states[batch_idx], batch.flatten_specs[batch_idx] = tree_flatten(hidden_state)

        if context.enable_grad:
            for idx, item in enumerate(batch.flatten_states[batch_idx]):
                if isinstance(item, torch.Tensor):
                    batch.flatten_states[batch_idx][idx] = item.detach().requires_grad_(item.requires_grad)

    batch.forward_events[batch_idx] = [torch.cuda.Event()] # type: ignore[reportArgumentType]
    batch.flatten_states[batch_idx] = async_d2h(
        device.compute_stream, device.downstream, batch.forward_events[batch_idx], batch.flatten_states[batch_idx], context.enable_grad
    )
    if batch_idx == context.num_microbatches - 1:
        for layer_id in layer_ids:
            free_layer(model.layers[layer_id])

@torch.no_grad()
def run_backward(layer_ids: range,
                 context: RoundPipeRunContext, device: 'DeviceManager') -> None:
    if not torch.autograd._is_checkpoint_valid():
        raise RuntimeError(
            "The behavior of RoundPipe is consistent with "
            "torch.utils.checkpoint with use_reentrant=True. It is incompatible"
            " with .grad() or passing an `inputs` parameter to .backward()."
        )
    model = context.model
    batch_idx = context.microbatch_id
    if batch_idx == 0:
        for layer_id in layer_ids:
            upload_layer(model.layers[layer_id], device.upstream, device.compute_stream, True)
    flatten_inputs_gpu = async_h2d(
        device.compute_stream, device.upstream, [], context.flatten_inputs[layer_ids[0]], True
    )
    device.upstream.wait_stream(device.compute_stream)
    hidden_state = tree_unflatten(flatten_inputs_gpu, context.flatten_specs[layer_ids[0]]) # type: ignore[reportArgumentType]
    
    with torch.random.fork_rng(
        devices=[device.device.index], enabled=context.preserve_rng_state, device_type='cuda'
    ):
        if context.preserve_rng_state:
            context.restore_rng_state(layer_ids[0], device)
        for layer_id in layer_ids:
            with torch.enable_grad(), torch.cuda.stream(device.compute_stream), \
                 torch.autocast('cuda', **context.device_autocast_kwargs), \
                 torch.autocast('cpu', **context.cpu_autocast_kwargs), \
                 annotate(f'{model.name}L[{layer_id}]B[{batch_idx}]Re'), \
                 model.model_timer.time_backward(layer_ids[0], device.compute_stream):
                try:
                    if layer_id == 0:
                        args, kwargs = hidden_state
                        hidden_state = model.layers[layer_id].forward(*args, **kwargs)
                    else:
                        hidden_state = model.layers[layer_id].forward(hidden_state)
                except Exception as e:
                    e.args = e.args + (f'The above error occurred in {model.name} layer {layer_id} during recomputation in backward pass.',)
                    raise
    
    flatten_outputs_gpu, _ = tree_flatten(hidden_state)
    flatten_grad_outputs_gpu = async_h2d(device.compute_stream, device.upstream, context.output_backward_events, context.grad_states)
    device.upstream.wait_stream(device.compute_stream)
    
    outputs_requires_grad = []
    outputs_grad = []
    for out, grad_out in zip(flatten_outputs_gpu, flatten_grad_outputs_gpu):
        if isinstance(out, torch.Tensor) and out.requires_grad:
            outputs_requires_grad.append(out)
            outputs_grad.append(grad_out)
    with annotate(f'{model.name}L[{layer_ids[0]}, {layer_ids[-1]}]B[{batch_idx}]Bwd'), \
         model.model_timer.time_backward(layer_ids[0], device.compute_stream):
        try:
            torch.autograd.backward(outputs_requires_grad, outputs_grad)
        except Exception as e:
            e.args = e.args + (f'The above error occurred in {model.name} layers {layer_ids} during backward pass.',)
            raise
    flatten_grad_inputs_gpu = [
        inp.grad if isinstance(inp, torch.Tensor) else None
        for inp in flatten_inputs_gpu
    ]

    context.output_backward_events = context.input_backward_events if layer_ids[0] == 0 else [torch.cuda.Event()] # type: ignore[reportAttributeAccessIssue]
    context.grad_states = async_d2h(
        device.compute_stream, device.downstream, context.output_backward_events, flatten_grad_inputs_gpu
    )
    if batch_idx == context.num_microbatches - 1:
        for layer_id in layer_ids:
            download_layer(model.layers[layer_id], device.downstream)

class RoundPipeBatchedBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, roundpipe_context: 'List[RoundPipeRunContext]',
                batch: 'Batch', tag: torch.Tensor, *all_inputs: Any) -> Any:
        for batch_idx, context in enumerate(roundpipe_context):
            context.input_backward_events = batch.backward_events[batch_idx]
            context.output_backward_events = [torch.cuda.Event()] # type: ignore[reportAttributeAccessIssue]
            batch.backward_events[batch_idx] = context.output_backward_events

        tensor_inputs = []
        ctx.tensor_indices = []
        for batch_idx, context in enumerate(roundpipe_context):
            for layer_idx, layer_input in enumerate(context.flatten_inputs):
                for i, item in enumerate(layer_input):
                    if isinstance(item, torch.Tensor):
                        tensor_inputs.append(item)
                        ctx.tensor_indices.append((batch_idx, layer_idx, i))
                        context.flatten_inputs[layer_idx][i] = None
        ctx.save_for_backward(*tensor_inputs)

        all_outputs, outputs_spec = tree_flatten(batch.flatten_states)
        ctx.outputs_spec = outputs_spec
        ctx.roundpipe_contexts = roundpipe_context
        return outputs_spec, *all_outputs

    @staticmethod
    def backward(ctx, _, *grad_outputs: Any) -> Any: # type: ignore[reportIncompatibleMethodOverride]
        from .device import get_next_device
        run_contexts: List[RoundPipeRunContext] = ctx.roundpipe_contexts
        grad_states = tree_unflatten(grad_outputs, ctx.outputs_spec)
        for batch_idx, context in enumerate(run_contexts):
            context.grad_states = grad_states[batch_idx]

        for idx, (batch_idx, layer_idx, item_idx) in enumerate(ctx.tensor_indices):
            run_contexts[batch_idx].flatten_inputs[layer_idx][item_idx] = ctx.saved_tensors[idx]

        for layer_ids in run_contexts[0].execute_plan.bwd_plan:
            device = get_next_device()
            device.launch_backward(layer_ids, run_contexts)

        grad_inputs = [item for context in run_contexts
                       for item in context.grad_states]
        for context in run_contexts:
            del context.grad_states
        for batch_idx, layer_idx, item_idx in ctx.tensor_indices:
            run_contexts[batch_idx].flatten_inputs[layer_idx][item_idx] = None
        return (None, None, None, *grad_inputs)

class RoundPipeMicrobatchBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, roundpipe_context: RoundPipeRunContext,
                batch: 'Batch', tag: torch.Tensor, *all_inputs: Any) -> Any:
        roundpipe_context.input_backward_events = batch.backward_events[roundpipe_context.microbatch_id]
        roundpipe_context.output_backward_events = [torch.cuda.Event()] # type: ignore[reportAttributeAccessIssue]
        batch.backward_events[roundpipe_context.microbatch_id] = roundpipe_context.output_backward_events

        tensor_inputs = []
        ctx.tensor_indices = []
        for layer_idx, layer_input in enumerate(roundpipe_context.flatten_inputs):
            for i, item in enumerate(layer_input):
                if isinstance(item, torch.Tensor):
                    tensor_inputs.append(item)
                    ctx.tensor_indices.append((layer_idx, i))
                    roundpipe_context.flatten_inputs[layer_idx][i] = None
        ctx.save_for_backward(*tensor_inputs)

        ctx.output_len = len(batch.flatten_states[roundpipe_context.microbatch_id])
        ctx.output_require_grad_idx = []
        output_require_grad = []
        for idx, item in enumerate(batch.flatten_states[roundpipe_context.microbatch_id]):
            if isinstance(item, torch.Tensor) and item.requires_grad:
                ctx.output_require_grad_idx.append(idx)
                output_require_grad.append(item)

        ctx.roundpipe_context = roundpipe_context
        return tag, ctx.output_require_grad_idx, *output_require_grad

    @staticmethod
    def backward(ctx, device_id: torch.Tensor, _, *grad_outputs: Any) -> Any: # type: ignore[reportIncompatibleMethodOverride]
        from .device import get_next_device, device_list
        context: RoundPipeRunContext = ctx.roundpipe_context
        context.grad_states = [None] * ctx.output_len
        for idx, grad_out in zip(ctx.output_require_grad_idx, grad_outputs):
            context.grad_states[idx] = grad_out

        for idx, (layer_idx, item_idx) in enumerate(ctx.tensor_indices):
            context.flatten_inputs[layer_idx][item_idx] = ctx.saved_tensors[idx]

        layer_ids = context.execute_plan.bwd_plan[0]
        if context.microbatch_id == 0:
            device = get_next_device()
            device_id = torch.tensor(device.id, dtype=torch.float32)
        else:
            device = device_list[int(device_id.item())]
        run_backward(layer_ids, context, device)

        grad_inputs = context.grad_states
        del context.grad_states
        for layer_idx, item_idx in ctx.tensor_indices:
            context.flatten_inputs[layer_idx][item_idx] = None
        return (None, None, device_id, *grad_inputs)
