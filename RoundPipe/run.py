"""Core runtime helpers for executing RoundPipe forward/backward passes."""

from beartype.typing import * # type: ignore[reportWildcardImportFromLibrary]
import traceback

import torch
from torch.utils.checkpoint import _get_autocast_kwargs
from torch.utils._pytree import tree_unflatten, tree_flatten, TreeSpec

from .batch import Batch
from .profile import annotate
from .scheduler import ModelExecutePlan
from .threads import thread_exception_print_lock
from .transfer import async_h2d, async_d2h, upload_layers, download_layer, free_layer

if TYPE_CHECKING:
    from .device import DeviceManager
    from .RoundPipe import RoundPipe
else:
    from typing_extensions import TypeAliasType
    DeviceManager = TypeAliasType('DeviceManager', 'RoundPipe.device.DeviceManager')
    RoundPipe = TypeAliasType('RoundPipe', 'RoundPipe.RoundPipe')

class RoundPipeRunContext:
    """Per-microbatch state shared between forward and backward passes.

    Attributes:
        model: The running ``RoundPipe`` instance.
        execute_plan: Plan describing fwd/bwd ordering.
        enable_grad: Whether to store data for backward pass.
        microbatch_id: Index of the microbatch this context tracks.
        num_microbatches: Total number of microbatches scheduled.
        preserve_rng_state: Whether to snapshot/restore RNG streams.
        device_autocast_kwargs: Settings applied to CUDA autocast.
        cpu_autocast_kwargs: Settings applied to CPU autocast.
        flatten_inputs: Cached flattened inputs for recompute.
        flatten_specs: Tree specs that rebuild flattened inputs.
        device_rng_states: Saved CUDA RNG states per layer when requested.
        cpu_rng_states: Saved CPU RNG states per layer when requested.
        input_backward_events: Events to record when gradients are ready.
        output_backward_events: Events to wait on before backward compute.
        grad_states: Gradient tensors for current backward pass.
    """
    model: 'RoundPipe'
    execute_plan: ModelExecutePlan
    enable_grad: bool
    microbatch_id: int
    num_microbatches: int
    preserve_rng_state: bool
    device_autocast_kwargs: dict
    cpu_autocast_kwargs: dict
    flatten_inputs: List[List[Any]]
    flatten_specs: List[Optional[TreeSpec]]
    device_rng_states: List[Optional[torch.Tensor]]
    cpu_rng_states: List[Optional[torch.Tensor]]
    input_backward_events: Sequence[torch.cuda.Event]
    output_backward_events: Sequence[torch.cuda.Event]
    grad_states: List[Any]

    def __init__(self, model: 'RoundPipe',
                 execute_plan: ModelExecutePlan,
                 enable_grad: bool,
                 microbatch_id: int,
                 num_microbatches: int,
                 preserve_rng_state: bool) -> None:
        """Initialize per-microbatch caches and RNG bookkeeping.

        Args:
            model: The running ``RoundPipe`` instance.
            execute_plan: Plan describing fwd/bwd ordering.
            enable_grad: Whether to store data for backward pass.
            microbatch_id: Microbatch index for this context.
            num_microbatches: Total number of microbatches in the batch.
            preserve_rng_state: Whether to snapshot RNG for recomputation.
        """
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

    def save_input(self, layer_id: int, batch: Batch, device: 'DeviceManager') -> None:
        """Stash flattened inputs (and optionally RNG) for backward recompute.
        
        If gradients are not enabled or the layer is not the first layer of a 
        backward stage, this is a no-op.

        Args:
            layer_id: Layer index whose inputs should be cached.
            batch: Batch holding the flattened tensors to snapshot.
            device: Device manager whose streams guard the transfer.
        """
        if not (self.enable_grad and self.execute_plan.backward_need_input(layer_id)):
            return

        self.flatten_inputs[layer_id] = batch.flatten_states[self.microbatch_id]
        self.flatten_specs[layer_id] = batch.flatten_specs[self.microbatch_id]

        device.wait_stream(device.downstream, device.compute_stream)
        with torch.cuda.stream(device.downstream):
            for idx, item in enumerate(self.flatten_inputs[layer_id]):
                if isinstance(item, torch.Tensor) and item.device != torch.device('cpu'):
                    device.mem_manager.record_stream(item, device.compute_stream, device.downstream)
                    self.flatten_inputs[layer_id][idx] = item.to('cpu', non_blocking=True)

        if self.preserve_rng_state:
            with device.device:
                self.device_rng_states[layer_id] = torch.cuda.get_rng_state()
            self.cpu_rng_states[layer_id] = torch.get_rng_state()

    def restore_rng_state(self, layer_id: int, device: 'DeviceManager') -> None:
        """Restore the RNG snapshot captured during `save_input`.

        Args:
            layer_id: Layer index whose RNG states should be restored.
            device: Device manager that owns the CUDA stream.

        Raises:
            AssertionError: If RNG state was not captured as expected.
        """
        if not self.preserve_rng_state:
            return

        device_states: torch.Tensor = self.device_rng_states[layer_id] # type: ignore[reportAssignmentType]
        cpu_states: torch.Tensor = self.cpu_rng_states[layer_id] # type: ignore[reportAssignmentType]
        assert device_states is not None or cpu_states is not None, "RNG states were not saved properly."

        with device.device:
            torch.cuda.set_rng_state(device_states)
        torch.set_rng_state(cpu_states)

@torch.no_grad()
def run_forward(layer_group_id: int, batch: Batch,
                context: RoundPipeRunContext, device: 'DeviceManager') -> None:
    """Upload layers, execute forward compute, and copy outputs back to host.

    Args:
        layer_group_id: Index of the layer group being executed.
        batch: Batch object containing flattened microbatch inputs.
        context: Microbatch-specific execution context.
        device: Device manager that streams data and compute.

    Returns:
        Results are written into ``batch.flatten_states`` in-place.
    """
    model = context.model
    batch_idx = context.microbatch_id
    layer_ids = context.execute_plan.fwd_plan[layer_group_id]
    grad_context = torch.enable_grad() if context.enable_grad else torch.no_grad()
    if batch_idx == 0:
        upload_layers([model.layers[layer_id] for layer_id in layer_ids], device, False)
    context.execute_plan.forward_wait_for(layer_group_id - 1)
    for layer_id in layer_ids:
        context.save_input(layer_id, batch, device)
        if layer_id == layer_ids[0]:
            batch.flatten_states[batch_idx] = async_h2d(
                device, batch.forward_events[batch_idx], batch.flatten_states[batch_idx], context.enable_grad
            )
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
            except Exception:
                with thread_exception_print_lock:
                    traceback.print_exc()
                    print(f'The above error occurred in {model.name} layer {layer_id} during forward pass.')
                raise SystemExit(1)
        batch.flatten_states[batch_idx], batch.flatten_specs[batch_idx] = tree_flatten(hidden_state)

        if context.enable_grad:
            for idx, item in enumerate(batch.flatten_states[batch_idx]):
                if isinstance(item, torch.Tensor):
                    batch.flatten_states[batch_idx][idx] = item.detach().requires_grad_(item.requires_grad)

    batch.forward_events[batch_idx] = [torch.cuda.Event()] # type: ignore[reportArgumentType]
    batch.flatten_states[batch_idx] = async_d2h(
        device, batch.forward_events[batch_idx], batch.flatten_states[batch_idx], context.enable_grad
    )
    device.mem_manager.flush()
    if batch_idx == context.num_microbatches - 1:
        for layer_id in layer_ids:
            free_layer(model.layers[layer_id], device) 
    context.execute_plan.forward_notify(layer_group_id)

@torch.no_grad()
def run_backward(layer_group_id: int,
                 context: RoundPipeRunContext, device: 'DeviceManager') -> None:
    """Recompute saved inputs, propagate gradients, and ship grads to CPU.

    Args:
        layer_group_id: Index of the backward layer group to execute.
        context: Microbatch-specific execution context.
        device: Device manager providing streams for recompute/backward.

    Returns:
        Results are written into ``context.grad_states`` in-place.

    Raises:
        RuntimeError: If checkpoint semantics are violated by the caller.
    """
    if not torch.autograd._is_checkpoint_valid():
        raise RuntimeError(
            "The behavior of RoundPipe is consistent with "
            "torch.utils.checkpoint with use_reentrant=True. It is incompatible"
            " with .grad() or passing an `inputs` parameter to .backward()."
        )
    model = context.model
    batch_idx = context.microbatch_id
    layer_ids = context.execute_plan.bwd_plan[layer_group_id]
    if batch_idx == 0:
        upload_layers([model.layers[layer_id] for layer_id in layer_ids], device, True)
    flatten_inputs_gpu = async_h2d(
        device, [], context.flatten_inputs[layer_ids[0]], True
    )
    context.flatten_inputs[layer_ids[0]].clear() # Free CPU memory
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
                except Exception:
                    with thread_exception_print_lock:
                        traceback.print_exc()
                        print(f'The above error occurred in {model.name} layer {layer_id} during recomputation in backward pass.')
                    raise SystemExit(1)
    
    flatten_outputs_gpu, _ = tree_flatten(hidden_state)
    context.execute_plan.backward_wait_for(layer_group_id - 1)
    flatten_grad_outputs_gpu = async_h2d(device, context.output_backward_events, context.grad_states)
    
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
        except Exception:
            with thread_exception_print_lock:
                traceback.print_exc()
                print(f'The above error occurred in {model.name} layers {layer_ids} during backward pass.')
            raise SystemExit(1)
    flatten_grad_inputs_gpu = [
        inp.grad if isinstance(inp, torch.Tensor) else None
        for inp in flatten_inputs_gpu
    ]

    context.output_backward_events = context.input_backward_events if layer_ids[0] == 0 else [torch.cuda.Event()] # type: ignore[reportAttributeAccessIssue]
    context.grad_states = async_d2h(
        device, context.output_backward_events, flatten_grad_inputs_gpu
    )
    device.mem_manager.flush()
    if batch_idx == context.num_microbatches - 1:
        for layer_id in layer_ids:
            download_layer(model.layers[layer_id], device)
            download_finish_event: torch.cuda.Event = torch.cuda.Event() # type: ignore[reportAssignmentType]
            download_finish_event.record(device.downstream)
            model.layer_gradient_ready_events[layer_id] = download_finish_event
    context.execute_plan.backward_notify(layer_group_id)

@torch.no_grad()
def run_forward_backward(batch: Batch, context: RoundPipeRunContext,
                         loss_fn: Callable[[Any, Any], Union[Sequence[torch.Tensor], torch.Tensor]],
                         device: 'DeviceManager') -> None:
    """Execute a fused forward/backward pass for training workloads.

    Args:
        batch: Batch object containing flattened activations and labels.
        context: Microbatch-specific execution context.
        loss_fn: Callable that calculates loss from outputs and labels.
        device: Device manager providing compute/transfer streams.

    Returns:
        Results are written into ``batch.flatten_states``, ``batch.loss_list``
            and ``context.grad_states`` in-place.
    """
    model = context.model
    batch_idx = context.microbatch_id
    layer_ids = context.execute_plan.bwd_plan[0]
    if batch_idx == 0:
        upload_layers([model.layers[layer_id] for layer_id in layer_ids], device, True)
    flatten_inputs_gpu = async_h2d(
        device, batch.forward_events[batch_idx], batch.flatten_states[batch_idx], True
    )
    batch.flatten_states[batch_idx].clear() # Free CPU memory
    hidden_state = tree_unflatten(flatten_inputs_gpu, batch.flatten_specs[batch_idx]) # type: ignore[reportArgumentType]

    for layer_id in layer_ids:
        with torch.enable_grad(), torch.cuda.stream(device.compute_stream), \
             torch.autocast('cuda', **context.device_autocast_kwargs), \
             torch.autocast('cpu', **context.cpu_autocast_kwargs), \
             annotate(f'{model.name}L[{layer_id}]B[{batch_idx}]Fwd'), \
             model.model_timer.time_backward(layer_ids[0], device.compute_stream):
            try:
                if layer_id == 0:
                    args, kwargs = hidden_state
                    hidden_state = model.layers[layer_id].forward(*args, **kwargs)
                else:
                    hidden_state = model.layers[layer_id].forward(hidden_state)
            except Exception:
                with thread_exception_print_lock:
                    traceback.print_exc()
                    print(f'The above error occurred in {model.name} layer {layer_id} during forward pass.')
                raise SystemExit(1)

    batch.flatten_states[batch_idx], batch.flatten_specs[batch_idx] = tree_flatten(hidden_state)
    batch.forward_events[batch_idx] = [torch.cuda.Event()] # type: ignore[reportArgumentType]
    batch.flatten_states[batch_idx] = async_d2h(
        device, batch.forward_events[batch_idx], batch.flatten_states[batch_idx], False
    )

    flatten_label, flatten_label_spec = tree_flatten(batch.label_list[batch_idx])
    flatten_label_gpu = async_h2d(
        device, [], flatten_label, False
    )
    label_gpu = tree_unflatten(flatten_label_gpu, flatten_label_spec)
    with torch.enable_grad(), torch.cuda.stream(device.compute_stream), \
         torch.autocast('cuda', **context.device_autocast_kwargs), \
         torch.autocast('cpu', **context.cpu_autocast_kwargs):
        loss = loss_fn(hidden_state, label_gpu)
    if isinstance(loss, torch.Tensor):
        batch.loss_list[batch_idx] = loss.detach()
    else:
        batch.loss_list[batch_idx] = [l.detach() for l in loss]

    with annotate(f'{model.name}L[{layer_ids[0]}, {layer_ids[-1]}]B[{batch_idx}]Bwd'), \
         model.model_timer.time_backward(layer_ids[0], device.compute_stream):
        try:
            torch.autograd.backward(loss)
        except Exception:
            with thread_exception_print_lock:
                traceback.print_exc()
                print(f'The above error occurred in {model.name} layers {layer_ids} during backward pass.')
            raise SystemExit(1)
    flatten_grad_inputs_gpu = [
        inp.grad if isinstance(inp, torch.Tensor) else None
        for inp in flatten_inputs_gpu
    ]

    context.output_backward_events = context.input_backward_events if layer_ids[0] == 0 else [torch.cuda.Event()] # type: ignore[reportAttributeAccessIssue]
    context.grad_states = async_d2h(
        device, context.output_backward_events, flatten_grad_inputs_gpu
    )
    device.mem_manager.flush()
    if batch_idx == context.num_microbatches - 1:
        for layer_id in layer_ids:
            download_layer(model.layers[layer_id], device)
            download_finish_event: torch.cuda.Event = torch.cuda.Event() # type: ignore[reportAssignmentType]
            download_finish_event.record(device.downstream)
            model.layer_gradient_ready_events[layer_id] = download_finish_event
    context.execute_plan.backward_notify(0)

class RoundPipeBatchedBackward(torch.autograd.Function):
    """Autograd node that launches backward passes for all microbatches."""
    @staticmethod
    def forward(ctx: Any, roundpipe_context: List[RoundPipeRunContext],
                batch: Batch, tag: torch.Tensor, *all_inputs: Any) -> Any:
        """Prepare shared backward state for all microbatches.

        Args:
            ctx: Autograd context (provided by PyTorch).
            roundpipe_context: Execution contexts per microbatch.
            batch: Batch object carrying flattened outputs and events.
            tag: Gradient anchor tensor to ensure output requires grad.
            *all_inputs: Flattened tensors produced during forward.

        Returns:
            Tuple with gradient indices followed by tensors requiring grad.
        """
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

        ctx.output_len = [len(flatten_outputs) for flatten_outputs in batch.flatten_states]
        ctx.output_require_grad_idx = []
        output_require_grad = []
        for batch_idx, flatten_outputs in enumerate(batch.flatten_states):
            for idx, item in enumerate(flatten_outputs):
                if isinstance(item, torch.Tensor) and item.requires_grad:
                    ctx.output_require_grad_idx.append((batch_idx, idx))
                    output_require_grad.append(item)

        ctx.roundpipe_contexts = roundpipe_context
        ctx.launched_backward = False
        return ctx.output_require_grad_idx, *output_require_grad

    @staticmethod
    def backward(ctx: Any, _, *grad_outputs: Any) -> Any: # type: ignore[reportIncompatibleMethodOverride]
        """Launch backward passes for all microbatches from saved state.

        Args:
            ctx: Autograd context populated in ``forward``.
            *grad_outputs: Gradients for the outputs that required grad.

        Returns:
            Gradients mapping back to ``all_inputs`` in the forward path.

        Raises:
            RuntimeError: If a double backward is attempted.
        """
        if ctx.launched_backward:
            raise RuntimeError("RoundPipe do not support double backward.")
        ctx.launched_backward = True

        from .device import get_next_device
        run_contexts: List[RoundPipeRunContext] = ctx.roundpipe_contexts
        for context, output_len in zip(run_contexts, ctx.output_len):
            context.grad_states = [None] * output_len
        for (batch_idx, idx), grad_out in zip(ctx.output_require_grad_idx, grad_outputs):
            run_contexts[batch_idx].grad_states[idx] = grad_out

        for idx, (batch_idx, layer_idx, item_idx) in enumerate(ctx.tensor_indices):
            run_contexts[batch_idx].flatten_inputs[layer_idx][item_idx] = ctx.saved_tensors[idx]

        for layer_group_id in range(len(run_contexts[0].execute_plan.bwd_plan)):
            device = get_next_device()
            device.launch_backward(layer_group_id, run_contexts)
        run_contexts[0].execute_plan.backward_wait_complete(len(run_contexts))

        grad_inputs = [item for context in run_contexts
                       for item in context.grad_states]
        for context in run_contexts:
            del context.grad_states
        return (None, None, None, *grad_inputs)

class RoundPipeMicrobatchBackward(torch.autograd.Function):
    """Autograd node that launches backward pass for a single microbatch."""
    @staticmethod
    def forward(ctx: Any, roundpipe_context: RoundPipeRunContext,
                batch: Batch, tag: torch.Tensor, *all_inputs: Any) -> Any:
        """Prepare backward state for a single microbatch.

        Args:
            ctx: Autograd context (provided by PyTorch).
            roundpipe_context: Execution context tied to one microbatch.
            batch: Batch with flattened outputs + events for this microbatch.
            tag: Gradient anchor to create dependency between microbatches
                thus ensuring correct backward order.
            *all_inputs: Flattened arguments provided during forward.

        Returns:
            Tuple ``(tag, grad_indices, *output_tensors)`` consumed later.
        """
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
        ctx.launched_backward = False
        return tag, ctx.output_require_grad_idx, *output_require_grad

    @staticmethod
    def backward(ctx: Any, device_id: torch.Tensor, _, *grad_outputs: Any) -> Any: # type: ignore[reportIncompatibleMethodOverride]
        """Kick off backward recomputation for a single microbatch.

        Args:
            ctx: Autograd context populated in ``forward``.
            device_id: Tensor encoding which CUDA device to reuse.
            *grad_outputs: Gradients for the tracked outputs.

        Returns:
            Gradients corresponding to the saved forward inputs.

        Raises:
            RuntimeError: If a double backward is attempted.
        """
        if ctx.launched_backward:
            raise RuntimeError("RoundPipe do not support double backward.")
        ctx.launched_backward = True

        from .device import get_next_device, device_list
        context: RoundPipeRunContext = ctx.roundpipe_context
        context.grad_states = [None] * ctx.output_len
        for idx, grad_out in zip(ctx.output_require_grad_idx, grad_outputs):
            context.grad_states[idx] = grad_out

        for idx, (layer_idx, item_idx) in enumerate(ctx.tensor_indices):
            context.flatten_inputs[layer_idx][item_idx] = ctx.saved_tensors[idx]

        if context.microbatch_id == 0:
            device = get_next_device()
            device_id = torch.tensor(device.id, dtype=torch.float32)
        else:
            device = device_list[int(device_id.item())]
        run_backward(0, context, device)

        grad_inputs = context.grad_states
        del context.grad_states
        return (None, None, device_id, *grad_inputs)

class RoundPipeInputBackward(torch.autograd.Function):
    """Autograd node that reconnects RoundPipe gradients to user inputs."""
    @staticmethod
    def forward(ctx: Any, roundpipe_context: List[RoundPipeRunContext], *all_inputs: Any) -> Any:
        """Anchor upstream gradients spanning multiple microbatches.

        Args:
            ctx: Autograd context provided by PyTorch.
            roundpipe_context: List of contexts participating in training.
            *all_inputs: Flattened inputs to track gradients for.

        Returns:
            Dummy scalar tensor that participates in autograd graphs.
        """
        ctx.roundpipe_contexts = roundpipe_context
        return torch.tensor(0.0)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any: # type: ignore[reportIncompatibleMethodOverride]
        """Return gradients captured in each ``RoundPipeRunContext``.

        Args:
            ctx: Autograd context populated during ``forward``.
            *grad_outputs: Gradients w.r.t. the dummy scalar (unused).

        Returns:
            Tuple matching ``(None, *flattened_input_grads)`` so that upstream
                PyTorch graphs receive gradients for their original inputs.
        """
        run_contexts: List[RoundPipeRunContext] = ctx.roundpipe_contexts
        grad_inputs = [item for context in run_contexts
                       for item in context.grad_states]
        for context in run_contexts:
            del context.grad_states
        return (None, *grad_inputs)
