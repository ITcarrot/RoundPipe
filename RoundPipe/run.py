from typing import * # type: ignore[reportWildcardImportFromLibrary]
import contextlib

import torch
from torch.utils.checkpoint import _infer_device_type, _get_autocast_kwargs, _get_device_module, get_device_states, set_device_states
from torch.utils._pytree import tree_flatten, tree_unflatten

from RoundPipe.transfer import async_d2h, async_h2d, upload_layer, download_layer, free_layer

if TYPE_CHECKING:
    from RoundPipe.batch import Batch
    from RoundPipe.device import DeviceManager
    from RoundPipe.RoundPipe import RoundPipe
    from RoundPipe.RunConfig import FullRoundPipeRunConfig
    from torch.utils._pytree import TreeSpec
    
    class CheckpointRunContext:
        batch_idx: int
        num_microbatch: int
        layers: List[torch.nn.Module]
        layer_ids: Iterable[int]
        device: DeviceManager
        input_backward_events: Sequence[Optional[torch.cuda.Event]]
        flatten_spec: TreeSpec
        inputs: List[Any]
        tensor_indices: List[int]
        save_for_backward: Callable[[torch.Tensor], None]
        saved_tensors: List[torch.Tensor]
        preserve_rng_state: bool
        device_type: str
        device_autocast_kwargs: Optional[dict]
        cpu_autocast_kwargs: dict
        fwd_cpu_state: torch.Tensor
        had_device_in_fwd: bool
        fwd_devices: List[int]
        fwd_device_states: List[torch.Tensor]
        output_backward_event: List[torch.cuda.Event]

class CheckpointRun(torch.autograd.Function):
    @staticmethod
    def forward(ctx: 'CheckpointRunContext',
                device: 'DeviceManager',
                model: 'RoundPipe',
                run_config: 'FullRoundPipeRunConfig',
                batch: 'Batch',
                layer_ids: Iterable[int], batch_idx: int,
                device_order_tag: torch.Tensor,
                *flatten_inputs_cpu: Any):
        ctx.batch_idx = batch_idx
        ctx.num_microbatch = run_config.num_microbatch
        ctx.layers = model.layers
        ctx.layer_ids = layer_ids
        ctx.device = device
        input_forward_events = batch.transfer_events[batch_idx][0]
        ctx.input_backward_events = batch.transfer_events[batch_idx][1]
        ctx.flatten_spec = batch.flatten_specs[batch_idx]
        # === Copy from torch.utils.checkpoint: CheckpointFunction: forward ===
        ctx.inputs = []
        tensor_inputs = []
        ctx.tensor_indices = []
        for i, arg in enumerate(flatten_inputs_cpu):
            if isinstance(arg, torch.Tensor):
                tensor_inputs.append(arg)
                ctx.tensor_indices.append(i)
                ctx.inputs.append(None)
            else:
                ctx.inputs.append(arg)
        ctx.save_for_backward(*tensor_inputs)
        # === End copy ===
        if batch_idx == 0:
            for idx in ctx.layer_ids:
                with model.model_timer.time(idx, device.upstream):
                    upload_layer(ctx.layers[idx], device.upstream, device.compute_stream, False)
        flatten_inputs_gpu = async_h2d(device.compute_stream, device.upstream, input_forward_events, flatten_inputs_cpu)
        device.upstream.wait_stream(device.compute_stream)
        # === Copy from torch.utils.checkpoint: CheckpointFunction: forward ===
        ctx.preserve_rng_state = run_config.preserve_rng_state
        ctx.device_type = _infer_device_type(*flatten_inputs_gpu)
        ctx.device_autocast_kwargs, ctx.cpu_autocast_kwargs = _get_autocast_kwargs(
            ctx.device_type
        )
        if ctx.preserve_rng_state:
            ctx.fwd_cpu_state = torch.get_rng_state()
            ctx.had_device_in_fwd = False
            device_module = _get_device_module(ctx.device_type)
            if getattr(device_module, "_initialized", False):
                ctx.had_device_in_fwd = True
                ctx.fwd_devices, ctx.fwd_device_states = get_device_states(*flatten_inputs_gpu)
        # === End copy ===
        hidden_state = tree_unflatten(flatten_inputs_gpu, ctx.flatten_spec)
        with torch.no_grad(), torch.cuda.stream(device.compute_stream):
            for idx in ctx.layer_ids:
                with model.model_timer.time(idx):
                    if idx == 0:
                        args, kwargs = hidden_state
                        hidden_state = ctx.layers[idx].forward(*args, **kwargs)
                    else:
                        hidden_state = ctx.layers[idx].forward(hidden_state)
        flatten_outputs_gpu, flatten_spec = tree_flatten(hidden_state)

        output_forward_event: List[torch.cuda.Event] = [torch.cuda.Event()] # type: ignore[reportAssignmentType]
        ctx.output_backward_event = [torch.cuda.Event()] # type: ignore[reportAttributeAccessIssue]
        flatten_outputs_cpu = async_d2h(device.compute_stream, device.downstream, output_forward_event, flatten_outputs_gpu)
        if batch_idx == ctx.num_microbatch - 1:
            for idx in ctx.layer_ids:
                free_layer(ctx.layers[idx])

        batch.flatten_specs[batch_idx] = flatten_spec
        batch.transfer_events[batch_idx] = (output_forward_event, ctx.output_backward_event)
        return device_order_tag.detach(), *flatten_outputs_cpu
    
    @staticmethod
    def backward(ctx: 'CheckpointRunContext', device_order_tag: torch.Tensor, # type: ignore[override]
                 *flatten_grad_outputs_cpu: torch.Tensor):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "The behavior of pipeMT checkpoint is consistent with "
                "torch.utils.checkpoint with use_reentrant=True. It is incompatible"
                " with .grad() or passing an `inputs` parameter to .backward()."
            )
        batch_idx = ctx.batch_idx
        device = ctx.device
        flatten_inputs_cpu = list(ctx.inputs)
        for i, idx in enumerate(ctx.tensor_indices):
            flatten_inputs_cpu[idx] = ctx.saved_tensors[i]
        
        if batch_idx == ctx.num_microbatch - 1:
            for idx in ctx.layer_ids:
                upload_layer(ctx.layers[idx], device.upstream, device.compute_stream, True)
        flatten_inputs_gpu = async_h2d(device.compute_stream, device.upstream, [], flatten_inputs_cpu)
        device.upstream.wait_stream(device.compute_stream)
        for idx, arg in enumerate(flatten_inputs_gpu):
            if isinstance(arg, torch.Tensor):
                arg.requires_grad_(flatten_inputs_cpu[idx].requires_grad)
        # === Copy from torch.utils.checkpoint: CheckpointFunction: forward ===
        rng_devices = []
        if ctx.preserve_rng_state and ctx.had_device_in_fwd:
            rng_devices = ctx.fwd_devices
        with torch.random.fork_rng(
            devices=rng_devices, enabled=ctx.preserve_rng_state, device_type=ctx.device_type
        ):
            if ctx.preserve_rng_state:
                torch.set_rng_state(ctx.fwd_cpu_state)
                if ctx.had_device_in_fwd:
                    set_device_states(ctx.fwd_devices, ctx.fwd_device_states, device_type=ctx.device_type)
            device_autocast_ctx = torch.autocast(
                device_type=ctx.device_type, **ctx.device_autocast_kwargs
            ) if torch.amp.is_autocast_available(ctx.device_type) else contextlib.nullcontext()
            with torch.enable_grad(), device_autocast_ctx, torch.amp.autocast("cpu", **ctx.cpu_autocast_kwargs):  # type: ignore[attr-defined]
        # === End copy ===
                hidden_state = tree_unflatten(flatten_inputs_gpu, ctx.flatten_spec)
                with torch.cuda.stream(device.compute_stream):
                    for idx in ctx.layer_ids:
                        if idx == 0:
                            args, kwargs = hidden_state
                            hidden_state = ctx.layers[idx].forward(*args, **kwargs)
                        else:
                            hidden_state = ctx.layers[idx].forward(hidden_state)
                flatten_outputs_gpu, _ = tree_flatten(hidden_state)
        
        flatten_grad_outputs_gpu = async_h2d(device.compute_stream, device.upstream, ctx.output_backward_event, flatten_grad_outputs_cpu)
        device.upstream.wait_stream(device.compute_stream)
        
        outputs_require_grad = []
        outputs_grad = []
        for idx, tensor in enumerate(flatten_outputs_gpu):
            if isinstance(tensor, torch.Tensor) and tensor.requires_grad:
                outputs_require_grad.append(tensor)
                outputs_grad.append(flatten_grad_outputs_gpu[idx])
        torch.autograd.backward(outputs_require_grad, outputs_grad)
        flatten_grad_inputs_gpu = [
            inp.grad if isinstance(inp, torch.Tensor) else None
            for inp in flatten_inputs_gpu
        ]
        
        flatten_grad_inputs_cpu = async_d2h(device.compute_stream, device.downstream, ctx.input_backward_events, flatten_grad_inputs_gpu)
        if batch_idx == 0:
            for idx in ctx.layer_ids:
                download_layer(ctx.layers[idx], device.downstream)
        
        return None, None, None, None, None, None, device_order_tag, *flatten_grad_inputs_cpu
