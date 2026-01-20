"""Core runtime helpers for executing RoundPipe forward/backward passes."""

from typing_extensions import *
import traceback
import functools
import copy
import sys

import torch
from torch.utils.checkpoint import _get_autocast_kwargs
from torch.utils._pytree import tree_unflatten, tree_flatten, TreeSpec

from .context import ForwardCtx, RecomputeCtx
from .profile import annotate
from .threads import thread_exception_print_lock
from .transfer import async_h2d, async_d2h, upload_layers, download_layer

if TYPE_CHECKING:
    from .batch import Batch
    from .device import DeviceManager
    from .roundpipe import RoundPipe
    from .scheduler import ModelTracker
else:
    Batch = TypeAliasType("Batch", "roundpipe.batch.Batch")
    DeviceManager = TypeAliasType("DeviceManager", "roundpipe.device.DeviceManager")
    RoundPipe = TypeAliasType("RoundPipe", "roundpipe.roundpipe.RoundPipe")
    ModelTracker = TypeAliasType("ModelTracker", "roundpipe.scheduler.ModelTracker")


class RoundPipeRunContext:
    """Per-microbatch state shared between forward and backward passes.

    Attributes:
        model: The running ``RoundPipe`` instance.
        gpu_fwd_layers: GPU-resident layers for current forward batch.
        gpu_bwd_layers: GPU-resident layers for current backward batch.
        tracker: Tracker describing fwd/bwd ordering across layers.
        enable_grad: Whether to store data for backward pass.
        microbatch_id: Index of the microbatch this context tracks.
        num_microbatches: Total number of microbatches scheduled.
        preserve_rng_state: Whether to snapshot/restore RNG streams.
        device_autocast_kwargs: Settings applied to CUDA autocast.
        cpu_autocast_kwargs: Settings applied to CPU autocast.
        flatten_inputs: Cached flattened inputs for recompute.
        flatten_specs: Tree specs that rebuild flattened inputs.
        recompute_data: Saved data for backward recompute.
        recompute_data_specs: Tree specs that rebuild recompute data.
        saved_buffers: Buffers saved for recomputation. Only applied at
            microbatch 0.
        download_event: Events to signal when saved data download is done.
        device_rng_states: Saved CUDA RNG states per layer when requested.
        cpu_rng_states: Saved CPU RNG states per layer when requested.
        input_backward_events: Events to record when gradients are ready.
        output_backward_events: Events to wait on before backward compute.
        grad_states: Gradient tensors for current backward pass.
    """

    model: "RoundPipe"
    gpu_fwd_layers: List[torch.nn.Module]
    gpu_bwd_layers: List[torch.nn.Module]
    tracker: "ModelTracker"
    enable_grad: bool
    microbatch_id: int
    num_microbatches: int
    preserve_rng_state: bool
    device_autocast_kwargs: dict
    cpu_autocast_kwargs: dict
    flatten_inputs: List[List[Any]]
    flatten_specs: List[Optional[TreeSpec]]
    recompute_data: List[List[Any]]
    recompute_data_specs: List[Optional[TreeSpec]]
    saved_buffers: Dict[torch.Tensor, torch.Tensor]
    download_event: List[torch.cuda.Event]
    device_rng_states: List[Optional[torch.Tensor]]
    cpu_rng_states: List[Optional[torch.Tensor]]
    input_backward_events: Sequence[torch.cuda.Event]
    output_backward_events: Sequence[torch.cuda.Event]
    grad_states: List[Optional[torch.Tensor]]

    def __init__(
        self,
        model: "RoundPipe",
        gpu_fwd_layers: List[torch.nn.Module],
        gpu_bwd_layers: List[torch.nn.Module],
        tracker: "ModelTracker",
        enable_grad: bool,
        microbatch_id: int,
        num_microbatches: int,
        preserve_rng_state: bool,
    ) -> None:
        """Initialize per-microbatch caches and RNG bookkeeping.

        Args:
            model: The running ``RoundPipe`` instance.
            gpu_fwd_layers: A list for sharing GPU-resident layers
                among forward microbatches.
            gpu_bwd_layers: A list for sharing GPU-resident layers
                among backward microbatches.
            tracker: Tracker describing fwd/bwd ordering across layers.
            enable_grad: Whether to store data for backward pass.
            microbatch_id: Microbatch index for this context.
            num_microbatches: Total number of microbatches in the batch.
            preserve_rng_state: Whether to snapshot RNG for recomputation.
        """
        self.model = model
        self.gpu_fwd_layers = gpu_fwd_layers
        self.gpu_bwd_layers = gpu_bwd_layers
        self.tracker = tracker
        self.enable_grad = enable_grad
        self.microbatch_id = microbatch_id
        self.num_microbatches = num_microbatches
        self.preserve_rng_state = preserve_rng_state

        device_autocast_kwargs, self.cpu_autocast_kwargs = _get_autocast_kwargs("cuda")
        assert device_autocast_kwargs is not None, "Failed to get CUDA autocast kwargs."
        self.device_autocast_kwargs = device_autocast_kwargs
        # RoundPipe moves param across GPUs; disable autocast caching.
        self.device_autocast_kwargs["cache_enabled"] = False
        self.cpu_autocast_kwargs["cache_enabled"] = False

        self.flatten_inputs = [[] for _ in range(model.num_layers)]
        self.flatten_specs = [None for _ in range(model.num_layers)]
        self.recompute_data = [[] for _ in range(model.num_layers)]
        self.recompute_data_specs = [None for _ in range(model.num_layers)]
        self.saved_buffers = {}
        if microbatch_id == 0 and enable_grad:
            for buffer in model.model.buffers():
                self.saved_buffers[buffer] = torch.empty_like(buffer, pin_memory=True)

        self.download_event = [
            cast(torch.cuda.Event, torch.cuda.Event()) for _ in range(model.num_layers)
        ]
        self.device_rng_states = [None for _ in range(model.num_layers)]
        self.cpu_rng_states = [None for _ in range(model.num_layers)]

    def save_input(
        self, layer_id: int, batch: "Batch", device: "DeviceManager"
    ) -> None:
        """Stash flattened inputs (and optionally RNG) for backward recompute.

        If gradients are not enabled or the layer is not the first layer of a
        backward stage, this is a no-op.

        Args:
            layer_id: Layer index whose inputs should be cached.
            batch: Batch holding the flattened tensors to snapshot.
            device: Device manager whose streams guard the transfer.
        """
        if not (self.enable_grad and self.tracker.backward_need_input(layer_id)):
            return

        self.flatten_inputs[layer_id] = copy.copy(
            batch.flatten_states[self.microbatch_id]
        )
        self.flatten_specs[layer_id] = batch.flatten_specs[self.microbatch_id]

        device.wait_stream(device.downstream, device.compute_stream)
        with torch.cuda.stream(device.downstream):
            for idx, item in enumerate(self.flatten_inputs[layer_id]):
                if isinstance(item, torch.Tensor):
                    if item.device != torch.device("cpu"):
                        device.mem_manager.record_stream(
                            item, device.compute_stream, device.downstream
                        )
                        self.flatten_inputs[layer_id][idx] = item.to(
                            "cpu", non_blocking=True
                        ).requires_grad_(item.requires_grad)
                else:
                    self.flatten_inputs[layer_id][idx] = copy.deepcopy(item)
            self.download_event[layer_id].record(device.downstream)

        if self.preserve_rng_state:
            with device.device:
                self.device_rng_states[layer_id] = torch.cuda.get_rng_state()
            self.cpu_rng_states[layer_id] = torch.get_rng_state()

    def save_buffer(self, layer: torch.nn.Module) -> None:
        """Save layer buffers for recomputation. Only called at microbatch 0.

        Args:
            layer: Layer whose buffers should be saved.
        """
        if self.microbatch_id != 0 or not self.enable_grad:
            return
        for buffer in layer.buffers():
            self.saved_buffers[buffer].copy_(buffer.data)

    def save_for_recompute(
        self, layer_id: int, device: "DeviceManager", *data: Any
    ) -> None:
        """Save data for backward recompute. Tensor to be saved cannot require
        gradients. This function can be called at most once from each layer.
        If forward gradients are not enabled, this is a no-op.

        Args:
            layer_id: Layer index whose data should be cached.
            device: Device manager whose streams guard the transfer.
            *data: Data to save for recomputation.
        """
        if not self.enable_grad:
            return
        assert (
            self.recompute_data_specs[layer_id] is None
        ), f"Recompute data for layer {layer_id} has already been saved."

        flatten_data, flatten_spec = tree_flatten(data)
        device.wait_stream(device.downstream, device.compute_stream)
        with torch.cuda.stream(device.downstream):
            for idx, item in enumerate(flatten_data):
                if isinstance(item, torch.Tensor):
                    assert (
                        not item.requires_grad
                    ), "Tensors saved for recompute cannot require gradients."
                    assert (
                        item.device == device.device
                    ), "Tensors saved for recompute must reside on the compute device."
                    device.mem_manager.record_stream(
                        item, device.compute_stream, device.downstream
                    )
                    flatten_data[idx] = item.to("cpu", non_blocking=True)
                else:
                    flatten_data[idx] = copy.deepcopy(item)
            self.download_event[layer_id].record(device.downstream)

        self.recompute_data[layer_id] = flatten_data
        self.recompute_data_specs[layer_id] = flatten_spec

    def fetch_recompute_data(self, layer_id: int, device: "DeviceManager") -> None:
        """Load data saved for backward recompute to GPU.

        Args:
            layer_id: Layer index whose data should be retrieved.
            device: Device manager whose streams guard the transfer.
        """
        if self.recompute_data_specs[layer_id] is None:
            return
        with torch.cuda.stream(device.upstream):
            self.download_event[layer_id].wait(device.upstream)
            for idx, item in enumerate(self.recompute_data[layer_id]):
                if isinstance(item, torch.Tensor):
                    self.recompute_data[layer_id][idx] = item.to(
                        device.device, non_blocking=True
                    )
                    device.mem_manager.record_stream(
                        self.recompute_data[layer_id][idx],
                        device.upstream,
                        device.compute_stream,
                    )

    def cut_recompute_data(self, layer_id: int) -> Any:
        """Retrieve and clear data saved for backward recompute.

        Args:
            layer_id: Layer index whose data should be retrieved.

        Returns:
            The data previously saved via `save_for_recompute`.
        """
        flatten_data = self.recompute_data[layer_id]
        flatten_spec = self.recompute_data_specs[layer_id]
        self.recompute_data[layer_id] = []
        self.recompute_data_specs[layer_id] = None
        if flatten_spec is None:
            return None
        return tree_unflatten(flatten_data, flatten_spec)

    def restore_rng_state(self, layer_id: int, device: "DeviceManager") -> None:
        """Restore the RNG snapshot captured during `save_input`.

        Args:
            layer_id: Layer index whose RNG states should be restored.
            device: Device manager that owns the CUDA stream.

        Raises:
            AssertionError: If RNG state was not captured as expected.
        """
        if not self.preserve_rng_state:
            return

        device_states = self.device_rng_states[layer_id]
        cpu_states = self.cpu_rng_states[layer_id]
        assert (
            device_states is not None and cpu_states is not None
        ), "RNG states were not saved properly."

        with device.device:
            torch.cuda.set_rng_state(device_states)
        torch.set_rng_state(cpu_states)


@torch.no_grad()
def run_forward(
    device: "DeviceManager",
    context: RoundPipeRunContext,
    layer_group_id: int,
    batch: "Batch",
) -> None:
    """Upload layers, execute forward compute, and copy outputs back to host.
    !!! info
        This function will run on a separate thread managed by the DeviceManager.
        Multiple threads may run concurrently on different devices.
        Be aware of thread-safety when using and modifying this function.
        All data access must limit to the input parameters and the specified model layers.

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
    layer_ids = context.tracker.fwd_plan[layer_group_id]
    grad_context = torch.enable_grad() if context.enable_grad else torch.no_grad()
    if batch_idx == 0:
        gpu_layers = upload_layers(
            [model.layers[layer_id] for layer_id in layer_ids],
            [model.layer_attrs[layer_id] for layer_id in layer_ids],
            device,
            False,
        )
        for layer_id, gpu_layer in zip(layer_ids, gpu_layers):
            context.save_buffer(model.layers[layer_id])
            context.gpu_fwd_layers[layer_id] = gpu_layer
    context.tracker.forward_wait_for(layer_group_id - 1)
    for layer_id in layer_ids:
        context.save_input(layer_id, batch, device)
        if layer_id == layer_ids[0]:
            batch.flatten_states[batch_idx] = async_h2d(
                device,
                batch.forward_events[batch_idx],
                batch.flatten_states[batch_idx],
                context.enable_grad,
            )
        hidden_state = tree_unflatten(
            batch.flatten_states[batch_idx], batch.flatten_specs[batch_idx]
        )
        with grad_context, torch.cuda.stream(device.compute_stream), torch.autocast(
            "cuda", **context.device_autocast_kwargs
        ), torch.autocast("cpu", **context.cpu_autocast_kwargs), annotate(
            f"{model.name}L[{layer_id}]B[{batch_idx}]Fwd"
        ), ForwardCtx(
            functools.partial(context.save_for_recompute, layer_id, device)
        ), model.model_timer.time_fwd(
            "fwd", layer_id, device.compute_stream
        ):
            try:
                if layer_id == 0:
                    args, kwargs = hidden_state
                    hidden_state = context.gpu_fwd_layers[layer_id].forward(
                        *args, **kwargs
                    )
                else:
                    hidden_state = context.gpu_fwd_layers[layer_id].forward(
                        hidden_state
                    )
            except Exception:
                with thread_exception_print_lock:
                    traceback.print_exc()
                    print(
                        f"The above error occurred in {model.name} layer {layer_id} during forward pass.",
                        file=sys.stderr,
                    )
                raise SystemExit(1)
        batch.flatten_states[batch_idx], batch.flatten_specs[batch_idx] = tree_flatten(
            hidden_state
        )

        if context.enable_grad:
            for idx, item in enumerate(batch.flatten_states[batch_idx]):
                if isinstance(item, torch.Tensor):
                    batch.flatten_states[batch_idx][idx] = item.detach().requires_grad_(
                        item.requires_grad
                    )

    batch.forward_events[batch_idx] = [cast(torch.cuda.Event, torch.cuda.Event())]
    batch.flatten_states[batch_idx] = async_d2h(
        device,
        batch.forward_events[batch_idx],
        batch.flatten_states[batch_idx],
        context.enable_grad,
    )
    device.mem_manager.flush()
    if batch_idx == context.num_microbatches - 1:
        for layer_id in layer_ids:
            download_layer(
                model.layers[layer_id],
                context.gpu_fwd_layers[layer_id],
                model.layer_attrs[layer_id],
                device,
                True,
                False,
            )
            context.gpu_fwd_layers[layer_id] = torch.nn.Module()  # Free GPU memory
    context.tracker.forward_notify(layer_group_id)


@torch.no_grad()
def run_backward(
    device: "DeviceManager", context: RoundPipeRunContext, layer_group_id: int
) -> None:
    """Recompute saved inputs, propagate gradients, and ship grads to CPU.
    !!! info
        This function will run on a separate thread managed by the DeviceManager
        or Pytorch autograd. Multiple threads may run concurrently on different devices.
        Be aware of thread-safety when using and modifying this function.
        All data access must limit to the input parameters and the specified model layers.

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
    layer_ids = context.tracker.bwd_plan[layer_group_id]
    if batch_idx == 0:
        gpu_layers = upload_layers(
            [model.layers[layer_id] for layer_id in layer_ids],
            [model.layer_attrs[layer_id] for layer_id in layer_ids],
            device,
            True,
            context.saved_buffers,
        )
        for layer_id, gpu_layer in zip(layer_ids, gpu_layers):
            context.gpu_bwd_layers[layer_id] = gpu_layer
    for layer_id in layer_ids:
        context.fetch_recompute_data(layer_id, device)
    flatten_inputs_gpu = async_h2d(
        device,
        [context.download_event[layer_ids[0]]],
        context.flatten_inputs[layer_ids[0]],
        True,
    )
    context.flatten_inputs[layer_ids[0]].clear()  # Free CPU memory
    flatten_input_spec = cast(TreeSpec, context.flatten_specs[layer_ids[0]])
    hidden_state = tree_unflatten(flatten_inputs_gpu, flatten_input_spec)

    with torch.random.fork_rng(
        devices=[device.device.index],
        enabled=context.preserve_rng_state,
        device_type="cuda",
    ):
        if context.preserve_rng_state:
            context.restore_rng_state(layer_ids[0], device)
        for layer_id in layer_ids:
            with torch.enable_grad(), torch.cuda.stream(
                device.compute_stream
            ), torch.autocast("cuda", **context.device_autocast_kwargs), torch.autocast(
                "cpu", **context.cpu_autocast_kwargs
            ), annotate(
                f"{model.name}L[{layer_id}]B[{batch_idx}]Re"
            ), RecomputeCtx(
                context.cut_recompute_data(layer_id)
            ), model.model_timer.time_fwd(
                "re", layer_id, device.compute_stream
            ):
                try:
                    if layer_id == 0:
                        args, kwargs = hidden_state
                        hidden_state = context.gpu_bwd_layers[layer_id].forward(
                            *args, **kwargs
                        )
                    else:
                        hidden_state = context.gpu_bwd_layers[layer_id].forward(
                            hidden_state
                        )
                except Exception:
                    with thread_exception_print_lock:
                        traceback.print_exc()
                        print(
                            f"The above error occurred in {model.name} layer {layer_id} during recomputation in backward pass.",
                            file=sys.stderr,
                        )
                    raise SystemExit(1)

    flatten_outputs_gpu, _ = tree_flatten(hidden_state)
    context.tracker.backward_wait_for(layer_group_id - 1)
    flatten_grad_outputs_gpu = async_h2d(
        device, context.output_backward_events, context.grad_states
    )

    outputs_requires_grad = []
    outputs_grad = []
    for out, grad_out in zip(flatten_outputs_gpu, flatten_grad_outputs_gpu):
        if isinstance(out, torch.Tensor) and out.requires_grad:
            outputs_requires_grad.append(out)
            outputs_grad.append(grad_out)
    with annotate(
        f"{model.name}L[{layer_ids[0]}, {layer_ids[-1]}]B[{batch_idx}]Bwd"
    ), model.model_timer.time_bwd(layer_ids, device.compute_stream):
        try:
            torch.autograd.backward(outputs_requires_grad, outputs_grad)
        except Exception:
            with thread_exception_print_lock:
                traceback.print_exc()
                print(
                    f"The above error occurred in {model.name} layers {layer_ids} during backward pass.\n"
                    f"Outputs requiring grad: {[out.shape for out in outputs_requires_grad]}\n"
                    f"Provided gradients: {[grad.shape if isinstance(grad, torch.Tensor) else type(grad) for grad in outputs_grad]}",
                    file=sys.stderr,
                )
            raise SystemExit(1)
    flatten_grad_inputs_gpu = [
        inp.grad if isinstance(inp, torch.Tensor) else None
        for inp in flatten_inputs_gpu
    ]

    context.output_backward_events = (
        context.input_backward_events
        if layer_ids[0] == 0
        else [cast(torch.cuda.Event, torch.cuda.Event())]
    )
    context.grad_states = async_d2h(
        device, context.output_backward_events, flatten_grad_inputs_gpu
    )
    device.mem_manager.flush()
    if batch_idx == context.num_microbatches - 1:
        for layer_id in layer_ids:
            download_layer(
                model.layers[layer_id],
                context.gpu_bwd_layers[layer_id],
                model.layer_attrs[layer_id],
                device,
                False,
                True,
            )
            context.gpu_bwd_layers[layer_id] = torch.nn.Module()  # Free GPU memory
    context.tracker.backward_notify(layer_group_id)


@torch.no_grad()
def run_forward_backward(
    device: "DeviceManager",
    context: RoundPipeRunContext,
    batch: "Batch",
    loss_fn: Callable[[Any, Any], Union[Sequence[torch.Tensor], torch.Tensor]],
    return_outputs: bool,
) -> None:
    """Execute a fused forward/backward pass for training workloads.
    !!! info
        This function will run on a separate thread managed by the DeviceManager.
        Multiple threads may run concurrently on different devices.
        Be aware of thread-safety when using and modifying this function.
        All data access must limit to the input parameters and the specified model layers.

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
    layer_ids = context.tracker.bwd_plan[0]
    if batch_idx == 0:
        gpu_layers = upload_layers(
            [model.layers[layer_id] for layer_id in layer_ids],
            [model.layer_attrs[layer_id] for layer_id in layer_ids],
            device,
            True,
        )
        for layer_id, gpu_layer in zip(layer_ids, gpu_layers):
            context.gpu_bwd_layers[layer_id] = gpu_layer
    flatten_inputs_gpu = async_h2d(
        device, batch.forward_events[batch_idx], batch.flatten_states[batch_idx], True
    )
    batch.flatten_states[batch_idx].clear()  # Free CPU memory
    hidden_state = tree_unflatten(flatten_inputs_gpu, batch.flatten_specs[batch_idx])

    for layer_id in layer_ids:
        with torch.enable_grad(), torch.cuda.stream(
            device.compute_stream
        ), torch.autocast("cuda", **context.device_autocast_kwargs), torch.autocast(
            "cpu", **context.cpu_autocast_kwargs
        ), annotate(
            f"{model.name}L[{layer_id}]B[{batch_idx}]Fwd"
        ), model.model_timer.time_fwd(
            "fwd", layer_id, device.compute_stream
        ), model.model_timer.time_fwd(
            "re", layer_id, device.compute_stream
        ):
            try:
                if layer_id == 0:
                    args, kwargs = hidden_state
                    hidden_state = context.gpu_bwd_layers[layer_id].forward(
                        *args, **kwargs
                    )
                else:
                    hidden_state = context.gpu_bwd_layers[layer_id].forward(
                        hidden_state
                    )
            except Exception:
                with thread_exception_print_lock:
                    traceback.print_exc()
                    print(
                        f"The above error occurred in {model.name} layer {layer_id} during forward pass.",
                        file=sys.stderr,
                    )
                raise SystemExit(1)

    batch.flatten_states[batch_idx], batch.flatten_specs[batch_idx] = tree_flatten(
        hidden_state if return_outputs else None
    )
    batch.forward_events[batch_idx] = [cast(torch.cuda.Event, torch.cuda.Event())]
    batch.flatten_states[batch_idx] = async_d2h(
        device, batch.forward_events[batch_idx], batch.flatten_states[batch_idx], False
    )

    flatten_label, flatten_label_spec = tree_flatten(batch.label_list[batch_idx])
    flatten_label_gpu = async_h2d(device, [], flatten_label, False)
    label_gpu = tree_unflatten(flatten_label_gpu, flatten_label_spec)
    with torch.enable_grad(), torch.cuda.stream(device.compute_stream), torch.autocast(
        "cuda", **context.device_autocast_kwargs
    ), torch.autocast("cpu", **context.cpu_autocast_kwargs):
        loss = loss_fn(hidden_state, label_gpu)
    del hidden_state  # Free GPU memory before backward

    device.wait_stream(device.downstream, device.compute_stream)
    with torch.cuda.stream(device.downstream):
        if isinstance(loss, torch.Tensor):
            batch.loss_list[batch_idx] = loss.to("cpu", non_blocking=True)
            device.mem_manager.record_stream(
                loss, device.compute_stream, device.downstream
            )
        else:
            batch.loss_list[batch_idx] = [l.to("cpu", non_blocking=True) for l in loss]
            for l in loss:
                device.mem_manager.record_stream(
                    l, device.compute_stream, device.downstream
                )
    batch.loss_ready.record(device.downstream)

    with annotate(
        f"{model.name}L[{layer_ids[0]}, {layer_ids[-1]}]B[{batch_idx}]Bwd"
    ), model.model_timer.time_bwd(layer_ids, device.compute_stream):
        try:
            torch.autograd.backward(loss)
        except Exception:
            with thread_exception_print_lock:
                traceback.print_exc()
                print(
                    f"The above error occurred in {model.name} layers {layer_ids} during backward pass.",
                    file=sys.stderr,
                )
            raise SystemExit(1)
    flatten_grad_inputs_gpu = [
        inp.grad if isinstance(inp, torch.Tensor) else None
        for inp in flatten_inputs_gpu
    ]

    context.output_backward_events = (
        context.input_backward_events
        if layer_ids[0] == 0
        else [cast(torch.cuda.Event, torch.cuda.Event())]
    )
    context.grad_states = async_d2h(
        device, context.output_backward_events, flatten_grad_inputs_gpu
    )
    device.mem_manager.flush()
    if batch_idx == context.num_microbatches - 1:
        for layer_id in layer_ids:
            download_layer(
                model.layers[layer_id],
                context.gpu_bwd_layers[layer_id],
                model.layer_attrs[layer_id],
                device,
                True,
                True,
            )
            context.gpu_bwd_layers[layer_id] = torch.nn.Module()  # Free GPU memory
    context.tracker.backward_notify(0)


class RoundPipeBatchedBackward(torch.autograd.Function):
    """Autograd node that launches backward passes for all microbatches."""

    @staticmethod
    def forward(
        ctx: Any,
        roundpipe_context: List[RoundPipeRunContext],
        batch: "Batch",
        tag: torch.Tensor,
        *all_inputs: Any,
    ) -> Tuple[List[Tuple[int, int]], Unpack[Tuple[torch.Tensor, ...]]]:
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
            context.output_backward_events = [
                cast(torch.cuda.Event, torch.cuda.Event())
            ]
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

        ctx.output_len = [
            len(flatten_outputs) for flatten_outputs in batch.flatten_states
        ]
        ctx.output_require_grad_idx = []
        output_require_grad: List[torch.Tensor] = []
        for batch_idx, flatten_outputs in enumerate(batch.flatten_states):
            for idx, item in enumerate(flatten_outputs):
                if isinstance(item, torch.Tensor) and item.requires_grad:
                    ctx.output_require_grad_idx.append((batch_idx, idx))
                    output_require_grad.append(item)

        ctx.roundpipe_contexts = roundpipe_context
        ctx.launched_backward = False
        return ctx.output_require_grad_idx, *output_require_grad

    @staticmethod
    def backward(  # pyright: ignore[reportIncompatibleMethodOverride]
        ctx: Any, _, *grad_outputs: Any
    ) -> Tuple[None, None, None, Unpack[Tuple[Optional[torch.Tensor], ...]]]:
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
        for (batch_idx, idx), grad_out in zip(
            ctx.output_require_grad_idx, grad_outputs
        ):
            run_contexts[batch_idx].grad_states[idx] = grad_out

        for idx, (batch_idx, layer_idx, item_idx) in enumerate(ctx.tensor_indices):
            run_contexts[batch_idx].flatten_inputs[layer_idx][item_idx] = (
                ctx.saved_tensors[idx]
            )

        for layer_group_id in range(len(run_contexts[0].tracker.bwd_plan)):
            device = get_next_device()
            device.launch_backward(
                layer_group_id,
                [
                    run_contexts[0].model.layer_attrs[i]
                    for i in run_contexts[0].tracker.bwd_plan[layer_group_id]
                ],
                run_contexts,
            )
        run_contexts[0].tracker.backward_wait_complete(len(run_contexts))

        grad_inputs = [item for context in run_contexts for item in context.grad_states]
        for context in run_contexts:
            del context.grad_states
        return (None, None, None, *grad_inputs)


class RoundPipeMicrobatchBackward(torch.autograd.Function):
    """Autograd node that launches backward pass for a single microbatch."""

    @staticmethod
    def forward(
        ctx: Any,
        roundpipe_context: RoundPipeRunContext,
        batch: "Batch",
        tag: torch.Tensor,
        *all_inputs: Any,
    ) -> Tuple[torch.Tensor, List[int], Unpack[Tuple[torch.Tensor, ...]]]:
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
        roundpipe_context.input_backward_events = batch.backward_events[
            roundpipe_context.microbatch_id
        ]
        roundpipe_context.output_backward_events = [
            cast(torch.cuda.Event, torch.cuda.Event())
        ]
        batch.backward_events[roundpipe_context.microbatch_id] = (
            roundpipe_context.output_backward_events
        )

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
        output_require_grad: List[torch.Tensor] = []
        for idx, item in enumerate(
            batch.flatten_states[roundpipe_context.microbatch_id]
        ):
            if isinstance(item, torch.Tensor) and item.requires_grad:
                ctx.output_require_grad_idx.append(idx)
                output_require_grad.append(item)

        ctx.roundpipe_context = roundpipe_context
        ctx.launched_backward = False
        return tag, ctx.output_require_grad_idx, *output_require_grad

    @staticmethod
    def backward(  # pyright: ignore[reportIncompatibleMethodOverride]
        ctx: Any, device_id: torch.Tensor, _, *grad_outputs: Any
    ) -> Tuple[None, None, torch.Tensor, Unpack[Tuple[Optional[torch.Tensor], ...]]]:
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
            for layer_id in context.tracker.bwd_plan[0]:
                context.model.layer_attrs[layer_id].backward_fence()
        else:
            device = device_list[int(device_id.item())]
        run_backward(device, context, 0)

        grad_inputs = context.grad_states
        del context.grad_states
        return (None, None, device_id, *grad_inputs)


class RoundPipeInputBackward(torch.autograd.Function):
    """Autograd node that reconnects RoundPipe gradients to user inputs."""

    @staticmethod
    def forward(
        ctx: Any, roundpipe_context: List[RoundPipeRunContext], *all_inputs: Any
    ) -> torch.Tensor:
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
    def backward(
        ctx: Any, *grad_outputs: Any
    ) -> Tuple[None, Unpack[Tuple[Optional[torch.Tensor], ...]]]:
        """Return gradients captured in each ``RoundPipeRunContext``.

        Args:
            ctx: Autograd context populated during ``forward``.
            *grad_outputs: Gradients w.r.t. the dummy scalar (unused).

        Returns:
            Tuple matching ``(None, *flattened_input_grads)`` so that upstream
                PyTorch graphs receive gradients for their original inputs.
        """
        run_contexts: List[RoundPipeRunContext] = ctx.roundpipe_contexts
        grad_inputs = [item for context in run_contexts for item in context.grad_states]
        for context in run_contexts:
            del context.grad_states
        return (None, *grad_inputs)
