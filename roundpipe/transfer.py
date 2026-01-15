"""Tensor transfer helpers that move activations/params between host/DEVICE.

Attributes:
    chunked_upload: Whether to split large params into chunks during upload.
    CHUNK_UPLOAD_SIZE: Size threshold (in bytes) for chunked uploads.
"""

from typing_extensions import *
import math

import torch

from .param import ParamAttribute

if TYPE_CHECKING:
    from .device import DeviceManager
else:
    DeviceManager = TypeAliasType("DeviceManager", "roundpipe.device.DeviceManager")

chunked_upload: bool = True
CHUNK_UPLOAD_SIZE: int = 256 * 1024 * 1024  # 256 MB


def async_d2h(
    device: "DeviceManager",
    transfer_finish_event: Iterable[torch.cuda.Event],
    device_tensors: Iterable[Union[torch.Tensor, Any]],
    keep_requires_grad: bool = False,
) -> List[Union[torch.Tensor, Any]]:
    """Copy tensors from device to host streams while preserving ordering.

    Args:
        device: Device manager owning the downstream stream.
        transfer_finish_event: Events to record when the copy completes.
        device_tensors: Iterable of tensors/objects residing on the device.
        keep_requires_grad: Whether to preserve ``requires_grad`` flags.

    Returns:
        List of tensors/objects now resident on the host.
    """
    host_tensors = []
    device.wait_stream(device.downstream, device.compute_stream)
    device.wait_stream(device.compute_stream, device.downstream)
    with torch.cuda.stream(device.downstream):
        for device_tensor in device_tensors:
            if isinstance(device_tensor, torch.Tensor):
                device.mem_manager.record_stream(
                    device_tensor, device.compute_stream, device.downstream
                )
                host_tensor = device_tensor.to(torch.device("cpu"), non_blocking=True)
                host_tensor.requires_grad_(
                    keep_requires_grad and device_tensor.requires_grad
                )
                host_tensors.append(host_tensor)
            else:
                host_tensors.append(device_tensor)
        for event in transfer_finish_event:
            event.record()
    return host_tensors


def async_h2d(
    device: "DeviceManager",
    host_ready_event: Iterable[torch.cuda.Event],
    host_tensors: Iterable[Union[torch.Tensor, Any]],
    keep_requires_grad: bool = False,
) -> List[Union[torch.Tensor, Any]]:
    """Copy tensors from host to device using the device's upload stream.

    Args:
        device: Device manager orchestrating the upload.
        host_ready_event: Events the host waited on before the copy.
        host_tensors: Iterable of host tensors/objects.
        keep_requires_grad: Whether to preserve ``requires_grad`` flags.

    Returns:
        List of tensors/objects now resident on the device.
    """
    device_tensors = []
    with torch.cuda.stream(device.upstream):
        for event in host_ready_event:
            event.wait()
        for host_tensor in host_tensors:
            if isinstance(host_tensor, torch.Tensor):
                host_tensor_requires_grad = host_tensor.requires_grad
                if not host_tensor.is_pinned():
                    # Re-mirror user tensors through a pinned buffer so CUDA can
                    # launch non-blocking copies even if the original tensor is pageable.
                    host_pinned = torch.empty_like(
                        host_tensor, device=torch.device("cpu"), pin_memory=True
                    )
                    host_pinned.copy_(host_tensor)
                    host_tensor = host_pinned
                device_tensor = host_tensor.to(device.device, non_blocking=True)
                device_tensor.requires_grad_(
                    keep_requires_grad and host_tensor_requires_grad
                )
                device.mem_manager.record_stream(
                    device_tensor, device.upstream, device.compute_stream
                )
                device_tensors.append(device_tensor)
            else:
                device_tensors.append(host_tensor)
    device.mark_upload()
    device.wait_stream(device.compute_stream, device.upstream)
    device.wait_stream(device.upstream, device.compute_stream)
    return device_tensors


def create_upload_pair(
    tensor_pair: List[Tuple[torch.Tensor, torch.Tensor]],
    src: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Allocate destination buffers and register chunk copies when needed.

    Args:
        tensor_pair: Collector that receives ``(src, dst)`` pairs.
        src: Source tensor currently resident in host memory.
        device: Target CUDA device.

    Returns:
        Destination tensor allocated on ``device``.
    """
    size = src.element_size() * src.nelement()
    dst = torch.empty_like(src, device=device)
    try:
        assert chunked_upload and size > CHUNK_UPLOAD_SIZE
        n_chunks = math.ceil(size / CHUNK_UPLOAD_SIZE)
        chunk_nelements = math.ceil(src.nelement() / n_chunks)
        chunked_src = src.view(-1).split(chunk_nelements)
        chunked_dst = dst.view(-1).split(chunk_nelements)
        tensor_pair.extend(zip(chunked_src, chunked_dst))
    except Exception:
        tensor_pair.append((src, dst))
    return dst


def upload_layers(
    layers: List[torch.nn.Module], device: "DeviceManager", upload_grad: bool
) -> torch.cuda.Event:
    """Move parameters/buffers (and optionally grads) onto the target device.

    Args:
        layers: Sequence of layers to upload.
        device: Device manager orchestrating the transfer.
        upload_grad: Whether to copy gradient buffers alongside parameters.

    Returns:
        Event that signals when the upload is complete.
    """
    from .scheduler import chunk_layer_params

    chunk_events = device.flush_upload_marks()
    if len(chunk_events) == 0:
        chunk_events.append(cast(torch.cuda.Event, torch.cuda.Event()))
    tensor_pair: List[Tuple[torch.Tensor, torch.Tensor]] = []
    with torch.cuda.stream(device.param_upstream):
        for layer in layers:
            for param in layer.parameters():
                param_attr = ParamAttribute.get(param)
                param.data = create_upload_pair(
                    tensor_pair, param_attr.data_cpu, device.device
                )
                if upload_grad and param.grad is not None:
                    param.grad = create_upload_pair(
                        tensor_pair, param.grad, device.device
                    )
                    param_attr.uploaded_grad = True
                else:
                    param_attr.uploaded_grad = False
            for buffer in layer.buffers():
                buffer_attr = ParamAttribute.get(buffer)
                buffer.data = create_upload_pair(
                    tensor_pair, buffer_attr.data_cpu, device.device
                )
    if len(tensor_pair) == 0:
        return cast(torch.cuda.Event, torch.cuda.Event())
    chunked_tensor_pairs = chunk_layer_params(tensor_pair, len(chunk_events))

    with torch.cuda.stream(device.param_upstream):
        for tensor_chunk, chunk_event in zip(chunked_tensor_pairs, chunk_events):
            chunk_event.wait()
            for src, dst in tensor_chunk:
                dst.copy_(src, non_blocking=True)
    finish_event = cast(torch.cuda.Event, torch.cuda.Event())
    finish_event.record(device.param_upstream)
    device.wait_stream(device.compute_stream, device.param_upstream)
    device.wait_stream(device.param_upstream, device.compute_stream)
    return finish_event


def free_layer(layer: torch.nn.Module, device: "DeviceManager"):
    """Swap layer tensors back to their CPU storage to free GPU memory.

    Args:
        layer: Module whose parameters/buffers should be restored to CPU.
    """
    for param in layer.parameters():
        device.mem_manager.free(
            param.data.untyped_storage(), device.param_upstream, device.compute_stream
        )
        param_attr = ParamAttribute.get(param)
        param.data = param_attr.data_cpu
    for buffer in layer.buffers():
        device.mem_manager.free(
            buffer.data.untyped_storage(), device.param_upstream, device.compute_stream
        )
        buffer_attr = ParamAttribute.get(buffer)
        buffer.data = buffer_attr.data_cpu


def download_layer(layer: torch.nn.Module, device: "DeviceManager"):
    """Copy layer params/buffers (and grads) back to the host asynchronously.
    Note that this only issues the copies; synchronization must be handled
    externally.

    Args:
        layer: Module whose tensors should be copied to host memory.
        device: Device manager orchestrating the transfer.
    """
    with torch.cuda.stream(device.downstream):
        for param in layer.parameters():
            device.mem_manager.free(
                param.data.untyped_storage(),
                device.param_upstream,
                device.compute_stream,
            )
            param_attr = ParamAttribute.get(param)
            param.data = param_attr.data_cpu
            if param.grad is not None:
                if param_attr.uploaded_grad:
                    device.mem_manager.free(
                        param.grad.untyped_storage(),
                        device.param_upstream,
                        device.compute_stream,
                        device.downstream,
                    )
                else:
                    device.mem_manager.free(
                        param.grad.untyped_storage(),
                        device.compute_stream,
                        device.downstream,
                    )
                if (
                    param_attr.data_grad is not None
                ):  # reuse allocated grad buffer if exists
                    param_attr.data_grad.copy_(param.grad, non_blocking=True)
                    param.grad = param_attr.data_grad
                else:
                    param.grad = param.grad.to(torch.device("cpu"), non_blocking=True)
        for buffer in layer.buffers():
            device.mem_manager.free(
                buffer.data.untyped_storage(),
                device.param_upstream,
                device.compute_stream,
                device.downstream,
            )
            buffer_attr = ParamAttribute.get(buffer)
            buffer_attr.data_cpu.copy_(buffer.data, non_blocking=True)
            buffer.data = buffer_attr.data_cpu


class PinnedUpload(torch.autograd.Function):
    """Autograd helper that enforces pinned host tensors before H2D copies."""

    @staticmethod
    def forward(ctx: Any, t: torch.Tensor, d: torch.device) -> torch.Tensor:
        """Ensure ``t`` resides in pinned memory before copying to device.

        Args:
            ctx: Autograd context (unused).
            t: Host tensor to transfer.
            d: Destination device.

        Returns:
            Tensor residing on device ``d``.
        """
        if not t.is_pinned():
            t_pinned = torch.empty_like(t, device=torch.device("cpu"), pin_memory=True)
            t_pinned.copy_(t)
            t = t_pinned
        return t.to(d, non_blocking=True)

    @staticmethod
    def backward(  # pyright: ignore[reportIncompatibleMethodOverride]
        ctx: Any, g: torch.Tensor
    ) -> Tuple[torch.Tensor, None]:
        """Move gradients back to pinned host memory.

        Args:
            ctx: Autograd context (unused).
            g: Gradient tensor on the device.

        Returns:
            Gradient tensor on CPU and ``None`` for the device argument.
        """
        g_host = torch.empty_like(g, device=torch.device("cpu"), pin_memory=True)
        g_host.copy_(g)
        return g_host, None


class RegisterBackwardEvent(torch.autograd.Function):
    """Records an event so backward consumers can synchronize on it."""

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, event: torch.cuda.Event) -> torch.Tensor:
        """Stash ``event`` so its completion gates backward consumption.

        Args:
            ctx: Autograd context storing the event handle.
            input: Tensor passed through untouched.
            event: CUDA event to signal when backward starts.

        Returns:
            The original ``input`` tensor.
        """
        ctx.event = event
        return input

    @staticmethod
    def backward(  # pyright: ignore[reportIncompatibleMethodOverride]
        ctx: Any, grad_outputs: torch.Tensor
    ) -> Tuple[torch.Tensor, None]:
        """Synchronize on the recorded event before returning gradients.

        Args:
            ctx: Autograd context containing the event handle.
            grad_outputs: Incoming gradient tensor.

        Returns:
            Tuple containing the guarded gradient and ``None`` for ``event``.
        """
        event: torch.cuda.Event = ctx.event
        event.synchronize()
        return grad_outputs, None
