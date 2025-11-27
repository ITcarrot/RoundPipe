from beartype.typing import * # type: ignore[reportWildcardImportFromLibrary]
import math

import torch

from .scheduler import chunk_layer_params

if TYPE_CHECKING:
    from .device import DeviceManager
else:
    from typing_extensions import TypeAliasType
    DeviceManager = TypeAliasType('DeviceManager', 'RoundPipe.device.DeviceManager')

chunked_upload = True
CHUNK_UPLOAD_SIZE = 256 * 1024 * 1024 # 256 MB

def async_d2h(device: 'DeviceManager',
              transfer_finish_event: Iterable[torch.cuda.Event],
              device_tensors: Iterable[Union[torch.Tensor, Any]],
              keep_requires_grad: bool = False
              ) -> List[Union[torch.Tensor, Any]]:
    host_tensors = []
    device.downstream.wait_stream(device.compute_stream)
    with torch.cuda.stream(device.downstream):
        for device_tensor in device_tensors:
            if isinstance(device_tensor, torch.Tensor):
                device_tensor.record_stream(device.downstream)
                host_tensor = device_tensor.to(torch.device('cpu'), non_blocking = True)
                host_tensor.requires_grad_(keep_requires_grad and device_tensor.requires_grad)
                host_tensors.append(host_tensor)
            else:
                host_tensors.append(device_tensor)
        for event in transfer_finish_event:
                event.record()
    return host_tensors

def async_h2d(device: 'DeviceManager',
              host_ready_event: Iterable[torch.cuda.Event],
              host_tensors: Iterable[Union[torch.Tensor, Any]],
              keep_requires_grad: bool = False
              ) -> List[Union[torch.Tensor, Any]]:
    device_tensors = []
    with torch.cuda.stream(device.upstream):
        for event in host_ready_event:
            event.wait()
        for host_tensor in host_tensors:
            if isinstance(host_tensor, torch.Tensor):
                host_tensor_requires_grad = host_tensor.requires_grad
                if not host_tensor.is_pinned():
                    host_pinned = torch.empty_like(host_tensor, device = torch.device('cpu'), pin_memory = True)
                    host_pinned.copy_(host_tensor)
                    host_tensor = host_pinned
                device_tensor = host_tensor.to(device.device, non_blocking = True)
                device_tensor.requires_grad_(keep_requires_grad and host_tensor_requires_grad)
                device_tensor.record_stream(device.compute_stream)
                device_tensors.append(device_tensor)
            else:
                device_tensors.append(host_tensor)
    device.compute_stream.wait_stream(device.upstream)
    device.mark_upload()
    return device_tensors

def create_upload_pair(tensor_pair: List[Tuple[torch.Tensor, torch.Tensor]],
                         src: torch.Tensor, device: torch.device) -> torch.Tensor:
    size = src.element_size() * src.nelement()
    dst = torch.empty_like(src, device = device)
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

def upload_layers(layers: List[torch.nn.Module],
                  device: 'DeviceManager', upload_grad: bool):
    chunk_events = device.flush_upload_marks()
    if len(chunk_events) == 0:
        chunk_events.append(torch.cuda.Event())  # type: ignore[reportAttributeAccessIssue]
    tensor_pair: List[Tuple[torch.Tensor, torch.Tensor]] = []
    with torch.cuda.stream(device.param_upstream):
        for layer in layers:
            for param in layer.parameters():
                param.data = create_upload_pair(tensor_pair, param.data_cpu, device.device) # type: ignore[attr-defined]
                param.data.record_stream(device.compute_stream)
                if upload_grad and param.grad is not None:
                    param.grad = create_upload_pair(tensor_pair, param.grad, device.device)
                    param.grad.record_stream(device.compute_stream)
            for buffer in layer.buffers():
                buffer.data = create_upload_pair(tensor_pair, buffer.data_cpu, device.device) # type: ignore[attr-defined]
                buffer.data.record_stream(device.compute_stream)
    if len(tensor_pair) == 0:
        return
    chunked_tensor_pairs = chunk_layer_params(tensor_pair, len(chunk_events))

    with torch.cuda.stream(device.param_upstream):
        for tensor_chunk, chunk_event in zip(chunked_tensor_pairs, chunk_events):
            chunk_event.wait()
            for src, dst in tensor_chunk:
                dst.copy_(src, non_blocking = True)
    device.compute_stream.wait_stream(device.param_upstream)

def free_layer(layer: torch.nn.Module):
    for param in layer.parameters():
        param.data = param.data_cpu # type: ignore[attr-defined]
    for buffer in layer.buffers():
        buffer.data = buffer.data_cpu # type: ignore[attr-defined]

def download_layer(layer: torch.nn.Module, transfer_stream: torch.cuda.Stream):
    with torch.cuda.stream(transfer_stream):
        for param in layer.parameters():
            param.data = param.data_cpu # type: ignore[attr-defined]
            if param.grad is not None:
                param.grad.record_stream(transfer_stream)
                param.grad = param.grad.to(torch.device('cpu'), non_blocking = True)
        for buffer in layer.buffers():
            buffer.data.record_stream(transfer_stream)
            buffer.data_cpu.copy_(buffer.data, non_blocking = True) # type: ignore[attr-defined]
            buffer.data = buffer.data_cpu # type: ignore[attr-defined]

class PinnedUpload(torch.autograd.Function):
    @staticmethod
    def forward(ctx, t: torch.Tensor, d: torch.device):
        if not t.is_pinned():
            t_pinned = torch.empty_like(t, device = torch.device('cpu'), pin_memory = True)
            t_pinned.copy_(t)
            t = t_pinned
        return t.to(d, non_blocking = True)

    @staticmethod
    def backward(ctx, g: torch.Tensor): # type: ignore[override]
        g_host = torch.empty_like(g, device = torch.device('cpu'), pin_memory = True)
        g_host.copy_(g)
        return g_host, None

class RegisterBackwardEvent(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, event: torch.cuda.Event) -> torch.Tensor:
        ctx.event = event
        return input

    @staticmethod
    def backward(ctx, grad_outputs: torch.Tensor): # type: ignore[override]
        event: torch.cuda.Event = ctx.event
        event.synchronize()
        return grad_outputs, None
