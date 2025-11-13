from typing import * # type: ignore[reportWildcardImportFromLibrary]

import torch

def async_d2h(compute_stream: torch.cuda.Stream,
              transfer_stream: torch.cuda.Stream,
              transfer_finish_event: Iterable[torch.cuda.Event],
              device_tensors: Iterable[Union[torch.Tensor, Any]],
              keep_requires_grad: bool = False
              ) -> List[Union[torch.Tensor, Any]]:
    host_tensors = []
    transfer_stream.wait_stream(compute_stream)
    with torch.cuda.stream(transfer_stream):
        for device_tensor in device_tensors:
            if isinstance(device_tensor, torch.Tensor):
                device_tensor.record_stream(transfer_stream)
                host_tensor = device_tensor.to(torch.device('cpu'), non_blocking = True)
                host_tensor.requires_grad_(keep_requires_grad and device_tensor.requires_grad)
                host_tensors.append(host_tensor)
            else:
                host_tensors.append(device_tensor)
        for event in transfer_finish_event:
                event.record()
    return host_tensors

def async_h2d(compute_stream: torch.cuda.Stream,
              transfer_stream: torch.cuda.Stream,
              host_ready_event: Iterable[torch.cuda.Event],
              host_tensors: Iterable[Union[torch.Tensor, Any]],
              keep_requires_grad: bool = False
              ) -> List[Union[torch.Tensor, Any]]:
    device_tensors = []
    with torch.cuda.stream(transfer_stream):
        for event in host_ready_event:
            event.wait()
        for host_tensor in host_tensors:
            if isinstance(host_tensor, torch.Tensor):
                host_tensor_requires_grad = host_tensor.requires_grad
                if not host_tensor.is_pinned():
                    host_pinned = torch.empty_like(host_tensor, device = torch.device('cpu'), pin_memory = True)
                    host_pinned.copy_(host_tensor)
                    host_tensor = host_pinned
                device_tensor = host_tensor.to(transfer_stream.device, non_blocking = True)
                device_tensor.requires_grad_(keep_requires_grad and host_tensor_requires_grad)
                device_tensor.record_stream(compute_stream)
                device_tensors.append(device_tensor)
            else:
                device_tensors.append(host_tensor)
    compute_stream.wait_stream(transfer_stream)
    return device_tensors

def upload_layer(layer: torch.nn.Module, transfer_stream: torch.cuda.Stream,
                 compute_stream: torch.cuda.Stream, upload_grad: bool):
    with torch.cuda.stream(transfer_stream):
        for param in layer.parameters():
            param.data = param.data_cpu.to(transfer_stream.device, non_blocking = True) # type: ignore[attr-defined]
            param.data.record_stream(compute_stream)
            if upload_grad and param.grad is not None:
                param.grad = param.grad.to(transfer_stream.device, non_blocking = True)
                param.grad.record_stream(compute_stream)
        for buffer in layer.buffers():
            buffer.data = buffer.data_cpu.to(transfer_stream.device, non_blocking = True) # type: ignore[attr-defined]
            buffer.data.record_stream(compute_stream)

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
