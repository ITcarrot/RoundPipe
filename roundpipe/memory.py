"""CPU memory managing utilities for RoundPipe."""

from typing_extensions import *

import weakref

import torch


def pin_module_alloc(mod: torch.nn.Module) -> None:
    """Pin module parameters and buffers to CPU memory.

    Args:
        mod: Module to pin.
    """
    for param in mod.parameters():
        param.data = param.data.pin_memory()
    for buffer in mod.buffers():
        buffer.data = buffer.data.pin_memory()


PAGE_SIZE = 4096


def pin_module_register(mod: torch.nn.Module) -> None:
    """Pin module parameters and buffers to CPU memory using cudaHostRegister.

    Args:
        mod: Module to pin.
    """
    storages: List[torch.UntypedStorage] = []
    for param in mod.parameters():
        if not param.data.is_pinned() and param.numel() > 0:
            storages.append(param.data.untyped_storage())
    for buffer in mod.buffers():
        if not buffer.data.is_pinned() and buffer.numel() > 0:
            storages.append(buffer.data.untyped_storage())
    if len(storages) == 0:
        return
    storages.sort(key=lambda s: s.data_ptr())
    cudart: Any = torch.cuda.cudart()

    ptr, size = storages[0].data_ptr(), storages[0].nbytes()
    for s in storages[1:]:
        s_ptr, s_size = s.data_ptr(), s.nbytes()
        if ptr + size + PAGE_SIZE <= s_ptr:
            ret = cudart.cudaHostRegister(ptr, size, 1)
            assert int(ret) == 0, f"Cuda error {int(ret)}"
            weakref.finalize(mod, lambda p: cudart.cudaHostUnregister(p), ptr)
            ptr, size = s_ptr, s_size
        else:
            size = max(size, s_ptr + s_size - ptr)
    ret = cudart.cudaHostRegister(ptr, size, 1)
    assert int(ret) == 0, f"Cuda error {int(ret)}"
    weakref.finalize(mod, lambda p: cudart.cudaHostUnregister(p), ptr)
