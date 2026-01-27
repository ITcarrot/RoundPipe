"""Generic helpers shared across RoundPipe components."""

from typing_extensions import *
import traceback
import weakref

import torch
import torch.nn as nn


def get_model_size(model: nn.Module, recurse: bool = True) -> int:
    """Return the combined parameter + buffer bytes for ``model``.

    Args:
        model: Module whose parameters/buffers are measured.
        recurse: Whether to include children recursively.

    Returns:
        Total size in bytes.
    """
    param_size = sum(p.numel() * p.element_size() for p in model.parameters(recurse))
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers(recurse))
    return param_size + buffer_size


def get_call_location(depth: int) -> str:
    """Get the call location in 'filename:lineno' format.

    Args:
        depth: The depth in the call stack to inspect.

    Returns:
        A string representing the call location.
    """
    from . import ENABLE_BEAR

    depth += 1  # Adjust for this function's frame
    if ENABLE_BEAR:
        depth *= 2  # Adjust for beartype frames
    filename, lineno, _, _ = traceback.extract_stack()[-1 - depth]
    return f'{filename.split("/")[-1]}:{lineno}'


def pin_tensor(tensor: torch.Tensor) -> None:
    """Pin a CPU tensor's memory using cudaHostRegister.

    Args:
        tensor: The CPU tensor to pin.
    """
    if tensor.device.type != "cpu":
        raise ValueError("Only CPU tensors can be pinned.")
    if tensor.is_pinned() or tensor.numel() == 0:
        return
    cudart: Any = torch.cuda.cudart()
    storage = tensor.data.untyped_storage()
    assert int(cudart.cudaHostRegister(storage.data_ptr(), storage.nbytes(), 1)) == 0
    weakref.finalize(
        storage, lambda ptr: cudart.cudaHostUnregister(ptr), storage.data_ptr()
    )
