"""Generic helpers shared across RoundPipe components."""

import traceback

import torch.nn as nn

from . import ENABLE_BEAR


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
    depth += 1  # Adjust for this function's frame
    if ENABLE_BEAR:
        depth *= 2  # Adjust for beartype frames
    filename, lineno, _, _ = traceback.extract_stack()[-1 - depth]
    return f'{filename.split("/")[-1]}:{lineno}'
