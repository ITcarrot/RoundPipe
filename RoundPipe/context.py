""""Context managers for RoundPipe."

Attributes:
    flags: Thread-local storage for context flags.
"""
from typing_extensions import *

import threading
import types

flags: threading.local = threading.local()

class RecomputeCtx:
    """Context manager to mark this scope is doing recompute."""
    def __enter__(self) -> None:
        flags.recompute = True

    def __exit__(self, exc_type: Optional[Type[BaseException]],
                 exc_value: Optional[BaseException],
                 traceback: Optional[types.TracebackType]) -> None:
        flags.recompute = False

def doing_recompute() -> bool:
    """Check if current scope is doing recompute."""
    return getattr(flags, 'recompute', False)

class OptimizerCtx:
    """Context manager to mark this scope is doing optimizer
        related operations. Under this scope, RoundPipe will
        redirect .parameters() and .named_parameters() to
        .optim_parameters() and .named_optim_parameters() respectively.
    """
    def __enter__(self) -> None:
        flags.optimizer = True

    def __exit__(self, exc_type: Optional[Type[BaseException]],
                 exc_value: Optional[BaseException],
                 traceback: Optional[types.TracebackType]) -> None:
        flags.optimizer = False

def doing_optimizer() -> bool:
    """Check if current scope is doing optimizer related operations."""
    return getattr(flags, 'optimizer', False)
