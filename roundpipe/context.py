""" "Context managers for RoundPipe."

Attributes:
    flags: Thread-local storage for context flags.
"""

from typing_extensions import *

import threading
import types

flags: threading.local = threading.local()


class ForwardCtx:
    """Context manager to mark this scope is doing forward pass.

    Attributes:
        save_for_recompute: A callable to save data for recomputation.
    """

    def __init__(self, save_for_recompute: Callable[..., None]) -> None:
        self.save_for_recompute: Callable[..., None] = save_for_recompute

    def __enter__(self) -> None:
        assert (
            getattr(flags, "forward", None) is None
        ), "Nested forward contexts are not allowed"
        flags.forward = self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[types.TracebackType],
    ) -> None:
        flags.forward = None


def save_for_recompute(*data: Any) -> None:
    """Save data for recomputation in current forward context.
    Tensor to be saved cannot require gradients. This function
    can be called at most once from each layer.
    If forward gradients are not enabled, this is a no-op.
    """
    forward_ctx: Optional[ForwardCtx] = getattr(flags, "forward", None)
    if forward_ctx is not None:
        forward_ctx.save_for_recompute(*data)


class RecomputeCtx:
    """Context manager to mark this scope is doing recompute.

    Attributes:
        recompute_data: Data saved for recomputation.
    """

    def __init__(self, recompute_data: Any) -> None:
        self.recompute_data: Any = recompute_data

    def __enter__(self) -> None:
        assert (
            getattr(flags, "recompute", None) is None
        ), "Nested recompute contexts are not allowed"
        flags.recompute = self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[types.TracebackType],
    ) -> None:
        flags.recompute = None


def doing_recompute() -> bool:
    """Check if current scope is doing recompute."""
    return getattr(flags, "recompute", None) is not None


def get_recompute_data() -> Any:
    """Get data saved for recomputation in current recompute context."""
    recompute_ctx: Optional[RecomputeCtx] = getattr(flags, "recompute", None)
    assert recompute_ctx is not None, "Not in recompute context"
    return recompute_ctx.recompute_data


class OptimizerCtx:
    """Context manager to mark this scope is doing optimizer
    related operations. Under this scope, RoundPipe will
    redirect .parameters() and .named_parameters() to
    .optim_parameters() and .named_optim_parameters() respectively.
    """

    def __enter__(self) -> None:
        flags.optimizer = True

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[types.TracebackType],
    ) -> None:
        flags.optimizer = False


def doing_optimizer() -> bool:
    """Check if current scope is doing optimizer related operations."""
    return getattr(flags, "optimizer", False)
