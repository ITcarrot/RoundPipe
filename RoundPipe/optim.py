"""Optimizer execute stream and related functions.

Attributes:
    kernel_queue: Queue of optimizer kernel tasks.
    optim_thread: Daemon thread that executes optimizer tasks.
"""
from beartype.typing import * # pyright: ignore[reportWildcardImportFromLibrary]

import queue

from .threads import RoundPipeThread

kernel_queue: queue.Queue[Tuple[Callable, Tuple, Dict[str, Any]]] = queue.Queue()
def controller() -> None:
    """Optimizer Stream thread function."""
    while True:
        fn, args, kwargs = kernel_queue.get()
        optim_thread.is_active = True
        fn(*args, **kwargs)
        optim_thread.is_active = False

optim_thread: RoundPipeThread = RoundPipeThread(
    target=controller,
    name="RoundPipe Optimizer Stream"
)

def launch_optim_kernel(fn: Callable, *args: Any, **kwargs: Any) -> None:
    """Launch an optimizer kernel on the optimizer stream.

    Args:
        fn: Callable that launches the optimizer kernel.
        *args: Positional arguments forwarded to ``fn``.
        **kwargs: Keyword arguments forwarded to ``fn``.
    """
    kernel_queue.put((fn, args, kwargs))
