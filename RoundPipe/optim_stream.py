"""Optimizer execute stream and related functions.

Attributes:
    kernel_queue: Queue of optimizer kernel tasks.
    optim_shutdown: Flag to signal optimizer stream shutdown.
    optim_active: Lock to indicate if the optimizer stream is active.
    optim_thread: Daemon thread that executes optimizer tasks.
"""
from typing_extensions import *

import queue
import threading
import _thread
import atexit

from .threads import RoundPipeThread

kernel_queue: queue.Queue[Tuple[Callable, Tuple, Dict[str, Any]]] = queue.Queue()
optim_shutdown = False
optim_active: _thread.LockType = threading.Lock()
def controller() -> None:
    """Optimizer Stream thread function."""
    while not optim_shutdown:
        fn, args, kwargs = kernel_queue.get()
        with optim_active:
            optim_thread.is_active = True
            fn(*args, **kwargs)
            optim_thread.is_active = False

optim_thread: RoundPipeThread = RoundPipeThread(
    target=controller,
    name="RoundPipe Optimizer Stream"
)

def shutdown_optim() -> None:
    """Shut down the optimizer stream."""
    global optim_shutdown
    optim_shutdown = True
    optim_active.acquire() # Ensure no tasks are running
atexit.register(shutdown_optim)

def launch_optim_kernel(fn: Callable, *args: Any, **kwargs: Any) -> None:
    """Launch an optimizer kernel on the optimizer stream.

    Args:
        fn: Callable that launches the optimizer kernel.
        *args: Positional arguments forwarded to ``fn``.
        **kwargs: Keyword arguments forwarded to ``fn``.
    """
    kernel_queue.put((fn, args, kwargs))

def synchronize_optim() -> None:
    """Synchronize the optimizer stream with the main thread."""
    event = threading.Event()
    launch_optim_kernel(event.set)
    event.wait()

def on_optim_stream() -> bool:
    """Check if the current thread is the optimizer stream.

    Returns:
        True if the current thread is the optimizer stream, False otherwise.
    """
    return threading.current_thread() is optim_thread
