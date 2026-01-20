"""Optimizer execute stream and related functions.

Attributes:
    KernelQueueType: queue.Queue[Tuple[Callable, Tuple, Dict[str, Any]]]
    kernel_queue: Queue of optimizer kernel tasks.
    optim_shutdown: Flag to signal optimizer stream shutdown.
    optim_active: Lock to indicate if the optimizer stream is active.
    optim_thread: Daemon thread that executes optimizer tasks.
"""

from typing_extensions import *

import sys
import queue
import threading
import _thread
import atexit

import torch

from .device import get_num_devices
from .threads import RoundPipeThread

if sys.version_info >= (3, 9):
    KernelQueueType = queue.Queue[Union[Tuple[Callable, Tuple, Dict[str, Any]], object]]
else:
    KernelQueueType = queue.Queue
kernel_queue: KernelQueueType = queue.Queue()
OPTIM_STOP = object()


def controller() -> None:
    """Optimizer Stream thread function."""
    num_cpu = torch.get_num_threads()
    num_gpu = get_num_devices()
    if num_cpu > num_gpu * 4:
        torch.set_num_threads(num_cpu - num_gpu)

    while True:
        job = kernel_queue.get()
        if job is OPTIM_STOP:
            break
        fn, args, kwargs = cast(Tuple[Callable, Tuple, Dict[str, Any]], job)
        optim_thread.is_active = True
        fn(*args, **kwargs)
        optim_thread.is_active = False


optim_thread: RoundPipeThread = RoundPipeThread(
    target=controller, name="RoundPipe Optimizer Stream"
)


@atexit.register
def shutdown_optim() -> None:
    """Shut down the optimizer stream."""
    kernel_queue.put(OPTIM_STOP)
    optim_thread.join()


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
