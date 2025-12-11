"""Thread helpers that keep RoundPipe worker threads observable and safe.

Attributes:
    roundpipe_threads: List of all RoundPipe threads created so far.
    thread_exception_print_lock: Lock to prevent interleaved exception prints.
    KeyboardInterruptRoundPipeThreads: KeyboardInterrupt raised when RoundPipe
        waits for its worker threads to finish.
"""

from beartype.typing import * # pyright: ignore[reportWildcardImportFromLibrary]

import threading
import traceback
import os
import time
import sys
import types
import _thread

class RoundPipeThread(threading.Thread):
    """Daemon thread wrapper that reports uncaught exceptions before exit.

    Attributes:
        is_active: Flag indicating whether the thread currently executes
            user work (used for debugging dumps).
    """

    def __init__(self, target: Callable, name: str, **kwargs: Any):
        """Wrap a target callable so crashes are surfaced immediately.

        Args:
            target: Callable that performs the thread's work.
            name: Friendly name to help when dumping thread stacks.
            **kwargs: Additional ``threading.Thread`` keyword arguments.
        """
        def exception_wrapper(*args, **kwds):
            try:
                target(*args, **kwds)
            except BaseException as e:
                if isinstance(e, Exception):
                    with thread_exception_print_lock:
                        print(f"Exception in {name}:")
                        traceback.print_exc()
                time.sleep(0.1)
                os._exit(1)

        super().__init__(target=exception_wrapper, name=name, daemon=True, **kwargs)
        self.is_active: bool = False
        roundpipe_threads.append(self)
        self.start()

roundpipe_threads: List[RoundPipeThread] = []
thread_exception_print_lock: _thread.LockType = threading.Lock()

KeyboardInterruptRoundPipeThreads: KeyboardInterrupt \
    = KeyboardInterrupt("KeyboardInterrupt when RoundPipe is waiting for its worker threads to finish their work. "
                        "Check traceback of worker threads for more details.")

def is_threading_internal(frame: types.FrameType) -> bool:
    """Return whether ``frame`` originates from Python or RoundPipe threading.

    Args:
        frame: Frame to inspect.

    Returns:
        ``True`` if the frame belongs to threading internals, else ``False``.
    """
    filename = frame.f_code.co_filename
    if filename.endswith("/threading.py"):
        return True
    if filename.endswith("/RoundPipe/threads.py"):
        return True
    return False

def print_trimmed_traceback(frame: Optional[types.FrameType]):
    """Print a traceback that omits internal threading frames.

    Args:
        frame: Frame whose stack should be printed.
    """
    trimmed_stack = []
    cur_frame = frame
    while cur_frame:
        trimmed_stack.append(cur_frame)
        cur_frame = cur_frame.f_back

    begin_idx = -1
    for begin_idx in range(len(trimmed_stack) - 1, -1, -1):
        if not is_threading_internal(trimmed_stack[begin_idx]):
            break
    traceback.print_stack(frame, begin_idx + 1)

def dump_all_active_threads():
    """Print trimmed stack traces for all currently active RoundPipe threads."""
    cur_frames = sys._current_frames()
    print("\n=== Dumping all active RoundPipe threads ===")
    for t in roundpipe_threads:
        if t.is_active and t.ident is not None:
            print(f"\n--- Thread: {t.name} (id={t.ident}) ---")
            print_trimmed_traceback(cur_frames[t.ident])
            print('')
    print("=== End dumping all active RoundPipe threads ===\n")
