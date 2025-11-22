from beartype.typing import * # type: ignore[reportWildcardImport]

import threading
import traceback
import os
import time
import sys
import types

class RoundPipeThread(threading.Thread):
    def __init__(self, target, name, **kwargs):
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
        self.is_active = False
        roundpipe_threads.append(self)
        self.start()

roundpipe_threads: List[RoundPipeThread] = []
thread_exception_print_lock = threading.Lock()

KeyboardInterruptRoundPipeThreads \
    = KeyboardInterrupt("KeyboardInterrupt when RoundPipe is waiting for its worker threads to finish their work. "
                        "Check traceback of worker threads for more details.")

def is_threading_internal(frame: types.FrameType):
    filename = frame.f_code.co_filename
    if filename.endswith("/threading.py"):
        return True
    if filename.endswith("/RoundPipe/threads.py"):
        return True
    return False

def print_trimmed_traceback(frame: Optional[types.FrameType]):
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
    cur_frames = sys._current_frames()
    print("\n=== Dumping all active RoundPipe threads ===")
    for t in roundpipe_threads:
        if t.is_active and t.ident is not None:
            print(f"\n--- Thread: {t.name} (id={t.ident}) ---")
            print_trimmed_traceback(cur_frames[t.ident])
            print('')
    print("=== End dumping all active RoundPipe threads ===\n")
