import threading
import traceback
import os
import time
import torch.nn as nn

def get_model_size(model: nn.Module, recurse: bool = True) -> int:
    param_size = sum(p.numel() * p.element_size() for p in model.parameters(recurse))
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers(recurse))
    return param_size + buffer_size

thread_exception_print_lock = threading.Lock()
def thread_exception_handler(args: threading.ExceptHookArgs) -> None:
    if args.exc_type is not SystemExit:
        with thread_exception_print_lock:
            print(f"Unhandled exception in thread {args.thread.name if args.thread else 'unknown'}:")
            traceback.print_exception(args.exc_type, args.exc_value, args.exc_traceback)
    time.sleep(0.1)
    os._exit(1)
threading.excepthook = thread_exception_handler
