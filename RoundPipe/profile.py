import os
import contextlib
from typing import * # type: ignore[reportWildcardImportFromLibrary]

PROFILER_TYPE = None

if os.environ.get('NSYS_PROFILING_SESSION_ID'):
    PROFILER_TYPE = 'nsys'
    import nvtx

def annotate(name: str, color: Optional[str] = None):
    if PROFILER_TYPE == 'nsys':
        return nvtx.annotate(name, color = color) # type: ignore[reportPossiblyUnboundVariable]
    else:
        return contextlib.nullcontext()
