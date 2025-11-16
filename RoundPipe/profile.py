from beartype.typing import * # type: ignore[reportWildcardImportFromLibrary]

import os
import contextlib

PROFILER_TYPE = None

if os.environ.get('NSYS_PROFILING_SESSION_ID'):
    PROFILER_TYPE = 'nsys'

def annotate(name: str, color: Optional[str] = None):
    if PROFILER_TYPE == 'nsys':
        import nvtx
        return nvtx.annotate(name, color = color)
    else:
        return contextlib.nullcontext()
