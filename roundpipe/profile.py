"""Profiling utilities for RoundPipe.

Attributes:
    PROFILER_TYPE: Type of profiler detected from environment variables.
"""

from typing_extensions import *

import os
import contextlib

PROFILER_TYPE: Optional[str] = None

if os.environ.get("NSYS_PROFILING_SESSION_ID"):
    PROFILER_TYPE = "nsys"


def annotate(name: str, color: Optional[str] = None) -> ContextManager:
    """Return a context manager that instruments profiler annotations.

    Args:
        name: Label that appears in profiling timelines.
        color: Color that appear in profiling timelines.

    Returns:
        The annotation context when profiling is enabled, otherwise
            `contextlib.nullcontext` so callers can use ``with`` uniformly.
    """
    if PROFILER_TYPE == "nsys":
        import nvtx

        return nvtx.annotate(name, color=color)
    else:
        return contextlib.nullcontext()
