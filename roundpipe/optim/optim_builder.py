"""Build, load and cache CPP implementation of optimizer functions.

Attributes:
    loaded_optim_functions: A dictionary caching loaded optimizer functions.
"""

from typing_extensions import *

import hashlib
import pathlib
import warnings

from torch.utils.cpp_extension import load, verify_ninja_availability
from cpuinfo import get_cpu_info

SRC_PATH = pathlib.Path(__file__).parent / "csrc"
CPP_FLAGS = ["-Ofast", "-fopenmp", "-mtune=native", "-march=native"]
# CPP_FLAGS.append("-fopt-info-vec-all=vec.log")

loaded_optim_functions: dict[str, Callable] = {}


def get_cpu_flags_hash() -> str:
    """Get a hash string representing the CPU flags.

    Returns:
        A string hash of the CPU flags.
    """
    try:
        info = get_cpu_info()
        flags = info.get("flags", [])
        flags_str = ",".join(sorted(flags))
        assert len(flags_str) > 0
        return hashlib.md5(flags_str.encode()).hexdigest()[:8]
    except Exception:
        warnings.warn(
            "Unable to get CPU flags for optimizer compilation. "
            "We cannot guarantee the cached optimizers are valid."
        )
        return "unknown"


def load_optim_function(name: str) -> None:
    """Compile and load the optimizer function from source by name.

    Args:
        name: The name of the optimizer function.
    """
    if name in loaded_optim_functions:
        return
    verify_ninja_availability()
    mod_name = f"roundpipe_optim_{name}_{get_cpu_flags_hash()}"
    source_files = [str(SRC_PATH / f"{name}.cpp")]
    loaded_module = load(name=mod_name, sources=source_files, extra_cflags=CPP_FLAGS)
    loaded_optim_functions[name] = getattr(loaded_module, name)


def get_optim_function(name: str) -> Callable:
    """Get the optimizer function by name.

    Args:
        name: The name of the optimizer function.

    Returns:
        The optimizer function.
    """
    load_optim_function(name)
    return loaded_optim_functions[name]
