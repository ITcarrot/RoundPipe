import sys
from importlib.metadata import version
from beartype import BeartypeConf
from beartype.claw import beartype_this_package

beartype_version = tuple(map(int, version("beartype").split(".")[:2]))
ENABLE_BEAR = beartype_version >= (0, 22)
if ENABLE_BEAR:
    beartype_this_package(conf=BeartypeConf(violation_type=UserWarning))
else:
    print(
        "[info] Upgrade beartype to >= 0.22 can enable runtime type checking for RoundPipe.",
        file=sys.stderr,
    )

from .context import (
    doing_recompute,
    OptimizerCtx,
    get_recompute_data,
    save_for_recompute,
)
from .grad_scaler import GradScaler
from . import optim
from .roundpipe import RoundPipe
from .run_config import RoundPipeRunConfig
from .scheduler import ModelExecutePlan
from .wrapper import wrap_model_to_roundpipe

__version__ = "0.1.0"
__all__ = [
    "doing_recompute",
    "OptimizerCtx",
    "get_recompute_data",
    "save_for_recompute",
    "GradScaler",
    "optim",
    "RoundPipe",
    "RoundPipeRunConfig",
    "ModelExecutePlan",
    "wrap_model_to_roundpipe",
]
