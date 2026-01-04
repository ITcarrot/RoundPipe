from beartype import BeartypeConf
from beartype.claw import beartype_this_package
beartype_this_package(conf=BeartypeConf(violation_type=UserWarning))

from .context import doing_recompute, OptimizerCtx, get_recompute_data, save_for_recompute
from .grad_scaler import GradScaler
from . import optim
from .RoundPipe import RoundPipe
from .RunConfig import RoundPipeRunConfig
from .wrapper import wrap_model_to_roundpipe

__version__ = '0.1.0'
__all__ = [
    'doing_recompute', 'OptimizerCtx', 'get_recompute_data', 'save_for_recompute',
    'GradScaler',
    'optim',
    'RoundPipe',
    'RoundPipeRunConfig',
    'wrap_model_to_roundpipe'
]
