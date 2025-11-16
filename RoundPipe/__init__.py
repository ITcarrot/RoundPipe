from beartype import BeartypeConf
from beartype.claw import beartype_this_package
beartype_this_package(conf=BeartypeConf(violation_type=UserWarning))

from .RoundPipe import RoundPipe
from .wrapper import wrap_model_to_roundpipe
from .RunConfig import RoundPipeRunConfig

__version__ = '0.1.0'
__all__ = ['RoundPipe', 'RoundPipeRunConfig', 'wrap_model_to_roundpipe']
