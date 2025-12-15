'''Module defining parameter attributes for RoundPipe.'''
from beartype.typing import * # pyright: ignore[reportWildcardImportFromLibrary]

import torch

class ParamAttribute:
    '''Class storing parameter attributes for RoundPipe.
    
    Attributes:
        data_cpu: The CPU copy of the parameter data.
        data_version: Which data_optim version used in data_cpu.
        data_grad: A temporary location to hold reference to gradient.
        data_optim: The optimizer copy of the parameter data.
        uploaded_grad: Whether the gradient is uploaded from cpu.
    '''
    @classmethod
    def set(cls, t: torch.Tensor) -> None:
        '''Attach a ParamAttribute to a tensor.'''
        attr = cls(t.data)
        t.roundpipe_param_attr = attr # pyright: ignore[reportAttributeAccessIssue]

    @classmethod
    def has(cls, t: torch.Tensor) -> bool:
        '''Check if a tensor has a ParamAttribute attached.'''
        return hasattr(t, 'roundpipe_param_attr')

    @classmethod
    def get(cls, t: torch.Tensor) -> 'ParamAttribute':
        '''Get the ParamAttribute attached to a tensor.'''
        return t.roundpipe_param_attr # pyright: ignore[reportAttributeAccessIssue]

    def __init__(self, data: torch.Tensor) -> None:
        '''Initialize the ParamAttribute with the given data.
        
        Args:
            data: The `.data` object of the parameter tensor.
        '''
        self.data_cpu: torch.Tensor = data
        self.data_version: int = data._version
        self.data_grad: Optional[torch.Tensor] = None
        self.data_optim: torch.Tensor = data
        self.uploaded_grad: bool = False

    def optim_inited(self) -> bool:
        '''Check if the optimizer copy has been initialized.'''
        return self.data_optim is not self.data_cpu
