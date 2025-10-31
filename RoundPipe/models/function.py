from typing import * # type: ignore[reportWildcardImportFromLibrary]

import torch.nn as nn

from RoundPipe import RoundPipe

class FunctionWrapper(nn.Module):
    def __init__(self, func: Callable) -> None:
        super().__init__()
        self.func = func

    def forward(self, *args, **kwargs):
        return self.func(*args, **kwargs)

def wrap_function(func: Callable, **roundpipe_kwargs: Any) -> RoundPipe:
    return RoundPipe(FunctionWrapper(func), **roundpipe_kwargs)
