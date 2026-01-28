from typing_extensions import *
import types

import torch
import torch.nn as nn

from . import DISABLE_TORCH_COMPILE
from ..roundpipe import RoundPipe


@torch.compile(disable=DISABLE_TORCH_COMPILE)
def CompileCrossEntropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int,
) -> torch.Tensor:
    return nn.functional.cross_entropy(
        logits.float(), labels, ignore_index=ignore_index, reduction="sum"
    )


def CompileForCausalLMLoss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    vocab_size: int,
    num_items_in_batch: Union[torch.Tensor, int, None] = None,
    ignore_index: int = -100,
    shift_labels: Optional[torch.Tensor] = None,
    **kwargs: Any,
) -> torch.Tensor:
    if shift_labels is None:
        # Shift so that tokens < n predict n
        labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    logits = logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(logits.device)
    if num_items_in_batch is None:
        num_items_in_batch = (shift_labels != ignore_index).sum()

    return CompileCrossEntropy(logits, shift_labels, ignore_index) / num_items_in_batch


LOSS_REPLACE = {}

try:
    from transformers.loss.loss_utils import ForCausalLMLoss

    LOSS_REPLACE[ForCausalLMLoss] = CompileForCausalLMLoss
except ImportError:
    pass


class FunctionWrapper(nn.Module):
    def __init__(self, func: Callable) -> None:
        super().__init__()
        self.func = func

    def forward(self, *args, **kwargs):
        return self.func(*args, **kwargs)


EXPECTED_MODEL_CLASS = types.FunctionType


def wrap_model(func: types.FunctionType, **roundpipe_kwargs: Any) -> RoundPipe:
    if func in LOSS_REPLACE:
        func = LOSS_REPLACE[func]
    return RoundPipe(FunctionWrapper(func), **roundpipe_kwargs)
