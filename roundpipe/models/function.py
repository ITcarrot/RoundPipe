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


@torch.compile(disable=DISABLE_TORCH_COMPILE)
def CompileLinearCrossEntropy(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    labels: torch.Tensor,
    ignore_index: int,
):
    logits = nn.functional.linear(hidden_states, weight, bias)
    return nn.functional.cross_entropy(
        logits.float(), labels, ignore_index=ignore_index, reduction="sum"
    )


TOKEN_CHUNK_SIZE = 4096


class ChunkedCompileLinearCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        labels: torch.Tensor,
        ignore_index: int,
        global_requires_grad: bool,
    ) -> torch.Tensor:
        hidden_states = hidden_states.detach().requires_grad_(ctx.needs_input_grad[0])
        weight = weight.detach().requires_grad_(ctx.needs_input_grad[1])
        if bias is not None:
            bias = bias.detach().requires_grad_(ctx.needs_input_grad[2])
        loss_sum = torch.zeros((), device=hidden_states.device)
        grad_context = torch.enable_grad() if global_requires_grad else torch.no_grad()
        with grad_context:
            for hidden_state, shift_label in zip(
                hidden_states.split(TOKEN_CHUNK_SIZE, dim=0),
                labels.split(TOKEN_CHUNK_SIZE, dim=0),
            ):
                loss = CompileLinearCrossEntropy(
                    hidden_state, weight, bias, shift_label, ignore_index
                )
                if loss.requires_grad:
                    loss.backward()
                loss_sum += loss.detach()
        grads: List[torch.Tensor] = []
        if hidden_states.grad is not None:
            grads.append(hidden_states.grad)
        if weight.grad is not None:
            grads.append(weight.grad)
        if bias is not None and bias.grad is not None:
            grads.append(bias.grad)
        ctx.save_for_backward(*grads)
        return loss_sum

    @staticmethod
    def backward(  # pyright: ignore[reportIncompatibleMethodOverride]
        ctx: Any, grad_loss: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], ...]:
        grads: Tuple[torch.Tensor, ...] = ctx.saved_tensors
        grads_idx = 0
        grad_outputs = []
        for i in range(3):
            if ctx.needs_input_grad[i]:
                grad_outputs.append(grads[grads_idx].mul_(grad_loss))
                grads_idx += 1
            else:
                grad_outputs.append(None)
        return *grad_outputs, None, None, None


def ChunkedCompileLinearForCausalLMLoss(
    hidden_states: torch.Tensor,
    lm_head: nn.Linear,
    labels: torch.Tensor,
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
    hidden_states = hidden_states.view(-1, hidden_states.size(-1))
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(hidden_states.device)
    if num_items_in_batch is None:
        num_items_in_batch = (shift_labels != ignore_index).sum()

    loss = cast(
        torch.Tensor,
        ChunkedCompileLinearCrossEntropy.apply(
            hidden_states,
            lm_head.weight,
            lm_head.bias,
            shift_labels,
            ignore_index,
            torch.is_grad_enabled(),
        ),
    )
    return loss / num_items_in_batch


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
