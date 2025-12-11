from beartype.typing import * # type: ignore[reportWildcardImportFromLibrary]
import types
import math

import torch
import torch.nn as nn

from ..RoundPipe import RoundPipe

class ChunkedCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, logits: torch.Tensor, labels: torch.Tensor,
                num_items_in_batch: torch.Tensor, ignore_index: int) -> torch.Tensor:
        CHUNK_SIZE = 64 * 1024 * 1024  # 64M logits
        ctx.ignore_index = ignore_index
        ctx.save_for_backward(logits, labels, num_items_in_batch)
        ctx.chunk_tokens = math.ceil(CHUNK_SIZE / logits.size(1))

        chunked_logits = logits.split(ctx.chunk_tokens, dim=0)
        chunked_labels = labels.split(ctx.chunk_tokens, dim=0)
        loss = torch.tensor(0.0, device=logits.device, dtype=torch.float32)
        for logit_chunk, label_chunk in zip(chunked_logits, chunked_labels):
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logit_chunk = logit_chunk.float()
            loss_chunk = nn.functional.cross_entropy(logit_chunk, label_chunk, ignore_index=ignore_index, reduction='sum')
            loss += loss_chunk / num_items_in_batch

        return loss

    @staticmethod
    def backward(ctx: Any, grad_loss: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]: # type: ignore[override]
        logits, labels, num_items_in_batch = ctx.saved_tensors
        grad_logits = torch.empty_like(logits)

        chunked_logits = logits.split(ctx.chunk_tokens, dim=0)
        chunked_grad = grad_logits.split(ctx.chunk_tokens, dim=0)
        chunked_labels = labels.split(ctx.chunk_tokens, dim=0)
        with torch.enable_grad():
            for logit_chunk, grad_chunk, label_chunk in zip(chunked_logits, chunked_grad, chunked_labels):
                # Upcast to float if we need to compute the loss to avoid potential precision issues
                logit_chunk = logit_chunk.float().detach()
                logit_chunk.requires_grad_(True)
                loss_chunk = nn.functional.cross_entropy(logit_chunk, label_chunk, ignore_index=ctx.ignore_index, reduction='sum')
                torch.autograd.backward(loss_chunk, grad_loss / num_items_in_batch)
                grad_chunk.copy_(logit_chunk.grad)

        return grad_logits, None, None, None

def ChunkedForCausalLMLoss(
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
    elif isinstance(num_items_in_batch, int):
        num_items_in_batch = torch.tensor(num_items_in_batch, device=logits.device)

    return ChunkedCrossEntropy.apply(logits, shift_labels, num_items_in_batch, ignore_index) # pyright: ignore[reportReturnType]

LOSS_REPLACE = {}

try:
    from transformers.loss.loss_utils import ForCausalLMLoss
    LOSS_REPLACE[ForCausalLMLoss] = ChunkedForCausalLMLoss
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
