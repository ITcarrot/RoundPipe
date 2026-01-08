from typing_extensions import *

import itertools
import random

import pytest
import torch

from roundpipe.optim import Adam


def run_optim(
    cls: type[torch.optim.Optimizer],
    ref_cls: type[torch.optim.Optimizer],
    *optim_args: Any,
    **optim_kwargs: Any
) -> None:
    params: List[torch.Tensor] = []
    for n in range(1, 11):
        params.append(
            torch.randn(n, n + 1, n + 2, dtype=torch.float32, requires_grad=True)
        )
    for n in range(1025):
        params.append(torch.randn(n, dtype=torch.float32, requires_grad=True))
    for n in range(1025):
        params.append(torch.randn(n, dtype=torch.complex64, requires_grad=True))
    random.shuffle(params)
    ref_params = [p.clone().detach().requires_grad_(True) for p in params]

    optimizer = cls(params, *optim_args, **optim_kwargs)
    ref_optimizer = ref_cls(ref_params, *optim_args, **optim_kwargs)
    for _ in range(10):
        for p, ref_p in zip(params, ref_params):
            p.grad = torch.randn_like(p)
            ref_p.grad = p.grad.clone()
        optimizer.step()
        ref_optimizer.step()
        for p, ref_p in zip(params, ref_params):
            assert torch.allclose(p, ref_p, atol=1e-6)


@pytest.mark.parametrize(
    "lr_betas, weight_decay, amsgrad, maximize, decoupled_weight_decay",
    itertools.product(
        [
            (0.01, 0.9, 0.999),
            (torch.tensor(0.001), torch.tensor(0.8), torch.tensor(0.95)),
        ],
        [0.0, 0.01],
        [True, False],
        [True, False],
        [True, False],
    ),
)
def test_Adam(
    lr_betas: Union[
        Tuple[float, float, float], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ],
    weight_decay: float,
    amsgrad: bool,
    maximize: bool,
    decoupled_weight_decay: bool,
) -> None:
    lr, *betas = lr_betas
    run_optim(
        Adam,
        torch.optim.Adam,
        lr=lr,
        betas=tuple(betas),
        eps=1e-4,
        weight_decay=weight_decay,
        amsgrad=amsgrad,
        maximize=maximize,
        decoupled_weight_decay=decoupled_weight_decay,
    )
