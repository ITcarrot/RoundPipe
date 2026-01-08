from typing_extensions import *

import warnings

import torch
from torch.optim.optimizer import Optimizer, ParamsT, _get_scalar_dtype

from .optim_builder import get_optim_function, load_optim_function

class Adam(Optimizer):
    """Implements Adam algorithm with fp32 stepping on CPU.
    """
    def __init__(self, params: ParamsT, lr: Union[float, torch.Tensor] = 1e-3,
        betas: tuple[Union[float, torch.Tensor], Union[float, torch.Tensor]] = (0.9, 0.999),
        eps: float = 1e-8, weight_decay: float = 0., amsgrad: bool = False, *,
        foreach: Optional[bool] = None, maximize: bool = False, capturable: bool = False,
        differentiable: bool = False, fused: Optional[bool] = None, decoupled_weight_decay: bool = False,
    ):
        """
        Args:
            params: iterable of parameters or named_parameters to optimize or
                iterable of dicts defining parameter groups. When using named_parameters,
                all parameters in all groups should be named
            lr: learning rate.
            betas: coefficients used for computing running averages of gradient and its square
            eps: term added to the denominator to improve numerical stability
            weight_decay: weight decay coefficient
            amsgrad: whether to use the AMSGrad variant of this algorithm from the paper
                `On the Convergence of Adam and Beyond`
            maximize: maximize the objective with respect to the params, instead of minimizing
            decoupled_weight_decay: if True, this optimizer is equivalent to AdamW and the
                algorithm will not accumulate weight decay in the momentum nor variance.
            foreach: Compatible placeholder for PyTorch's Adam optimizer.
            capturable: Compatible placeholder for PyTorch's Adam optimizer.
            differentiable: Compatible placeholder for PyTorch's Adam optimizer.
            fused: Compatible placeholder for PyTorch's Adam optimizer.
        """
        load_optim_function("adam")
        assert capturable is False, "capturable=True is not supported."
        assert differentiable is False, "differentiable=True is not supported."
        if fused is not None:
            warnings.warn("The fused option is not supported and will be ignored.", UserWarning)
        if foreach is not None:
            warnings.warn("The foreach option is not supported and will be ignored.", UserWarning)

        if isinstance(lr, torch.Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if isinstance(betas[0], torch.Tensor) and betas[0].numel() != 1:
                raise ValueError("Tensor betas[0] must be 1-element")
        if isinstance(betas[1], torch.Tensor) and betas[1].numel() != 1:
                raise ValueError("Tensor betas[1] must be 1-element")

        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            amsgrad=amsgrad, maximize=maximize,
            decoupled_weight_decay=decoupled_weight_decay,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state: dict[str, Any]):
        """Sets the state of the optimizer.
        This method is used when loading a saved optimizer state.
        It ensures that all necessary keys are present in the state dictionary

        Args:
            state: The state dictionary to set.
        """
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)
            group.setdefault("maximize", False)
            group.setdefault("decoupled_weight_decay", False)
            for p in group["params"]:
                p_state = self.state.get(p, [])
                if len(p_state) != 0 and not torch.is_tensor(p_state["step"]):
                    step_val = float(p_state["step"])
                    p_state["step"] = torch.tensor(step_val, dtype=_get_scalar_dtype())

    def _init_group(self, group: Dict[str, Any], params_with_grad: List[torch.Tensor],
        grads: List[torch.Tensor], exp_avgs: List[torch.Tensor],
        exp_avg_sqs: List[torch.Tensor], max_exp_avg_sqs: List[torch.Tensor],
        state_steps: List[torch.Tensor],
    ):
        """Initializes the state for each parameter group.
        Results are stored in the provided lists inplace.

        Args:
            group: The parameter group to initialize.
            params_with_grad: List to store parameters with gradients.
            grads: List to store gradients of the parameters.
            exp_avgs: List to store exponential moving averages of gradients.
            exp_avg_sqs: List to store exponential moving averages of squared gradients.
            max_exp_avg_sqs: List to store maximum exponential moving averages of squared gradients.
            state_steps: List to store the step count for each parameter.
        """
        for p in group["params"]:
            if p.grad is not None:
                if p.grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )
                params_with_grad.append(p)
                grads.append(p.grad)

                state = self.state[p]
                # Lazy state initialization
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0, dtype=_get_scalar_dtype())
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    if group["amsgrad"]:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["max_exp_avg_sq"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )

                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])
                if group["amsgrad"]:
                    max_exp_avg_sqs.append(state["max_exp_avg_sq"])
                state_steps.append(state["step"])

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], Any]] = None) -> Any:
        """Performs a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.

        Returns:
            The loss value if a closure is provided, otherwise None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad: List[torch.Tensor] = []
            grads: List[torch.Tensor] = []
            exp_avgs: List[torch.Tensor] = []
            exp_avg_sqs: List[torch.Tensor] = []
            max_exp_avg_sqs: List[torch.Tensor] = []
            state_steps: List[torch.Tensor] = []
            beta1, beta2 = group["betas"]

            self._init_group(group, params_with_grad, grads, exp_avgs,
                exp_avg_sqs, max_exp_avg_sqs, state_steps,
            )

            adam(params_with_grad, grads, exp_avgs, exp_avg_sqs,
                 max_exp_avg_sqs, state_steps, amsgrad=group["amsgrad"],
                 beta1=beta1, beta2=beta2, lr=group["lr"],
                 weight_decay=group["weight_decay"],
                 eps=group["eps"], maximize=group["maximize"],
                 decoupled_weight_decay=group["decoupled_weight_decay"],
            )

        return loss

def adam(params: list[torch.Tensor], grads: list[torch.Tensor],
         exp_avgs: list[torch.Tensor], exp_avg_sqs: list[torch.Tensor],
         max_exp_avg_sqs: list[torch.Tensor], state_steps: list[torch.Tensor],
         foreach: Optional[bool] = None, capturable: bool = False,
         differentiable: bool = False, fused: Optional[bool] = None,
         grad_scale: Optional[torch.Tensor] = None, found_inf: Optional[torch.Tensor] = None,
         has_complex: bool = False, decoupled_weight_decay: bool = False, *,
         amsgrad: bool, beta1: Union[torch.Tensor, float], beta2: Union[torch.Tensor, float],
         lr: Union[float, torch.Tensor], weight_decay: float, eps: float, maximize: bool,
):
    """Functional API that performs Adam algorithm computation.

    See `roundpipe.optim.Adam` for details.
    """
    assert not capturable, "capturable=True is not supported."
    assert not differentiable, "differentiable=True is not supported."
    if fused is not None:
        warnings.warn("The fused option is not supported and will be ignored.", UserWarning)
    if foreach is not None:
        warnings.warn("The foreach option is not supported and will be ignored.", UserWarning)
    assert grad_scale is None and found_inf is None, "integrated grad scaling is not supported."

    lr, beta1, beta2 = float(lr), float(beta1), float(beta2)
    step_int: List[int] = []
    for step_t in state_steps:
        step_t += 1
        step_int.append(int(step_t.item()))
    for tensor_list in (params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs):
        for i, t in enumerate(tensor_list):
            if torch.is_complex(t):
                tensor_list[i] = t = torch.view_as_real(t)
            assert t.is_cpu, "RoundPipe Adam only supports CPU tensors."
            assert t.dtype is torch.float32, "RoundPipe Adam only supports float32 tensors."
            assert t.is_contiguous(), "All tensors must be contiguous."

    adam_kernel = get_optim_function("adam")
    adam_kernel(params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs,
                step_int, amsgrad, beta1, beta2, lr,
                weight_decay, eps, maximize, decoupled_weight_decay)
