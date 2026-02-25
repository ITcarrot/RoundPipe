# Optimizer

RoundPipe supports any optimizer from PyTorch and correctly synchronizes optimizer state in distributed training. For PyTorch optimizers that support a `fused` implementation, we recommend passing `fused=True` when creating the optimizer for significantly better performance. However, PyTorch's built-in optimizers have poor CPU performance, so we maintain CPU-optimized implementations of selected optimizers, compiled specifically for the host CPU to accelerate optimizer updates.

The following are the optimizer implementations maintained in RoundPipe, listed in alphabetical order.

## roundpipe.optim.Adam

```python
class roundpipe.optim.Adam(
    params: ParamsT,
    lr: Union[float, torch.Tensor] = 1e-3,
    betas: Tuple[Union[float, torch.Tensor], Union[float, torch.Tensor]] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    amsgrad: bool = False,
    *,
    foreach: Optional[bool] = None,
    maximize: bool = False,
    capturable: bool = False,
    differentiable: bool = False,
    fused: Optional[bool] = None,
    decoupled_weight_decay: bool = False,
)
```

Implements the Adam optimization algorithm, performing parameter updates on CPU in fp32 precision. This Adam implementation is compiled and optimized for CPU execution, offering better performance than PyTorch's native CPU Adam.

The interface is designed to be API-compatible with `torch.optim.Adam` and can be used as a drop-in replacement.

**Parameters:**

- `params`: An iterable of parameters or dicts defining parameter groups.
- `lr`: Learning rate. Defaults to `1e-3`.
- `betas`: Coefficients used for computing running averages of gradient and its square. Defaults to `(0.9, 0.999)`.
- `eps`: Term added to the denominator to improve numerical stability. Defaults to `1e-8`.
- `weight_decay`: Weight decay coefficient. Defaults to `0.0`.
- `amsgrad`: Whether to use the AMSGrad variant from the paper *On the Convergence of Adam and Beyond*. Defaults to `False`.
- `maximize`: Whether to maximize the objective function (instead of minimizing). Defaults to `False`.
- `decoupled_weight_decay`: If `True`, equivalent to the AdamW algorithm where weight decay does not accumulate in the momentum and variance terms. Defaults to `False`.
- `foreach`: Compatibility placeholder parameter; ignored with a warning if provided.
- `capturable`: Compatibility placeholder parameter; setting to `True` is not supported.
- `differentiable`: Compatibility placeholder parameter; setting to `True` is not supported.
- `fused`: Compatibility placeholder parameter; ignored with a warning if provided.

**Limitations:**

- Only supports `float32` tensors on CPU.
- Sparse gradients are not supported.
- All tensors must be contiguous.
