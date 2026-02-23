# 优化器

RoundPipe 支持 PyTorch 中的任何优化器，并且在分布式训练中能够正确地同步优化器状态。对于支持`fused`实现的 PyTorch 优化器，我们建议您在创建优化器时传入`fused=True`，可以显著提升性能。不过 PyTorch 中的优化器在 CPU 上的性能较差，因此我们维护了部分优化器的 CPU 实现，通过针对运行设备的 CPU 进行编译优化，加快优化器的更新速度。

以下是 RoundPipe 中维护的优化器实现，按照字母顺序排列。

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

实现 Adam 优化算法，使用 fp32 精度在 CPU 上执行参数更新。此 Adam 实现针对 CPU 执行进行了编译优化，相比 PyTorch 原生的 CPU Adam 有更好的性能。

接口设计与 `torch.optim.Adam` 兼容，可直接替换使用。

**参数：**

- `params`：参数迭代器或定义参数组的字典迭代器。
- `lr`：学习率。默认为 `1e-3`。
- `betas`：用于计算梯度及其平方的运行均值的系数。默认为 `(0.9, 0.999)`。
- `eps`：添加到分母以提高数值稳定性的项。默认为 `1e-8`。
- `weight_decay`：权重衰减系数。默认为 `0.0`。
- `amsgrad`：是否使用论文 *On the Convergence of Adam and Beyond* 中的 AMSGrad 变体。默认为 `False`。
- `maximize`：是否最大化目标函数（而非最小化）。默认为 `False`。
- `decoupled_weight_decay`：如果为 `True`，等效于 AdamW 算法，权重衰减不会累积在动量和方差中。默认为 `False`。
- `foreach`：兼容占位参数，传入时会被忽略并产生警告。
- `capturable`：兼容占位参数，不支持设为 `True`。
- `differentiable`：兼容占位参数，不支持设为 `True`。
- `fused`：兼容占位参数，传入时会被忽略并产生警告。

**限制：**

- 仅支持 CPU 上的 `float32` 张量。
- 不支持稀疏梯度。
- 所有张量必须是连续的（contiguous）。
