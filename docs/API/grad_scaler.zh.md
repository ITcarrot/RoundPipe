# 梯度缩放器

在混合精度训练中，使用低精度浮点数（如 `float16`）计算时，梯度值可能因数值范围有限而出现下溢（变为零）。梯度缩放通过在反向传播前放大损失值、在优化器更新前缩小梯度来解决这一问题，从而在保持训练精度的同时利用低精度带来的性能优势。

由于 RoundPipe 的优化器更新是异步执行的（在后台线程中与 GPU 计算并行运行），PyTorch 原生的 `torch.amp.GradScaler` 无法直接适配这种执行模式。因此，RoundPipe 重新实现了 `GradScaler`，使其能够正确处理主线程和优化器线程之间的缩放因子同步。

RoundPipe 的 `GradScaler` 接口设计与 `torch.amp.GradScaler` 完全一致，可以无缝替换。

## roundpipe.GradScaler

```python
class roundpipe.GradScaler(
    init_scale: float = 2.0 ** 16,
    growth_factor: float = 2.0,
    backoff_factor: float = 0.5,
    growth_interval: int = 2000,
    enabled: bool = True,
)
```

梯度缩放器，帮助简便地执行梯度缩放的各个步骤。

**参数：**

- `init_scale`：初始缩放因子。默认为 `2^16 = 65536`。
- `growth_factor`：在连续 `growth_interval` 次迭代未出现 inf/NaN 梯度时，缩放因子的增长倍数。默认为 `2.0`。
- `backoff_factor`：在某次迭代出现 inf/NaN 梯度时，缩放因子的回退倍数。默认为 `0.5`。
- `growth_interval`：连续多少次迭代未出现 inf/NaN 梯度后，缩放因子乘以 `growth_factor`。默认为 `2000`。
- `enabled`：是否启用梯度缩放。设为 `False` 时，`step` 直接调用优化器的 `step()`，其他方法变为空操作。

**示例：**

```python
from roundpipe import RoundPipe, GradScaler
from roundpipe.optim import Adam

scaler = GradScaler()
model = RoundPipe(my_model.to(torch.float16), optim_dtype=torch.float32)
optimizer = Adam(model.optim_parameters(), lr=1e-3)

for data, labels in dataloader:
    loss = model.forward_backward(
        input_args=(data,),
        label=labels,
        loss_fn=lambda out, lbl: scaler.scale(
            criterion(out.float(), lbl)
        ) / num_microbatch,
    )
    model.step(lambda: (scaler.step(optimizer), optimizer.zero_grad()))
    scaler.update()
```

### GradScaler.scale

```python
GradScaler.scale(
    outputs: Union[torch.Tensor, Iterable[torch.Tensor]],
) -> Union[torch.Tensor, Iterable[torch.Tensor]]
```

将张量或张量列表乘以缩放因子。

通常在 `loss_fn` 中调用，对损失值进行缩放，使得反向传播产生的梯度处于 `float16` 可表示的数值范围内。

**参数：**

- `outputs`：需要缩放的输出张量或张量可迭代对象。

**返回值：**

- 缩放后的张量。如果未启用缩放，原样返回。

### GradScaler.unscale_

```python
GradScaler.unscale_(optimizer: Optimizer) -> None
```

将优化器中的梯度张量除以缩放因子（反缩放）。

`unscale_` 是可选的，用于在反向传播和 `step` 之间需要修改或检查梯度的场景。如果未显式调用 `unscale_`，梯度将在 `step` 中自动反缩放。

如果从主线程调用此方法，反缩放操作将在优化器线程上执行并同步。

**参数：**

- `optimizer`：拥有待反缩放梯度的优化器。

### GradScaler.step

```python
GradScaler.step(
    optimizer: Optimizer,
    *args: Any,
    **kwargs: Any,
) -> Optional[float]
```

执行 `unscale_` 后进行参数更新（如果梯度不包含 inf/NaN）。

此方法依次执行两个操作：

1. 如果尚未对该优化器调用过 `unscale_`，则自动调用。同时检查梯度是否包含 inf/NaN。
2. 如果未发现 inf/NaN 梯度，调用 `optimizer.step()` 进行参数更新；否则跳过本次更新，避免损坏参数。

**参数：**

- `optimizer`：执行梯度更新的优化器。
- `*args`：转发给 `optimizer.step()` 的位置参数。
- `**kwargs`：转发给 `optimizer.step()` 的关键字参数。

**返回值：**

- 如果未启用缩放，返回 `optimizer.step()` 的返回值。如果启用缩放且在优化器线程上执行，返回该返回值；否则返回 `None`。

### GradScaler.update

```python
GradScaler.update(new_scale: Optional[Union[float, torch.Tensor]] = None) -> None
```

更新缩放因子。此方法必须从主线程调用。

如果之前的优化器步骤被跳过（因为梯度包含 inf/NaN），缩放因子将乘以 `backoff_factor` 以减小。如果连续 `growth_interval` 次迭代未跳过，缩放因子将乘以 `growth_factor` 以增大。

**参数：**

- `new_scale`：手动设置的新缩放因子。如果不为 `None`，使用此值替代自动计算的缩放因子。该值会被复制到内部张量，后续对传入张量的修改不会影响缩放器。

### GradScaler.get_scale

```python
GradScaler.get_scale() -> float
```

返回当前的缩放因子。

根据调用所在的线程（主线程或优化器线程），返回对应线程使用的缩放因子。

**返回值：**

- 当前缩放因子的 Python `float` 值。如果未启用缩放，返回 `1.0`。

### GradScaler.get_growth_factor

```python
GradScaler.get_growth_factor(up_to_date: bool = False) -> float
```

获取缩放因子的增长倍数。

**参数：**

- `up_to_date`：如果为 `True`，确保返回最新值（会阻塞并与优化器线程同步）。否则，可能返回上一次 `update()` 之前的值。

### GradScaler.set_growth_factor

```python
GradScaler.set_growth_factor(new_factor: float) -> None
```

设置新的增长倍数。

### GradScaler.get_backoff_factor

```python
GradScaler.get_backoff_factor(up_to_date: bool = False) -> float
```

获取缩放因子的回退倍数。

**参数：**

- `up_to_date`：如果为 `True`，确保返回最新值（会阻塞并与优化器线程同步）。否则，可能返回上一次 `update()` 之前的值。

### GradScaler.set_backoff_factor

```python
GradScaler.set_backoff_factor(new_factor: float) -> None
```

设置新的回退倍数。

### GradScaler.get_growth_interval

```python
GradScaler.get_growth_interval(up_to_date: bool = False) -> int
```

获取增长间隔。

**参数：**

- `up_to_date`：如果为 `True`，确保返回最新值（会阻塞并与优化器线程同步）。否则，可能返回上一次 `update()` 之前的值。

### GradScaler.set_growth_interval

```python
GradScaler.set_growth_interval(new_interval: int) -> None
```

设置新的增长间隔。
