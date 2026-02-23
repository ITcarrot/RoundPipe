# 运行时上下文

RoundPipe 使用基于线程局部变量的上下文管理器来跟踪当前的执行阶段（前向传播、重计算、优化器更新）。用户可以通过这些 API 在模型的前向传播中保存数据以供重计算使用，或者在优化器相关操作时正确获取参数。

## roundpipe.save_for_recompute

```python
roundpipe.save_for_recompute(*data: Any) -> None
```

在当前前向传播上下文中保存数据，供后续重计算阶段使用。

RoundPipe 在反向传播时需要重新执行前向传播来恢复中间激活值。如果模型的某些中间结果会导致 GPU - CPU 同步（例如 `torch.nonzero()`），可以在前向传播时通过此函数保存，避免重计算时引入同步开销。

**参数：**

- `*data`：需要保存的任意数据。保存的张量不能需要梯度（`requires_grad=False`）。

**注意事项：**

- 每个层的前向传播中最多调用一次。
- 如果当前未启用梯度计算，则此函数为空操作。
- 保存的数据可以在重计算阶段通过 `get_recompute_data()` 获取。

**示例：**

```python
import torch
import torch.nn as nn
from roundpipe import save_for_recompute, doing_recompute, get_recompute_data

class MyLayer(nn.Module):
    def forward(self, x):
        if doing_recompute():
            # 重计算阶段：直接使用保存的 mask
            mask, = get_recompute_data()
        else:
            # 前向传播阶段：生成并保存 mask
            mask = torch.nonzero(x)
            save_for_recompute(mask)
        return x, mask
```

## roundpipe.doing_recompute

```python
roundpipe.doing_recompute() -> bool
```

检查当前作用域是否处于重计算阶段。

RoundPipe 在反向传播时会重新执行前向传播来恢复激活值，此函数可用于在前向传播代码中区分是首次前向计算还是重计算阶段，从而执行不同的逻辑。

**返回值：**

- `bool`：如果当前处于重计算上下文中，返回 `True`；否则返回 `False`。

## roundpipe.get_recompute_data

```python
roundpipe.get_recompute_data() -> tuple
```

获取在前向传播中通过 `save_for_recompute()` 保存的数据。

**返回值：**

- `tuple`：保存的数据。即使只保存了一个元素，也会以元组形式返回。

**注意事项：**

- 此函数只能在重计算上下文中调用。如果在重计算上下文外调用，将抛出 `AssertionError`。

## roundpipe.OptimizerCtx

```python
class roundpipe.OptimizerCtx
```

上下文管理器，用于标记当前作用域正在执行优化器相关操作。

在此上下文中，`RoundPipe` 会将 `.parameters()` 和 `.named_parameters()` 重定向到 `.optim_parameters()` 和 `.optim_named_parameters()`。这使得用户可以使用标准 PyTorch 优化器创建方式，而无需显式调用优化器专用的参数接口。

**使用场景：**

当创建优化器或执行其他需要访问优化器参数的操作时，应使用此上下文管理器包裹相关代码。

**示例：**

```python
from roundpipe import RoundPipe, OptimizerCtx
from roundpipe.optim import Adam

model = RoundPipe(my_model)

# 方式一：使用 OptimizerCtx（推荐，与 PyTorch 习惯一致）
with OptimizerCtx():
    optimizer = Adam(model.parameters(), lr=0.001)

# 方式二：直接使用 optim_parameters（等效）
optimizer = Adam(model.optim_parameters(), lr=0.001)
```
