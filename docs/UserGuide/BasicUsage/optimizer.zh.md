# 创建优化器

RoundPipe 支持任意 PyTorch 优化器。由于 RoundPipe 内部管理参数在 CPU 和 GPU 之间的传输以及优化器参数的数据类型转换，创建优化器时需要使用 RoundPipe 提供的参数接口。

有两种等价的方式创建优化器：

```python
from roundpipe import RoundPipe, OptimizerCtx
from roundpipe.optim import Adam

model = RoundPipe(my_model.to(torch.float16), optim_dtype=torch.float32)

# 方式一：使用 OptimizerCtx（推荐，与 PyTorch 习惯一致）
with OptimizerCtx():
    optimizer = Adam(model.parameters(), lr=1e-3)

# 方式二：直接使用 optim_parameters()（等价）
optimizer = Adam(model.optim_parameters(), lr=1e-3)
```

两种方式完全等价，选择你觉得更自然的即可。

## OptimizerCtx 的作用

RoundPipe 内部维护两套参数：

- **模型参数**（低精度）：用于 GPU 上的前向和反向计算
- **优化器参数**（高精度）：用于 CPU 上的优化器更新

标准 PyTorch 优化器通过 `model.parameters()` 获取参数。但在 RoundPipe 中，`model.parameters()` 默认返回的是模型参数（低精度），而优化器需要的是优化器参数（高精度）。

`OptimizerCtx` 上下文管理器解决了这个问题：在其作用域内，`model.parameters()` 和 `model.named_parameters()` 会被自动重定向到 `model.optim_parameters()` 和 `model.optim_named_parameters()`，返回优化器参数。

这意味着在 `OptimizerCtx` 内，你可以像使用普通 PyTorch 模型一样创建优化器，无需关心 RoundPipe 的内部参数管理：

```python
with OptimizerCtx():
    # 这里 model.parameters() 返回的是 FP32 优化器参数
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # 也支持参数组
    optimizer = torch.optim.AdamW([
        {'params': model.parameters(), 'lr': 1e-4},
    ])
```

## RoundPipe CPU 优化器

RoundPipe 的优化器更新在 CPU 上执行（因为优化器参数存储在 CPU 内存中）。PyTorch 内置优化器的 CPU 性能较差，因此 RoundPipe 提供了经过 CPU 指令集优化的优化器实现，通过编译器向量化获得更好的性能。

目前提供的优化器见 [API 文档](../../API/optimizer.md)。

使用方式与 PyTorch 原生优化器完全一致：

```python
from roundpipe.optim import Adam

with OptimizerCtx():
    # 用法与 torch.optim.Adam 完全相同
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=0.01)

    # 支持 decoupled_weight_decay（等价于 AdamW）
    optimizer = Adam(model.parameters(), lr=1e-3,
                     weight_decay=0.01, decoupled_weight_decay=True)
```

推荐优先使用 RoundPipe 提供的优化器，也可以直接使用 PyTorch 原生优化器，功能完全正常，只是 CPU 上的更新速度会稍慢。对于支持 `fused` 参数的 PyTorch 优化器，建议传入 `fused=True` 以获得更好的性能。
