# 重计算数据保存

## 使用场景

RoundPipe 默认采用激活重计算：反向传播时重新执行前向计算来恢复中间激活值。大多数情况下，重计算产生的结果与原始前向完全一致（配合 `preserve_rng_state=True` 保证随机行为一致）。

但某些模型层的 forward 中包含**不稳定的操作**：即使输入相同，两次执行也可能产生不同的结果，或者某些操作会触发 GPU-CPU 同步导致性能下降。典型场景：

- **MoE 模型的 expert 路由**：路由决策可能依赖于 `torch.topk` 的选择，重计算时需要保证路由结果一致
- **动态形状操作**：如 `torch.nonzero()` 返回的 tensor 形状取决于输入值，且会触发 GPU-CPU 同步
- **条件分支**：forward 中根据中间计算结果选择不同的执行路径

对于这些场景，RoundPipe 提供了 `save_for_recompute` API，允许在前向传播时保存关键数据，在重计算时直接使用保存的数据而非重新计算。

## API

### save_for_recompute(*data)

在前向传播阶段保存数据，供后续重计算阶段使用。

```python
from roundpipe import save_for_recompute

save_for_recompute(routing_indices, expert_mask)
```

- 每层的 forward 中最多调用一次
- 保存的 tensor 必须不需要梯度（`requires_grad=False`）
- 如果当前未启用梯度计算，此函数为空操作

### doing_recompute()

检查当前是否处于重计算阶段。

```python
from roundpipe import doing_recompute

if doing_recompute():
    # 当前是重计算阶段
    ...
else:
    # 当前是正常前向传播
    ...
```

### get_recompute_data()

在重计算阶段获取之前通过 `save_for_recompute` 保存的数据。

```python
from roundpipe import get_recompute_data

data = get_recompute_data()  # 返回 tuple
```

- 只能在重计算上下文中调用，否则会抛出 `AssertionError`
- 即使只保存了一个值，也以 tuple 形式返回

## 示例

### MoE expert 路由重放

```python
import torch
import torch.nn as nn
from roundpipe import save_for_recompute, doing_recompute, get_recompute_data

class MoELayer(nn.Module):
    def __init__(self, num_experts, hidden_size, expert_size):
        super().__init__()
        self.gate = nn.Linear(hidden_size, num_experts)
        self.experts = nn.ModuleList([
            nn.Linear(hidden_size, expert_size)
            for _ in range(num_experts)
        ])

    def forward(self, x):
        if doing_recompute():
            # 重计算阶段：直接使用保存的路由结果
            selected_experts, = get_recompute_data()
        else:
            # 正常前向：计算路由并保存
            gate_logits = self.gate(x)
            routing_weights = torch.softmax(gate_logits, dim=-1)
            selected_experts = torch.topk(routing_weights, k=2, dim=-1).indices
            save_for_recompute(selected_experts)

        # 使用路由结果执行 expert 计算
        # ...
        return output
```

### 避免 GPU-CPU 同步

```python
class SparseLayer(nn.Module):
    def forward(self, x):
        if doing_recompute():
            mask, = get_recompute_data()
        else:
            # torch.nonzero() 会触发 GPU-CPU 同步（需要知道非零元素数量）
            # 保存结果避免重计算时再次同步
            mask = torch.nonzero(x > 0.5)
            save_for_recompute(mask)

        # 使用 mask 进行稀疏计算
        return x[mask]
```

### 条件分支

```python
class ConditionalLayer(nn.Module):
    def forward(self, x):
        if doing_recompute():
            use_branch_a, = get_recompute_data()
        else:
            # 根据输入统计量选择分支
            use_branch_a = (x.mean() > 0).item()  # 触发 GPU-CPU 同步
            save_for_recompute(use_branch_a)

        if use_branch_a:
            return self.branch_a(x)
        else:
            return self.branch_b(x)
```
