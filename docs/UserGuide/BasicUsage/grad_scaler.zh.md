# 梯度缩放器（GradScaler）

## 为什么需要 GradScaler

在混合精度训练中，模型使用 FP16 进行前向和反向计算以提高性能。但 FP16 的数值范围有限，较小的梯度值会下溢为零，导致训练不稳定甚至无法收敛。

GradScaler 通过**梯度缩放**解决这个问题：

1. 在计算 loss 后，将 loss 乘以一个较大的缩放因子（如 65536）
2. 反向传播产生的梯度也会被相应放大，避免下溢
3. 在优化器更新前，将梯度除以缩放因子，恢复真实值
4. 如果检测到梯度中出现 inf/NaN（说明缩放因子过大），跳过本次更新并减小缩放因子

## 基本用法

RoundPipe 提供了自己的 `GradScaler` 实现，接口与 `torch.amp.GradScaler` 完全兼容，但额外支持了 RoundPipe 的异步优化器执行模型。

一个完整的混合精度训练循环：

```python
from roundpipe import RoundPipe, OptimizerCtx, GradScaler
from roundpipe.optim import Adam

model = RoundPipe(my_model.to(torch.float16), optim_dtype=torch.float32)
with OptimizerCtx():
    optimizer = Adam(model.parameters(), lr=1e-3)
scaler = GradScaler()

for data, labels in dataloader:
    # 1. forward_backward：在 loss_fn 中调用 scaler.scale()
    loss = model.forward_backward(
        input_args=(data.to(torch.float16),),
        label=labels,
        loss_fn=lambda outputs, labels: scaler.scale(
            criterion(outputs.float(), labels)
        ) / num_microbatch,
    )

    # 2. step：在 step_fn 中调用 scaler.step()
    model.step(lambda: (
        scaler.step(optimizer),
        optimizer.zero_grad(),
    ))

    # 3. update：更新缩放因子
    scaler.update()

    # 获取真实 loss 值（除以缩放因子）
    real_loss = loss.item() / scaler.get_scale()
```

各步骤说明：

- **`scaler.scale(loss)`**：将 loss 乘以缩放因子。放在 `loss_fn` 内部，这样每个 microbatch 的反向传播都会使用缩放后的 loss。
- **`scaler.step(optimizer)`**：自动执行 `unscale_`（将梯度除以缩放因子）+ 检查 inf/NaN + `optimizer.step()`。如果梯度中有 inf/NaN，会跳过本次更新。
- **`scaler.update()`**：根据本次迭代是否跳过更新来调整缩放因子。必须在主线程调用。

## 与 PyTorch GradScaler 的区别

RoundPipe 的 `GradScaler` 与 PyTorch 原生 `torch.amp.GradScaler` 的主要区别在于**线程安全**。

RoundPipe 的优化器更新默认在后台线程异步执行（通过 `model.step(is_async=True)`）。`scaler.scale()` 在 GPU 计算线程中被调用，而 `scaler.step()` 和 `scaler.update()` 在优化器线程中执行。PyTorch 原生的 `GradScaler` 不支持这种跨线程使用模式，因此必须使用 RoundPipe 提供的版本。

!!! warning "不要使用 PyTorch 原生 GradScaler"
    ```python
    # ❌ 错误：PyTorch 原生 GradScaler 不支持异步优化器
    scaler = torch.amp.GradScaler()

    # ✅ 正确：使用 RoundPipe 的 GradScaler
    from roundpipe import GradScaler
    scaler = GradScaler()
    ```

### 手动 unscale

如果需要在优化器更新前对梯度进行额外操作（如梯度裁剪），可以手动调用 `unscale_`：

```python
model.step(lambda: (
    scaler.unscale_(optimizer),
    torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], max_norm=1.0),
    scaler.step(optimizer),
    optimizer.zero_grad(),
))
```

如果没有手动调用 `unscale_`，`scaler.step()` 会自动执行，无需额外操作。
