# 训练循环

本节介绍如何使用 RoundPipe 编写完整的训练循环，包括前向反向传播、优化器更新、同步以及推理评估。

## forward_backward()

`forward_backward()` 是 RoundPipe 的核心训练 API，它将前向传播和反向传播融合在一起执行。相比分别调用 `forward()` 和 `backward()`，融合执行允许前向和反向在流水线中交替进行，消除流水线气泡，性能显著更优。

```python
loss = model.forward_backward(
    input_args=(images,),
    input_kwargs={},
    label=labels,
    loss_fn=my_loss_fn,
    return_outputs=False,
    run_config=RoundPipeRunConfig(),
)
```

**参数详解**：

- **`input_args`**：模型前向传播的位置参数，以 tuple 形式传入。例如模型的 `forward(x)` 接受一个输入，则传 `input_args=(x,)`。

- **`input_kwargs`**：模型前向传播的关键字参数，以 dict 形式传入。例如 HuggingFace 模型通常接受 `input_ids` 和 `attention_mask`：

    ```python
    loss = model.forward_backward(
        input_kwargs={"input_ids": ids, "attention_mask": mask},
        label=labels,
        loss_fn=my_loss_fn,
    )
    ```

- **`label`**：标签数据，会被自动拆分为与输入对应的微批次，传递给 `loss_fn`。

- **`loss_fn`**：损失函数，接受 `(outputs, labels)` 两个参数，返回一个 loss tensor。RoundPipe 会对每个微批次独立调用 `loss_fn` 并执行反向传播，最终返回所有微批次 loss 的总和。

!!! info "loss计算"
    由于 loss 是所有微批次的总和，如果你希望得到与整个 batch 上计算等价的平均 loss，需要在 `loss_fn` 中除以微批次数量：

    ```python
    loss_fn=lambda outputs, labels: criterion(outputs, labels) / num_microbatch
    ```

    配合 GradScaler 使用时：

    ```python
    loss_fn=lambda outputs, labels: scaler.scale(
        criterion(outputs.float(), labels)
    ) / num_microbatch
    ```

- **`return_outputs`**：是否同时返回模型输出。默认 `False` 只返回 loss；设为 `True` 时返回 `(loss, outputs)` 元组。注意返回输出会增加额外的内存开销和同步等待。

    ```python
    # 只需要 loss
    loss = model.forward_backward(...)

    # 同时需要输出（如计算准确率）
    loss, outputs = model.forward_backward(..., return_outputs=True)
    ```

- **`run_config`**：本次调用的运行配置，会覆盖模型级别的默认配置。详见 [RoundPipeRunConfig 调参](../AdvancedUsage/run_config.md)。

## model.step()

`model.step()` 执行优化器更新。它接受一个 callable，在优化器线程中执行：

```python
def step_fn():
    optimizer.step()
    optimizer.zero_grad()

model.step(step_fn)
```

**`step_fn` 中的优化器操作**：

`step_fn` 是一个无参数的 callable，你可以在其中执行任意优化器相关操作。常见模式：

```python
# 基本用法
def step_fn():
    optimizer.step()
    optimizer.zero_grad()

model.step(step_fn)

# 配合 GradScaler
def step_fn():
    scaler.step(optimizer)
    optimizer.zero_grad()

model.step(step_fn)

# 配合梯度裁剪
def step_fn():
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], max_norm=1.0)
    scaler.step(optimizer)
    optimizer.zero_grad()

model.step(step_fn)
```

!!! info "parameter()重定向"
    在 `step_fn` 中访问 `model.parameters()` 和 `model.named_parameters()` 会自动重定向到优化器参数（高精度），确保优化器更新中访问的是正确的参数集合。

!!! warning "数据竞争"
    `step_fn` 在优化器线程中执行，与 GPU 计算并行。`step_fn` 中只应访问优化器参数和梯度，访问其他共享数据需要注意数据竞争问题。

**`is_async` 参数**：

- `is_async=True`（默认）：`step()` 立即返回，优化器更新在后台线程异步执行。下一次迭代使用的参数会落后一步（staleness-1），这在实践中不影响收敛。这是推荐的模式，因为优化器更新的用时被完全隐藏在下一步的 GPU 计算下。
- `is_async=False`：`step()` 阻塞等待优化器更新完成后才返回。每次迭代使用最新参数，但会显著降低性能，通常不推荐使用。

## model.synchronize()

`synchronize()` 等待所有异步操作完成，并将优化器参数同步回模型参数。调用后：

- 模型参数反映最新的优化器更新结果
- 参数的 `.grad` 属性包含累积的梯度

**何时需要调用**：

```python
# 1. 评估前：确保模型参数是最新的
model.synchronize()
model.eval()
with torch.no_grad():
    output = model(test_data)

# 2. 保存 checkpoint 前
model.synchronize()
torch.save(model.model.state_dict(), "checkpoint.pt")

# 3. 需要检查梯度时
model.synchronize()
for name, param in model.model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad norm = {param.grad.norm()}")
```

在正常的训练循环中（`forward_backward` → `step` 循环），不需要调用 `synchronize()`，RoundPipe 内部会自动处理参数一致性。

## 推理 / 评估运行

评估时使用 `model.eval()` + `torch.no_grad()` + `model.forward()`：

```python
model.synchronize()  # 确保参数是最新的
model.eval()

correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(torch.float16)
        outputs = model(images)  # 使用 forward()，不需要反向传播
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
model.train()  # 切回训练模式
```

**`forward()` vs `forward_backward()`**：

| | `forward()` | `forward_backward()` |
|---|---|---|
| 用途 | 推理 / 评估 | 训练 |
| 反向传播 | 不执行，但支持调用 | 自动执行 |
| 返回值 | 模型输出 | loss（或 loss + 输出） |
| 调用方式 | `model(x)` 或 `model.forward(x)` | `model.forward_backward(...)` |
| 梯度 | 可选（配合 `torch.no_grad()`） | 自动累积 |

`forward()` 支持与普通 PyTorch 模型相同的调用方式：

```python
# 以下两种方式等价
output = model(x, attention_mask=mask)
output = model.forward(x, attention_mask=mask)

# 也可以通过 run_config 覆盖配置
output = model(x, roundpipe_run_config=RoundPipeRunConfig(num_microbatch=2))

# 输出同样支持 backward
loss = criterion(output, labels)
loss.backward()
```
