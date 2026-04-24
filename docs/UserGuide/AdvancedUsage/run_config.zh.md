# 运行配置调参

`RoundPipeRunConfig` 是 RoundPipe 的运行配置类，控制微批次数量、重计算粒度、输出设备等行为。

## 模型级配置 vs 调用级覆盖

配置可以在两个层级指定：

- **模型级**：创建 `RoundPipe` 时通过 `model_run_config` 传入，作为该模型的默认配置
- **调用级**：调用 `forward()` 或 `forward_backward()` 时传入，覆盖模型级默认值

```python
from roundpipe import RoundPipe, RoundPipeRunConfig

# 模型级：设置默认 num_microbatch=4
model = RoundPipe(
    my_model,
    model_run_config=RoundPipeRunConfig(num_microbatch=4),
)

# 调用级：本次调用使用 num_microbatch=8，覆盖模型级默认值
loss = model.forward_backward(
    input_args=(data,),
    label=labels,
    loss_fn=loss_fn,
    run_config=RoundPipeRunConfig(num_microbatch=8),
)
```

当两个层级都指定了同一参数时，调用级优先。如果两个层级都未指定（均为 `None`），则使用内置默认值。

## num_microbatch

微批次数量，控制将输入数据拆分为多少个微批次进行流水线执行。

**默认值**：`GPU 数量 + 1`

**如何选择**：

- `num_microbatch` 越大，每个微批次的数据量越小，GPU 峰值显存占用越低
- 但过小的微批次会降低 GPU 计算效率（kernel launch 开销占比增大，矩阵运算无法充分利用 GPU 并行度）
- 通常设为 GPU 数量 + 1 以上，可以保证流水线无气泡（GPU 空闲等待）。如果显存紧张，可以尝试增大 `num_microbatch`，但不宜过大。

**典型场景**：

```python
# 显存充足，追求最高吞吐
config = RoundPipeRunConfig(num_microbatch=torch.cuda.device_count() + 1)

# 显存紧张（长序列 / 大模型），用更多微批次降低显存
config = RoundPipeRunConfig(num_microbatch=16)
```

## recompute_grain

反向传播时激活重计算的粒度。

- **`"stage"`（默认）**：以 stage 为单位重计算。一个 stage 内的所有层先全部重计算前向，再执行反向。显存占用较高（需要同时持有 stage 内所有层的激活），但数据传输次数少。
- **`"layer"`**：以单层为单位重计算。每层独立重计算前向并立即执行反向，然后释放激活。峰值显存更低，但每层都需要单独上传层的输入，增加了数据传输开销。

```python
# 默认：stage 粒度
config = RoundPipeRunConfig(recompute_grain="stage")

# 显存紧张时切换到 layer 粒度
config = RoundPipeRunConfig(recompute_grain="layer")
```

**选择建议**：

- 优先使用 `"stage"`（默认），性能更好
- 如果遇到 GPU OOM，先尝试增大 `num_microbatch`
- 如果仍然 OOM，再切换到 `"layer"`

## output_device

模型输出张量放置的设备。

- **CPU（默认）**：输出传回 CPU 内存。适合大多数训练场景，因为 loss 计算通常在 `loss_fn` 内部完成（在 GPU 上），训练循环中不需要直接操作输出。
- **GPU**：输出保留在 GPU 上。适合需要在 GPU 上对输出做后处理的场景。

```python
# 默认：输出放在 CPU
config = RoundPipeRunConfig(output_device=None)  # 等价于 CPU

# 输出放在 GPU 0
config = RoundPipeRunConfig(output_device=torch.device("cuda:0"))
```

!!! note
    将输出放在 GPU 上会额外占用显存。如果只是为了计算 loss，不需要设置 `output_device`，因为 `forward_backward()` 的 `loss_fn` 已经在 GPU 上执行。
