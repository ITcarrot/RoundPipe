# 解决 OOM

训练大模型时，GPU 或 CPU 内存不足（Out of Memory）是最常见的问题。本节介绍如何诊断 OOM 类型以及对应的解决方案。

## GPU OOM

GPU OOM 表现为 `torch.cuda.OutOfMemoryError` 或 `CUDA out of memory` 错误。以下方法按推荐优先级排列：

### 1. 增大 num_microbatch

最直接有效的方法。增大 `num_microbatch` 会减小每个微批次的数据量，从而降低单次计算的 GPU 显存峰值。

```python
# 从默认值开始逐步增大
config = RoundPipeRunConfig(num_microbatch=16)  # 或 32、64...
```

代价是每个微批次变小后，GPU 计算效率可能下降。通常增大到默认值的 2~4 倍就足够解决大部分 OOM 问题。

### 2. 切换 recompute_grain 到 "layer"

将重计算粒度从默认的 `"stage"` 切换到 `"layer"`，可以显著降低 GPU 峰值显存：

```python
config = RoundPipeRunConfig(recompute_grain="layer")
```

`"stage"` 模式下，一个 stage 内所有层的激活需要同时驻留在 GPU 上；`"layer"` 模式下，每层独立重计算并立即执行反向，峰值显存大幅降低。代价是每层需要单独上传模型层输入，数据传输量增加。

### 3. 减少 execute_plan 中每个 stage 的层数

如果使用自定义执行计划，可以将每个 stage 包含的层数减少，降低单个 stage 的参数和激活显存占用。详见[自定义执行计划](execute_plan.md)。

也可以通过调整 `ModelExecutePlan.auto()` 的 `model_memory_limit` 参数来限制每个 stage 的显存：

```python
from roundpipe import ModelExecutePlan

# 降低显存限制，迫使自动划分产生更小的 stage
plan = ModelExecutePlan.auto(
    "fused", model,
    model_memory_limit=8.0,  # 限制为 8GB（默认是 GPU 显存的 60%）
)
```

## CPU OOM

CPU OOM 表现为进程被系统 OOM Killer 杀死（`Killed`）、`MemoryError`，或系统开始大量使用 swap 导致训练极度缓慢。

### 1. pin_model 选择 "register" 或 "off"

默认的 `pin_model="alloc"` 使用 PyTorch 的 `pin_memory` 分配锁页内存，PyTorch 会将每次分配对齐到 2 的幂次，可能导致 CPU 内存占用接近模型大小的 2 倍。

```python
# 减少 CPU 内存占用（参数速度下降约 10%）
model = RoundPipe(my_model, pin_model="register")

# 模型超过 CPU 内存时，配合 mmap 加载
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    # 使用 mmap 加载，Linux 会按需从磁盘读取并自动替换内存页
)
model = RoundPipe(my_model_sequential, pin_model="off")
```

三种模式的 CPU 内存占用对比：

| pin_model | CPU 内存占用 | H2D 传输性能 | 适用场景 |
|-----------|------------|-------------|---------|
| `"alloc"` | 最高（约 1.5× 模型大小） | 最快 | 默认，CPU 内存充足时 |
| `"register"` | 中等（约 1× 模型大小） | 下降约 10% | 大模型，CPU 内存较紧张 |
| `"off"` | 最低（按需加载） | 最慢 | 超大模型 LoRA 微调，模型超过 CPU 内存 |

### 2. 减小 num_microbatch，使用多轮 gradient accumulation

每个微批次的中间激活（层间传递的数据）会缓存在 CPU 内存中。`num_microbatch` 越大，同时缓存的激活越多，CPU 内存占用越高。

如果 CPU 内存紧张，可以减小 `num_microbatch` 并通过多次 `forward_backward` 调用手动实现梯度累积：

```python
# 原来：一次处理大 batch，num_microbatch=16
loss = model.forward_backward(input_args=(large_batch,), ...)

# 改为：分两次处理，每次 num_microbatch=8
config = RoundPipeRunConfig(num_microbatch=8)
loss1 = model.forward_backward(input_args=(batch_part1,), ..., run_config=config)
loss2 = model.forward_backward(input_args=(batch_part2,), ..., run_config=config)
# 梯度自动累积，然后统一更新
model.step(lambda: (optimizer.step(), optimizer.zero_grad()))
```
