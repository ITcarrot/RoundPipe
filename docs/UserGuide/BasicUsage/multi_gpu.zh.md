# 多 GPU 训练

## 自动多卡

RoundPipe 全自动支持一机多卡训练，**零代码改动**。同一份训练脚本在单卡和多卡环境下行为一致，RoundPipe 会自动检测所有可用的 CUDA 设备并将流水线阶段分发到各 GPU 上执行。

```python
# 这段代码在 1 卡和 8 卡上都能运行，无需任何修改
model = RoundPipe(my_model.to(torch.float16), optim_dtype=torch.float32)

for data, labels in dataloader:
    loss = model.forward_backward(
        input_args=(data,),
        label=labels,
        loss_fn=my_loss_fn,
    )
    model.step(lambda: (optimizer.step(), optimizer.zero_grad()))
```

不需要 `torch.distributed.init_process_group()`，不需要 `DistributedDataParallel`，不需要设置 `RANK` 或 `WORLD_SIZE` 环境变量。RoundPipe 在单进程内管理所有 GPU。

如果需要限制使用的 GPU，通过环境变量 `CUDA_VISIBLE_DEVICES` 控制：

```bash
# 只使用 GPU 0 和 GPU 1
CUDA_VISIBLE_DEVICES=0,1 python train.py

# 只使用单卡
CUDA_VISIBLE_DEVICES=0 python train.py
```

## 多卡与单卡的行为一致性

只要显式指定 `num_microbatch`，RoundPipe 可以保证程序语义不受卡数影响。也就是说，在相同的 `num_microbatch` 下，1 卡和 8 卡的训练结果（loss、梯度、参数更新）是一致的。

```python
# 固定 num_microbatch，确保跨卡数一致性
model = RoundPipe(
    my_model.to(torch.float16),
    optim_dtype=torch.float32,
    model_run_config=RoundPipeRunConfig(num_microbatch=9),
)
```

如果不指定 `num_microbatch`，默认值为 `GPU 数量 + 1`，此时不同卡数下的微批次数量不同，训练语义可能会有差异（因为 loss 是各微批次 loss 的总和，一致性取决于 loss_fn 的实现）。

**num_microbatch 调整建议**：

- **默认值**：`num_devices + 1`（例如 8 卡时为 9）。这是保证流水线无气泡的最小值。
- **如果固定这一值**：应当保证始终大于等于默认值（即大于等于最大可能使用的 GPU 数量 + 1）。如果 `num_microbatch` 小于 GPU 数量 + 1，部分 GPU 会在某些时刻空闲，产生流水线气泡。
- **增大 num_microbatch**：减小每个微批次的数据量，降低 GPU 峰值显存占用。适合训练长序列或显存紧张的场景。但过大的 `num_microbatch` 会导致每个微批次太小，降低 GPU 计算效率。
