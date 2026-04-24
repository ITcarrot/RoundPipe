# Multi-GPU Training

## Automatic Multi-GPU

RoundPipe supports multi-GPU training out of the box with **zero code changes**. The same training script behaves identically on one GPU and on eight. RoundPipe automatically detects all available CUDA devices and dispatches pipeline stages across them.

```python
# This code runs on 1 GPU or 8 GPUs — no modifications needed
model = RoundPipe(my_model.to(torch.float16), optim_dtype=torch.float32)

for data, labels in dataloader:
    loss = model.forward_backward(
        input_args=(data,),
        label=labels,
        loss_fn=my_loss_fn,
    )
    model.step(lambda: (optimizer.step(), optimizer.zero_grad()))
```

No `torch.distributed.init_process_group()`. No `DistributedDataParallel`. No `RANK` or `WORLD_SIZE` environment variables. RoundPipe manages all GPUs within a single process.

To restrict which GPUs are used, set the `CUDA_VISIBLE_DEVICES` environment variable:

```bash
# Use only GPU 0 and GPU 1
CUDA_VISIBLE_DEVICES=0,1 python train.py

# Use a single GPU
CUDA_VISIBLE_DEVICES=0 python train.py
```

## Consistent Behavior Across GPU Counts

When `num_microbatch` is set explicitly, RoundPipe guarantees that program semantics are independent of the number of GPUs. That is, with the same `num_microbatch`, training on 1 GPU and 8 GPUs produces consistent results (loss, gradients, and parameter updates).

```python
# Fix num_microbatch to ensure consistency across GPU counts
model = RoundPipe(
    my_model.to(torch.float16),
    optim_dtype=torch.float32,
    model_run_config=RoundPipeRunConfig(num_microbatch=9),
)
```

If `num_microbatch` is not specified, the default is `GPU count + 1`. In that case, different GPU counts lead to different microbatch counts, and training semantics may differ (since the loss is the sum of per-microbatch losses, and consistency depends on how `loss_fn` is implemented).

**Guidelines for choosing num_microbatch**:

- **Default**: `num_devices + 1` (e.g., 9 for 8 GPUs). This is the minimum value that keeps the pipeline bubble-free.
- **If you fix this value**: it should always be at least `max GPU count + 1`. If `num_microbatch` is less than the GPU count + 1, some GPUs will idle at times, creating pipeline bubbles.
- **Increasing num_microbatch**: reduces per-microbatch data size, lowering GPU peak memory usage. Useful for long sequences or tight memory budgets. However, overly large `num_microbatch` makes each microbatch too small, reducing GPU compute efficiency.
