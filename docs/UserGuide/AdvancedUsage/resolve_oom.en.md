# Resolving OOM Errors

Out-of-memory errors — on GPU or CPU — are the most common issue when training large models. This section covers how to diagnose and fix each type.

## GPU OOM

GPU OOM manifests as `torch.cuda.OutOfMemoryError` or a `CUDA out of memory` message. The following solutions are listed in recommended priority order.

### 1. Increase num_microbatch

The most straightforward fix. A larger `num_microbatch` reduces per-microbatch data size, lowering the GPU's peak memory usage.

```python
# Start from the default and increase gradually
config = RoundPipeRunConfig(num_microbatch=16)  # or 32, 64, ...
```

The trade-off is that smaller microbatches may lower GPU compute efficiency. In most cases, 2-4x the default value is enough to resolve the problem.

### 2. Switch recompute_grain to "layer"

Changing the recomputation granularity from the default `"stage"` to `"layer"` can significantly reduce GPU peak memory:

```python
config = RoundPipeRunConfig(recompute_grain="layer")
```

In `"stage"` mode, activations for all layers in a stage must reside on the GPU simultaneously. In `"layer"` mode, each layer is recomputed and backpropagated independently, then its activations are released — dramatically lowering peak memory. The trade-off is increased data transfer, since each layer's input must be uploaded separately.

### 3. Reduce layers per stage in the execution plan

If you're using a custom execution plan, reduce the number of layers per stage to lower each stage's parameter and activation footprint. See [Custom Execution Plans](execute_plan.md).

You can also adjust `model_memory_limit` in `ModelExecutePlan.auto()` to constrain per-stage memory:

```python
from roundpipe import ModelExecutePlan

# Lower the memory limit to force smaller stages
plan = ModelExecutePlan.auto(
    "fused", model,
    model_memory_limit=8.0,  # Limit to 8 GB (default is 60% of GPU VRAM)
)
```

## CPU OOM

CPU OOM typically shows up as the process being killed by the system's OOM killer (`Killed`), a `MemoryError`, or the system swapping heavily and making training extremely slow.

### 1. Choose "register" or "off" for pin_model

The default `pin_model="alloc"` uses PyTorch's `pin_memory` to allocate page-locked memory. PyTorch aligns each allocation to a power of 2, which can nearly double CPU memory usage compared to the actual model size.

```python
# Reduce CPU memory usage (~10% slower transfers)
model = RoundPipe(my_model, pin_model="register")

# When the model exceeds CPU memory, use mmap loading
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    # mmap loading: Linux reads from disk on demand and automatically swaps memory pages
)
model = RoundPipe(my_model_sequential, pin_model="off")
```

Comparison of the three modes:

| pin_model | CPU memory usage | H2D transfer speed | When to use |
|-----------|-----------------|-------------------|-------------|
| `"alloc"` | Highest (~1.5x model size) | Fastest | Default; when CPU memory is plentiful |
| `"register"` | Moderate (~1x model size) | ~10% slower | Large models with tight CPU memory |
| `"off"` | Lowest (on-demand loading) | Slowest | Very large model LoRA fine-tuning; model exceeds CPU memory |

### 2. Reduce num_microbatch and use manual gradient accumulation

Intermediate activations (data passed between layers) for each microbatch are cached in CPU memory. More microbatches mean more cached activations and higher CPU memory usage.

If CPU memory is tight, lower `num_microbatch` and perform gradient accumulation manually across multiple `forward_backward` calls:

```python
# Before: one pass with a large batch, num_microbatch=16
loss = model.forward_backward(input_args=(large_batch,), ...)

# After: two passes, each with num_microbatch=8
config = RoundPipeRunConfig(num_microbatch=8)
loss1 = model.forward_backward(input_args=(batch_part1,), ..., run_config=config)
loss2 = model.forward_backward(input_args=(batch_part2,), ..., run_config=config)
# Gradients accumulate automatically; then update once
model.step(lambda: (optimizer.step(), optimizer.zero_grad()))
```
