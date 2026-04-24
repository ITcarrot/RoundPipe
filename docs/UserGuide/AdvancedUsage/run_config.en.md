# RoundPipeRunConfig

`RoundPipeRunConfig` is RoundPipe's runtime configuration class. It controls the number of microbatches, recomputation granularity, output device placement, and other behaviors.

## Model-Level Config vs Per-Call Overrides

Configuration can be specified at two levels:

- **Model-level**: Passed via `model_run_config` when creating a `RoundPipe` instance. Acts as the default for all calls on that model.
- **Per-call**: Passed to `forward()` or `forward_backward()`. Overrides the model-level defaults for that call only.

```python
from roundpipe import RoundPipe, RoundPipeRunConfig

# Model-level: set default num_microbatch=4
model = RoundPipe(
    my_model,
    model_run_config=RoundPipeRunConfig(num_microbatch=4),
)

# Per-call: use num_microbatch=8 for this call, overriding the model-level default
loss = model.forward_backward(
    input_args=(data,),
    label=labels,
    loss_fn=loss_fn,
    run_config=RoundPipeRunConfig(num_microbatch=8),
)
```

When both levels specify the same parameter, the per-call value takes precedence. If neither level specifies a value (both are `None`), the built-in default is used.

## num_microbatch

Number of microbatches. Controls how many pieces the input data is split into for pipeline execution.

**Default**: `GPU count + 1`

**How to choose**:

- Larger `num_microbatch` → smaller per-microbatch data → lower GPU peak memory.
- Too-small microbatches reduce GPU compute efficiency (kernel-launch overhead dominates, and matrix operations can't fully utilize GPU parallelism).
- Setting it to at least `GPU count + 1` ensures a bubble-free pipeline (no GPU idle time). If memory is tight, increase `num_microbatch` — but don't go excessively high.

**Typical scenarios**:

```python
# Plenty of VRAM — maximize throughput
config = RoundPipeRunConfig(num_microbatch=torch.cuda.device_count() + 1)

# Tight memory (long sequences / large models) — lower peak memory with more microbatches
config = RoundPipeRunConfig(num_microbatch=16)
```

## recompute_grain

Granularity of activation recomputation during the backward pass.

- **`"stage"` (default)**: Recompute at the stage level. All layers in a stage recompute their forward pass first, then backward runs. Higher memory usage (activations for all layers in the stage must coexist), but fewer data transfers.
- **`"layer"`**: Recompute one layer at a time. Each layer independently recomputes its forward pass, runs backward immediately, and releases its activations. Lower peak memory, but each layer requires a separate upload of its input, increasing data transfer overhead.

```python
# Default: stage granularity
config = RoundPipeRunConfig(recompute_grain="stage")

# Tight memory: switch to layer granularity
config = RoundPipeRunConfig(recompute_grain="layer")
```

**Recommendations**:

- Prefer `"stage"` (default) for better performance.
- If you hit GPU OOM, try increasing `num_microbatch` first.
- If OOM persists, switch to `"layer"`.

## output_device

Device placement for model output tensors.

- **CPU (default)**: Outputs are transferred back to CPU memory. Suitable for most training scenarios, since loss computation typically happens inside `loss_fn` (on the GPU) and the training loop doesn't need to manipulate outputs directly.
- **GPU**: Outputs stay on the GPU. Useful when you need to post-process outputs on the GPU.

```python
# Default: output on CPU
config = RoundPipeRunConfig(output_device=None)  # equivalent to CPU

# Output on GPU 0
config = RoundPipeRunConfig(output_device=torch.device("cuda:0"))
```

!!! note
    Keeping outputs on the GPU consumes additional VRAM. If you only need the loss, there's no need to set `output_device` — `forward_backward()`'s `loss_fn` already runs on the GPU.
