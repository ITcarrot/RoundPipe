# Custom Execution Plans

## What Is an Execution Plan

An execution plan (`ModelExecutePlan`) defines how model layers are grouped and ordered during forward and backward passes — essentially, it determines which layers belong to the same pipeline stage.

An execution plan has two attributes:

- **`fwd_plan`**: Forward pass grouping. Type: `List[range]`, where each `range` specifies the layer indices in one stage.
- **`bwd_plan`**: Backward pass grouping. Type: `List[range]`.

### The Concept of Stages

A stage is the basic scheduling unit for the pipeline. All layers in a stage are uploaded to a GPU together and executed sequentially. Stage design directly affects training efficiency:

- **Stages too large**: A single stage's parameters and activations consume too much VRAM, potentially causing OOM.
- **Stages unbalanced**: The slowest stage becomes the bottleneck, and other GPUs sit idle.

RoundPipe's asymmetric partitioning allows forward and backward passes to use different stage layouts. Since forward computation takes roughly 1/3 the time of the backward pass, forward stages typically contain more layers and backward stages fewer, balancing execution time across stages.

**Example**: An execution plan for a 4-layer model:

```python
from roundpipe import ModelExecutePlan

plan = ModelExecutePlan()

# Execution plan when using forward()
plan.fwd_plan = [range(0, 2), range(2, 4)] # Forward: 2 stages, 2 layers each
plan.bwd_plan = [range(3, 4), range(2, 3), # Backward: 4 stages, 1 layer each
                 range(1, 2), range(0, 1)]
```

When using `forward_backward()`, the first backward stage fuses part of the forward computation (avoiding redundant recomputation), so the first backward stage's layers should not overlap with the forward plan:

```python
# Execution plan when using forward_backward()
plan = ModelExecutePlan()
plan.fwd_plan = [range(0, 3)]              # Forward covers only the first 3 layers
plan.bwd_plan = [range(3, 4), range(2, 3), # First backward stage starts at layer 4
                 range(1, 2), range(0, 1)]
```

## Automatic Tuning

In most cases, you don't need to build an execution plan manually. `ModelExecutePlan.auto()` generates a near-optimal partition based on actual per-layer computation times and memory usage.

```python
from roundpipe import ModelExecutePlan, RoundPipeRunConfig

# Auto-generate the plan (run a few iterations first to collect timing data)
plan = ModelExecutePlan.auto("fused", model)

# Use the generated plan
loss = model.forward_backward(
    input_args=(data,),
    label=labels,
    loss_fn=loss_fn,
    run_config=RoundPipeRunConfig(execute_plan=plan),
)
```

`auto()` parameters:

- **`run_type`**: Execution mode.
    - `"infer"`: Forward-only inference.
    - `"train"`: Separate forward and backward (training based on `forward()`).
    - `"fused"`: Fused forward-backward (training based on `forward_backward()` — the most common choice).
- **`min_stages`**: Minimum number of stages. Defaults to the GPU count. More stages mean fewer pipeline bubbles but smaller stages.
- **`upper_threshold`**: Load-balancing tolerance. Defaults to 1.1, meaning a stage is allowed to take up to 1.1x the time of the longest individual layer. Increasing this allows more flexible partitions but may increase memory usage.
- **`model_memory_limit`**: Estimated available GPU memory (GB). Defaults to 60% of the smallest GPU's VRAM. Because RoundPipe prefetches one stage's parameters, each stage's memory limit is half this value.

### How Auto-Tuning Works

`auto()` optimizes based on:

1. **Timing data**: RoundPipe automatically measures per-layer forward, backward, and recomputation times during execution, using a moving average. On the first run, a default partition is used; subsequent iterations can regenerate a better plan based on actual timings.
2. **Memory constraints**: Ensures each stage's total parameter and gradient size stays within the memory limit.

**Joint optimization across multiple models**:

If training involves multiple RoundPipe models (e.g., encoder + decoder), pass them all to `auto()` for joint optimization:

```python
plan1, plan2 = ModelExecutePlan.auto("fused", model1, model2)
```

## Manual Execution Plans

### When to Use

- Auto-tuning results are unsatisfactory (e.g., unstable per-layer timings).
- You need precise control over each stage's memory footprint.
- Debugging or profiling with a fixed partition.

### Goal

The core objective when building a manual plan is to **balance execution time across stages**. The slowest stage determines the pipeline's throughput; idle time in other stages is wasted.

### How to Build a Plan

1. Enable verbose timing to get per-layer measurements:

    ```python
    from roundpipe.timer import ModelTimer
    ModelTimer.VERBOSE = True
    # Run a few iterations; per-layer fwd/re/bwd times will be printed to stderr
    ```

2. Group adjacent layers so that each group's total time is roughly equal:

    ```python
    plan = ModelExecutePlan()
    # Suppose: 24 transformer layers + 1 lm_head layer
    # Forward: each layer ≈ 2 ms, lm_head ≈ 6 ms
    # Partition into 4 stages, each ≈ 14 ms
    plan.fwd_plan = [
        range(0, 7),    # layers 0-6: 7×2 = 14 ms
        range(7, 14),   # layers 7-13: 7×2 = 14 ms
        range(14, 21),  # layers 14-20: 7×2 = 14 ms
        range(21, 24),  # layers 21-24: 3×2 + 6 = 12 ms (includes lm_head)
    ]
    # Backward is similar, but each layer takes ≈ 6 ms (3× forward)
    plan.bwd_plan = [
        range(24, 25),  # lm_head backward
        range(22, 24),
        range(20, 22),
        # ...
        range(0, 2),
    ]
    ```

### Validation Rules

An execution plan must satisfy these conditions, or RoundPipe will raise an error:

- The union of all forward `range`s must cover every layer exactly once (0 to L-1).
- The same applies to the backward plan.
- Layer indices in the forward plan must be in ascending order (shallow to deep).
- Layer indices in the backward plan must be in descending order (deep to shallow).
- When using `forward_backward()`, the last layer of the forward plan + 1 must equal the first layer of the first backward stage — meaning the final layers participate only in backward (their forward computation is treated as recomputation).
