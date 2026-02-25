# Logging

RoundPipe's logging features provide tools for monitoring pipeline scheduling and performance, helping developers understand model behavior and identify bottlenecks during training. Below are some of the logging-related features and configuration options.

## ModelTimer.VERBOSE

```python
roundpipe.ModelTimer.VERBOSE: bool = False
```

A class-level flag that controls whether `ModelTimer` prints per-layer timing updates to `stderr` after each iteration.

When set to `True`, after each iteration's timing events are processed, the timer prints one line per layer per action type (`fwd`, `re`, `bwd`) to `stderr`, showing both the newly recorded time and the current smoothed estimate.

**Output format:**

```
Layer {layer_idx} {action}  new record: {time:.3f} ms  new estimate: {time:.3f} ms
```

**Example:**

```python
from roundpipe.timer import ModelTimer

ModelTimer.VERBOSE = True
```

After an iteration completes, output similar to the following will appear on `stderr`:

```
Layer 0 fwd  new record: 1.234 ms  new estimate: 1.200 ms
Layer 0 re   new record: 1.180 ms  new estimate: 1.150 ms
Layer 0 bwd  new record: 2.456 ms  new estimate: 2.400 ms
Layer 1 fwd  new record: 1.567 ms  new estimate: 1.530 ms
...
```

This is useful for debugging pipeline scheduling and identifying slow layers.
