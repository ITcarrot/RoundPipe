# Logging and Profiling

## Per-Layer Timing

RoundPipe has built-in per-layer timing that reports the execution time of each layer's forward pass, backward pass, and recomputation. This is invaluable for identifying pipeline bottlenecks and tuning execution plans.

To enable timing output:

```python
from roundpipe.timer import ModelTimer

ModelTimer.VERBOSE = True
```

Once enabled, timing information is printed to `stderr` after each iteration:

```
Layer 0 fwd  new record: 1.234 ms  new estimate: 1.200 ms
Layer 0 re   new record: 1.180 ms  new estimate: 1.150 ms
Layer 0 bwd  new record: 2.456 ms  new estimate: 2.400 ms
Layer 1 fwd  new record: 1.567 ms  new estimate: 1.530 ms
Layer 1 re   new record: 1.520 ms  new estimate: 1.500 ms
Layer 1 bwd  new record: 2.890 ms  new estimate: 2.850 ms
...
```

Field meanings:

- **Layer N**: Layer index.
- **fwd / re / bwd**: Forward / recomputation / backward.
- **new record**: Actual measured time for this iteration.
- **new estimate**: Moving-average estimate (used by RoundPipe's automatic stage partitioning).
