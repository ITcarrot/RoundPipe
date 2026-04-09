# 日志与性能分析

## 模型层计时

RoundPipe 内置了 per-layer 计时功能，可以输出每层前向、反向、重计算的执行时间。这对于分析流水线瓶颈、调优执行计划非常有用。

开启计时输出：

```python
from roundpipe.timer import ModelTimer

ModelTimer.VERBOSE = True
```

开启后，每次迭代完成后会在 `stderr` 输出每层的计时信息：

```
Layer 0 fwd  new record: 1.234 ms  new estimate: 1.200 ms
Layer 0 re   new record: 1.180 ms  new estimate: 1.150 ms
Layer 0 bwd  new record: 2.456 ms  new estimate: 2.400 ms
Layer 1 fwd  new record: 1.567 ms  new estimate: 1.530 ms
Layer 1 re   new record: 1.520 ms  new estimate: 1.500 ms
Layer 1 bwd  new record: 2.890 ms  new estimate: 2.850 ms
...
```

各字段含义：

- **Layer N**：层索引
- **fwd / re / bwd**：前向 / 重计算 / 反向
- **new record**：本次迭代的实际测量时间
- **new estimate**：滑动平均后的估计时间（RoundPipe 用此值进行自动阶段划分）
