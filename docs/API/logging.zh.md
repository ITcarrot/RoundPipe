# 运行日志

RoundPipe 的运行日志功能提供了对流水线调度和性能的监控工具，帮助开发者了解模型在训练过程中的行为和性能瓶颈。以下是一些与运行日志相关的功能和配置选项。

## ModelTimer.VERBOSE

```python
roundpipe.ModelTimer.VERBOSE: bool = False
```

类级别标志，控制 `ModelTimer` 是否在每次迭代后将逐层计时信息输出到 `stderr`。

设置为 `True` 后，每次迭代的计时事件处理完成时，计时器会为每一层的每种操作类型（`fwd`、`re`、`bwd`）向 `stderr` 输出一行，展示本次记录的时间和当前的滑动平均估计值。

**输出格式：**

```
Layer {layer_idx} {action}  new record: {time:.3f} ms  new estimate: {time:.3f} ms
```

**示例：**

```python
from roundpipe.timer import ModelTimer

ModelTimer.VERBOSE = True
```

迭代完成后，`stderr` 上将出现类似以下的输出：

```
Layer 0 fwd  new record: 1.234 ms  new estimate: 1.200 ms
Layer 0 re   new record: 1.180 ms  new estimate: 1.150 ms
Layer 0 bwd  new record: 2.456 ms  new estimate: 2.400 ms
Layer 1 fwd  new record: 1.567 ms  new estimate: 1.530 ms
...
```

适用于调试流水线调度和定位慢层。
