# 自定义执行计划

## 什么是执行计划

执行计划（`ModelExecutePlan`）定义了模型各层在前向和反向传播中的分组和执行顺序，它决定了哪些层被分配到同一个流水线阶段（stage）。

一个执行计划包含两个属性：

- **`fwd_plan`**：前向传播的执行分组，类型为 `List[range]`，每个 `range` 表示一个 stage 包含的层索引
- **`bwd_plan`**：反向传播的执行分组，类型为 `List[range]`

### stage 的概念

Stage 是流水线调度的基本单位。同一个 stage 内的所有层会被一起上传到 GPU 并连续执行。stage 的设计直接影响训练效率：

- **stage 太大**：单个 stage 的参数和激活占用过多 GPU 显存，可能导致 OOM
- **stage 不均衡**：最慢的 stage 成为瓶颈，其他 GPU 空闲等待

RoundPipe 的非对称划分允许前向和反向使用不同的 stage 划分。由于前向计算约为反向的 1/3 时间，前向 stage 通常包含更多层，反向 stage 包含更少层，使得每个 stage 的执行时间尽可能相等。

**示例**：一个 4 层模型的执行计划：

```python
from roundpipe import ModelExecutePlan

plan = ModelExecutePlan()

# 使用 forward() 时的执行计划
plan.fwd_plan = [range(0, 2), range(2, 4)] # 前向：2 个 stage，每个 2 层
plan.bwd_plan = [range(3, 4), range(2, 3), # 反向：4 个 stage，每个 1 层
                 range(1, 2), range(0, 1)]
```

使用 `forward_backward()` 时，反向的第一个 stage 会融合部分前向计算（省去冗余重计算），因此第一个反向 stage 的层不应与前向 plan 重叠：

```python
# 使用 forward_backward() 时的执行计划
plan = ModelExecutePlan()
plan.fwd_plan = [range(0, 3)]              # 前向只覆盖前 3 层
plan.bwd_plan = [range(3, 4), range(2, 3), # 反向第一个 stage 从第 4 层开始
                 range(1, 2), range(0, 1)]
```

## 自动调优

大多数情况下，不需要手动构造执行计划。`ModelExecutePlan.auto()` 会根据模型各层的实际计算时间和显存占用自动生成近似最优的划分方案。

```python
from roundpipe import ModelExecutePlan, RoundPipeRunConfig

# 自动生成执行计划（建议先运行多次迭代以收集计时数据）
plan = ModelExecutePlan.auto("fused", model)

# 使用生成的计划
loss = model.forward_backward(
    input_args=(data,),
    label=labels,
    loss_fn=loss_fn,
    run_config=RoundPipeRunConfig(execute_plan=plan),
)
```

`auto()` 的参数：

- **`run_type`**：运行类型
    - `"infer"`：仅前向推理
    - `"train"`：分离的前向和反向（基于 `forward()` 的训练）
    - `"fused"`：融合前向反向（基于 `forward_backward()` 的训练，最常用）
- **`min_stages`**：最小 stage 数量，默认为 GPU 数量。stage 数量越多，流水线气泡越少，但每个 stage 越小
- **`upper_threshold`**：stage 负载均衡的上限比例，默认 1.1，含义为允许某个 stage 的执行时间超过执行时间最长的层的比例。增大此值允许更灵活的划分但可能增加显存占用
- **`model_memory_limit`**：估计的 GPU 可用显存（GB），默认为最小 GPU 显存的 60%。RoundPipe 会预取一个 stage 的参数，因此每个 stage 的显存限制为此值的一半

### 自动调优的依据

`auto()` 基于以下信息进行优化：

1. **计时数据**：RoundPipe 在模型执行过程中自动测量每层的前向、反向、重计算时间，使用滑动平均。首次运行时使用默认划分，后续迭代可以基于实际计时数据重新生成更优的计划。
2. **显存约束**：确保每个 stage 的参数和梯度总量不超过显存限制。

**多模型联合优化**：

如果训练涉及多个 RoundPipe 模型（如 encoder + decoder），可以传入多个模型进行联合优化：

```python
plan1, plan2 = ModelExecutePlan.auto("fused", model1, model2)
```

## 手动指定执行计划

### 适用场景

- 自动调优的结果不理想（如某些层的计时数据不稳定）
- 需要精确控制每个 stage 的显存占用
- 调试或性能分析时需要固定划分方案

### 构造目标

手动构造执行计划时，核心目标是**让每个 stage 的执行时间尽可能均衡**。最慢的 stage 决定了整个流水线的吞吐量，其他 stage 的空闲时间就是浪费。

### 如何构造 plan

1. 先用 `ModelTimer.VERBOSE = True` 获取每层的计时数据：

    ```python
    from roundpipe.timer import ModelTimer
    ModelTimer.VERBOSE = True
    # 运行几次迭代后，stderr 会输出每层的 fwd/re/bwd 时间
    ```

2. 根据计时数据，将相邻层分组，使每组的总时间接近：

    ```python
    plan = ModelExecutePlan()
    # 假设有 24 层 transformer + 1 层 lm_head
    # 前向：每层约 2ms，lm_head 约 6ms
    # 分为 4 个 stage，每个约 14ms
    plan.fwd_plan = [
        range(0, 7),    # 层 0-6: 7×2 = 14ms
        range(7, 14),   # 层 7-13: 7×2 = 14ms
        range(14, 21),  # 层 14-20: 7×2 = 14ms
        range(21, 24),  # 层 21-24: 3×2 + 6 = 12ms（含 lm_head）
    ]
    # 反向类似，但每层时间约 6ms（3× 前向）
    plan.bwd_plan = [
        range(24, 25),  # lm_head 反向
        range(22, 24),
        range(20, 22),
        # ...
        range(0, 2),
    ]
    ```

### 验证规则

执行计划必须满足以下条件，否则 RoundPipe 会报错：

- 前向 plan 的所有 range 合并后必须恰好覆盖所有层（0 到 L-1），每层恰好出现一次
- 反向 plan 同样必须恰好覆盖所有层
- 前向 plan 中的层索引必须递增（从浅到深）
- 反向 plan 中的层索引必须递减（从深到浅）
- 使用 `forward_backward()` 时，前向 plan 的最后一层 + 1 应等于反向 plan 第一个 stage 的起始层，即最后几层只参与进行反向传播（前向计算当作重计算）
