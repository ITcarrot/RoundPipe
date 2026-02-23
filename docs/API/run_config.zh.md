# 运行设置

## roundpipe.RoundPipeRunConfig

```python
class roundpipe.RoundPipeRunConfig(
    requires_grad: Optional[bool] = None,
    output_device: Optional[torch.device] = None,
    preserve_rng_state: Optional[bool] = None,
    recompute_grain: Optional[Literal["stage", "layer"]] = None,
    num_microbatch: Optional[int] = None,
    split_input: Union[
        Tuple[Optional[Tuple], Optional[Dict[str, Any]]],
        Callable[[Tuple, Dict[str, Any], int], Tuple[List[Tuple], List[Dict[str, Any]]]],
        None,
    ] = None,
    split_label: Union[Any, Callable[[Any, int], List[Any]], None] = None,
    merge_output: Union[Any, Callable[[List[Any]], Any], bool, None] = None,
    execute_plan: Optional[ModelExecutePlan] = None,
)
```

RoundPipe 模型的运行时配置。

用户可以在两个层级指定配置：

- **模型级别**：在创建 `RoundPipe` 时通过 `model_run_config` 参数传入，作为该模型的默认配置。
- **调用级别**：在调用 `forward()` 或 `forward_backward()` 时传入，覆盖模型级别的默认值。

当两个层级都指定了某个参数时，调用级别的配置优先。如果两个层级都未指定（均为 `None`），则使用以下默认值：

**参数：**

- `requires_grad`：是否启用梯度计算。默认跟随全局的 `torch.is_grad_enabled()` 设置。
- `output_device`：输出张量放置的设备。默认为 CPU。如果需要在 GPU 上直接使用输出，可以指定为 CUDA 设备。
- `preserve_rng_state`：是否保存和恢复随机数生成器状态。默认为 `True`。开启后可以保证重计算阶段与前向传播的随机行为一致（例如 Dropout）。
- `recompute_grain`：反向传播重计算的粒度。默认为 `"stage"`。
    - `"stage"`：以流水线阶段为单位进行重计算。
    - `"layer"`：在每个阶段内按层进行重计算和反向传播。`"layer"` 粒度可以减少峰值显存占用，但可能增加一些数据传输开销。
- `num_microbatch`：将输入切分为多少个微批次。默认为 `CUDA 设备数量 + 1`。增加微批次数量可以减小显存占用，但过高可能导致每一微批次的计算量过小，降低 GPU 利用率；低于默认值将导致流水线气泡。
- `split_input`：指定如何将输入参数切分为微批次。默认为自动切分。详见[切分输入](#split-input)。
- `split_label`：指定如何将标签切分为微批次。默认为自动切分。详见[切分标签](#split-label)。
- `merge_output`：指定如何将微批次的输出合并为单个输出。默认为自动合并。详见[合并输出](#merge-output)。
- `execute_plan`：可选的 `ModelExecutePlan`，用于指定执行策略。默认由 RoundPipe 自动调优。详见[设置运行计划](#set-execute-plan)。

## 切分输入 {#split-input}

RoundPipe 需要将输入数据切分为多个微批次以实现流水线并行。`split_input` 参数控制切分方式。

### 自动切分机制

当 `split_input` 为 `None` 时（默认），RoundPipe 会自动推断切分方式。输入参数可以是单个值，也可以是 Python 自带的嵌套结构（元组、列表、字典），自动切分会递归地对结构中的每个叶子节点应用以下规则：
- 对于 `torch.Tensor`：如果张量维度大于 0，沿第 0 维（batch 维度）均匀切分。
- 对于标量张量、非张量类型：在所有微批次之间复制。
- 对于自定义变量类型，除非用户使用`torch.utils._pytree.register_pytree_node`定义了展开方法，否则会被视为不可切分，在所有微批次之间复制，即便其中包含张量。

自动切分适用于大多数标准场景。但如果模型的输入结构较为复杂（例如某些张量不应按 batch 维度切分），则需要手动指定切分方案。

### 编写切分方案

`split_input` 可以设为 `Tuple[Optional[Tuple], Optional[Dict[str, Any]]]`，分别指定位置参数和关键字参数的切分方案（spec）。位置参数或关键字参数的 spec 可以为 `None`（表示自动切分）或一个与输入结构对应的 spec 结构。切分方案的结构应与输入参数一一对应，每个位置使用以下标记之一：

- `torch.distributed.pipelining.microbatch.TensorChunkSpec(dim)`：沿指定维度切分张量。
- `torch.distributed.pipelining.microbatch._Replicate`：在所有微批次间复制。

**示例：**

```python
from torch.distributed.pipelining.microbatch import TensorChunkSpec, _Replicate
from roundpipe import RoundPipeRunConfig

# 位置参数中第一个按 batch 切分，第二个复制
config = RoundPipeRunConfig(
    split_input=(
        (TensorChunkSpec(0), _Replicate),  # args spec
        None,                               # kwargs spec（自动推断）
    )
)
```

### 编写自定义切分函数

当内置的切分方案无法满足需求时，可以传入一个自定义函数：

```python
def custom_split(
    args: Tuple, kwargs: Dict[str, Any], num_microbatch: int
) -> Tuple[List[Tuple], List[Dict[str, Any]]]:
```

函数接收三个参数：

1. `args`：原始位置参数元组。
2. `kwargs`：原始关键字参数字典。
3. `num_microbatch`：微批次数量。

函数应返回一个元组 `(args_list, kwargs_list)`，其中每个列表的长度等于微批次数量。

**示例：**

```python
def custom_split(args, kwargs, num_microbatch):
    images, masks = args
    chunks_images = images.chunk(num_microbatch)
    chunks_masks = masks.chunk(num_microbatch)
    args_list = [(img, mask) for img, mask in zip(chunks_images, chunks_masks)]
    kwargs_list = [kwargs] * num_microbatch  # 关键字参数复制
    return args_list, kwargs_list

config = RoundPipeRunConfig(split_input=custom_split)
```

## 切分标签 {#split-label}

在使用 `forward_backward()` 时，RoundPipe 同样需要将标签（label）切分为多个微批次，使每个微批次的标签与对应的输入匹配。`split_label` 参数控制切分方式。

### 自动切分机制

当 `split_label` 为 `None` 时（默认），RoundPipe 会自动推断切分方式，行为与输入参数的自动切分类似。

### 编写切分方案

`split_label` 可以设为一个与标签结构对应的 spec，spec编写方法和输入参数的切分spec类似。

**示例：**

```python
from torch.distributed.pipelining.microbatch import TensorChunkSpec, _Replicate
from roundpipe import RoundPipeRunConfig

# 标签为 (标量张量, 形状为 (3, 12, 4) 的张量, 整数)
# 第二个张量需要沿第 1 维切分
config = RoundPipeRunConfig(
    split_label=(_Replicate, TensorChunkSpec(1), _Replicate)
)
```

### 编写自定义切分函数

当内置的切分方案无法满足需求时，可以传入一个自定义函数：

```python
def custom_split_label(
    label: Any, num_microbatch: int
) -> List[Any]:
```

函数接收两个参数：

1. `label`：原始标签。
2. `num_microbatch`：微批次数量。

函数应返回一个列表，长度等于微批次数量，每个元素为对应微批次的标签。

**示例：**

```python
def custom_split_label(label, num_microbatch):
    targets, weights = label
    chunks_targets = targets.chunk(num_microbatch)
    chunks_weights = weights.chunk(num_microbatch)
    return [(t, w) for t, w in zip(chunks_targets, chunks_weights)]

config = RoundPipeRunConfig(split_label=custom_split_label)
```

## 合并输出 {#merge-output}

RoundPipe 在流水线执行完成后，需要将各微批次的输出合并为一个完整的输出。`merge_output` 参数控制合并方式。

### 自动合并机制

当 `merge_output` 为 `None` 或 `True` 时（默认），RoundPipe 会自动推断合并方式。输出可以是单个值，也可以是 Python 自带的嵌套结构（元组、列表、字典），自动合并会递归地对结构中的每个叶子节点应用以下规则：

- **张量**：沿第 0 维（batch 维度）拼接（`torch.cat`）。
- **标量张量**：取所有微批次的平均值。
- **非张量类型**：会检查所有微批次的值是否相同，如果相同则返回该值，否则抛出错误。
- 对于自定义变量类型，除非用户使用`torch.utils._pytree.register_pytree_node`定义了展开方法，否则会被视为不可分割，按非张量类型处理，即便其中包含张量。

### 编写合并方案

`merge_output` 可以设为一个与输出结构对应的 spec，每个位置使用以下的标记之一：

- `torch.distributed.pipelining.microbatch.TensorChunkSpec(dim)`：沿指定维度拼接张量（`torch.cat`）。
- `torch.distributed.pipelining.microbatch._Replicate`：输出保证这个变量在各输出见相同。
- `torch.distributed.pipelining.microbatch._CustomReducer`：符合 PyTorch 定义的规约方法。

`_CustomReducer`示例：

```python
from torch.distributed.pipelining.microbatch import _CustomReducer

sum_reducer = _CustomReducer(torch.tensor(0.0), lambda x, y: x + y)
```

### 编写自定义合并函数

可以传入一个自定义函数来完全控制合并逻辑：

```python
def custom_merge(outputs: List[Any]) -> Any:
```

函数接收一个列表，包含各微批次的输出，返回合并后的结果。

```python
def custom_merge(outputs):
    # outputs 是一个列表，每个元素是一个微批次的模型输出
    logits = torch.cat([out.logits for out in outputs], dim=0)
    return logits

config = RoundPipeRunConfig(merge_output=custom_merge)
```

### 不合并输出

将 `merge_output` 设为 `False` 时，输出不会被合并，而是将输出的每一基本变量以 `RoundPipePackedData` 的形式返回。基本变量是指展开所有 Python 自带的嵌套结构（元组、列表、字典）后的叶子节点。对于自定义变量类型，除非用户使用`torch.utils._pytree.register_pytree_node`定义了展开方法，否则同样视为基本变量。

`RoundPipePackedData` 是 Python `list` 的子类，保存了各微批次的输出以及对应的 CUDA 传输事件。通常情况下，用户不需要直接操作 `RoundPipePackedData`，如有需要，可以使用`.synchronize()`方法等待所有输出传输至 CPU 内存，之后直接将其视为一个普通的列表来访问各微批次的输出。

`RoundPipe` 模型的输出需要直接传递给下一个 `RoundPipe` 模型时，使用 `merge_output=False` 可以避免不必要的同步和数据拷贝，从而实现模型间的流水线衔接。`wrap_model_to_roundpipe` 在递归包装非最后一层的 `nn.ModuleList` 模块时，会自动设置 `merge_output=False`。

## 设置运行计划 {#set-execute-plan}

### roundpipe.ModelExecutePlan

```python
class roundpipe.ModelExecutePlan:
    def __init__(self) -> None:
        self.fwd_plan: List[range] = []
        self.bwd_plan: List[range] = []
```

RoundPipe 模型的执行计划，定义了前向和反向传播中各层的执行顺序和分组方式。

执行计划决定了哪些层被分配到同一个流水线阶段（stage），每个阶段的层会被一起加载到 GPU 上执行。合理的执行计划可以平衡各阶段的计算负载、控制 GPU 显存占用。

不同于传统的流水线并行训练框架，RoundPipe 不需要前向传播和反向传播严格成对地执行，因此前向计划和反向计划可以独立设计；不需要阶段数是 GPU 数量的倍数。只要总阶段数大于 GPU 数量，所有阶段用时相近，即可达成较好的并行效率。

**属性：**

- `fwd_plan`：前向传播的执行顺序，`List[range]` 类型。每个 `range` 表示一个阶段中包含的层编号。
- `bwd_plan`：反向传播的执行顺序，`List[range]` 类型。

**示例：**

对于一个包含 4 层的模型，以下是使用`forward`方法的一个可能的执行计划：

```python
plan = ModelExecutePlan()
plan.fwd_plan = [range(0, 2), range(2, 4)]  
plan.bwd_plan = [range(3, 4), range(2, 3), range(1, 2), range(0, 1)]
```

使用`forward_backward`方法时，最后一个阶段将不会进行重计算以节约时间，因此反向传播计划中的第一个阶段的模型层不应出现在前向传播计划中：

```python
plan = ModelExecutePlan()
plan.fwd_plan = [range(0, 3)]  
plan.bwd_plan = [range(3, 4), range(2, 3), range(1, 2), range(0, 1)]
```

#### ModelExecutePlan.auto

```python
@classmethod
ModelExecutePlan.auto(
    run_type: Literal["infer", "train", "fused"],
    *models: RoundPipe,
    min_stages: int = get_num_devices(),
    upper_threshold: float = 1.1,
    model_memory_limit: float = get_min_gpu_memory() * 0.6,
) -> Union[ModelExecutePlan, List[ModelExecutePlan]]
```

根据模型的计算时间和内存大小自动生成执行计划。

当传入单个模型时，返回单个 `ModelExecutePlan`；传入多个模型时，返回计划列表，各模型的计划会协同优化以达到整体负载均衡。

**参数：**

- `run_type`：运行类型。
    - `"infer"`：仅前向推理。
    - `"train"`：独立的前向传播和反向传播（用于`forward`训练）。
    - `"fused"`：融合的前向和反向传播（用于`forward_backward`训练）。
- `*models`：一个或多个 `RoundPipe` 模型。
- `min_stages`：最少的流水线阶段数。这是一个提示值，实际阶段数可能因模型大小而低于此值。默认为 GPU 设备数。
- `upper_threshold`：阶段负载均衡的上限比率。限制任一阶段与最慢层之间的最大允许比值。增大此值提供更灵活的阶段划分，但可能消耗更多 GPU 显存。默认为 `1.1`。
- `model_memory_limit`：模型参数和梯度可用的 GPU 显存估算值（GB）。RoundPipe 会预取一个阶段的模型参数到 GPU 显存，因此每个阶段的显存限制为 `model_memory_limit / 2`。默认为最小 GPU 显存的 60%。

**返回值：**

- 单个模型时返回 `ModelExecutePlan`，多个模型时返回 `List[ModelExecutePlan]`。

RounPipe 在模型运行时会对每一层的计算时间进行在线测量并取滑动平均值。根据计时结果，RoundPipe 会尽量使得每个阶段的计算时间尽可能接近，以实现最佳的流水线效率。

**示例：**

```python
from roundpipe import RoundPipe, RoundPipeRunConfig, ModelExecutePlan

model = RoundPipe(my_model)

# 自动生成训练计划
plan = ModelExecutePlan.auto("fused", model)

# 使用生成的计划
loss = model.forward_backward(
    input_args=(data,),
    label=labels,
    loss_fn=loss_fn,
    run_config=RoundPipeRunConfig(execute_plan=plan),
)

# 多模型协同优化
plan1, plan2 = ModelExecutePlan.auto("fused", model1, model2)
output1 = model1.forward(
    input_args=(data1,),
    run_config=RoundPipeRunConfig(execute_plan=plan1, merge_output=False),
)
loss = model2.forward_backward(
    input_args=(output1,),
    label=labels,
    loss_fn=loss_fn,
    run_config=RoundPipeRunConfig(execute_plan=plan2),
)
```
