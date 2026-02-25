# 模型相关 API

## roundpipe.RoundPipe

```python
class roundpipe.RoundPipe(
    model: nn.Module,
    optim_dtype: Optional[torch.dtype] = None,
    name: Optional[str] = None,
    model_run_config: RoundPipeRunConfig = RoundPipeRunConfig(),
    pin_model: Literal["alloc", "register", "off"] = "alloc",
)
```

将 `nn.Module` 包装为 RoundPipe 的流水线执行运行时。这是 RoundPipe 的核心类，负责将模型拆分为层、管理参数在 CPU 和 GPU 之间的传输、以及协调前向和反向传播的流水线执行。

**参数：**

- `model`：要包装的模型。可以是 `nn.Sequential` 或任意模型。`nn.Sequential` 模型会被直接拆分为各个层级；非 Sequential 模型则会被当作单个层处理。
- `optim_dtype`：优化器参数的数据类型。默认与模型参数的数据类型相同。常用的设置是将模型参数保持为 `torch.float16`，优化器参数使用 `torch.float32` 以保证数值稳定性。
- `name`：可选的标识符，用于日志和跟踪信息中的显示。默认根据调用位置自动生成（格式为 `文件名:行号`）。
- `model_run_config`：模型级别的默认运行设置，在每次调用 `forward()` 或 `forward_backward()` 时作为基线配置。详见 [运行设置](run_config.md)。
- `pin_model`：模型参数的内存锁页策略：
    - `"alloc"`：使用 PyTorch 的 `pin_memory` 锁页内存。这是默认选项，通常有最好的主机-设备传输性能，但由于 PyTorch 会将所有分配对齐到 2 的幂次，可能导致最多 2 倍的 CPU 内存占用。
    - `"register"`：使用 `cudaHostRegister` 锁页内存。减少超大模型的 CPU 内存占用，但主机-设备传输性能约降低 10%。仅在 CUDA 环境下可用。
    - `"off"`：不锁页内存。适用于 LoRA 微调超出 CPU 内存的模型。结合模型加载时的 `mmap`，Linux 可以按需从磁盘加载数据，并在内存不足时自动驱逐已用数据。

### RoundPipe.forward

```python
RoundPipe.forward(
    *args: Any,
    roundpipe_run_config: RoundPipeRunConfig = RoundPipeRunConfig(),
    **kwargs: Any,
) -> Any
```

执行前向传播。输入数据会被自动切分为微批次，在流水线中并行执行，完成后将各微批次的输出合并返回。

**参数：**

- `*args`：传递给底层模型的位置参数。
- `roundpipe_run_config`：本次调用的运行设置，会覆盖模型级别的默认配置。
- `**kwargs`：传递给底层模型的关键字参数。

**返回值：**

- 合并后的输出结果。具体合并方式由 `merge_output` 配置决定。

### RoundPipe.forward_backward

```python
RoundPipe.forward_backward(
    input_args: Tuple[Any, ...] = (),
    input_kwargs: Dict[str, Any] = {},
    label: Any = None,
    loss_fn: Callable[[Any, Any], Union[Sequence[torch.Tensor], torch.Tensor]] = lambda outputs, labels: outputs,
    return_outputs: bool = False,
    run_config: RoundPipeRunConfig = RoundPipeRunConfig(),
) -> Union[Tuple[Union[List[torch.Tensor], torch.Tensor], Any], List[torch.Tensor], torch.Tensor]
```

执行融合的前向和反向传播。相比先调用 `forward()` 再手动执行 `backward()`，此方法使用优化的调度策略，允许前向传播和反向传播在流水线中同时执行，从而消灭流水线气泡。

**参数：**

- `input_args`：前向传播的位置参数。
- `input_kwargs`：前向传播的关键字参数。
- `label`：标签数据，按照 `loss_fn` 的期望格式提供。
- `loss_fn`：损失函数。接收 `(outputs, labels)` 两个参数，返回一个损失张量或损失张量序列。
- `return_outputs`：是否同时返回模型输出。
- `run_config`：本次调用的运行设置。

**返回值：**

- 如果 `return_outputs=False`（默认），返回所有微批次损失的总和。
- 如果 `return_outputs=True`，返回 `(loss_sum, merged_outputs)` 元组。

该函数会对各个微批次分别求损失后分别反向传播并累积梯度，相当于对所有微批次的损失求和作为输入训练样本的损失。调用此函数的语义等价于：

```python
def forward_backward(
    input_args: Tuple[Any, ...] = (),
    input_kwargs: Dict[str, Any] = {},
    label: Any = None,
    loss_fn: Callable[[Any, Any], Union[Sequence[torch.Tensor], torch.Tensor]] = lambda outputs, labels: outputs,
    return_outputs: bool = False,
    run_config: RoundPipeRunConfig = RoundPipeRunConfig(),
) -> Union[Tuple[Union[List[torch.Tensor], torch.Tensor], Any], List[torch.Tensor], torch.Tensor]:
    split_input_args, split_input_kwargs = split_input(
        input_args, input_kwargs
    )
    split_labels = split_label(labels)
    losses, outputs = [], []
    for input_args_mb, input_kwargs_mb, label_mb in zip(
        split_input_args, split_input_kwargs, split_labels
    ):
        output_mb = model.forward(input_args_mb, input_kwargs_mb)
        loss_mb = loss_fn(output_mb, label_mb)
        torch.autograd.backward(loss_mb)
        # 当 loss_mb 是一个张量时，上一行代码其实就是
        # loss_mb.backward()
        losses.append(loss_mb)
        outputs.append(output_mb)
    if return_outputs:
        return sum(losses), merge_output(outputs)
    else:
        return sum(losses)
```

### RoundPipe.optim_parameters

```python
RoundPipe.optim_parameters() -> Iterator[torch.nn.Parameter]
```

返回适用于优化器的参数迭代器。

RoundPipe 内部管理模型参数的存储位置和数据类型。通过此方法获取的参数以优化器所需的格式（由 `optim_dtype` 指定）存储，适合直接传递给优化器。

**返回值：**

- 参数迭代器，每个参数均为优化器专用的参数副本。

### RoundPipe.optim_named_parameters

```python
RoundPipe.optim_named_parameters(
    prefix: str = "",
    remove_duplicate: bool = True,
) -> Iterator[Tuple[str, torch.nn.Parameter]]
```

返回带名称的优化器参数迭代器。

**参数：**

- `prefix`：参数名称的前缀。
- `remove_duplicate`：是否跳过重复的参数。

**返回值：**

- 生成 `(name, parameter)` 元组的迭代器。

### RoundPipe.step

```python
RoundPipe.step(
    step_fn: Callable[..., None],
    is_async: bool = True,
    *args: Any,
    **kwargs: Any,
) -> None
```

使用提供的步骤函数执行优化器更新。

默认的异步模式下，`step` 会立即返回，优化器更新在后台线程中执行。训练迭代将使用延迟一步的参数，这在实际训练中通常不影响收敛。同步模式会等待优化器更新完成后才返回，确保每次迭代使用最新参数，但会显著降低性能，一般不推荐使用。

!!! warning "数据竞争"
    `step_fn` 中的数据访问应仅限于优化器参数。访问其他数据需要注意潜在的数据竞争问题。

**参数：**

- `step_fn`：执行一步优化更新的可调用对象。
- `is_async`：是否异步执行。默认为 `True`。
- `*args`：转发给 `step_fn` 的位置参数。
- `**kwargs`：转发给 `step_fn` 的关键字参数。

### RoundPipe.synchronize

```python
RoundPipe.synchronize() -> None
```

同步优化器参数和反向传播的梯度到模型参数。

调用此方法后，模型参数将反映最新的优化器更新结果，参数的 `.grad` 属性将包含累积的梯度。这在需要检查参数值或梯度时非常有用（例如在评估前或保存检查点前）。

## roundpipe.wrap_model_to_roundpipe

```python
roundpipe.wrap_model_to_roundpipe(
    model: nn.Module,
    use_sequential_preset: Optional[bool] = None,
    lower_threshold: int = 16 * 1024,
    upper_threshold: Optional[int] = None,
    skip_modules: Container[nn.Module] = [],
    override_config: Dict[nn.Module, RoundPipeRunConfig] = {},
    model_run_config: RoundPipeRunConfig = RoundPipeRunConfig(),
    name: Optional[str] = None,
    **roundpipe_kwargs: Any,
) -> Union[RoundPipe, AutoRoundPipe]
```

使用递归启发式方法或预设方案，将模型自动包装为 RoundPipe 实例。

此函数会尝试以下策略：

1. 如果 `use_sequential_preset` 不为 `False`，首先尝试使用 RoundPipe 内置的模型预设将模型转换为等效的 Sequential 结构。
2. 如果没有可用的预设，则递归遍历模型的子模块，根据大小阈值和模块类型决定如何包装。
3. 对于最终无法拆分为 Sequential 的模型，返回 `AutoRoundPipe` 实例。

**参数：**

- `model`：要包装的根模块。
- `use_sequential_preset`：是否使用内置的 Sequential 预设。`None` 或 `True` 会尝试使用预设，`False` 跳过预设查找。当设为 `None` 时，如果找到匹配的预设会打印提示信息。
- `lower_threshold`：直接包装为 `RoundPipe` 的最小模块大小（字节）。小于此阈值的模块会以 `num_microbatch=1` 的方式包装。
- `upper_threshold`：拆分子模块的最大大小阈值。默认为模型总大小除以 `(GPU 数量 + 1)`。
- `skip_modules`：不需要包装的模块列表。
- `override_config`：为特定模块指定的运行设置覆盖。
- `model_run_config`：`RoundPipe` 实例的默认运行设置。
- `name`：模块名称。如果为 `None`，根据调用位置自动生成。
- `**roundpipe_kwargs`：转发给 `RoundPipe` 构造函数的额外参数。

**拆分策略解析：**

当没有匹配的预设时，函数递归遍历模型的子模块树。对于每个遇到的模块，按照以下优先级决定处理方式：

1. **跳过**：如果模块在 `skip_modules` 列表中，直接返回原模块，不做任何包装。**这个模块将会在 CPU 上执行。**

2. **直接包装为多层 `RoundPipe`**：如果模块是 `nn.Sequential`，将其直接包装为 `RoundPipe`，其中每个子模块作为一个独立的流水线层。

3. **直接包装为单层 `RoundPipe`**：如果满足以下任一条件：
    - 模块**自身参数**（非递归）的大小 ≥ `lower_threshold`（默认 16 KB）
    - 模块**总大小**（递归）≤ `upper_threshold`

    并且模块实现了自己的 `forward` 方法，则将整个模块包装为一个单层的 `RoundPipe`。

    特殊情况：如果模块总大小 < `lower_threshold`，包装时会将 `num_microbatch` 设置为 1，因为对极小的模块进行微批次拆分没有性能收益。

4. **逐元素包装 `nn.ModuleList`**：如果模块是 `nn.ModuleList`，将列表中的每个元素分别包装为独立的 `RoundPipe` 实例。默认情况下，我们认为 ModuleList 的模块是连续调用的，非末尾元素的 `merge_output` 设为 `False`，以实现流水线并行计算。如果需要访问 ModuleList 中间层的输出，可以在 `override_config` 中为对应模块设置 `merge_output=True`，但这会引入额外的同步并阻止框架并行计算。

5. **递归进入子模块**：如果以上条件都不满足（即模块太大、不是 Sequential/ModuleList、且自身没有足够多的参数），则递归遍历其 `named_children()`，对每个子模块重复上述判断。**模块本身的`forward`将会在 CPU 上执行。**

6. **其它特殊处理**：对于 HuggingFace 的 `PreTrainedModel`，还会额外包装其 `loss_function`。

**返回值：**

- 如果成功使用预设或递归方法将模型转换为 `RoundPipe`，返回 `RoundPipe` 实例；否则返回 `AutoRoundPipe` 实例。`AutoRoundPipe` 是一个占位类，表示模型已被包装但无法拆分为 Sequential 结构，用户仍然可以使用 RoundPipe 的前向传播和优化器功能，但无法享受层级拆分带来的性能提升，也不能调用`forward_backward()` 方法。

**异常：**

- `NotImplementedError`：如果 `use_sequential_preset=True` 但该模型类型没有对应的预设。

### RoundPipe.set_original_model

```python
RoundPipe.set_original_model(original_model: nn.Module) -> None
```

设置原始模型的引用，用于属性访问的代理转发。

当模型通过 `wrap_model_to_roundpipe` 被转换为 Sequential 结构后，原始模型上的属性（如 HuggingFace 模型的 `vocab_size`、`config` 等）仍然可以通过此机制访问。`wrap_model_to_roundpipe` 会自动调用此方法，通常不需要手动调用。

**参数：**

- `original_model`：原始未包装的模型。

### 类成员转发

RoundPipe 重写了 `__getattr__`、`__setattr__` 和 `__delattr__`，实现了对被包装模型的属性透明访问：

- **读取**：当访问 RoundPipe 实例上不存在的属性时，如果设置了`original_model`会从中查找，否侧从构造 RoundPipe 变量时的传入模型上查找。
- **写入**：初始化完成后，对 RoundPipe 实例的属性写入会转发到 `original_model`，如果没有则转发到构造 RoundPipe 变量时的传入模型上。
- **删除**：属性删除同样会传播到 `original_model` 或构造传入模型。

这一机制使得 RoundPipe 包装后的模型可以像原始模型一样使用，无需关心包装层的存在。
