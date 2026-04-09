# 自定义 Microbatch 切分

RoundPipe 将输入数据拆分为多个微批次（microbatch）进行流水线执行。默认情况下，RoundPipe 会自动推断切分方式，但对于非标准的输入/输出格式，可能需要手动指定切分和合并策略。

## 默认切分行为

### 输入自动切分

当 `split_input` 为 `None`（默认）时，RoundPipe 递归遍历输入的嵌套结构（tuple、list、dict），对每个叶节点应用以下规则：

- **多维 Tensor**：沿第 0 维（batch 维度）均匀切分
- **标量 Tensor 和非 Tensor 类型**：复制到每个微批次

```python
# 示例：自动切分行为
# 输入 args = (images,)，images.shape = (12, 3, 224, 224)
# num_microbatch = 4
# → 每个微批次得到 images.shape = (3, 3, 224, 224)
```

### Label 自动切分

`split_label` 为 `None`（默认）时，行为与输入自动切分相同。

### Output 自动合并

`merge_output` 为 `None` 或 `True`（默认）时，递归遍历输出结构，对每个叶节点：

- **多维 Tensor**：沿第 0 维拼接（`torch.cat`）
- **标量 Tensor**：取所有微批次的平均值
- **非 Tensor 类型**：检查所有微批次的值是否相等，相等则返回该值，否则报错

## split_input

当默认切分不满足需求时，可以通过 `split_input` 手动指定。有两种方式：

### 方式一：Split Spec

使用 `TensorChunkSpec` 和 `_Replicate` 标记每个输入的切分方式：

```python
from torch.distributed.pipelining.microbatch import TensorChunkSpec, _Replicate
from roundpipe import RoundPipeRunConfig

# 场景：模型接受 (images, scale_factor)
# images 沿 batch 维切分，scale_factor 复制到每个微批次
config = RoundPipeRunConfig(
    split_input=(
        (TensorChunkSpec(0), _Replicate),  # args spec
        None,                               # kwargs spec（自动推断）
    )
)

# 场景：某个输入需要沿非 batch 维切分
# 例如 (tokens, position_ids)，position_ids 沿 dim=1 切分
config = RoundPipeRunConfig(
    split_input=(
        (TensorChunkSpec(0), TensorChunkSpec(1)),  # args spec
        None,
    )
)
```

Spec 的结构必须与输入的结构一一对应。`args` spec 是一个 tuple，`kwargs` spec 是一个 dict（或 `None` 表示自动推断）。

### 方式二：自定义函数

当 spec 无法满足需求时，可以提供完全自定义的切分函数：

```python
def custom_split(args, kwargs, num_microbatch):
    """
    args: 原始位置参数 tuple
    kwargs: 原始关键字参数 dict
    num_microbatch: 微批次数量
    返回: (args_list, kwargs_list)，每个 list 长度等于 num_microbatch
    """
    images, masks = args
    chunks_images = images.chunk(num_microbatch)
    chunks_masks = masks.chunk(num_microbatch)
    args_list = [(img, mask) for img, mask in zip(chunks_images, chunks_masks)]
    kwargs_list = [kwargs] * num_microbatch  # kwargs 复制到每个微批次
    return args_list, kwargs_list

config = RoundPipeRunConfig(split_input=custom_split)
```

一个更复杂的例子——输入是 dict，需要对不同字段采用不同的切分策略：

```python
def split_transformer_input(args, kwargs, num_microbatch):
    # kwargs = {"input_ids": ..., "attention_mask": ..., "position_ids": ...}
    input_ids_chunks = kwargs["input_ids"].chunk(num_microbatch)
    mask_chunks = kwargs["attention_mask"].chunk(num_microbatch)
    pos_chunks = kwargs["position_ids"].chunk(num_microbatch)
    args_list = [()] * num_microbatch
    kwargs_list = [
        {"input_ids": ids, "attention_mask": mask, "position_ids": pos}
        for ids, mask, pos in zip(input_ids_chunks, mask_chunks, pos_chunks)
    ]
    return args_list, kwargs_list

config = RoundPipeRunConfig(split_input=split_transformer_input)
```

## split_label

与 `split_input` 类似，`split_label` 控制标签的切分方式。

### Split Spec

```python
from torch.distributed.pipelining.microbatch import TensorChunkSpec, _Replicate

# label 是一个 tuple: (targets, sample_weights)
# targets 沿 batch 维切分，sample_weights 也沿 batch 维切分
config = RoundPipeRunConfig(
    split_label=(TensorChunkSpec(0), TensorChunkSpec(0))
)

# label 是一个 tuple: (targets, class_weights)
# targets 沿 batch 维切分，class_weights 复制（所有样本共享）
config = RoundPipeRunConfig(
    split_label=(TensorChunkSpec(0), _Replicate)
)
```

### 自定义函数

```python
def custom_split_label(label, num_microbatch):
    """
    label: 原始标签
    num_microbatch: 微批次数量
    返回: List[label]，长度等于 num_microbatch
    """
    targets, weights = label
    chunks_targets = targets.chunk(num_microbatch)
    chunks_weights = weights.chunk(num_microbatch)
    return [(t, w) for t, w in zip(chunks_targets, chunks_weights)]

config = RoundPipeRunConfig(split_label=custom_split_label)
```

## merge_output

控制如何将各微批次的输出合并为最终输出。

### Split Spec

与切分类似，可以用 spec 指定每个输出字段的合并方式：

```python
from torch.distributed.pipelining.microbatch import TensorChunkSpec, _Replicate, _CustomReducer

# 输出是 (logits, hidden_states)
# logits 沿 batch 维拼接，hidden_states 也沿 batch 维拼接
config = RoundPipeRunConfig(
    merge_output=(TensorChunkSpec(0), TensorChunkSpec(0))
)

# 使用自定义 reducer：对 loss 求和
sum_reducer = _CustomReducer(torch.tensor(0.0), lambda x, y: x + y)
config = RoundPipeRunConfig(merge_output=sum_reducer)
```

### 自定义函数

```python
def custom_merge(outputs):
    """
    outputs: List[output]，每个元素是一个微批次的模型输出
    返回: 合并后的输出
    """
    # 例如：HuggingFace 模型输出是一个对象，只需要 logits
    logits = torch.cat([out.logits for out in outputs], dim=0)
    return logits

config = RoundPipeRunConfig(merge_output=custom_merge)
```

### 禁用合并

设置 `merge_output=False` 可以禁用输出合并。此时输出中的每个叶变量会以 `RoundPipePackedData`（一个 list 子类）的形式返回，包含各微批次的输出和对应的 CUDA 传输事件。

```python
config = RoundPipeRunConfig(merge_output=False)
output = model(data, roundpipe_run_config=config)
# output 中的 tensor 是 RoundPipePackedData 对象
output.synchronize()  # 等待所有传输完成
# 然后可以像普通 list 一样访问各微批次的输出
```

这在将一个 RoundPipe 模型的输出直接传给另一个 RoundPipe 模型时很有用，可以避免不必要的同步和数据拷贝。`wrap_model_to_roundpipe` 在递归包装非最终模块时会自动设置 `merge_output=False`。
