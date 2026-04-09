# 注册自定义数据类型

## 问题背景

RoundPipe 在训练过程中需要对模型各层的输入输出执行以下操作：

- **切分与合并**：将输入拆分为多个微批次，将输出从多个微批次合并
- **设备间传输**：在 CPU 和 GPU 之间传输各层的输入输出激活值

为了执行这些操作，RoundPipe 需要能够"展开"数据结构，找到其中包含的所有 tensor 并逐一操作。RoundPipe 使用 PyTorch 内置的 pytree 机制（`torch.utils._pytree`）来实现这一点。

默认情况下，pytree 只支持 Python 内置容器类型：`tuple`、`list`、`dict`、`namedtuple`、`OrderedDict`。如果模型的输入或输出包含**自定义数据类型**（如 dataclass、自定义类），pytree 会将其视为不可拆分的叶节点——其中包含的 tensor 不会被自动传输到正确的设备上，也不会被正确切分。

例如，假设模型某层的输出是一个自定义类：

```python
class ModelOutput:
    def __init__(self, logits, hidden_states):
        self.logits = logits            # tensor
        self.hidden_states = hidden_states  # tensor
```

如果不注册展开函数，RoundPipe 会将整个 `ModelOutput` 对象视为一个叶节点。当 RoundPipe 尝试将其从 GPU 传输到 CPU 时，内部的 tensor 不会被移动，导致后续层在错误的设备上访问数据。

## 注册展开函数

通过 `torch.utils._pytree.register_pytree_node` 注册自定义类型的展开（flatten）和重建（unflatten）函数，让 pytree 能够识别并处理自定义类型。

### 基本接口

```python
from torch.utils._pytree import register_pytree_node

register_pytree_node(
    cls,                # 要注册的类
    flatten_fn,         # 展开函数：cls -> (children, context)
    unflatten_fn,       # 重建函数：(children, context) -> cls
)
```

- **`flatten_fn(obj)`**：接受一个自定义类型的实例，返回 `(children, context)`。`children` 是一个 list，包含所有需要被 pytree 递归处理的子元素（通常是 tensor）；`context` 是重建时需要的额外信息（如字段名、非 tensor 属性等）。
- **`unflatten_fn(children, context)`**：接受 `children` 和 `context`，重建出原始类型的实例。

### 示例：注册 dataclass

```python
from dataclasses import dataclass
import torch
from torch.utils._pytree import register_pytree_node

@dataclass
class TransformerOutput:
    logits: torch.Tensor
    hidden_states: torch.Tensor
    loss: float = 0.0  # 非 tensor 字段

def flatten_transformer_output(obj):
    # children: 需要被 pytree 处理的元素（tensor）
    children = [obj.logits, obj.hidden_states]
    # context: 重建时需要的额外信息
    context = {"loss": obj.loss}
    return children, context

def unflatten_transformer_output(children, context):
    logits, hidden_states = children
    return TransformerOutput(
        logits=logits,
        hidden_states=hidden_states,
        loss=context["loss"],
    )

register_pytree_node(
    TransformerOutput,
    flatten_transformer_output,
    unflatten_transformer_output,
)
```

注册后，RoundPipe 就能正确处理 `TransformerOutput`：

- 切分时，`logits` 和 `hidden_states` 会沿 batch 维度被切分为多个微批次
- 传输时，两个 tensor 会被正确地在 CPU 和 GPU 之间移动
- 合并时，各微批次的 tensor 会被拼接，`loss` 字段会检查一致性

### 示例：注册包含可选字段的类

```python
class FlexibleOutput:
    def __init__(self, logits, attentions=None):
        self.logits = logits
        self.attentions = attentions  # 可能为 None

def flatten_flexible(obj):
    children = [obj.logits]
    has_attentions = obj.attentions is not None
    if has_attentions:
        children.append(obj.attentions)
    context = {"has_attentions": has_attentions}
    return children, context

def unflatten_flexible(children, context):
    if context["has_attentions"]:
        logits, attentions = children
    else:
        logits = children[0]
        attentions = None
    return FlexibleOutput(logits=logits, attentions=attentions)

register_pytree_node(FlexibleOutput, flatten_flexible, unflatten_flexible)
```

### 示例：注册嵌套结构

如果自定义类型中包含其他自定义类型或容器，只需将它们放入 `children`，pytree 会递归处理：

```python
class BatchResult:
    def __init__(self, outputs: list, metadata: dict):
        self.outputs = outputs    # list of tensors
        self.metadata = metadata  # dict，可能包含 tensor

def flatten_batch_result(obj):
    # 将 outputs 和 metadata 都放入 children，pytree 会递归展开
    children = [obj.outputs, obj.metadata]
    context = None
    return children, context

def unflatten_batch_result(children, context):
    outputs, metadata = children
    return BatchResult(outputs=outputs, metadata=metadata)

register_pytree_node(BatchResult, flatten_batch_result, unflatten_batch_result)
```

### 注意事项

- 注册必须在创建 `RoundPipe` 实例**之前**完成，通常放在模型定义文件的顶部
- `children` 中的元素顺序必须与 `unflatten_fn` 中的解包顺序一致
- `context` 必须是可序列化的（不能包含 tensor 或其他不可 pickle 的对象）
- HuggingFace Transformers 的模型输出类（如 `CausalLMOutputWithPast`）已经被 transformers 自动注册，无需手动处理
