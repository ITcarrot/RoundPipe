# Registering Custom Data Types

## Background

During training, RoundPipe needs to perform the following operations on each layer's inputs and outputs:

- **Splitting and merging**: split inputs into microbatches and merge outputs back.
- **Device transfers**: move activations between CPU and GPU.

To do this, RoundPipe needs to "unpack" data structures, find every tensor inside, and operate on them individually. It relies on PyTorch's built-in pytree mechanism (`torch.utils._pytree`) for this.

By default, pytree only recognizes Python built-in container types: `tuple`, `list`, `dict`, `namedtuple`, and `OrderedDict`. If a model's inputs or outputs contain **custom data types** (e.g., dataclasses, custom classes), pytree treats them as opaque leaf nodes — tensors inside them won't be transferred to the right device or split correctly.

For example, suppose a layer produces an output using a custom class:

```python
class ModelOutput:
    def __init__(self, logits, hidden_states):
        self.logits = logits            # tensor
        self.hidden_states = hidden_states  # tensor
```

Without registering an unpack function, RoundPipe sees the entire `ModelOutput` object as a single leaf. When it tries to transfer the object from GPU to CPU, the internal tensors are not moved, causing downstream layers to access data on the wrong device.

## Registering Flatten/Unflatten Functions

Use `torch.utils._pytree.register_pytree_node` to register flatten and unflatten functions for your custom type, so that pytree can recognize and process it.

### Interface

```python
from torch.utils._pytree import register_pytree_node

register_pytree_node(
    cls,                # The class to register
    flatten_fn,         # Flatten: cls -> (children, context)
    unflatten_fn,       # Unflatten: (children, context) -> cls
)
```

- **`flatten_fn(obj)`**: Takes an instance of the custom type and returns `(children, context)`. `children` is a list of all elements that pytree should recursively process (typically tensors); `context` is any extra information needed for reconstruction (e.g., field names, non-tensor attributes).
- **`unflatten_fn(children, context)`**: Takes `children` and `context` and reconstructs the original type.

### Example: Registering a Dataclass

```python
from dataclasses import dataclass
import torch
from torch.utils._pytree import register_pytree_node

@dataclass
class TransformerOutput:
    logits: torch.Tensor
    hidden_states: torch.Tensor
    loss: float = 0.0  # Non-tensor field

def flatten_transformer_output(obj):
    # children: elements for pytree to process (tensors)
    children = [obj.logits, obj.hidden_states]
    # context: extra info needed for reconstruction
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

After registration, RoundPipe correctly handles `TransformerOutput`:

- During splitting, `logits` and `hidden_states` are split along the batch dimension into microbatches.
- During transfers, both tensors are moved between CPU and GPU correctly.
- During merging, per-microbatch tensors are concatenated, and the `loss` field is checked for consistency.

### Example: Optional Fields

```python
class FlexibleOutput:
    def __init__(self, logits, attentions=None):
        self.logits = logits
        self.attentions = attentions  # May be None

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

### Example: Nested Structures

If your custom type contains other custom types or containers, simply include them in `children` — pytree handles them recursively:

```python
class BatchResult:
    def __init__(self, outputs: list, metadata: dict):
        self.outputs = outputs    # list of tensors
        self.metadata = metadata  # dict, may contain tensors

def flatten_batch_result(obj):
    # Put both into children; pytree will recurse into them
    children = [obj.outputs, obj.metadata]
    context = None
    return children, context

def unflatten_batch_result(children, context):
    outputs, metadata = children
    return BatchResult(outputs=outputs, metadata=metadata)

register_pytree_node(BatchResult, flatten_batch_result, unflatten_batch_result)
```

### Important Notes

- Registration must happen **before** creating the `RoundPipe` instance — typically at the top of the model definition file.
- The order of elements in `children` must match the unpacking order in `unflatten_fn`.
- `context` must be serializable (no tensors or other non-picklable objects).
- HuggingFace Transformers model output classes (e.g., `CausalLMOutputWithPast`) are already registered by the transformers library — no manual registration needed.
