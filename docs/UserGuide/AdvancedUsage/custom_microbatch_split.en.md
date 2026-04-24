# Custom Microbatch Splitting

RoundPipe splits input data into microbatches for pipeline execution. By default it infers the splitting strategy automatically, but non-standard input/output formats may require you to specify custom split and merge strategies.

## Default Splitting Behavior

### Automatic Input Splitting

When `split_input` is `None` (default), RoundPipe recursively traverses the nested input structure (tuples, lists, dicts) and applies these rules to each leaf node:

- **Multi-dimensional tensors**: split evenly along dimension 0 (the batch dimension).
- **Scalar tensors and non-tensor types**: replicated to every microbatch.

```python
# Example: automatic splitting
# input args = (images,), images.shape = (12, 3, 224, 224)
# num_microbatch = 4
# → each microbatch gets images.shape = (3, 3, 224, 224)
```

### Automatic Label Splitting

When `split_label` is `None` (default), the same rules as input splitting apply.

### Automatic Output Merging

When `merge_output` is `None` or `True` (default), RoundPipe recursively traverses the output structure and applies these rules to each leaf node:

- **Multi-dimensional tensors**: concatenated along dimension 0 (`torch.cat`).
- **Scalar tensors**: averaged across microbatches.
- **Non-tensor types**: checked for equality across all microbatches; returns the value if equal, raises an error otherwise.

## split_input

When the defaults don't fit your needs, use `split_input` to specify a custom strategy. There are two approaches:

### Approach 1: Split Spec

Use `TensorChunkSpec` and `_Replicate` to annotate how each input should be split:

```python
from torch.distributed.pipelining.microbatch import TensorChunkSpec, _Replicate
from roundpipe import RoundPipeRunConfig

# Scenario: model takes (images, scale_factor)
# Split images along the batch dim; replicate scale_factor to every microbatch
config = RoundPipeRunConfig(
    split_input=(
        (TensorChunkSpec(0), _Replicate),  # args spec
        None,                               # kwargs spec (auto-infer)
    )
)

# Scenario: an input needs splitting along a non-batch dimension
# e.g., (tokens, position_ids) where position_ids splits along dim=1
config = RoundPipeRunConfig(
    split_input=(
        (TensorChunkSpec(0), TensorChunkSpec(1)),  # args spec
        None,
    )
)
```

The spec structure must match the input structure one-to-one. The `args` spec is a tuple and the `kwargs` spec is a dict (or `None` for auto-inference).

### Approach 2: Custom Function

When specs aren't expressive enough, provide a fully custom split function:

```python
def custom_split(args, kwargs, num_microbatch):
    """
    args: original positional argument tuple
    kwargs: original keyword argument dict
    num_microbatch: number of microbatches
    Returns: (args_list, kwargs_list), each of length num_microbatch
    """
    images, masks = args
    chunks_images = images.chunk(num_microbatch)
    chunks_masks = masks.chunk(num_microbatch)
    args_list = [(img, mask) for img, mask in zip(chunks_images, chunks_masks)]
    kwargs_list = [kwargs] * num_microbatch  # replicate kwargs to every microbatch
    return args_list, kwargs_list

config = RoundPipeRunConfig(split_input=custom_split)
```

A more complex example — input is a dict with per-field splitting strategies:

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

Similar to `split_input`, `split_label` controls how labels are split.

### Split Spec

```python
from torch.distributed.pipelining.microbatch import TensorChunkSpec, _Replicate

# Label is a tuple: (targets, sample_weights)
# Both split along the batch dimension
config = RoundPipeRunConfig(
    split_label=(TensorChunkSpec(0), TensorChunkSpec(0))
)

# Label is a tuple: (targets, class_weights)
# targets split along batch dim; class_weights replicated (shared across samples)
config = RoundPipeRunConfig(
    split_label=(TensorChunkSpec(0), _Replicate)
)
```

### Custom Function

```python
def custom_split_label(label, num_microbatch):
    """
    label: original label
    num_microbatch: number of microbatches
    Returns: List[label] of length num_microbatch
    """
    targets, weights = label
    chunks_targets = targets.chunk(num_microbatch)
    chunks_weights = weights.chunk(num_microbatch)
    return [(t, w) for t, w in zip(chunks_targets, chunks_weights)]

config = RoundPipeRunConfig(split_label=custom_split_label)
```

## merge_output

Controls how per-microbatch outputs are combined into the final output.

### Split Spec

Similar to splitting, you can use specs to annotate how each output field should be merged:

```python
from torch.distributed.pipelining.microbatch import TensorChunkSpec, _Replicate, _CustomReducer

# Output is (logits, hidden_states)
# Both concatenated along the batch dimension
config = RoundPipeRunConfig(
    merge_output=(TensorChunkSpec(0), TensorChunkSpec(0))
)

# Custom reducer: sum the losses
sum_reducer = _CustomReducer(torch.tensor(0.0), lambda x, y: x + y)
config = RoundPipeRunConfig(merge_output=sum_reducer)
```

### Custom Function

```python
def custom_merge(outputs):
    """
    outputs: List[output], one per microbatch
    Returns: merged output
    """
    # Example: HuggingFace model output is an object; only logits are needed
    logits = torch.cat([out.logits for out in outputs], dim=0)
    return logits

config = RoundPipeRunConfig(merge_output=custom_merge)
```

### Disabling Merging

Set `merge_output=False` to disable output merging. Each leaf variable in the output will be returned as a `RoundPipePackedData` (a list subclass) containing per-microbatch outputs along with their CUDA transfer events.

```python
config = RoundPipeRunConfig(merge_output=False)
output = model(data, roundpipe_run_config=config)
# Tensors in the output are RoundPipePackedData objects
output.synchronize()  # Wait for all transfers to complete
# Then access per-microbatch outputs like a regular list
```

This is useful when piping one RoundPipe model's output directly into another, avoiding unnecessary synchronization and data copies. `wrap_model_to_roundpipe` automatically sets `merge_output=False` when recursively wrapping non-final modules.
