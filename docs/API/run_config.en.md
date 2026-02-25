# Run Config

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

Runtime configuration for RoundPipe models.

Configuration can be specified at two levels:

- **Model level**: Passed via the `model_run_config` parameter when creating a `RoundPipe` instance, serving as the default configuration for that model.
- **Call level**: Passed when calling `forward()` or `forward_backward()`, overriding the model-level defaults.

When both levels specify the same parameter, the call-level configuration takes precedence. If neither level specifies a parameter (both are `None`), the following defaults are used:

**Parameters:**

- `requires_grad`: Whether to enable gradient computation. Defaults to the global `torch.is_grad_enabled()` setting.
- `output_device`: Device where output tensors are placed. Defaults to CPU. Specify a CUDA device if you need to use outputs directly on GPU.
- `preserve_rng_state`: Whether to save and restore the random number generator state. Defaults to `True`. When enabled, ensures that the recomputation phase reproduces the same random behavior as the forward pass (e.g., Dropout).
- `recompute_grain`: Granularity of backward recomputation. Defaults to `"stage"`.
    - `"stage"`: Recompute at the pipeline stage level.
    - `"layer"`: Recompute and perform backward pass per layer within a stage. `"layer"` granularity can reduce peak GPU memory usage, but may increase data transfer overhead.
- `num_microbatch`: Number of microbatches to split the input into. Defaults to `number of CUDA devices + 1`. Increasing the number of microbatches reduces GPU memory usage, but setting it too high may result in each microbatch being too small for efficient GPU utilization; setting it below the default will cause pipeline bubbles.
- `split_input`: Specifies how to split input arguments into microbatches. Defaults to automatic splitting. See [Split Input](#split-input).
- `split_label`: Specifies how to split labels into microbatches. Defaults to automatic splitting. See [Split Label](#split-label).
- `merge_output`: Specifies how to merge microbatch outputs into a single output. Defaults to automatic merging. See [Merge Output](#merge-output).
- `execute_plan`: An optional `ModelExecutePlan` specifying the execution strategy. Defaults to auto-tuned by RoundPipe. See [Execution Plan](#set-execute-plan).

## Split Input {#split-input}

RoundPipe needs to split input data into multiple microbatches for pipelined parallel execution. The `split_input` parameter controls the splitting behavior.

### Automatic Splitting

When `split_input` is `None` (default), RoundPipe automatically infers the splitting strategy. Input arguments can be a single value or a nested Python structure (tuples, lists, dicts). Automatic splitting recursively applies the following rules to each leaf node in the structure:

- **`torch.Tensor`**: If the tensor has more than 0 dimensions, it is evenly split along dimension 0 (batch dimension).
- **Scalar tensors and non-tensor types**: Replicated across all microbatches.
- **Custom types**: Unless the user has registered an unflattening method via `torch.utils._pytree.register_pytree_node`, custom types are treated as unsplittable and replicated across all microbatches, even if they contain tensors.

Automatic splitting works for most standard use cases. If the model's input structure is more complex (e.g., some tensors should not be split along the batch dimension), you need to specify a manual splitting scheme.

### Writing Split Specs

`split_input` can be set to `Tuple[Optional[Tuple], Optional[Dict[str, Any]]]`, specifying the splitting specs for positional and keyword arguments respectively. Either the positional or keyword argument spec can be `None` (indicating automatic splitting) or a spec structure that corresponds to the input structure. Each position in the spec uses one of the following markers:

- `torch.distributed.pipelining.microbatch.TensorChunkSpec(dim)`: Split the tensor along the specified dimension.
- `torch.distributed.pipelining.microbatch._Replicate`: Replicate across all microbatches.

**Example:**

```python
from torch.distributed.pipelining.microbatch import TensorChunkSpec, _Replicate
from roundpipe import RoundPipeRunConfig

# First positional arg split along batch dim, second replicated
config = RoundPipeRunConfig(
    split_input=(
        (TensorChunkSpec(0), _Replicate),  # args spec
        None,                               # kwargs spec (auto-infer)
    )
)
```

### Writing Custom Split Functions

When built-in split specs cannot meet your needs, you can provide a custom function:

```python
def custom_split(
    args: Tuple, kwargs: Dict[str, Any], num_microbatch: int
) -> Tuple[List[Tuple], List[Dict[str, Any]]]:
```

The function receives three parameters:

1. `args`: The original positional arguments tuple.
2. `kwargs`: The original keyword arguments dict.
3. `num_microbatch`: The number of microbatches.

The function should return a tuple `(args_list, kwargs_list)`, where each list has a length equal to the number of microbatches.

**Example:**

```python
def custom_split(args, kwargs, num_microbatch):
    images, masks = args
    chunks_images = images.chunk(num_microbatch)
    chunks_masks = masks.chunk(num_microbatch)
    args_list = [(img, mask) for img, mask in zip(chunks_images, chunks_masks)]
    kwargs_list = [kwargs] * num_microbatch  # replicate kwargs
    return args_list, kwargs_list

config = RoundPipeRunConfig(split_input=custom_split)
```

## Split Label {#split-label}

When using `forward_backward()`, RoundPipe also needs to split labels into multiple microbatches so that each microbatch's label matches its corresponding input. The `split_label` parameter controls the splitting behavior.

### Automatic Splitting

When `split_label` is `None` (default), RoundPipe automatically infers the splitting strategy, following the same behavior as input argument automatic splitting.

### Writing Split Specs

`split_label` can be set to a spec that corresponds to the label structure. The spec writing method is similar to that for input argument split specs.

**Example:**

```python
from torch.distributed.pipelining.microbatch import TensorChunkSpec, _Replicate
from roundpipe import RoundPipeRunConfig

# Label is (scalar tensor, tensor of shape (3, 12, 4), integer)
# The second tensor needs to be split along dimension 1
config = RoundPipeRunConfig(
    split_label=(_Replicate, TensorChunkSpec(1), _Replicate)
)
```

### Writing Custom Split Functions

When built-in split specs cannot meet your needs, you can provide a custom function:

```python
def custom_split_label(
    label: Any, num_microbatch: int
) -> List[Any]:
```

The function receives two parameters:

1. `label`: The original label.
2. `num_microbatch`: The number of microbatches.

The function should return a list of length equal to the number of microbatches, where each element is the label for the corresponding microbatch.

**Example:**

```python
def custom_split_label(label, num_microbatch):
    targets, weights = label
    chunks_targets = targets.chunk(num_microbatch)
    chunks_weights = weights.chunk(num_microbatch)
    return [(t, w) for t, w in zip(chunks_targets, chunks_weights)]

config = RoundPipeRunConfig(split_label=custom_split_label)
```

## Merge Output {#merge-output}

After pipeline execution, RoundPipe needs to merge the outputs from all microbatches into a single complete output. The `merge_output` parameter controls the merging behavior.

### Automatic Merging

When `merge_output` is `None` or `True` (default), RoundPipe automatically infers the merging strategy. Output can be a single value or a nested Python structure (tuples, lists, dicts). Automatic merging recursively applies the following rules to each leaf node:

- **Tensors**: Concatenated along dimension 0 (batch dimension) via `torch.cat`.
- **Scalar tensors**: Averaged across all microbatches.
- **Non-tensor types**: All microbatch values are checked for equality; if equal, that value is returned; otherwise an error is raised.
- **Custom types**: Unless the user has registered an unflattening method via `torch.utils._pytree.register_pytree_node`, custom types are treated as indivisible and handled like non-tensor types, even if they contain tensors.

### Writing Merge Specs

`merge_output` can be set to a spec that corresponds to the output structure, with each position using one of the following markers:

- `torch.distributed.pipelining.microbatch.TensorChunkSpec(dim)`: Concatenate tensors along the specified dimension via `torch.cat`.
- `torch.distributed.pipelining.microbatch._Replicate`: The output should be identical across microbatches.
- `torch.distributed.pipelining.microbatch._CustomReducer`: A custom reducer following the PyTorch-defined reduction interface.

`_CustomReducer` example:

```python
from torch.distributed.pipelining.microbatch import _CustomReducer

sum_reducer = _CustomReducer(torch.tensor(0.0), lambda x, y: x + y)
```

### Writing Custom Merge Functions

You can provide a custom function to fully control the merging logic:

```python
def custom_merge(outputs: List[Any]) -> Any:
```

The function receives a list containing each microbatch's output and returns the merged result.

```python
def custom_merge(outputs):
    # outputs is a list where each element is one microbatch's model output
    logits = torch.cat([out.logits for out in outputs], dim=0)
    return logits

config = RoundPipeRunConfig(merge_output=custom_merge)
```

### No Merging

Setting `merge_output` to `False` disables output merging. Instead, each leaf variable in the output is returned as a `RoundPipePackedData` object. Leaf variables are the nodes obtained after unflattening all built-in Python nested structures (tuples, lists, dicts). Custom types are also treated as leaf variables unless the user has registered an unflattening method via `torch.utils._pytree.register_pytree_node`.

`RoundPipePackedData` is a subclass of Python `list` that stores the outputs from each microbatch along with the corresponding CUDA transfer events. Normally, you do not need to interact with `RoundPipePackedData` directly. If needed, you can call `.synchronize()` to wait for all outputs to be transferred to CPU memory, and then treat it as a regular list to access individual microbatch outputs.

When the output of one `RoundPipe` model needs to be passed directly to another `RoundPipe` model, using `merge_output=False` avoids unnecessary synchronization and data copying, enabling pipelined chaining between models. `wrap_model_to_roundpipe` automatically sets `merge_output=False` when recursively wrapping non-final `nn.ModuleList` modules.

## Execution Plan {#set-execute-plan}

### roundpipe.ModelExecutePlan

```python
class roundpipe.ModelExecutePlan:
    def __init__(self) -> None:
        self.fwd_plan: List[range] = []
        self.bwd_plan: List[range] = []
```

Execution plan for a RoundPipe model, defining the execution order and grouping of layers during forward and backward passes.

The execution plan determines which layers are assigned to the same pipeline stage. Layers in the same stage are loaded onto the GPU and executed together. A well-designed execution plan can balance the computational load across stages and control GPU memory usage.

Unlike traditional pipeline-parallel training frameworks, RoundPipe does not require forward and backward passes to be executed in strict pairs, so forward and backward plans can be designed independently. The number of stages does not need to be a multiple of the number of GPUs. As long as the total number of stages exceeds the number of GPUs and all stages have similar execution times, good parallel efficiency can be achieved.

**Attributes:**

- `fwd_plan`: Forward pass execution order, of type `List[range]`. Each `range` represents the layer indices included in one stage.
- `bwd_plan`: Backward pass execution order, of type `List[range]`.

**Example:**

For a model with 4 layers, here is a possible execution plan when using the `forward` method:

```python
plan = ModelExecutePlan()
plan.fwd_plan = [range(0, 2), range(2, 4)]
plan.bwd_plan = [range(3, 4), range(2, 3), range(1, 2), range(0, 1)]
```

When using the `forward_backward` method, the last stage is not recomputed to save time, so the first stage in the backward plan should not overlap with layers in the forward plan:

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

Automatically generate an execution plan based on the model's computation time and memory size.

When a single model is passed, returns a single `ModelExecutePlan`; when multiple models are passed, returns a list of plans that are co-optimized for overall load balancing.

**Parameters:**

- `run_type`: Run type.
    - `"infer"`: Forward inference only.
    - `"train"`: Separate forward and backward passes (for `forward`-based training).
    - `"fused"`: Fused forward and backward pass (for `forward_backward`-based training).
- `*models`: One or more `RoundPipe` models.
- `min_stages`: Minimum number of pipeline stages. This is a hint value; the actual number of stages may be lower depending on the model size. Defaults to the number of GPU devices.
- `upper_threshold`: Upper ratio for stage load balancing. Limits the maximum allowed ratio between any stage and the slowest layer. Increasing this value provides more flexible stage partitioning but may consume more GPU memory. Defaults to `1.1`.
- `model_memory_limit`: Estimated GPU memory (in GB) available for model parameters and gradients. RoundPipe prefetches one stage's model parameters to GPU memory, so the memory limit per stage is `model_memory_limit / 2`. Defaults to 60% of the smallest GPU's memory.

**Returns:**

- A single `ModelExecutePlan` for a single model, or a `List[ModelExecutePlan]` for multiple models.

RoundPipe measures each layer's computation time online during model execution and uses a sliding average. Based on these timing results, RoundPipe tries to make each stage's computation time as equal as possible for optimal pipeline efficiency.

**Example:**

```python
from roundpipe import RoundPipe, RoundPipeRunConfig, ModelExecutePlan

model = RoundPipe(my_model)

# Auto-generate a training plan
plan = ModelExecutePlan.auto("fused", model)

# Use the generated plan
loss = model.forward_backward(
    input_args=(data,),
    label=labels,
    loss_fn=loss_fn,
    run_config=RoundPipeRunConfig(execute_plan=plan),
)

# Multi-model co-optimized planning
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
