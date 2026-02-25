# Model APIs

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

Wraps an `nn.Module` with RoundPipe's pipelined execution runtime. This is the core class of RoundPipe, responsible for splitting a model into layers, managing parameter transfers between CPU and GPU, and coordinating pipelined forward and backward passes.

**Parameters:**

- `model`: The model to wrap. Can be `nn.Sequential` or an arbitrary model. An `nn.Sequential` model is split directly into individual layers; a non-Sequential model is treated as a single layer.
- `optim_dtype`: Data type for optimizer parameters. Defaults to the same type as the model parameters. A common setup is to keep model parameters in `torch.float16` and use `torch.float32` for optimizer parameters to ensure numerical stability.
- `name`: Optional identifier for display in logs and traces. Defaults to an auto-generated string based on the call site (format: `filename:line`).
- `model_run_config`: Model-level default run configuration, used as the baseline for each call to `forward()` or `forward_backward()`. See [Run Config](run_config.md) for details.
- `pin_model`: Memory pinning strategy for model parameters:
    - `"alloc"`: Use PyTorch's `pin_memory` for pinned memory. This is the default and usually provides the best host-to-device transfer performance, but may cause up to 2x CPU memory usage because PyTorch pads all allocations to a power of 2.
    - `"register"`: Use `cudaHostRegister` for pinned memory. Reduces CPU memory usage for very large models, but host-to-device transfer performance degrades by approximately 10%. Only available with CUDA.
    - `"off"`: Do not pin memory. Suitable for LoRA fine-tuning models that exceed CPU memory. Combined with `mmap` during model loading, Linux can load data on demand from disk and automatically evict used data when out of memory.

### RoundPipe.forward

```python
RoundPipe.forward(
    *args: Any,
    roundpipe_run_config: RoundPipeRunConfig = RoundPipeRunConfig(),
    **kwargs: Any,
) -> Any
```

Execute a forward pass. The input data is automatically split into microbatches, executed in parallel through the pipeline, and the outputs from all microbatches are merged before being returned.

**Parameters:**

- `*args`: Positional arguments forwarded to the underlying model.
- `roundpipe_run_config`: Per-call run configuration that overrides the model-level defaults.
- `**kwargs`: Keyword arguments forwarded to the underlying model.

**Returns:**

- The merged output. The specific merging behavior is determined by the `merge_output` configuration.

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

Execute a fused forward and backward pass. Compared to calling `forward()` followed by a manual `backward()`, this method uses an optimized scheduling strategy that allows forward and backward passes to execute concurrently in the pipeline, eliminating pipeline bubbles.

**Parameters:**

- `input_args`: Positional arguments for the forward pass.
- `input_kwargs`: Keyword arguments for the forward pass.
- `label`: Label data, provided in the format expected by `loss_fn`.
- `loss_fn`: Loss function. Takes `(outputs, labels)` as arguments and returns a loss tensor or a sequence of loss tensors.
- `return_outputs`: Whether to also return model outputs.
- `run_config`: Per-call run configuration.

**Returns:**

- If `return_outputs=False` (default), returns the sum of losses across all microbatches.
- If `return_outputs=True`, returns a `(loss_sum, merged_outputs)` tuple.

Calling this function will calculate loss and perform backward for each microbatch seperately. This is equivalent to using the sum of losses from each microbatch as the loss of the input training sample. Specifically, the semantics of this function is:

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
        # When loss_mb is a tensor, the above line is the same as
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

Return an iterator over parameters suitable for optimizer consumption.

RoundPipe internally manages the storage location and data type of model parameters. Parameters obtained through this method are stored in the format required by the optimizer (as specified by `optim_dtype`), and can be passed directly to an optimizer.

**Returns:**

- A parameter iterator, where each parameter is an optimizer-specific copy.

### RoundPipe.optim_named_parameters

```python
RoundPipe.optim_named_parameters(
    prefix: str = "",
    remove_duplicate: bool = True,
) -> Iterator[Tuple[str, torch.nn.Parameter]]
```

Return an iterator over named optimizer parameters.

**Parameters:**

- `prefix`: Prefix to prepend to parameter names.
- `remove_duplicate`: Whether to skip duplicate parameters.

**Returns:**

- An iterator yielding `(name, parameter)` tuples.

### RoundPipe.step

```python
RoundPipe.step(
    step_fn: Callable[..., None],
    is_async: bool = True,
    *args: Any,
    **kwargs: Any,
) -> None
```

Execute an optimizer update using the provided step function.

In the default asynchronous mode, `step` returns immediately and the optimizer update runs in a background thread. Training iterations will use parameters that are one step behind, which typically does not affect convergence in practice. Synchronous mode waits for the optimizer update to complete before returning, ensuring each iteration uses the latest parameters, but significantly reduces performance and is generally not recommended.

!!! warning "Data Races"
    Data access in `step_fn` should be limited to optimizer parameters only. Accessing other data requires awareness of potential data race issues.

**Parameters:**

- `step_fn`: A callable that performs one optimization step.
- `is_async`: Whether to execute asynchronously. Defaults to `True`.
- `*args`: Positional arguments forwarded to `step_fn`.
- `**kwargs`: Keyword arguments forwarded to `step_fn`.

### RoundPipe.synchronize

```python
RoundPipe.synchronize() -> None
```

Synchronize optimizer parameters and backward gradients back to model parameters.

After calling this method, model parameters will reflect the latest optimizer update results, and the `.grad` attribute of parameters will contain the accumulated gradients. This is useful when you need to inspect parameter values or gradients (e.g., before evaluation or saving a checkpoint).

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

Automatically wrap a model into a RoundPipe instance using recursive heuristics or built-in presets.

This function attempts the following strategies:

1. If `use_sequential_preset` is not `False`, it first tries to use a built-in model preset to convert the model into an equivalent Sequential structure.
2. If no preset is available, it recursively traverses the model's submodules, deciding how to wrap each based on size thresholds and module types.
3. For models that ultimately cannot be split into a Sequential structure, it returns an `AutoRoundPipe` instance.

**Parameters:**

- `model`: The root module to wrap.
- `use_sequential_preset`: Whether to use a built-in Sequential preset. `None` or `True` attempts to use a preset; `False` skips preset lookup. When set to `None`, a message is printed if a matching preset is found.
- `lower_threshold`: Minimum module size (in bytes) for direct wrapping as a `RoundPipe`. Modules smaller than this threshold are wrapped with `num_microbatch=1`.
- `upper_threshold`: Maximum size threshold for splitting submodules. Defaults to the total model size divided by `(number of GPUs + 1)`.
- `skip_modules`: A list of modules that should not be wrapped.
- `override_config`: Run configuration overrides for specific modules.
- `model_run_config`: Default run configuration for `RoundPipe` instances.
- `name`: Module name. If `None`, auto-generated based on the call site.
- `**roundpipe_kwargs`: Additional keyword arguments forwarded to the `RoundPipe` constructor.

**Wrapping Strategy Details:**

When no matching preset is found, the function recursively traverses the model's submodule tree. For each module encountered, it decides the action based on the following priority:

1. **Skip**: If the module is in the `skip_modules` list, the original module is returned without any wrapping. **This module will execute on CPU.**

2. **Wrap directly as a multi-layer `RoundPipe`**: If the module is an `nn.Sequential`, it is wrapped directly as a `RoundPipe`, with each child module as an independent pipeline layer.

3. **Wrap directly as a single-layer `RoundPipe`**: If any of the following conditions are met:
    - The module's **own parameters** (non-recursive) are >= `lower_threshold` (default 16 KB)
    - The module's **total size** (recursive) is <= `upper_threshold`

    and the module implements its own `forward` method, the entire module is wrapped as a single-layer `RoundPipe`.

    Special case: If the module's total size is < `lower_threshold`, it is wrapped with `num_microbatch` set to 1, since splitting extremely small modules into microbatches provides no performance benefit.

4. **Wrap `nn.ModuleList` element-wise**: If the module is an `nn.ModuleList`, each element is wrapped as a separate `RoundPipe` instance. By default, ModuleList modules are assumed to be called sequentially, so `merge_output` is set to `False` for non-final elements to enable pipelined parallel computation. If you need to access intermediate layer outputs, you can set `merge_output=True` for the corresponding module in `override_config`, but this introduces additional synchronization and prevents the framework from parallelizing computation.

5. **Recurse into submodules**: If none of the above conditions are met (i.e., the module is too large, not a Sequential/ModuleList, and does not have enough of its own parameters), the function recurses into `named_children()`, repeating the above checks for each child. **The module's own `forward` will execute on CPU.**

6. **Other special handling**: For HuggingFace `PreTrainedModel` instances, the `loss_function` is additionally wrapped.

**Returns:**

- If the model is successfully converted using a preset or recursive wrapping, returns a `RoundPipe` instance; otherwise returns an `AutoRoundPipe` instance. `AutoRoundPipe` is a placeholder class indicating the model has been wrapped but could not be split into a Sequential structure. Users can still use RoundPipe's forward pass and optimizer features, but cannot benefit from layer-level splitting performance gains and cannot call `forward_backward()`.

**Raises:**

- `NotImplementedError`: If `use_sequential_preset=True` but no preset exists for the model type.

### RoundPipe.set_original_model

```python
RoundPipe.set_original_model(original_model: nn.Module) -> None
```

Set a reference to the original model for proxied attribute access.

When a model is converted into a Sequential structure by `wrap_model_to_roundpipe`, attributes on the original model (such as HuggingFace model's `vocab_size`, `config`, etc.) can still be accessed through this mechanism. `wrap_model_to_roundpipe` calls this method automatically, so manual invocation is typically unnecessary.

**Parameters:**

- `original_model`: The original unwrapped model.

### Attribute Forwarding

RoundPipe overrides `__getattr__`, `__setattr__`, and `__delattr__` to provide transparent attribute access to the wrapped model:

- **Read**: When accessing an attribute that does not exist on the RoundPipe instance, it is looked up on `original_model` if set, else on the model used to create the RoundPipe instance.
- **Write**: After initialization, attribute writes on the RoundPipe instance are forwarded to  `original_model` or `model`.
- **Delete**: Attribute deletions are similarly propagated to `original_model` or `model`.

This mechanism allows a RoundPipe-wrapped model to be used just like the original model, without needing to be aware of the wrapping layer.
