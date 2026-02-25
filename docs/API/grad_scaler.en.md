# Gradient Scaler

In mixed precision training, using low-precision floating point numbers (e.g., `float16`) can cause gradient values to underflow (become zero) due to the limited numerical range. Gradient scaling addresses this by amplifying the loss before backward propagation and dividing the gradients before the optimizer update, thereby maintaining training accuracy while leveraging the performance benefits of low precision.

Because RoundPipe's optimizer updates execute asynchronously (running in a background thread in parallel with GPU computation), PyTorch's native `torch.amp.GradScaler` cannot directly accommodate this execution model. RoundPipe therefore provides its own `GradScaler` implementation that correctly handles scale factor synchronization between the main thread and the optimizer thread.

RoundPipe's `GradScaler` interface is fully compatible with `torch.amp.GradScaler` and can be used as a drop-in replacement.

## roundpipe.GradScaler

```python
class roundpipe.GradScaler(
    init_scale: float = 2.0 ** 16,
    growth_factor: float = 2.0,
    backoff_factor: float = 0.5,
    growth_interval: int = 2000,
    enabled: bool = True,
)
```

Gradient scaler that helps perform the steps of gradient scaling conveniently.

**Parameters:**

- `init_scale`: Initial scale factor. Defaults to `2^16 = 65536`.
- `growth_factor`: Factor by which the scale is multiplied when no inf/NaN gradients occur for `growth_interval` consecutive iterations. Defaults to `2.0`.
- `backoff_factor`: Factor by which the scale is multiplied when inf/NaN gradients occur in an iteration. Defaults to `0.5`.
- `growth_interval`: Number of consecutive iterations without inf/NaN gradients before the scale is multiplied by `growth_factor`. Defaults to `2000`.
- `enabled`: Whether to enable gradient scaling. When set to `False`, `step` directly calls the optimizer's `step()`, and other methods become no-ops.

**Example:**

```python
from roundpipe import RoundPipe, GradScaler
from roundpipe.optim import Adam

scaler = GradScaler()
model = RoundPipe(my_model.to(torch.float16), optim_dtype=torch.float32)
optimizer = Adam(model.optim_parameters(), lr=1e-3)

for data, labels in dataloader:
    loss = model.forward_backward(
        input_args=(data,),
        label=labels,
        loss_fn=lambda out, lbl: scaler.scale(
            criterion(out.float(), lbl)
        ) / num_microbatch,
    )
    model.step(lambda: (scaler.step(optimizer), optimizer.zero_grad()))
    scaler.update()
```

### GradScaler.scale

```python
GradScaler.scale(
    outputs: Union[torch.Tensor, Iterable[torch.Tensor]],
) -> Union[torch.Tensor, Iterable[torch.Tensor]]
```

Multiply a tensor or list of tensors by the scale factor.

Typically called within `loss_fn` to scale the loss value so that gradients produced by backward propagation fall within the representable range of `float16`.

**Parameters:**

- `outputs`: Output tensors or iterable of tensors to scale.

**Returns:**

- The scaled tensors. If scaling is not enabled, the inputs are returned unmodified.

### GradScaler.unscale_

```python
GradScaler.unscale_(optimizer: Optimizer) -> None
```

Divide ("unscale") the optimizer's gradient tensors by the scale factor.

`unscale_` is optional, serving cases where you need to modify or inspect gradients between the backward pass and `step`. If `unscale_` is not called explicitly, gradients will be unscaled automatically during `step`.

If called from the main thread, the unscale operation is executed on the optimizer thread and synchronized.

**Parameters:**

- `optimizer`: Optimizer that owns the gradients to be unscaled.

### GradScaler.step

```python
GradScaler.step(
    optimizer: Optimizer,
    *args: Any,
    **kwargs: Any,
) -> Optional[float]
```

Invoke `unscale_` followed by a parameter update (if gradients do not contain inf/NaN).

This method performs two operations in sequence:

1. If `unscale_` has not already been called for this optimizer, it is called automatically. Gradients are also checked for inf/NaN values.
2. If no inf/NaN gradients are found, `optimizer.step()` is called to update parameters; otherwise the update is skipped to avoid corrupting parameters.

**Parameters:**

- `optimizer`: Optimizer that applies the gradient update.
- `*args`: Positional arguments forwarded to `optimizer.step()`.
- `**kwargs`: Keyword arguments forwarded to `optimizer.step()`.

**Returns:**

- If scaling is disabled, returns the return value of `optimizer.step()`. If scaling is enabled and executing on the optimizer thread, returns that return value; otherwise returns `None`.

### GradScaler.update

```python
GradScaler.update(new_scale: Optional[Union[float, torch.Tensor]] = None) -> None
```

Update the scale factor. This method must be called from the main thread.

If the previous optimizer step was skipped (because gradients contained inf/NaN), the scale is multiplied by `backoff_factor` to reduce it. If `growth_interval` consecutive iterations occurred without skipping, the scale is multiplied by `growth_factor` to increase it.

**Parameters:**

- `new_scale`: Manually set a new scale value. If not `None`, this value is used instead of the automatically computed scale. The value is copied to an internal tensor, so subsequent modifications to the passed tensor will not affect the scaler.

### GradScaler.get_scale

```python
GradScaler.get_scale() -> float
```

Return the current scale factor.

Depending on which thread this is called from (main thread or optimizer thread), the scale factor for the corresponding thread is returned.

**Returns:**

- The current scale factor as a Python `float`. Returns `1.0` if scaling is disabled.

### GradScaler.get_growth_factor

```python
GradScaler.get_growth_factor(up_to_date: bool = False) -> float
```

Get the scale growth factor.

**Parameters:**

- `up_to_date`: If `True`, ensures the latest value is returned (blocks and synchronizes with the optimizer thread). Otherwise, may return a value from before the previous `update()`.

### GradScaler.set_growth_factor

```python
GradScaler.set_growth_factor(new_factor: float) -> None
```

Set a new growth factor.

### GradScaler.get_backoff_factor

```python
GradScaler.get_backoff_factor(up_to_date: bool = False) -> float
```

Get the scale backoff factor.

**Parameters:**

- `up_to_date`: If `True`, ensures the latest value is returned (blocks and synchronizes with the optimizer thread). Otherwise, may return a value from before the previous `update()`.

### GradScaler.set_backoff_factor

```python
GradScaler.set_backoff_factor(new_factor: float) -> None
```

Set a new backoff factor.

### GradScaler.get_growth_interval

```python
GradScaler.get_growth_interval(up_to_date: bool = False) -> int
```

Get the growth interval.

**Parameters:**

- `up_to_date`: If `True`, ensures the latest value is returned (blocks and synchronizes with the optimizer thread). Otherwise, may return a value from before the previous `update()`.

### GradScaler.set_growth_interval

```python
GradScaler.set_growth_interval(new_interval: int) -> None
```

Set a new growth interval.
