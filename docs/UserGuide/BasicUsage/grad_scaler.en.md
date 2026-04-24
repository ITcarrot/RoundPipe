# GradScaler

## Why GradScaler Is Needed

In mixed-precision training, the model uses FP16 for forward and backward computation to improve performance. However, FP16 has a limited numerical range — small gradient values can underflow to zero, causing training instability or failure to converge.

GradScaler addresses this through **gradient scaling**:

1. After computing the loss, it multiplies the loss by a large scale factor (e.g., 65536).
2. Backpropagation produces proportionally larger gradients, preventing underflow.
3. Before the optimizer update, gradients are divided by the scale factor to restore their true values.
4. If inf/NaN values are detected in the gradients (indicating the scale factor is too large), the update is skipped and the scale factor is reduced.

## Basic Usage

RoundPipe provides its own `GradScaler` implementation. The interface is fully compatible with `torch.amp.GradScaler`, but it additionally supports RoundPipe's asynchronous optimizer execution model.

A complete mixed-precision training loop:

```python
from roundpipe import RoundPipe, OptimizerCtx, GradScaler
from roundpipe.optim import Adam

model = RoundPipe(my_model.to(torch.float16), optim_dtype=torch.float32)
with OptimizerCtx():
    optimizer = Adam(model.parameters(), lr=1e-3)
scaler = GradScaler()

for data, labels in dataloader:
    # 1. forward_backward: call scaler.scale() inside loss_fn
    loss = model.forward_backward(
        input_args=(data.to(torch.float16),),
        label=labels,
        loss_fn=lambda outputs, labels: scaler.scale(
            criterion(outputs.float(), labels)
        ) / num_microbatch,
    )

    # 2. step: call scaler.step() inside step_fn
    model.step(lambda: (
        scaler.step(optimizer),
        optimizer.zero_grad(),
    ))

    # 3. update: adjust the scale factor
    scaler.update()

    # Get the true loss value (divide by scale factor)
    real_loss = loss.item() / scaler.get_scale()
```

What each step does:

- **`scaler.scale(loss)`**: Multiplies the loss by the scale factor. Placed inside `loss_fn` so that every microbatch's backward pass uses the scaled loss.
- **`scaler.step(optimizer)`**: Automatically runs `unscale_` (divides gradients by the scale factor) + checks for inf/NaN + calls `optimizer.step()`. If inf/NaN is found, the update is skipped.
- **`scaler.update()`**: Adjusts the scale factor based on whether the update was skipped. Must be called on the main thread.

## Differences from PyTorch's GradScaler

The key difference between RoundPipe's `GradScaler` and PyTorch's native `torch.amp.GradScaler` is **thread safety**.

RoundPipe's optimizer updates run asynchronously in a background thread by default (via `model.step(is_async=True)`). `scaler.scale()` is called on the GPU computation thread, while `scaler.step()` and `scaler.update()` run on the optimizer thread. PyTorch's native `GradScaler` does not support this cross-thread usage pattern, so you must use RoundPipe's version.

!!! warning "Do not use PyTorch's native GradScaler"
    ```python
    # ❌ Wrong: PyTorch's GradScaler does not support async optimizers
    scaler = torch.amp.GradScaler()

    # ✅ Correct: use RoundPipe's GradScaler
    from roundpipe import GradScaler
    scaler = GradScaler()
    ```

### Manual Unscaling

If you need to perform additional operations on gradients before the optimizer update (e.g., gradient clipping), call `unscale_` manually:

```python
def step_fn():
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], max_norm=1.0)
    scaler.step(optimizer)
    optimizer.zero_grad()

model.step(step_fn)
```

If you do not call `unscale_` manually, `scaler.step()` performs it automatically — no extra action needed.
