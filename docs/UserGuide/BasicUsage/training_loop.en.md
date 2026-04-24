# Training Loop

This section covers how to write a complete training loop with RoundPipe, including forward/backward passes, optimizer updates, synchronization, and inference/evaluation.

## forward_backward()

`forward_backward()` is RoundPipe's core training API. It fuses forward and backward passes into a single call. Compared to calling `forward()` and `backward()` separately, fused execution allows forward and backward stages to interleave in the pipeline, eliminating pipeline bubbles and significantly improving performance.

```python
loss = model.forward_backward(
    input_args=(images,),
    input_kwargs={},
    label=labels,
    loss_fn=my_loss_fn,
    return_outputs=False,
    run_config=RoundPipeRunConfig(),
)
```

**Parameters**:

- **`input_args`**: Positional arguments for the model's forward pass, as a tuple. For example, if `forward(x)` takes one input, pass `input_args=(x,)`.

- **`input_kwargs`**: Keyword arguments for the model's forward pass, as a dict. For example, HuggingFace models typically accept `input_ids` and `attention_mask`:

    ```python
    loss = model.forward_backward(
        input_kwargs={"input_ids": ids, "attention_mask": mask},
        label=labels,
        loss_fn=my_loss_fn,
    )
    ```

- **`label`**: Label data. Automatically split into microbatches aligned with the inputs and passed to `loss_fn`.

- **`loss_fn`**: Loss function that takes `(outputs, labels)` and returns a loss tensor. RoundPipe calls `loss_fn` independently for each microbatch and backpropagates, then returns the sum of all microbatch losses.

!!! info "Loss computation"
    Since the returned loss is the sum across microbatches, divide by the microbatch count inside `loss_fn` if you want the mean loss equivalent to computing over the full batch:

    ```python
    loss_fn=lambda outputs, labels: criterion(outputs, labels) / num_microbatch
    ```

    When using GradScaler:

    ```python
    loss_fn=lambda outputs, labels: scaler.scale(
        criterion(outputs.float(), labels)
    ) / num_microbatch
    ```

- **`return_outputs`**: Whether to also return the model outputs. Defaults to `False` (returns loss only). Set to `True` to get a `(loss, outputs)` tuple. Returning outputs adds extra memory overhead and synchronization.

    ```python
    # Loss only
    loss = model.forward_backward(...)

    # Loss and outputs (e.g., for computing accuracy)
    loss, outputs = model.forward_backward(..., return_outputs=True)
    ```

- **`run_config`**: Run configuration for this call, overriding the model-level defaults. See [RoundPipeRunConfig Tuning](../AdvancedUsage/run_config.md).

## model.step()

`model.step()` performs the optimizer update. It takes a callable that runs on the optimizer thread:

```python
def step_fn():
    optimizer.step()
    optimizer.zero_grad()

model.step(step_fn)
```

**What goes in `step_fn`**:

`step_fn` is a zero-argument callable where you perform any optimizer-related operations. Common patterns:

```python
# Basic usage
def step_fn():
    optimizer.step()
    optimizer.zero_grad()

model.step(step_fn)

# With GradScaler
def step_fn():
    scaler.step(optimizer)
    optimizer.zero_grad()

model.step(step_fn)

# With gradient clipping
def step_fn():
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], max_norm=1.0)
    scaler.step(optimizer)
    optimizer.zero_grad()

model.step(step_fn)
```

!!! info "parameter() redirection"
    Inside `step_fn`, calls to `model.parameters()` and `model.named_parameters()` are automatically redirected to the optimizer parameters (high precision), ensuring the optimizer update accesses the correct parameter set.

!!! warning "Data races"
    `step_fn` runs on the optimizer thread in parallel with GPU computation. Only access optimizer parameters and gradients inside `step_fn`; accessing other shared data requires care to avoid data races.

**The `is_async` parameter**:

- `is_async=True` (default): `step()` returns immediately and the optimizer update runs asynchronously in a background thread. Parameters used in the next iteration are one step behind (staleness-1), which does not affect convergence in practice. This is the recommended mode because the optimizer update time is fully hidden behind the next step's GPU computation.
- `is_async=False`: `step()` blocks until the optimizer update finishes. Every iteration uses up-to-date parameters, but performance drops significantly. Typically not recommended.

## model.synchronize()

`synchronize()` waits for all asynchronous operations to complete and syncs the optimizer parameters back to the model parameters. After the call:

- Model parameters reflect the latest optimizer update.
- The `.grad` attribute on parameters contains the accumulated gradients.

**When to call it**:

```python
# 1. Before evaluation — ensure parameters are up to date
model.synchronize()
model.eval()
with torch.no_grad():
    output = model(test_data)

# 2. Before saving a checkpoint
model.synchronize()
torch.save(model.model.state_dict(), "checkpoint.pt")

# 3. When you need to inspect gradients
model.synchronize()
for name, param in model.model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad norm = {param.grad.norm()}")
```

During normal training (`forward_backward` → `step` loop), you do **not** need to call `synchronize()` — RoundPipe handles parameter consistency internally.

## Inference / Evaluation

For evaluation, use `model.eval()` + `torch.no_grad()` + `model.forward()`:

```python
model.synchronize()  # Ensure parameters are up to date
model.eval()

correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(torch.float16)
        outputs = model(images)  # Use forward() — no backward pass needed
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
model.train()  # Switch back to training mode
```

**`forward()` vs `forward_backward()`**:

| | `forward()` | `forward_backward()` |
|---|---|---|
| Purpose | Inference / evaluation | Training |
| Backward pass | Not performed, but supported | Automatic |
| Return value | Model output | Loss (or loss + output) |
| Calling convention | `model(x)` or `model.forward(x)` | `model.forward_backward(...)` |
| Gradients | Optional (use with `torch.no_grad()`) | Automatically accumulated |

`forward()` supports the same calling conventions as a regular PyTorch model:

```python
# These two are equivalent
output = model(x, attention_mask=mask)
output = model.forward(x, attention_mask=mask)

# You can also override the config per-call
output = model(x, roundpipe_run_config=RoundPipeRunConfig(num_microbatch=2))

# The output supports backward as usual
loss = criterion(output, labels)
loss.backward()
```
