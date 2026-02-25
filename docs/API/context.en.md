# Runtime Context

RoundPipe uses thread-local context managers to track the current execution phase (forward pass, recomputation, optimizer update). Users can use these APIs to save data during a model's forward pass for use during recomputation, or to correctly access parameters during optimizer-related operations.

## roundpipe.save_for_recompute

```python
roundpipe.save_for_recompute(*data: Any) -> None
```

Save data in the current forward pass context for use during the subsequent recomputation phase.

RoundPipe re-executes the forward pass during backward propagation to recover intermediate activations. If certain intermediate results in the model cause GPU-CPU synchronization (e.g., `torch.nonzero()`), you can save them during the forward pass using this function to avoid synchronization overhead during recomputation.

**Parameters:**

- `*data`: Arbitrary data to save. Saved tensors must not require gradients (`requires_grad=False`).

**Notes:**

- Can be called at most once per layer's forward pass.
- If gradient computation is not currently enabled, this function is a no-op.
- Saved data can be retrieved during the recomputation phase via `get_recompute_data()`.

**Example:**

```python
import torch
import torch.nn as nn
from roundpipe import save_for_recompute, doing_recompute, get_recompute_data

class MyLayer(nn.Module):
    def forward(self, x):
        if doing_recompute():
            # Recomputation phase: use the saved mask directly
            mask, = get_recompute_data()
        else:
            # Forward pass phase: generate and save the mask
            mask = torch.nonzero(x)
            save_for_recompute(mask)
        return x, mask
```

## roundpipe.doing_recompute

```python
roundpipe.doing_recompute() -> bool
```

Check whether the current scope is in the recomputation phase.

RoundPipe re-executes the forward pass during backward propagation to recover activations. This function can be used within forward pass code to distinguish between the initial forward computation and the recomputation phase, allowing different logic to be executed accordingly.

**Returns:**

- `bool`: `True` if currently within a recomputation context; `False` otherwise.

## roundpipe.get_recompute_data

```python
roundpipe.get_recompute_data() -> tuple
```

Retrieve data saved during the forward pass via `save_for_recompute()`.

**Returns:**

- `tuple`: The saved data. Even if only a single item was saved, it is returned as a tuple.

**Notes:**

- This function can only be called within a recomputation context. Calling it outside a recomputation context will raise an `AssertionError`.

## roundpipe.OptimizerCtx

```python
class roundpipe.OptimizerCtx
```

A context manager that marks the current scope as performing optimizer-related operations.

Within this context, `RoundPipe` redirects `.parameters()` and `.named_parameters()` to `.optim_parameters()` and `.optim_named_parameters()`. This allows users to create optimizers using the standard PyTorch pattern without explicitly calling the optimizer-specific parameter interfaces.

**Use Cases:**

Use this context manager when creating an optimizer or performing other operations that need access to optimizer parameters.

**Example:**

```python
from roundpipe import RoundPipe, OptimizerCtx
from roundpipe.optim import Adam

model = RoundPipe(my_model)

# Method 1: Using OptimizerCtx (recommended, consistent with PyTorch conventions)
with OptimizerCtx():
    optimizer = Adam(model.parameters(), lr=0.001)

# Method 2: Using optim_parameters directly (equivalent)
optimizer = Adam(model.optim_parameters(), lr=0.001)
```
