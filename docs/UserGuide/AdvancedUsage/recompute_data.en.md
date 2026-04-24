# Saving Data for Recomputation

## When You Need This

By default, RoundPipe uses activation recomputation: it re-executes the forward pass during backward to recover intermediate activations. In most cases, recomputation produces results identical to the original forward pass (with `preserve_rng_state=True` ensuring consistent random behavior).

However, some model layers contain **non-deterministic operations** in their `forward()`: even with the same input, two executions may produce different results, or certain operations trigger GPU-CPU synchronization that hurts performance. Typical scenarios:

- **MoE expert routing**: Routing decisions may depend on `torch.topk` selections that must stay consistent across recomputation.
- **Dynamic-shape operations**: Operations like `torch.nonzero()` return tensors whose shape depends on input values and trigger GPU-CPU sync.
- **Conditional branches**: A `forward()` that chooses different execution paths based on intermediate results.

For these cases, RoundPipe provides the `save_for_recompute` API, allowing you to save critical data during the forward pass and reuse it during recomputation instead of recomputing it.

## API

### save_for_recompute(*data)

Saves data during the forward pass for use during recomputation.

```python
from roundpipe import save_for_recompute

save_for_recompute(routing_indices, expert_mask)
```

- May be called at most once per layer's `forward()`.
- Saved tensors must not require gradients (`requires_grad=False`).
- This is a no-op when gradient computation is disabled.

### doing_recompute()

Checks whether the current execution is a recomputation pass.

```python
from roundpipe import doing_recompute

if doing_recompute():
    # Currently in a recomputation pass
    ...
else:
    # Normal forward pass
    ...
```

### get_recompute_data()

Retrieves data previously saved via `save_for_recompute` during a recomputation pass.

```python
from roundpipe import get_recompute_data

data = get_recompute_data()  # Returns a tuple
```

- Can only be called during recomputation; raises `AssertionError` otherwise.
- Always returns a tuple, even if only one value was saved.

## Examples

### MoE Expert Routing Replay

```python
import torch
import torch.nn as nn
from roundpipe import save_for_recompute, doing_recompute, get_recompute_data

class MoELayer(nn.Module):
    def __init__(self, num_experts, hidden_size, expert_size):
        super().__init__()
        self.gate = nn.Linear(hidden_size, num_experts)
        self.experts = nn.ModuleList([
            nn.Linear(hidden_size, expert_size)
            for _ in range(num_experts)
        ])

    def forward(self, x):
        if doing_recompute():
            # Recomputation: reuse the saved routing result
            selected_experts, = get_recompute_data()
        else:
            # Normal forward: compute routing and save it
            gate_logits = self.gate(x)
            routing_weights = torch.softmax(gate_logits, dim=-1)
            selected_experts = torch.topk(routing_weights, k=2, dim=-1).indices
            save_for_recompute(selected_experts)

        # Use routing result for expert computation
        # ...
        return output
```

### Avoiding GPU-CPU Synchronization

```python
class SparseLayer(nn.Module):
    def forward(self, x):
        if doing_recompute():
            mask, = get_recompute_data()
        else:
            # torch.nonzero() triggers GPU-CPU sync (needs the count of non-zero elements)
            # Save the result to avoid re-syncing during recomputation
            mask = torch.nonzero(x > 0.5)
            save_for_recompute(mask)

        # Use mask for sparse computation
        return x[mask]
```

### Conditional Branches

```python
class ConditionalLayer(nn.Module):
    def forward(self, x):
        if doing_recompute():
            use_branch_a, = get_recompute_data()
        else:
            # Choose branch based on input statistics
            use_branch_a = (x.mean() > 0).item()  # Triggers GPU-CPU sync
            save_for_recompute(use_branch_a)

        if use_branch_a:
            return self.branch_a(x)
        else:
            return self.branch_b(x)
```
