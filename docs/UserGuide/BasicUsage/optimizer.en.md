# Creating an Optimizer

RoundPipe works with any PyTorch optimizer. Because RoundPipe internally manages parameter transfers between CPU and GPU and handles optimizer-parameter dtype conversion, you need to use RoundPipe's parameter interface when creating the optimizer.

There are two equivalent ways:

```python
from roundpipe import RoundPipe, OptimizerCtx
from roundpipe.optim import Adam

model = RoundPipe(my_model.to(torch.float16), optim_dtype=torch.float32)

# Option 1: Use OptimizerCtx (recommended — feels like standard PyTorch)
with OptimizerCtx():
    optimizer = Adam(model.parameters(), lr=1e-3)

# Option 2: Use optim_parameters() directly (equivalent)
optimizer = Adam(model.optim_parameters(), lr=1e-3)
```

Both are completely equivalent — use whichever feels more natural.

## What OptimizerCtx Does

RoundPipe maintains two sets of parameters internally:

- **Model parameters** (low precision): used for forward and backward computation on the GPU.
- **Optimizer parameters** (high precision): used for optimizer updates on the CPU.

Standard PyTorch optimizers obtain parameters via `model.parameters()`. In RoundPipe, however, `model.parameters()` returns the model parameters (low precision) by default, while the optimizer needs the optimizer parameters (high precision).

`OptimizerCtx` solves this: within its scope, `model.parameters()` and `model.named_parameters()` are automatically redirected to `model.optim_parameters()` and `model.optim_named_parameters()`, returning the high-precision optimizer parameters.

This means you can create optimizers exactly as you would with a regular PyTorch model, without worrying about RoundPipe's internal parameter management:

```python
with OptimizerCtx():
    # model.parameters() returns FP32 optimizer parameters here
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Parameter groups are also supported
    optimizer = torch.optim.AdamW([
        {'params': model.parameters(), 'lr': 1e-4},
    ])
```

## RoundPipe CPU Optimizers

RoundPipe's optimizer updates run on the CPU (since optimizer parameters live in CPU memory). PyTorch's built-in optimizers have poor CPU performance, so RoundPipe provides optimizers that are optimized for CPU instruction sets through compiler vectorization.

See the [API reference](../../API/optimizer.md) for the full list of available optimizers.

Usage is identical to PyTorch's native optimizers:

```python
from roundpipe.optim import Adam

with OptimizerCtx():
    # Same interface as torch.optim.Adam
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=0.01)

    # Supports decoupled_weight_decay (equivalent to AdamW)
    optimizer = Adam(model.parameters(), lr=1e-3,
                     weight_decay=0.01, decoupled_weight_decay=True)
```

We recommend using RoundPipe's optimizers for best performance. You can also use PyTorch's native optimizers — they work correctly, just with slower CPU-side updates. For PyTorch optimizers that support a `fused` parameter, passing `fused=True` will improve performance.
