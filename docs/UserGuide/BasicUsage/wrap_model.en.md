# Wrap Model

The first step to training with RoundPipe is wrapping your model into a RoundPipe instance. There are two approaches: for models with built-in presets (e.g., Qwen3, Llama), you can wrap them in one line; for custom models, you convert them to `nn.Sequential` form and wrap manually.

## Using Model Presets

RoundPipe ships with built-in presets for popular large language models, automatically converting them into the Sequential structure required for pipeline execution. See the [Model Zoo](../../model_zoo.md) for the full list of supported models.

Use `wrap_model_to_roundpipe()` for one-line wrapping:

```python
from transformers import AutoModelForCausalLM
from roundpipe import wrap_model_to_roundpipe, RoundPipeRunConfig

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-1.7B",
    use_cache=False,          # KV cache must be disabled during training
    dtype=torch.float16,
    _attn_implementation="flash_attention_2",
)

model = wrap_model_to_roundpipe(
    model,
    use_sequential_preset=True,  # Force using preset lookup and conversion
    model_run_config=RoundPipeRunConfig(num_microbatch=4),  # Optional: override default run config
    optim_dtype=torch.float32,
    # Additional RoundPipe() constructor arguments can be passed here
)
```

`wrap_model_to_roundpipe()` automatically detects the model type. If a matching preset exists, it converts the model into an equivalent Sequential structure and returns a `RoundPipe` instance. After conversion, the original model's attributes (e.g., `model.vocab_size`, `model.config`) remain accessible.

## Custom Models

RoundPipe can train any deep neural network architecture. To enable correct distributed training with good performance, your model needs to follow a few conventions.

### nn.Sequential Representation

The model passed to RoundPipe must be organized as an `nn.Sequential`. RoundPipe treats each submodule in the Sequential as a model layer for scheduling, so how you partition the layers directly affects training efficiency.

We recommend organizing the model as **input adapter + repeated blocks + output adapter**. Split points should be at the model's "narrow waists" — where the data passed between layers is smallest. For transformer models, the typical split looks like:

```python
import torch.nn as nn

# Example: a simple transformer model
model = nn.Sequential(
    embedding_layer,        # Input adapter: token ids -> hidden states
    transformer_layer_0,    # Repeated blocks
    transformer_layer_1,
    transformer_layer_2,
    # ...
    transformer_layer_n,
    norm_and_lm_head,       # Output adapter: hidden states -> logits
)
```

Each transformer layer passes only the hidden-states tensor to the next, which is relatively small — an ideal split point.

**What to avoid**:

```python
# Not recommended: splitting within attention and MLP internals
model = nn.Sequential(
    layer_0_qkv_proj,
    layer_0_attention,
    layer_0_out_proj,
    layer_0_mlp_up_proj,
    layer_0_mlp_down_proj,
    layer_1_qkv_proj,
    # ...
)
```

While this works, it causes large activation tensors to be transferred between layers, increasing inter-GPU data transfer overhead and hurting training efficiency.

### Forward Function Checklist

#### Variable Access Restrictions

Because RoundPipe executes multiple layers in parallel across GPUs, there are restrictions on variable access inside `forward()`:

| Operation | Global variables | Regular instance variables | Parameters | Standalone module buffers | Shared module buffers |
|-----------|-----------------|---------------------------|------------|--------------------------|----------------------|
| Read | ✅ | ✅ | ✅ | ✅ | ✅ |
| Write | ❌ | ❌ | ❌ | ✅ | ❌ |

What each category means:

- **Global variables**: Variables defined outside the model. Multiple layers may run in different threads simultaneously, so writing to globals causes data races.
- **Regular instance variables**: Accessed via `self.xxx` but not wrapped with `nn.Parameter` or registered via `register_buffer`. Subject to the same concurrency risk.
- **Parameters**: Model parameters (`nn.Parameter`). RoundPipe manages their CPU-GPU transfers; they are read-only during forward execution. Parameter sharing within the same RoundPipe instance is supported, but not across different RoundPipe instances.
- **Standalone module buffers**: Registered via `register_buffer` and belonging to a single layer (not shared across submodules of `nn.Sequential`). RoundPipe transfers them alongside parameters, so they can be written safely.
- **Shared module buffers**: Registered via `register_buffer` but shared across multiple layers. Since they may be accessed concurrently by different layers, writing is not safe.

**Example**:

```python
class MyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(256, 256)
        self.register_buffer('acc', torch.zeros(256))  # Standalone buffer, safe to write
        self.call_count = 0  # Regular instance variable — do NOT write!

    def forward(self, x):
        # ✅ Correct: read parameters, write to standalone buffer
        out = self.linear(x)
        self.acc.add_(out.mean(dim=0).detach())
        return out

        # ❌ Wrong: writing to a regular instance variable
        # self.call_count += 1  # Data race!
```

#### Temporary Tensor Device Placement

Temporary tensors created in `forward()` must not hard-code a device. Infer the device from inputs or weights instead:

```python
class MyBlock(nn.Module):
    def forward(self, x):
        # ❌ Wrong: hard-coded device
        mask = torch.ones(x.shape[0], device='cuda:0')

        # ✅ Correct: infer device from input
        mask = torch.ones(x.shape[0], device=x.device)

        # ✅ Correct: infer device from weights
        bias = torch.zeros(self.linear.weight.shape[0],
                          device=self.linear.weight.device)
        return x
```

RoundPipe schedules different layers on different GPUs. Hard-coding `cuda:0` will create the tensor on the wrong device.

### Wrapping the Model

Once your `nn.Sequential` model is ready, wrap it with `RoundPipe()`:

```python
from roundpipe import RoundPipe, RoundPipeRunConfig

model = RoundPipe(
    model=my_sequential_model.to(torch.float16),
    optim_dtype=torch.float32,
    model_run_config=RoundPipeRunConfig(num_microbatch=4),
    pin_model="alloc",
)
```

**Parameter reference**:

`model_run_config` sets the model-level default run configuration. See [RoundPipeRunConfig Tuning](../AdvancedUsage/run_config.md) for details.

`pin_model` controls the page-locking strategy for model parameters in CPU memory, which affects CPU-to-GPU transfer performance:

| Option | Description | When to use |
|--------|-------------|-------------|
| `"alloc"` | Allocates pinned memory via PyTorch's `pin_memory` | **Default**. Best transfer performance, but CPU memory usage may roughly double (PyTorch aligns allocations to powers of 2) |
| `"register"` | Pins existing memory via `cudaHostRegister` | NVIDIA GPUs only. Useful for LoRA fine-tuning of large models when CPU memory is tight. ~10% slower transfers |
| `"off"` | No pinned memory | For LoRA fine-tuning of very large models (e.g., 235B) that exceed CPU memory, used together with `mmap` loading |

`optim_dtype` specifies the data type for optimizer parameters. A typical setup uses `torch.float16` for model parameters (saving VRAM and transfer bandwidth) and `torch.float32` for optimizer parameters (preserving numerical stability). If omitted, it defaults to the model parameter type.

### Automatic Model Wrapping

!!! warning "Experimental Feature"
    Automatic splitting is experimental. It does not support the fused `forward_backward()` and incurs a performance penalty.

For complex models that you prefer not to convert manually to Sequential form, you can try the automatic splitting mode of `wrap_model_to_roundpipe()`.
We strongly recommend writing the Sequential conversion manually for better performance and full feature support (including `forward_backward()`). If the model is a well-known open-source model not yet in the preset list, consider opening an issue or PR to add a preset.

```python
from roundpipe import wrap_model_to_roundpipe

model = wrap_model_to_roundpipe(
    model,
    use_sequential_preset=False,  # Skip preset lookup; use automatic splitting
    optim_dtype=torch.float32,
)
```

Automatic splitting recursively walks the model's submodule tree and decides how to wrap each module based on parameter-size thresholds. If the model ultimately cannot be split into Sequential form, it returns an `AutoRoundPipe` instance. This instance can still use RoundPipe's forward pass and optimizer features, but cannot use the fused `forward_backward()` and does not benefit from RoundPipe's pipeline scheduling optimizations.
