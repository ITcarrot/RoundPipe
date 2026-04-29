# 包装模型

使用 RoundPipe 训练模型的第一步是将模型包装为 RoundPipe 实例。RoundPipe 提供了两种方式：对于已有预设的模型（如 Qwen3、Llama）可以一键包装；对于自定义模型，需要将其转换为 `nn.Sequential` 形式后手动包装。

## 使用预设模型

RoundPipe 为常见的大语言模型提供了内置预设，可以自动将模型转换为适合流水线执行的 Sequential 结构。支持的模型列表见[预设模型清单](../../ModelZoo/model_presets.md)。

使用 `wrap_model_to_roundpipe()` 即可一键包装：

```python
from transformers import AutoModelForCausalLM
from roundpipe import wrap_model_to_roundpipe, RoundPipeRunConfig

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-1.7B",
    use_cache=False,          # 训练时必须关闭 KV cache
    dtype=torch.float16,
    _attn_implementation="flash_attention_2",
)

model = wrap_model_to_roundpipe(
    model,
    use_sequential_preset=True,  # 强制使用预设查找和转换
    model_run_config=RoundPipeRunConfig(num_microbatch=4),  # 可选，覆盖默认运行配置
    optim_dtype=torch.float32,
    # 可传入更多 RoundPipe() 构造函数参数
)
```

`wrap_model_to_roundpipe()` 会自动检测模型类型，如果存在匹配的预设，会将模型转换为等价的 Sequential 结构并返回一个 `RoundPipe` 实例。转换后，原始模型的属性（如 `model.vocab_size`、`model.config`）仍然可以正常访问。

## 自定义模型

RoundPipe 支持训练任意结构的深度神经网络模型，但为了能够让 RoundPipe 正确地进行分布式训练并保证训练性能，需要遵循一定的适配规范。

### nn.Sequential 表示

传入 RoundPipe 的模型需要以 `nn.Sequential` 的形式组织。RoundPipe 会将 `nn.Sequential` 中的每个子模块作为一个模型层（layer）进行调度，因此对模型层的划分与训练效率息息相关。

我们建议以**输入适配器 + 若干重复模型结构 + 输出适配器**来组织输入模型。切分点应选在模型的"窄处"，即层间传递的数据量最小的位置。对于 transformer 模型，典型的切分方式是：

```python
import torch.nn as nn

# 以一个简单的 transformer 模型为例
model = nn.Sequential(
    embedding_layer,        # 输入适配器：将 token id 转为 hidden states
    transformer_layer_0,    # 重复结构
    transformer_layer_1,
    transformer_layer_2,
    # ...
    transformer_layer_n,
    norm_and_lm_head,       # 输出适配器：将 hidden states 转为 logits
)
```

每个 transformer layer 之间传递的是 hidden states tensor，数据量相对较小，是理想的切分点。

**不推荐的切分方式**：

```python
# 不推荐：将 attention 和 MLP 内的各部分也拆开
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

虽然这种切分方式也能工作，但会导致每层之间传递大量的激活数据，增加 GPU 之间的数据传输开销，降低训练效率。

### forward 函数自查

#### 变量访问限制

由于 RoundPipe 会并行在多个 GPU 上执行多个模型层，因此对模型 forward 函数中的变量访问有限制：

| 操作 | 全局变量 | 普通类变量 | 参数 | 独立模块 buffer | 共享模块 buffer |
|------|---------|-----------|------|---------------|---------------|
| 读取 | ✅ | ✅ | ✅ | ✅ | ✅ |
| 写入 | ❌ | ❌ | ❌ | ✅ | ❌ |

各类变量的含义：

- **全局变量**：在模型定义外部定义的变量。由于多个层可能在不同线程中并行执行，写入全局变量会导致数据竞争。
- **普通类变量**：通过 `self.xxx` 访问但未使用 `nn.Parameter` 包装或 `register_buffer` 注册的变量。同样存在并发写入风险。
- **参数**：模型参数（`nn.Parameter`）。RoundPipe 管理参数在 CPU 和 GPU 之间的传输，forward 执行时参数是只读的。RoundPipe 支持在同一 RoundPipe 实例中共享参数，但不支持在不同 RoundPipe 实例之间共享参数。
- **独立模块 buffer**：通过 `register_buffer` 注册且仅属于一个 layer 的 buffer（不在 `nn.Sequential` 的不同子模块中共享）。RoundPipe 会随参数一起传输，可以安全写入。
- **共享模块 buffer**：通过 `register_buffer` 注册且在多个 layer 中共享的 buffer。由于可能被并行执行的不同层同时访问，不能写入。

**示例**：

```python
class MyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(256, 256)
        self.register_buffer('acc', torch.zeros(256))  # 独立 buffer，可写入
        self.call_count = 0  # 普通类变量，不可写入！

    def forward(self, x):
        # ✅ 正确：读取参数，写入独立 buffer
        out = self.linear(x)
        self.acc.add_(out.mean(dim=0).detach())
        return out

        # ❌ 错误：写入普通类变量
        # self.call_count += 1  # 数据竞争！
```

#### 临时张量位置

forward 函数中创建的临时张量不能写死设备位置，应该根据输入或权重的 device 推导：

```python
class MyBlock(nn.Module):
    def forward(self, x):
        # ❌ 错误：写死 device
        mask = torch.ones(x.shape[0], device='cuda:0')

        # ✅ 正确：从输入推导 device
        mask = torch.ones(x.shape[0], device=x.device)

        # ✅ 正确：从权重推导 device
        bias = torch.zeros(self.linear.weight.shape[0],
                          device=self.linear.weight.device)
        return x
```

RoundPipe 会将不同的层调度到不同的 GPU 上执行，写死 `cuda:0` 会导致张量被创建在错误的设备上。

### 包装模型

准备好 `nn.Sequential` 模型后，使用 `RoundPipe()` 进行包装：

```python
from roundpipe import RoundPipe, RoundPipeRunConfig

model = RoundPipe(
    model=my_sequential_model.to(torch.float16),
    optim_dtype=torch.float32,
    model_run_config=RoundPipeRunConfig(num_microbatch=4),
    pin_model="alloc",
)
```

**参数说明**：

`model_run_config` 用于设置模型级别的默认运行配置，详见 [RoundPipeRunConfig 调参](../AdvancedUsage/run_config.md)。

`pin_model` 控制 CPU 内存中模型参数的锁页策略，影响 CPU→GPU 的传输性能：

| 选项 | 说明 | 适用场景 |
|------|------|---------|
| `"alloc"` | 使用 PyTorch 的 `pin_memory` 分配锁页内存 | **默认选项**，传输性能最佳，但 CPU 内存占用可能翻倍（PyTorch 会将分配对齐到 2 的幂次） |
| `"register"` | 使用 `cudaHostRegister` 注册锁页内存 | 仅限 Nvidia GPU，LoRA 微调大模型场景，CPU 内存较紧张时使用，传输性能下降约 10% |
| `"off"` | 不使用锁页内存 | LoRA 微调超大模型（如 235B），模型超过 CPU 内存时配合 `mmap` 加载使用 |

`optim_dtype` 指定优化器参数的数据类型。典型配置是模型参数使用 `torch.float16`（节省显存和传输带宽），优化器参数使用 `torch.float32`（保证数值稳定性）。如果不指定，默认与模型参数类型相同。

### 自动包装模型

!!! warning "实验性功能"
    自动拆分是实验性功能，不支持融合前向和反向传播的 `forward_backward()`，且性能有损失。

对于不想手动转换为 Sequential 结构的复杂模型，可以尝试 `wrap_model_to_roundpipe()` 的自动拆分功能。
但我们强烈建议用户手动编写 Sequential 转换代码，以获得更好的性能和更丰富的功能支持（如 `forward_backward()`）。如果该模型是知名开源模型且不在预设列表中，欢迎提交 issue 或 PR 添加预设支持。

```python
from roundpipe import wrap_model_to_roundpipe

model = wrap_model_to_roundpipe(
    model,
    use_sequential_preset=False,  # 跳过预设查找，使用自动拆分
    optim_dtype=torch.float32,
)
```

自动拆分会递归遍历模型的子模块树，根据参数大小阈值决定如何包装每个模块。如果模型最终无法被拆分为 Sequential 结构，会返回一个 `AutoRoundPipe` 实例。它仍然可以使用 RoundPipe 的前向传播和优化器功能，但无法使用 `forward_backward()` 融合执行，也无法获得RoundPipe流水线调度策略带来的性能提升。
