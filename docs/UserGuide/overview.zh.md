# 概览

RoundPipe 是一个高性能的深度神经网络训练框架，专为在消费级 GPU 上训练大规模模型而设计。它基于纯 PyTorch 实现，原生支持 NVIDIA、AMD、华为昇腾等 PyTorch 兼容的 GPU 平台。

## RoundPipe 解决什么问题

- **GPU 显存不足以训练大模型**：消费级 GPU（如 RTX 4090）仅有 24GB 显存，而一个 8B 参数模型在混合精度训练下需要约 128GB 存储模型状态（参数、梯度、优化器状态），远超单卡容量。RoundPipe 将模型状态存储在 CPU 内存中，按需将计算任务分发到 GPU 执行，突破显存限制。
- **没有 NVLink 的 GPU 间通信瓶颈**：消费级服务器使用 PCIe 互联，带宽不到 NVLink 的 20%。数据并行方案需要在每次迭代中通过集合通信同步全部参数，导致大量时间花在通信上。RoundPipe 采用[分发式流水线并行](#dispatch-pp)，每个阶段只需将参数从 CPU 上传到单个 GPU，大幅减少GPU间通信开销。
- **但 RoundPipe 不适用于小模型（< 0.5B）的分布式训练**：对于单卡显存足以开展训练的小模型，数据并行方案更简单高效。RoundPipe 的优势在于模型训练需要超出**单一 GPU 显存**时的场景。

## 分发式流水线并行 {#dispatch-pp}

### 什么是流水线并行

流水线并行（Pipeline Parallelism）是一种模型并行策略。它将模型按层划分为多个**阶段**（stage），训练数据被拆分为多个**微批次**（microbatch），各阶段以流水线方式依次处理这些微批次。

以一个简单的例子说明：假设模型有 4 层，被划分为 2 个 stage（每个 stage 2 层），使用 GPipe 调度处理 2 个微批次：

```
时间 →

GPU 0 (stage 0): [mb0 F][mb1 F]              [mb1 B][mb0 B]
GPU 1 (stage 1):        [mb0 F][mb1 F][mb1 B][mb0 B]

F = 前向传播, B = 反向传播
```

GPU 0 完成 mb0 的 stage 0 前向后，GPU 1 开始处理 mb0 的 stage 1 前向，同时 GPU 0 可以处理 mb1 的 stage 0 前向，实现两个 GPU 并行工作。但在所有前向完成、反向开始之间，存在 GPU 空闲等待的**气泡**。

### RoundPipe 的分发式流水线并行

传统流水线并行中，每个 stage 绑定到固定的 GPU（stage 0 始终在 GPU 0 上执行，stage 1 始终在 GPU 1 上执行），这给阶段划分带来了限制。

RoundPipe 打破了 stage 与 GPU 的绑定。在 RoundPipe 中，模型参数存储在 CPU 内存中，每次执行一个 stage 时，将该 stage 的参数从 CPU 上传到 GPU，计算完成后释放。因此，任何 stage 都可以在任何 GPU 上执行。

RoundPipe 将 GPU 视为**无状态的计算资源池**，以轮询（round-robin）方式将 stage 动态分发到各 GPU：

```
时间 →
GPU 0: [F stage0] [F stage3] [B stage2] ...
GPU 1:   [F stage1] [B stage4] [B stage1] ...
GPU 2:     [F stage2] [B stage3] [B stage0] ...
```

这带来了几个关键优势：

1. **更灵活的阶段划分**：stage 数量可以是任意整数，不受 GPU 数量约束。而且前向和反向可以采用不同的划分方案（非对称划分），进一步优化负载均衡。
2. **更少流水线气泡**：RoundPipe 的轮询分发使得前向和反向 stage 可以互相紧密衔接，GPU 之间不需要等待同步。
3. **更低的显存占用**：任意时刻 GPU 上只有当前正在执行的一个 stage 的数据。

#### RoundPipe 中 stage 的概念和与 layer 的关系

- **Layer（层）**：模型的基本组成单元，对应传入 `nn.Sequential` 的每个子模块。例如一个 transformer 模型的每个 transformer block 是一个 layer。
- **Stage（阶段）**：一个或多个连续 layer 的组合，是流水线调度的基本单位。同一 stage 内的所有 layer 会被一起上传到 GPU 并连续执行。

RoundPipe 的前向和反向使用**非对称划分**：由于一个 transformer layer 的前向计算时间约为反向（含重计算）的 1/3，前向 stage 包含更多层，反向 stage 包含更少层，使得每个 stage 的执行时间尽可能相等。例如，一个 12 层模型可能被划分为：

- 前向：`[layer 0-2], [layer 3-5], [layer 6-8], [layer 9-11]`（每个 stage 3 层）
- 反向：`[layer 11], [layer 10], [layer 9], ..., [layer 0]`（每个 stage 1 层）

## 核心概念

### 串行编程，自动并行

RoundPipe 采用**单控制器架构**，用户只需编写串行的、单设备风格的训练脚本，RoundPipe 会自动将计算映射到所有可用的 GPU 上并行执行。

用户与 RoundPipe 交互的核心 API 只有两个：

- `model.forward_backward()`：执行前向和反向传播
- `model.step()`：执行优化器更新

一个典型的训练循环如下：

```python
for data, labels in dataloader:
    loss = model.forward_backward(
        input_args=(data,),
        label=labels,
        loss_fn=my_loss_fn,
    )
    model.step(lambda: (optimizer.step(), optimizer.zero_grad()))
```

这段代码看起来和普通的单卡训练几乎一样，但 RoundPipe 在内部自动完成了微批次拆分、多 GPU 流水线调度、参数传输、梯度累积、异步优化器更新等所有并行化工作。

### 以 CPU 为中心的参数存储

RoundPipe 采用**计算分发范式**：

- CPU 内存是模型状态和中间激活的**主存储位置**
- GPU 是**无状态的计算加速器**，只在执行某个 stage 时临时持有该 stage 的相关数据（显存即缓存）
- 计算完成后，梯度下载回 CPU，GPU 上的参数副本立即释放

这种设计使得可训练的模型规模大幅提升，显存限制仅与单一模型层有关，而非整个模型。用户使用时仅需要将模型参数和输入数据放在 CPU 上，由 RoundPipe 自动处理数据传输和计算调度，无需用户干预。

### 输入分割与 microbatch 执行

RoundPipe 将每个训练批次自动拆分为多个**微批次**（microbatch），然后以流水线方式处理。每个微批次独立完成前向传播、loss 计算和反向传播，梯度在所有微批次上累积：

```
输入批次 (batch_size=12, num_microbatch=3)
    ↓ 自动拆分
microbatch 0: [样本 0-3]   ──forward──→ output_0 ──loss_fn──→ loss_0 ──backward──→ 梯度累积
microbatch 1: [样本 4-7]   ──forward──→ output_1 ──loss_fn──→ loss_1 ──backward──→ 梯度累积
microbatch 2: [样本 8-11]  ──forward──→ output_2 ──loss_fn──→ loss_2 ──backward──→ 梯度累积
    ↓
返回 loss = loss_0 + loss_1 + loss_2
```

注意：每个微批次的 loss 是**独立计算、独立反向传播**的，最终返回的 loss 是所有微批次 loss 的总和，梯度也是各微批次梯度的累加。这等价于将整个 batch 的 loss 求和后反向传播。

如果希望得到与整个 batch 上计算等价的**平均** loss，需要在 `loss_fn` 中手动除以微批次数量：

```python
loss_fn=lambda outputs, labels: criterion(outputs, labels) / num_microbatch
```

### 激活重计算

反向传播需要用到前向传播的中间激活值（即每层的中间计算结果）。对于大模型和长序列，存储所有激活值会消耗大量显存。**激活重计算**（Activation Recomputation）是一种用计算换显存的技术：前向传播时不存储中间激活，只保存每层的输入；反向传播时重新执行一遍前向计算来恢复中间结果。虽然这带来了额外的计算开销，但显著降低了显存占用，使得可以以更大的 batch size 训练更大的模型，进而提高 GPU 利用率和整体训练效率。

RoundPipe 集成了完全激活重计算，即反向传播时每层都重新执行前向计算，无需用户额外修改代码。
