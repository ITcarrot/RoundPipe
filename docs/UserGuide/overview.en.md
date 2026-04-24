# Overview

RoundPipe is a high-performance deep neural network training framework designed for training large-scale models on consumer-grade GPUs. Built entirely on PyTorch, it natively supports any GPU platform compatible with PyTorch, including NVIDIA, AMD, and Huawei Ascend.

## RoundPipe For ...

- **GPU memory is too small for large models**: Consumer GPUs like the RTX 4090 have only 24 GB of VRAM, yet an 8B-parameter model requires roughly 128 GB to store its full training state (parameters, gradients, and optimizer states) in mixed precision — far exceeding a single card's capacity. RoundPipe keeps model state in CPU memory and dispatches computation to GPUs on demand, breaking through the VRAM barrier.
- **Communication bottleneck without NVLink**: Consumer-grade servers use PCIe interconnects, which offer less than 20% of NVLink bandwidth. Data-parallel approaches must synchronize all parameters via collective communication every iteration, spending most of their time on transfers. RoundPipe uses [Dispatch Pipeline Parallelism](#dispatch-pp), where each stage only needs to upload its parameters from CPU to a single GPU, dramatically reducing inter-GPU communication overhead.
- **RoundPipe is not intended for small models (< 0.5B)**: When a model fits comfortably in a single GPU's memory, data parallelism is simpler and more efficient. RoundPipe's advantage emerges when model training requires more than  **a single GPU's VRAM**.

## Dispatch Pipeline Parallelism {#dispatch-pp}

### What Is Pipeline Parallelism

Pipeline Parallelism is a model-parallel strategy that partitions a model by layers into multiple **stages**, splits training data into multiple **microbatches**, and processes those microbatches through the stages in a pipelined fashion.

A simple example: suppose a model has 4 layers partitioned into 2 stages (2 layers each), using GPipe scheduling with 2 microbatches:

```
Time →

GPU 0 (stage 0): [mb0 F][mb1 F]              [mb1 B][mb0 B]
GPU 1 (stage 1):        [mb0 F][mb1 F][mb1 B][mb0 B]

F = forward pass, B = backward pass
```

After GPU 0 finishes the stage-0 forward pass for mb0, GPU 1 begins the stage-1 forward for mb0, while GPU 0 can start the stage-0 forward for mb1 — both GPUs work in parallel. However, between the end of all forward passes and the start of backward passes, there are idle **pipeline bubbles** where GPUs sit waiting.

### RoundPipe's Dispatch Pipeline Parallelism

In traditional pipeline parallelism, each stage is pinned to a fixed GPU (stage 0 always runs on GPU 0, stage 1 on GPU 1), which constrains how stages can be partitioned.

RoundPipe breaks this stage-to-GPU binding. Model parameters live in CPU memory; each time a stage runs, its parameters are uploaded from CPU to a GPU, and released after computation. Any stage can therefore run on any GPU.

RoundPipe treats GPUs as a **stateless compute pool** and dispatches stages to them in round-robin order:

```
Time →
GPU 0: [F stage0] [F stage3] [B stage2] ...
GPU 1:   [F stage1] [B stage4] [B stage1] ...
GPU 2:     [F stage2] [B stage3] [B stage0] ...
```

This yields several key advantages:

1. **Flexible stage partitioning**: The number of stages can be any integer, independent of the GPU count. Forward and backward passes can even use different partitions (asymmetric partitioning) for better load balancing.
2. **Fewer pipeline bubbles**: Round-robin dispatch lets forward and backward stages interleave tightly, eliminating inter-GPU synchronization waits.
3. **Lower memory footprint**: At any moment, a GPU holds data for only the single stage currently executing.

#### Stages and Layers in RoundPipe

- **Layer**: The basic building block of the model — each submodule passed to `nn.Sequential`. For example, each transformer block is a layer.
- **Stage**: A group of one or more consecutive layers that forms the scheduling unit for the pipeline. All layers in a stage are uploaded to a GPU together and executed sequentially.

RoundPipe uses **asymmetric partitioning** for forward and backward passes: because a transformer layer's forward computation takes roughly 1/3 the time of its backward pass (including recomputation), forward stages contain more layers and backward stages fewer, so that every stage takes approximately the same time. For example, a 12-layer model might be partitioned as:

- Forward: `[layer 0-2], [layer 3-5], [layer 6-8], [layer 9-11]` (3 layers per stage)
- Backward: `[layer 11], [layer 10], [layer 9], ..., [layer 0]` (1 layer per stage)

## Core Concepts

### Sequential Code, Parallel Run

RoundPipe uses a **single-controller architecture**: you write a plain, single-device training script, and RoundPipe automatically maps the computation across all available GPUs.

The core API boils down to two calls:

- `model.forward_backward()` — run the forward and backward passes
- `model.step()` — run the optimizer update

A typical training loop looks like this:

```python
for data, labels in dataloader:
    loss = model.forward_backward(
        input_args=(data,),
        label=labels,
        loss_fn=my_loss_fn,
    )
    model.step(lambda: (optimizer.step(), optimizer.zero_grad()))
```

This reads almost identically to standard single-GPU training, but under the hood RoundPipe handles microbatch splitting, multi-GPU pipeline scheduling, parameter transfers, gradient accumulation, and asynchronous optimizer updates — all automatically.

### CPU-Centric Parameter Storage

RoundPipe follows a **compute-dispatch paradigm**:

- CPU memory is the **primary store** for model state and intermediate activations.
- GPUs are **stateless compute accelerators** that temporarily hold a stage's data only while executing it (VRAM acts as a cache).
- After computation, gradients are downloaded back to CPU and the GPU-side parameter copy is released immediately.

This design vastly increases the trainable model size: the VRAM limit depends only on the size of a single model layer, not the entire model. You simply keep model parameters and input data on the CPU; RoundPipe takes care of all data transfers and scheduling.

### Input Splitting and Microbatch Execution

RoundPipe automatically splits each training batch into multiple **microbatches** and processes them in a pipeline. Each microbatch independently performs its forward pass, loss computation, and backward pass, with gradients accumulated across all microbatches:

```
Input batch (batch_size=12, num_microbatch=3)
    ↓ automatic split
microbatch 0: [samples 0-3]   ──forward──→ output_0 ──loss_fn──→ loss_0 ──backward──→ accumulate grads
microbatch 1: [samples 4-7]   ──forward──→ output_1 ──loss_fn──→ loss_1 ──backward──→ accumulate grads
microbatch 2: [samples 8-11]  ──forward──→ output_2 ──loss_fn──→ loss_2 ──backward──→ accumulate grads
    ↓
returns loss = loss_0 + loss_1 + loss_2
```

Note: each microbatch's loss is **computed and backpropagated independently**. The returned loss is the sum of all microbatch losses, and the gradients are the sum of per-microbatch gradients. This is mathematically equivalent to computing the loss over the entire batch and backpropagating once.

If you want the **mean** loss (equivalent to averaging over the full batch), divide by the number of microbatches inside `loss_fn`:

```python
loss_fn=lambda outputs, labels: criterion(outputs, labels) / num_microbatch
```

### Activation Recomputation

Backward passes need the intermediate activations (per-layer computation results) from the forward pass. For large models and long sequences, storing all activations consumes enormous amounts of memory. **Activation recomputation** trades compute for memory: during the forward pass, only each layer's input is saved (not intermediate activations); during the backward pass, the forward computation is re-executed to recover intermediate results. Although this adds computational overhead, it significantly reduces memory usage, enabling larger batch sizes and larger models, which in turn improves GPU utilization and overall training efficiency.

RoundPipe integrates full activation recomputation — every layer's forward computation is re-executed during the backward pass — with no additional code changes required.
