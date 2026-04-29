# 训练自定义模型

!!! tip "交互式 Notebook"
    自己动手试试: [resnet_cifar100.zh.ipynb](https://github.com/ITcarrot/RoundPipe/blob/main/example/resnet_cifar100.zh.ipynb)

本教程演示如何使用 **RoundPipe** 在 CIFAR-100 数据集上从零训练一个自定义的 ResNet。

你将学到：

1. 如何把一个模型拆解成 `nn.Sequential` 结构，使它能被 RoundPipe 按层流水化执行；
2. 如何用 `RoundPipe` + `OptimizerCtx` 搭建混合精度训练（FP16 模型 + FP32 优化器）；
3. 如何用 `forward_backward()` 这种融合的前后向 API 训练，以及在评估前用 `synchronize()` 同步模型状态。

RoundPipe 在多 GPU 环境下可自动分配流水段，本教程的代码无论在 1 卡还是 8 卡上都能直接跑。

!!! note
    本教程仅用于演示 RoundPipe 的使用方法，由于 RoundPipe 并非为小模型设计，因此训练速度可能低于常规单机实现。

## 1. 环境准备

需要安装 PyTorch, torchvision, matplotlib, tqdm 和 roundpipe。

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

from roundpipe import RoundPipe, OptimizerCtx, GradScaler
from roundpipe.optim import Adam
```

## 2. 把 ResNet 表达为 `nn.Sequential`

RoundPipe 的核心思路是把模型拆成若干层，逐层在不同 GPU / 不同时刻执行。所以模型要写成 `nn.Sequential`，让 RoundPipe 自动按层切分流水段。

```python
# ResNet Basic Block
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


def make_blocks(in_channels, out_channels, blocks, stride):
    layers = []
    layers.append(BasicBlock(in_channels, out_channels, stride))
    for _ in range(1, blocks):
        layers.append(BasicBlock(out_channels, out_channels))
    return layers


def build_resnet(num_classes=100):
    layers = [
        nn.Conv2d(3, 32, 3, 1, 1, bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        *make_blocks(32, 64, 3, 2),
        *make_blocks(64, 128, 4, 2),
        *make_blocks(128, 256, 6, 2),
        *make_blocks(256, 512, 3, 2),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(512, num_classes),
    ]
    return nn.Sequential(*layers)
```

## 3. 准备 CIFAR-100 数据

训练集使用 RandomCrop + RandomHorizontalFlip + 归一化。测试集只做归一化。

```python
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, 4),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

trainset = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform_train)
testset  = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform_test)

trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testloader  = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
```

## 4. 用 RoundPipe 包装模型

将 `nn.Sequential` 模型直接传给 `RoundPipe`，在 `OptimizerCtx()` 里创建优化器使得优化器能指向正确的参数。

```python
model = RoundPipe(build_resnet().to(torch.float16), optim_dtype=torch.float32)

with OptimizerCtx():
    optimizer = Adam(model.parameters())

scaler = GradScaler()
criterion = nn.CrossEntropyLoss()
```

## 5. 评估函数

评估前必须调用 `model.synchronize()`：训练阶段优化器更新默认是异步执行的，GPU 使用的主参数比优化器处参数落后一步，`synchronize()` 会等待并把最新的参数同步到主参数。

```python
def evaluate(loader):
    model.eval()
    correct, total, loss_sum = 0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(torch.float16), labels
            outputs = model(images)
            loss = criterion(outputs.float(), labels)
            loss_sum += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    return loss_sum / total, correct / total
```

## 6. 训练循环

RoundPipe 的核心训练 API 是 `forward_backward(...)`：它把前向和反向融合在一次调用中，前向 / 反向阶段会在 GPU 之间错峰交错，从而尽量消除流水线气泡。

几个细节：

- `loss_fn` 除以 `(num_devices + 1)` 是把损失按 microbatch 数量归一（默认 `num_microbatch = num_devices + 1`）；
- `model.step(step_fn)` 异步触发优化器更新（默认 `is_async=True`），传入的函数会在优化器线程上运行；
- `scaler.update()` 调整下一步的 loss scale。

```python
train_losses, test_losses = [], []
train_accs,   test_accs   = [], []

def optimizer_step():
    scaler.step(optimizer)
    optimizer.zero_grad()

EPOCHS = 20
for epoch in range(EPOCHS):
    model.train()
    running_loss, correct, total = 0, 0, 0
    for images, labels in tqdm(trainloader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
        loss, outputs = model.forward_backward(
            input_args=(images.to(torch.float16),),
            label=labels,
            loss_fn=lambda outputs, labels: scaler.scale(
                criterion(outputs.float(), labels)
            ) / (torch.cuda.device_count() + 1),
            return_outputs=True,
        )
        model.step(optimizer_step)
        scaler.update()

        loss_item = loss.item() / scaler.get_scale()
        running_loss += loss_item * labels.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_acc = correct / total
    test_loss, test_acc = evaluate(testloader)

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)

    print(
        f"Epoch {epoch+1:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
        f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}"
    )
```

??? example "Output"

    ```
    Epoch 01/20 | Train Loss: 4.3328 | Train Acc: 0.0413 | Test Loss: 3.9271 | Test Acc: 0.0791
    Epoch 02/20 | Train Loss: 3.7637 | Train Acc: 0.1099 | Test Loss: 3.4888 | Test Acc: 0.1579
    Epoch 03/20 | Train Loss: 3.3601 | Train Acc: 0.1793 | Test Loss: 3.0630 | Test Acc: 0.2373
    Epoch 04/20 | Train Loss: 3.0545 | Train Acc: 0.2372 | Test Loss: 2.7539 | Test Acc: 0.2966
    Epoch 05/20 | Train Loss: 2.7554 | Train Acc: 0.2927 | Test Loss: 2.5448 | Test Acc: 0.3400
    Epoch 06/20 | Train Loss: 2.4864 | Train Acc: 0.3514 | Test Loss: 2.3281 | Test Acc: 0.3921
    Epoch 07/20 | Train Loss: 2.2699 | Train Acc: 0.3944 | Test Loss: 2.1084 | Test Acc: 0.4358
    Epoch 08/20 | Train Loss: 2.0877 | Train Acc: 0.4377 | Test Loss: 1.9628 | Test Acc: 0.4708
    Epoch 09/20 | Train Loss: 1.9232 | Train Acc: 0.4729 | Test Loss: 1.8592 | Test Acc: 0.4925
    Epoch 10/20 | Train Loss: 1.7882 | Train Acc: 0.5050 | Test Loss: 1.7536 | Test Acc: 0.5162
    Epoch 11/20 | Train Loss: 1.6731 | Train Acc: 0.5318 | Test Loss: 1.7105 | Test Acc: 0.5275
    Epoch 12/20 | Train Loss: 1.5630 | Train Acc: 0.5571 | Test Loss: 1.6222 | Test Acc: 0.5548
    Epoch 13/20 | Train Loss: 1.4762 | Train Acc: 0.5764 | Test Loss: 1.5683 | Test Acc: 0.5698
    Epoch 14/20 | Train Loss: 1.3741 | Train Acc: 0.6041 | Test Loss: 1.5597 | Test Acc: 0.5726
    Epoch 15/20 | Train Loss: 1.2957 | Train Acc: 0.6246 | Test Loss: 1.5099 | Test Acc: 0.5823
    Epoch 16/20 | Train Loss: 1.2177 | Train Acc: 0.6443 | Test Loss: 1.5081 | Test Acc: 0.5903
    Epoch 17/20 | Train Loss: 1.1500 | Train Acc: 0.6630 | Test Loss: 1.4463 | Test Acc: 0.6080
    Epoch 18/20 | Train Loss: 1.0774 | Train Acc: 0.6808 | Test Loss: 1.4315 | Test Acc: 0.6115
    Epoch 19/20 | Train Loss: 1.0130 | Train Acc: 0.6965 | Test Loss: 1.4786 | Test Acc: 0.6077
    Epoch 20/20 | Train Loss: 0.9505 | Train Acc: 0.7124 | Test Loss: 1.4499 | Test Acc: 0.6189
    ```

## 7. 可视化训练曲线

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(train_losses, label="Train Loss")
axes[0].plot(test_losses, label="Test Loss")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
axes[0].legend(); axes[0].set_title("Loss Curve")
axes[1].plot(train_accs, label="Train Acc")
axes[1].plot(test_accs, label="Test Acc")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
axes[1].set_title("Accuracy Curve")
plt.tight_layout()
plt.show()
```

## 小结

本教程展示了使用 RoundPipe 训练 ResNet 的最小完整流程：

1. **写成 Sequential**：把 ResNet 写成 `Conv -> BasicBlock x N -> ClassifierHead` 的 `nn.Sequential`，每个 block 自然成为一个流水段。
2. **混合精度**：模型放 FP16、优化器放 FP32，使用 `OptimizerCtx` + `roundpipe.optim.Adam`。
3. **融合前后向**：`forward_backward()` 把训练循环写得几乎和单卡 PyTorch 一样简单，但底层会自动做多 GPU 流水线调度；`model.step()` 异步触发优化器，`model.synchronize()` 在评估前同步参数。

如果你有更多 GPU，无需改任何代码，RoundPipe 会自动按设备数生成流水线计划。可以通过 `run_config=RoundPipeRunConfig(num_microbatch=...)` 进一步控制 microbatch 数量。
