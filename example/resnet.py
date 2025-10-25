import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import random
import numpy as np
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from RoundPipe import RoundPipe

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
                nn.BatchNorm2d(out_channels)
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
# Helper to create a list of blocks (not Sequential)
def make_blocks(in_channels, out_channels, blocks, stride):
    layers = []
    layers.append(BasicBlock(in_channels, out_channels, stride))
    for _ in range(1, blocks):
        layers.append(BasicBlock(out_channels, out_channels))
    return layers

class ResNet(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
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
            nn.Linear(512, num_classes)
        ]
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)

# Data preparation
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, 4),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

# Training setup
model = RoundPipe(ResNet().seq)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

train_losses, test_losses = [], []
train_accs, test_accs = [], []

def evaluate(loader):
    model.eval()
    correct, total, loss_sum = 0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images, labels
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_sum += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    return loss_sum / total, correct / total

# Training loop
epochs = 20
for epoch in range(epochs):
    model.train()
    running_loss, correct, total = 0, 0, 0
    for images, labels in trainloader:
        images, labels = images, labels
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * labels.size(0)
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
    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

import os, re
pattern = re.compile(
    r"Train Loss: ([\d.]+) \| Train Acc: ([\d.]+) \| Test Loss: ([\d.]+) \| Test Acc: ([\d.]+)"
)
script_dir = os.path.dirname(os.path.abspath(__file__))
ref_train_losses, ref_train_accs = [], []
ref_test_losses, ref_test_accs = [], []
with open(os.path.join(script_dir, 'resnet_cifar100_ref.log'), 'r') as f:
    for line in f:
        match = pattern.search(line)
        if match:
            ref_train_losses.append(float(match.group(1)))
            ref_train_accs.append(float(match.group(2)))
            ref_test_losses.append(float(match.group(3)))
            ref_test_accs.append(float(match.group(4)))

# Plotting
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(ref_train_losses, label='Ref Train Loss', linestyle='dashed')
plt.plot(ref_test_losses, label='Ref Test Loss', linestyle='dashed')
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curve')
plt.subplot(1,2,2)
plt.plot(ref_train_accs, label='Ref Train Acc', linestyle='dashed')
plt.plot(ref_test_accs, label='Ref Test Acc', linestyle='dashed')
plt.plot(train_accs, label='Train Acc')
plt.plot(test_accs, label='Test Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Curve')
plt.tight_layout()
plt.savefig(script_dir + '/resnet_cifar100_curves.png')
plt.show()