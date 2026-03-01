---
hide:
  - navigation
  - toc
---

<link rel="stylesheet" href="../assets/index.css">

<p align="center">
  <img src="../assets/banner.light.svg#only-light" alt="RoundPipe Banner" width="400">
  <img src="../assets/banner.dark.svg#only-dark" alt="RoundPipe Banner" width="400">
</p>

<p align="center" class="home-hero-subtitle">训练你的大模型</p>

<p align="center" class="home-hero-tagline">性能卓越<span class="space-1rem"></span>通用易用<span class="space-1rem"></span>专为消费级GPU设计</p>

<div class="home-hero-actions">
  <div class="glass-card home-install-command">
    <span style="color: var(--md-code-hl-color); user-select:none;">></span>
    <code id="install-command-text">pip install roundpipe</code>
    <svg id="copy-install-icon" width="1rem" height="1rem" viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg" fill="var(--md-code-hl-color)" style="cursor: pointer;"><path fill-rule="evenodd" clip-rule="evenodd" d="M4 4l1-1h5.414L14 6.586V14l-1 1H5l-1-1V4zm9 3l-3-3H5v10h8V7z"/><path fill-rule="evenodd" clip-rule="evenodd" d="M3 1L2 2v10l1 1V2h6.414l-1-1H3z"/></svg>
  </div>
  <a class="md-button glass-card" href="API/model">快速开始↗</a>
</div>

<script>
  (() => {
    const icon = document.getElementById('copy-install-icon');
    if (!navigator.clipboard) {
      icon.style.display = 'none';
      return;
    }
    icon.addEventListener('click', async () => {
      const code = document.getElementById('install-command-text');
      const text = code?.textContent?.trim();
      await navigator.clipboard.writeText(text);
    });
  })();
</script>

#

<div class="home-features">

  <a class="glass-card home-feature-card" href="#huge-model">
    <span class="feature-emoji">🧠</span>
    <h3>超大模型支持</h3>
    <p>仅需24GB显存，支持64K+超长上下文训练，支持32B大模型全量微调，支持235B大模型LoRA微调。</p>
  </a>

  <a class="glass-card home-feature-card" href="#performance">
    <span class="feature-emoji">⚡</span>
    <h3>释放显卡算力</h3>
    <p>完全释放4090算力，训练速度提升多达6倍，性能比肩A800 NVLINK。</p>
  </a>

  <a class="glass-card home-feature-card" href="#scaling">
    <span class="feature-emoji">📈</span>
    <h3>线性并行扩展</h3>
    <p>无需修改代码，节点内全自动多GPU线性扩展，且语义不变。</p>
  </a>

  <a class="glass-card home-feature-card" href="#simple-api">
    <span class="feature-emoji">✨</span>
    <h3>简单易用</h3>
    <p>串行编程接口，无需并行编程经验。支持Jupyter Notebook，轻松上手。</p>
  </a>

  <a class="glass-card home-feature-card" href="#simple-api">
    <span class="feature-emoji">🔧</span>
    <h3>灵活通用</h3>
    <p>支持任意深度神经网络，任意训练流程，任意参数更新策略。</p>
  </a>

  <a class="glass-card home-feature-card" href="#portable">
    <span class="feature-emoji">🔄</span>
    <h3>跨平台兼容</h3>
    <p>纯PyTorch实现，兼容Nvidia、AMD、昇腾等多种GPU平台。</p>
  </a>

</div>

<hr class="home-divider">

<!-- Section 1 – 超大模型支持 -->
<div class="home-section" id="huge-model">
  <div class="glass-card home-section-card">
    <div class="home-section-text">
      <h2>超大模型支持</h2>
      <p>仅需 24GB 显存，支持 64K+ 超长上下文训练</p>
      <p class="highlight"><strong>单卡即可全量微调 32B 大模型、LoRA 微调 235B 大模型</strong></p>
      <p>相比 PyTorch FSDP，最大输入序列长度至多提升 7+ 倍</p>
    </div>
    <div class="home-section-visual">
      <img src="../assets/index.fig1.light.svg#only-light" alt="超大模型支持">
      <img src="../assets/index.fig1.dark.svg#only-dark" alt="超大模型支持">
    </div>
  </div>
</div>

<!-- Section 2 – 释放显卡算力 -->
<div class="home-section home-section-reverse" id="performance">
  <div class="glass-card home-section-card">
    <div class="home-section-text">
      <h2>释放显卡算力</h2>
      <p>4090 在大模型训练中达到接近 A800 NVLINK 的吞吐量</p>
      <p>训练速度相比 FSDP Offload 提升多达 6 倍</p>
      <p>对于更大的模型，RoundPipe 吞吐量全面超越 FSDP</p>
    </div>
    <div class="home-section-visual">
      <img src="../assets/index.fig2.light.svg#only-light" alt="释放显卡算力">
      <img src="../assets/index.fig2.dark.svg#only-dark" alt="释放显卡算力">
    </div>
  </div>
</div>

<!-- Section 3 – 线性并行扩展 -->
<div class="home-section" id="scaling">
  <div class="glass-card home-section-card">
    <div class="home-section-text">
      <h2>线性并行扩展</h2>
      <p>无需修改任何代码，节点内全自动多 GPU 线性扩展</p>
      <p>吞吐量随 GPU 数量近似线性增长</p>
      <p>且每张 GPU 的最大输入序列长度保持不变</p>
    </div>
    <div class="home-section-visual">
      <img src="../assets/index.fig3.light.svg#only-light" alt="线性并行扩展">
      <img src="../assets/index.fig3.dark.svg#only-dark" alt="线性并行扩展">
    </div>
  </div>
</div>

<!-- Section 4 – 简单易用 & 灵活通用 -->
<div class="home-section home-section-reverse" id="simple-api">
  <div class="glass-card home-section-card">
    <div class="home-section-text">
      <h2>简单易用 & 灵活通用</h2>
      <p>串行编程接口，无需并行编程经验</p>
      <p>支持 Jupyter Notebook，所见即所得</p>
    </div>
    <div class="home-section-code" markdown="1">
```python
import torch
from roundpipe import RoundPipe, OptimizerCtx
# 任意深度神经网络
model = torch.nn.Sequential(layer1, layer2, layer3, ...)
# 任意 PyTorch 优化器
with OptimizerCtx():
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# 任意训练流程
for data in dataloader:
    loss = model.forward_backward(data)
    # 任意参数更新策略
    def step_fn():
        optimizer.step()
        optimizer.zero_grad()
    model.step(step_fn)
```
    </div>
  </div>
</div>

<!-- Section 5 – 跨平台兼容 -->
<div class="home-section" id="portable">
  <div class="glass-card home-section-card">
    <div class="home-section-text">
      <h2>跨平台兼容</h2>
      <p>纯 PyTorch 实现</p>
      <p>天然兼容 Nvidia、AMD、昇腾等多种 GPU 平台</p>
      <p>一份代码，多平台运行</p>
    </div>
    <div class="home-section-visual">
      <img src="../assets/index.fig5.light.svg#only-light" alt="跨平台兼容">
      <img src="../assets/index.fig5.dark.svg#only-dark" alt="跨平台兼容">
    </div>
  </div>
</div>
