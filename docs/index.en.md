---
hide:
  - navigation
  - toc
---

<link rel="stylesheet" href="assets/index.css">

<p align="center">
  <img src="assets/banner.light.svg#only-light" alt="RoundPipe Banner" width="400">
  <img src="assets/banner.dark.svg#only-dark" alt="RoundPipe Banner" width="400">
</p>

<p align="center" class="home-hero-subtitle">Train Your Large Models</p>

<p align="center" class="home-hero-tagline">High Performance<span class="space-1rem"></span>Easy to Use<span class="space-1rem"></span>Built for Gaming GPUs</p>

<div class="home-hero-actions">
  <div class="glass-card home-install-command">
    <span style="color: var(--md-code-hl-color); user-select:none;">></span>
    <code id="install-command-text">pip install roundpipe</code>
    <svg id="copy-install-icon" width="1rem" height="1rem" viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg" fill="var(--md-code-hl-color)" style="cursor: pointer;"><path fill-rule="evenodd" clip-rule="evenodd" d="M4 4l1-1h5.414L14 6.586V14l-1 1H5l-1-1V4zm9 3l-3-3H5v10h8V7z"/><path fill-rule="evenodd" clip-rule="evenodd" d="M3 1L2 2v10l1 1V2h6.414l-1-1H3z"/></svg>
  </div>
  <a class="md-button glass-card" href="API/model">Get started ↗</a>
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

  <div class="glass-card home-feature-card">
    <span class="feature-emoji">🧠</span>
    <h3>Built for huge models</h3>
    <p>On a single 24GB GPU, train with 64K+ long context, full fine-tune 32B models or LoRA fine-tune models up to 235B.</p>
  </div>

  <div class="glass-card home-feature-card">
    <span class="feature-emoji">⚡</span>
    <h3>High Performance</h3>
    <p>Push a 4090 close to A800 NVLINK-class throughput. Up to 6× faster than FSDP Offload in typical training workloads.</p>
  </div>

  <div class="glass-card home-feature-card">
    <span class="feature-emoji">📈</span>
    <h3>Linear scaling</h3>
    <p>Scale to multiple GPUs in-node without rewriting your training loop. Throughput grows linearly while your code stays the same.</p>
  </div>

  <div class="glass-card home-feature-card">
    <span class="feature-emoji">✨</span>
    <h3>Feels like PyTorch</h3>
    <p>A sequential programming interface with a low learning curve. Works well in Jupyter Notebook for rapid iteration.</p>
  </div>

  <div class="glass-card home-feature-card">
    <span class="feature-emoji">🔧</span>
    <h3>General by design</h3>
    <p>No constraints on layer structure, training flow, or parameter update strategy.</p>
  </div>

  <div class="glass-card home-feature-card">
    <span class="feature-emoji">🔄</span>
    <h3>Portable across accelerators</h3>
    <p>Pure PyTorch implementation. Runs across Nvidia, AMD, and Ascend GPU platforms.</p>
  </div>

</div>

<hr class="home-divider">

<!-- Section 1 – Huge model support -->
<div class="home-section">
  <div class="glass-card home-section-card">
    <div class="home-section-text">
      <h2>Train bigger than ever</h2>
      <p>64K+ long-context training on a single 24GB GPU</p>
      <p class="highlight"><strong>Full fine-tuning for 32B, LoRA for up to 235B</strong></p>
      <p>Up to 7× longer sequence length than PyTorch FSDP</p>
    </div>
    <div class="home-section-visual">
      <img src="assets/index.fig1.light.svg#only-light" alt="Huge model support">
      <img src="assets/index.fig1.dark.svg#only-dark" alt="Huge model support">
    </div>
  </div>
</div>

<!-- Section 2 – GPU throughput -->
<div class="home-section home-section-reverse">
  <div class="glass-card home-section-card">
    <div class="home-section-text">
      <h2>Extracts maximum performance</h2>
      <p>A 4090 can reach near A800 NVLINK-level throughput</p>
      <p>Up to 6× faster than FSDP Offload</p>
      <p>As models grow, RoundPipe keeps pulling ahead</p>
    </div>
    <div class="home-section-visual">
      <img src="assets/index.fig2.light.svg#only-light" alt="GPU throughput">
      <img src="assets/index.fig2.dark.svg#only-dark" alt="GPU throughput">
    </div>
  </div>
</div>

<!-- Section 3 – Linear scaling -->
<div class="home-section">
  <div class="glass-card home-section-card">
    <div class="home-section-text">
      <h2>Scale out without rewrites</h2>
      <p>100% automatic multi-GPU scaling within a node</p>
      <p>Throughput grows linearly with GPU count</p>
      <p>Max sequence length per GPU stays unchanged</p>
    </div>
    <div class="home-section-visual">
      <img src="assets/index.fig3.light.svg#only-light" alt="Linear scaling">
      <img src="assets/index.fig3.dark.svg#only-dark" alt="Linear scaling">
    </div>
  </div>
</div>

<!-- Section 4 – Simple API -->
<div class="home-section home-section-reverse">
  <div class="glass-card home-section-card">
    <div class="home-section-text">
      <h2>Simple API, flexible training</h2>
      <p>Sequential programming interface</p>
      <p>0 parallel programming</p>
      <p>Jupyter Notebook friendly</p>
    </div>
    <div class="home-section-code" markdown="1">
```python
import torch
from roundpipe import RoundPipe, OptimizerCtx
# Any deep neural network
model = torch.nn.Sequential(layer1, layer2, layer3, ...)
# Any PyTorch optimizer
with OptimizerCtx():
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# Any training loop
for data in dataloader:
    loss = model.forward_backward(data)
    # Any parameter update strategy
    def step_fn():
        optimizer.step()
        optimizer.zero_grad()
    model.step(step_fn)
```
    </div>
  </div>
</div>

<!-- Section 5 – Cross-platform -->
<div class="home-section">
  <div class="glass-card home-section-card">
    <div class="home-section-text">
      <h2>Portable by default</h2>
      <p>Pure PyTorch implementation</p>
      <p>Compatible with Nvidia, AMD, Ascend, and more</p>
      <p>Write once, train anywhere</p>
    </div>
    <div class="home-section-visual">
      <img src="assets/index.fig5.light.svg#only-light" alt="Cross-platform">
      <img src="assets/index.fig5.dark.svg#only-dark" alt="Cross-platform">
    </div>
  </div>
</div>
