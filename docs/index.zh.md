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
  <a class="md-button glass-card" href="Tutorial.zh.md">快速开始↗</a>
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
    <h3>超大模型支持</h3>
    <p>仅需24GB显存，支持64K+超长上下文训练，支持32B大模型全量微调，支持235B大模型LoRA微调。</p>
  </div>

  <div class="glass-card home-feature-card">
    <span class="feature-emoji">⚡</span>
    <h3>释放显卡算力</h3>
    <p>完全释放4090算力，训练速度提升2-3倍，性能比肩A800 NVLINK。</p>
  </div>

  <div class="glass-card home-feature-card">
    <span class="feature-emoji">📈</span>
    <h3>线性并行扩展</h3>
    <p>无需修改代码，节点内全自动多GPU线性扩展，且语义不变。</p>
  </div>

  <div class="glass-card home-feature-card">
    <span class="feature-emoji">✨</span>
    <h3>简单易用</h3>
    <p>串行编程接口，无需并行编程经验。支持Jupyter Notebook，轻松上手。</p>
  </div>

  <div class="glass-card home-feature-card">
    <span class="feature-emoji">🔧</span>
    <h3>灵活通用</h3>
    <p>支持任意深度神经网络，不限模型层内结构，不限训练流程，不限参数更新策略。</p>
  </div>

  <div class="glass-card home-feature-card">
    <span class="feature-emoji">🔄</span>
    <h3>跨平台兼容</h3>
    <p>纯Pytorch实现，兼容Nvidia、AMD等多种GPU平台。</p>
  </div>

</div>
