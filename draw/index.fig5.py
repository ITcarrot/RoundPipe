#!/usr/bin/env python3
"""Fig 5 – Cross-platform Throughput comparison."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
from common import (COLOR_4090, COLOR_910B, COLOR_W7800, THEMES,
                     save_fig, style_ax, add_legend, generate_all)

# ── Data (tokens/s) ─────────────────────────────────────────────
MODELS = ['Qwen3-1.7B', 'Llama3.1-8B', 'Qwen3-32B', 'Qwen3-235B\n(LoRA)']
DEVICES = ['W7800', '910B', '4090']
DATA = [
    [17851.94, 5915.46,  1449.89, 665.36],    # W7800
    [50598.56, 23253.48, 5028.43, 459.01],     # 910B
    [65416.83, 24274.54, 5516.22, 1819.61],    # 4090
]
DEVICE_COLORS = [COLOR_W7800, COLOR_910B, COLOR_4090]

LABELS = {
    'zh': dict(ylabel='训练吞吐量 (tokens/s)', devices=['AMD W7800', '昇腾 910B', 'RTX 4090']),
    'en': dict(ylabel='Training Throughput (tokens/s)', devices=['AMD W7800', 'Ascend 910B', 'RTX 4090']),
}


def draw(lang, theme):
    t = THEMES[theme]
    lab = LABELS[lang]
    fig, ax = plt.subplots(figsize=(8, 5))

    n_models = len(MODELS)
    n_dev = len(DEVICES)
    bar_w = 0.22
    x = np.arange(n_models)

    for i, (dev, color, row) in enumerate(zip(lab['devices'], DEVICE_COLORS, DATA)):
        offset = (i - n_dev / 2 + 0.5) * bar_w
        ax.bar(x + offset, row, bar_w,
               color=color, edgecolor='none',
               label=dev, linewidth=0.5, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, fontsize=9, color=t['text'])
    style_ax(ax, theme, ylabel=lab['ylabel'])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda v, _: f'{v/1000:.0f}k' if v >= 1000 else f'{v:.0f}'))
    add_legend(ax, theme, loc='upper right')
    fig.tight_layout()
    save_fig(fig, 5, lang, theme)


if __name__ == '__main__':
    generate_all(draw, 5)
