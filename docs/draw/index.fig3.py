#!/usr/bin/env python3
"""Fig 3 – Linear Parallel Scaling (throughput solid, seq length dashed)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
from common import (MODEL_COLORS, THEMES,
                     save_fig, style_ax, add_legend, generate_all)

# ── Data ─────────────────────────────────────────────────────────
GPUS = np.arange(1, 9)

MODELS_ZH = ['Qwen3-1.7B', 'Llama3-8B', 'Qwen3-32B', 'Qwen3-235B (LoRA)']
MODELS_EN = ['Qwen3-1.7B', 'Llama3-8B', 'Qwen3-32B', 'Qwen3-235B (LoRA)']

# Throughput (tokens/s) per GPU count
THROUGHPUT = [
    [8880.99,  17025.55, 25702.44, 33178.39, 42026.20, 48829.86, 57598.61, 65416.83],
    [3142.17,  6259.09,  9221.60,  12277.77, 15342.93, 18363.22, 21296.67, 24274.54],
    [740.46,   1476.11,  2203.82,  2897.08,  3599.81,  4290.25,  4913.34,  5516.22],
    [480.30,   807.87,   1088.40,  1281.49,  1509.48,  1635.75,  1742.93,  1819.61],
]

# Max input sequence length (k) – constant across GPU counts
MAX_SEQ_LEN = [73, 49, 28, 31]

LABELS = {
    'zh': dict(
        ylabel_left='吞吐量 (tokens/s)',
        ylabel_right='最大输入序列长度 (k tokens)',
        throughput_suffix=' 吞吐量',
        seqlen_suffix=' 序列长度',
    ),
    'en': dict(
        ylabel_left='Throughput (tokens/s)',
        ylabel_right='Max Sequence Length (k tokens)',
        throughput_suffix=' throughput',
        seqlen_suffix=' seq length',
    ),
}


def draw(lang, theme):
    t = THEMES[theme]
    lab = LABELS[lang]
    models = MODELS_ZH if lang == 'zh' else MODELS_EN
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    # Throughput – solid lines
    for i, (model, tp) in enumerate(zip(models, THROUGHPUT)):
        ax1.plot(GPUS, tp, '-o', color=MODEL_COLORS[i], linewidth=2,
                 markersize=4, label=model + lab['throughput_suffix'], zorder=3)

    # Max sequence length – dashed horizontal lines
    for i, (model, sl) in enumerate(zip(models, MAX_SEQ_LEN)):
        ax2.axhline(y=sl, color=MODEL_COLORS[i], linestyle='--', linewidth=1.2,
                     alpha=0.55, label=model + lab['seqlen_suffix'], zorder=2)

    # Style left axis
    style_ax(ax1, theme, ylabel=lab['ylabel_left'])
    ax1.set_xlabel('GPUs' if lang == 'en' else 'GPU 数量',
                   fontsize=10, color=t['text'], labelpad=8)
    ax1.set_xticks(GPUS)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda v, _: f'{v/1000:.0f}k' if v >= 1000 else f'{v:.0f}'))

    # Style right axis
    ax2.set_ylabel(lab['ylabel_right'], fontsize=10, color=t['text'], labelpad=8)
    ax2.tick_params(colors=t['text'], labelsize=9)
    ax2.set_ylim(0, 90)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0f}k'))
    for spine in ax2.spines.values():
        spine.set_visible(False)

    # Combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(h1 + h2, l1 + l2, fontsize=7, ncol=2,
                        loc='upper left', frameon=True,
                        facecolor=t['legend_frame'],
                        edgecolor='none', labelcolor=t['text'])

    fig.tight_layout()
    save_fig(fig, 3, lang, theme)


if __name__ == '__main__':
    generate_all(draw, 3)
