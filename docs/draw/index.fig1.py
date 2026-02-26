#!/usr/bin/env python3
"""Fig 1 – Maximum Input Sequence Length comparison."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
from common import (ROUNDPIPE_COLOR, FSDP_COLOR, THEMES,
                     save_fig, style_ax, add_legend, generate_all)

# ── Data (values in thousands, None = OOM) ──────────────────────
MODELS = ['Qwen3-1.7B', 'Llama3.1-8B', 'Qwen3-32B', 'Qwen3-235B\n(LoRA)']
FRAMEWORKS = [
    'A800 - FSDP',
    'A800 - RoundPipe',
    '4090 - RoundPipe',
    '4090 - FSDP Offload',
]
DATA = [
    [39,  29,  11,   None],   # A800 FSDP
    [288, 226, 126,  118],    # A800 RoundPipe
    [73,  49,  28,   31],     # 4090 RoundPipe
    [11,  11,  None, None],   # 4090 FSDP Offload
]
FW_STYLE = [
    dict(color=FSDP_COLOR,     hatch='///'),   # A800 FSDP
    dict(color=ROUNDPIPE_COLOR, hatch='///'),   # A800 RoundPipe
    dict(color=ROUNDPIPE_COLOR, hatch=''),      # 4090 RoundPipe
    dict(color=FSDP_COLOR,     hatch=''),       # 4090 FSDP Offload
]

LABELS = {
    'zh': dict(ylabel='最大输入序列长度 (k tokens)'),
    'en': dict(ylabel='Max Input Sequence Length (k tokens)'),
}


def draw(lang, theme):
    t = THEMES[theme]
    fig, ax = plt.subplots(figsize=(8, 5))

    n_models = len(MODELS)
    n_fw = len(FRAMEWORKS)
    bar_w = 0.18
    x = np.arange(n_models)

    for i, (fw, style, row) in enumerate(zip(FRAMEWORKS, FW_STYLE, DATA)):
        vals = [v if v is not None else 0 for v in row]
        offset = (i - n_fw / 2 + 0.5) * bar_w
        bars = ax.bar(
            x + offset, vals, bar_w,
            color=style['color'],
            hatch=style['hatch'],
            edgecolor=t['hatch_edge'] if style['hatch'] else 'none',
            label=fw, linewidth=0.6, zorder=3,
        )
        for j, (bar, val) in enumerate(zip(bars, row)):
            if val is None:
                ax.text(bar.get_x() + bar.get_width() / 2, 3,
                        'OOM', ha='center', va='bottom', fontsize=6.5,
                        color=t['oom_color'], fontweight='bold', rotation=90)
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 4,
                        f'{val}k', ha='center', va='bottom', fontsize=6.5,
                        color=t['text_secondary'])

    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, fontsize=9, color=t['text'])
    style_ax(ax, theme, ylabel=LABELS[lang]['ylabel'])
    add_legend(ax, theme, loc='upper left', ncol=2)
    fig.tight_layout()
    save_fig(fig, 1, lang, theme)


if __name__ == '__main__':
    generate_all(draw, 1)
