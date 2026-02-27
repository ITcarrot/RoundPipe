#!/usr/bin/env python3
"""Fig 1 – Maximum Input Sequence Length comparison."""
import sys, os

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
from common import (
    ROUNDPIPE_COLOR,
    FSDP_COLOR,
    FONT_SIZES,
    THEMES,
    save_fig,
    style_ax,
    add_legend,
    generate_all,
)

# ── Data (values in thousands, None = OOM) ──────────────────────
MODELS = ["Qwen3-1.7B", "Llama3.1-8B", "Qwen3-32B", "Qwen3-235B\n(LoRA)"]
FRAMEWORKS = [
    "4090 - FSDP Offload",
    "4090 - RoundPipe",
    "A800 - FSDP",
    "A800 - RoundPipe",
]
DATA = [
    [11, 11, None, None],  # 4090 FSDP Offload
    [73, 49, 28, 31],  # 4090 RoundPipe
    [39, 29, 11, None],  # A800 FSDP
    [288, 226, 126, 118],  # A800 RoundPipe
]
FW_STYLE = [
    dict(color=FSDP_COLOR, hatch=""),  # 4090 FSDP Offload
    dict(color=ROUNDPIPE_COLOR, hatch=""),  # 4090 RoundPipe
    dict(color=FSDP_COLOR, hatch="///"),  # A800 FSDP
    dict(color=ROUNDPIPE_COLOR, hatch="///"),  # A800 RoundPipe
]

LABELS = {
    "zh": dict(ylabel="最大输入序列长度 (k tokens)"),
    "en": dict(ylabel="Max Input Sequence Length (k tokens)"),
}


def draw(lang, theme):
    t = THEMES[theme]
    fig, ax = plt.subplots(figsize=(8, 5))

    n_models = len(MODELS)
    bar_w = 0.18
    gap = 0.06  # gap between A800 group and 4090 group
    x = np.arange(n_models)

    # Offsets: [bar0, bar1] gap [bar2, bar3], centered around 0
    # Each bar center is at: ±(0.5*bar_w + gap/2) and ±(1.5*bar_w + gap/2)
    half_gap = gap / 2
    offsets = [
        -1.5 * bar_w - half_gap,  # bar 0 (A800 FSDP)
        -0.5 * bar_w - half_gap,  # bar 1 (A800 RoundPipe)
        0.5 * bar_w + half_gap,  # bar 2 (4090 RoundPipe)
        1.5 * bar_w + half_gap,  # bar 3 (4090 FSDP Offload)
    ]

    for i, (fw, style, row) in enumerate(zip(FRAMEWORKS, FW_STYLE, DATA)):
        vals = [v if v is not None else 0 for v in row]
        bars = ax.bar(
            x + offsets[i],
            vals,
            bar_w,
            color=style["color"],
            hatch=style["hatch"],
            edgecolor=t["hatch_edge"] if style["hatch"] else "none",
            label=fw,
            linewidth=0.6,
            zorder=3,
        )
        for j, (bar, val) in enumerate(zip(bars, row)):
            if val is None:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    3,
                    "OOM",
                    ha="center",
                    va="bottom",
                    fontsize=FONT_SIZES["oom_label"],
                    color=FSDP_COLOR,
                    fontweight="bold",
                    rotation=90,
                )
            else:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 4,
                    f"{val}k",
                    ha="center",
                    va="bottom",
                    fontsize=FONT_SIZES["bar_label"],
                    color=t["text_axis"],
                    fontweight="bold",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, fontsize=FONT_SIZES["xtick_model"], color=t["text_axis"])
    style_ax(ax, theme, ylabel=LABELS[lang]["ylabel"])
    add_legend(ax, theme, loc="upper right", ncol=2)
    fig.tight_layout()
    save_fig(fig, 1, lang, theme)


if __name__ == "__main__":
    generate_all(draw, 1)
