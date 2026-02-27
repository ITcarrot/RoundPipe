#!/usr/bin/env python3
"""Fig 2 – Training Throughput comparison."""
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

# ── Data (tokens/s, None = OOM) ─────────────────────────────────
MODELS = ["Qwen3-1.7B", "Llama3.1-8B", "Qwen3-32B", "Qwen3-235B\n(LoRA)"]
FRAMEWORKS = [
    "A800 - FSDP",
    "A800 - RoundPipe",
    "4090 - RoundPipe",
    "4090 - FSDP Offload",
]
DATA = [
    [85829.26, 29148.46, 3454.77, None],  # A800 FSDP
    [84691.82, 28427.01, 6301.19, 1795.86],  # A800 RoundPipe
    [65416.83, 24274.54, 5516.22, 1819.61],  # 4090 RoundPipe
    [35073.58, 4070.81, None, None],  # 4090 FSDP Offload
]
FW_STYLE = [
    dict(color=FSDP_COLOR, hatch="///"),
    dict(color=ROUNDPIPE_COLOR, hatch="///"),
    dict(color=ROUNDPIPE_COLOR, hatch=""),
    dict(color=FSDP_COLOR, hatch=""),
]

LABELS = {
    "zh": dict(ylabel="训练吞吐量 (tokens/s)"),
    "en": dict(ylabel="Training Throughput (tokens/s)"),
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
            x + offset,
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
                    400,
                    "OOM",
                    ha="center",
                    va="bottom",
                    fontsize=FONT_SIZES["oom_label"],
                    color=t["oom_color"],
                    fontweight="bold",
                    rotation=90,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, fontsize=FONT_SIZES["xtick_model"], color=t["text"])
    style_ax(ax, theme, ylabel=LABELS[lang]["ylabel"])
    # Format y-axis with K suffix
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v/1000:.0f}k" if v >= 1000 else f"{v:.0f}")
    )
    add_legend(ax, theme, loc="upper right", ncol=2)
    fig.tight_layout()
    save_fig(fig, 2, lang, theme)


if __name__ == "__main__":
    generate_all(draw, 2)
