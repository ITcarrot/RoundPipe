#!/usr/bin/env python3
"""Fig 5 – Cross-platform Throughput comparison."""
import sys, os

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from common import (
    FONT_SIZES,
    THEMES,
    save_fig,
    style_ax,
    add_legend,
    generate_all,
)

# ── Data (tokens/s) ─────────────────────────────────────────────
MODELS = ["Qwen3-1.7B", "Llama3.1-8B", "Qwen3-32B", "Qwen3-235B\n(LoRA)"]
DEVICES = ["W7800", "910B", "4090"]
DATA = [
    [17851.94, 5915.46, 1449.89, 665.36],  # W7800
    [50598.56, 23253.48, 5028.43, 459.01],  # 910B
    [65416.83, 24274.54, 5516.22, 1819.61],  # 4090
]
DEVICE_COLOR_KEYS = ["w7800", "910b", "4090"]

YLABEL = "Training Throughput (tokens/s)"
DEVICE_LABELS = ["AMD W7800", "Ascend 910B", "RTX 4090"]


def draw(theme):
    t = THEMES[theme]
    fig, ax = plt.subplots(figsize=(8, 5))

    n_models = len(MODELS)
    n_dev = len(DEVICES)
    bar_w = 0.22
    x = np.arange(n_models)

    for i, (dev, color_key, row) in enumerate(
        zip(DEVICE_LABELS, DEVICE_COLOR_KEYS, DATA)
    ):
        offset = (i - n_dev / 2 + 0.5) * bar_w
        ax.bar(
            x + offset,
            row,
            bar_w,
            color=t["colors"][color_key],
            edgecolor="none",
            label=dev,
            linewidth=0.5,
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, fontsize=FONT_SIZES["xtick_model"], color=t["text"])
    style_ax(ax, theme, ylabel=YLABEL)
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"{v/1000:.0f}k" if v >= 1000 else f"{v:.0f}")
    )
    add_legend(ax, theme, loc="upper right")
    fig.tight_layout()
    save_fig(fig, 5, theme)


if __name__ == "__main__":
    generate_all(draw, 5)
