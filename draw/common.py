"""Shared styling and utilities for index page figures."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# ── Font sizes (single source of truth) ─────────────────────────
FONT_SIZES = {
    "axis_label": 16,  # ylabel / xlabel
    "tick": 12,  # axis tick labels
    "xtick_model": 16,  # x-axis model name labels
    "legend": 13,  # legend text
    "legend_dense": 10,  # legend with many items (fig3)
    "bar_label": 11,  # data labels on top of bars (e.g. "226k")
    "oom_label": 12,  # OOM text on bars
}

# ── Themes ──────────────────────────────────────────────────────
THEMES = {
    "light": dict(
        colors={
            "roundpipe": "#4A90D9",
            "fsdp": "#E8834A",
            "4090": "#4A90D9",
            "910b": "#E8834A",
            "w7800": "#5BAD5E",
            "model_colors": ["#4A90D9", "#E8834A", "#5BAD5E", "#C466DB"],
        },
        text="#333333",
        text_axis="#000000",
        text_secondary="#777777",
        grid="#E0E0E0",
        hatch_edge=(1, 1, 1, 0.7),
        legend_frame=(1, 1, 1, 0.85),
        oom_color="#AAAAAA",
    ),
    "dark": dict(
        colors={
            "roundpipe": "#2F5F8F",
            "fsdp": "#A85F33",
            "4090": "#2F5F8F",
            "910b": "#A85F33",
            "w7800": "#3B723D",
            "model_colors": ["#2F5F8F", "#A85F33", "#3B723D", "#82458F"],
        },
        # Avoid pure white in dark mode (too glaring on slate backgrounds).
        text="#B7BEC8",
        text_axis="#D2D8E0",
        text_secondary="#8C96A3",
        grid="#222A35",
        hatch_edge=(0, 0, 0, 0.5),
        legend_frame=(0.12, 0.12, 0.18, 0.85),
        oom_color="#59616D",
    ),
}

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")


def save_fig(fig, fig_num, theme):
    os.makedirs(ASSETS_DIR, exist_ok=True)
    # Keep filenames stable across themes.
    fname = f"index.fig{fig_num}.{theme}.svg"
    fpath = os.path.join(ASSETS_DIR, fname)
    fig.savefig(fpath, format="svg", transparent=True, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


def style_ax(ax, theme, ylabel="", xlabel=""):
    t = THEMES[theme]
    if ylabel:
        ax.set_ylabel(
            ylabel, fontsize=FONT_SIZES["axis_label"], color=t["text_axis"], labelpad=8
        )
    if xlabel:
        ax.set_xlabel(
            xlabel, fontsize=FONT_SIZES["axis_label"], color=t["text_axis"], labelpad=8
        )
    ax.tick_params(colors=t["text_axis"], labelsize=FONT_SIZES["tick"])
    ax.grid(axis="y", color=t["grid"], linestyle="--", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    # Half-frame: only left and bottom spines
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color(t["text_axis"])
        ax.spines[spine].set_linewidth(0.8)


def add_legend(ax, theme, **kwargs):
    t = THEMES[theme]
    defaults = dict(
        fontsize=FONT_SIZES["legend"],
        frameon=True,
        facecolor=t["legend_frame"],
        edgecolor="none",
        labelcolor=t["text"],
    )
    defaults.update(kwargs)
    ax.legend(**defaults)


def generate_all(draw_fn, fig_num):
    """Run draw_fn(theme) for all supported themes."""
    for theme in ("light", "dark"):
        draw_fn(theme)
    print(f"Fig{fig_num}: light/dark variants generated.")
