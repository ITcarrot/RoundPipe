"""Shared styling and utilities for index page figures."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mplfonts import use_font
import numpy as np
import os

use_font('Noto Sans CJK SC')
plt.rcParams['axes.unicode_minus'] = False

# ── Colors ──────────────────────────────────────────────────────
ROUNDPIPE_COLOR = '#4A90D9'
FSDP_COLOR = '#E8834A'

COLOR_4090 = '#4A90D9'
COLOR_910B = '#E8834A'
COLOR_W7800 = '#5BAD5E'

MODEL_COLORS = ['#4A90D9', '#E8834A', '#5BAD5E', '#C466DB']

# ── Themes ──────────────────────────────────────────────────────
THEMES = {
    'light': dict(
        text='#333333',
        text_secondary='#777777',
        grid='#E0E0E0',
        hatch_edge=(1, 1, 1, 0.7),
        legend_frame=(1, 1, 1, 0.85),
        oom_color='#AAAAAA',
    ),
    'dark': dict(
        text='#D4D4D4',
        text_secondary='#888888',
        grid='#333333',
        hatch_edge=(0, 0, 0, 0.5),
        legend_frame=(0.12, 0.12, 0.18, 0.85),
        oom_color='#666666',
    ),
}

ASSETS_DIR = os.path.join(os.path.dirname(__file__), '..', 'assets')


def save_fig(fig, fig_num, lang, theme):
    os.makedirs(ASSETS_DIR, exist_ok=True)
    fname = f'index.fig{fig_num}.{lang}.{theme}.svg'
    fpath = os.path.join(ASSETS_DIR, fname)
    fig.savefig(fpath, format='svg', transparent=True, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {fname}')


def style_ax(ax, theme, ylabel='', xlabel=''):
    t = THEMES[theme]
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, color=t['text'], labelpad=8)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10, color=t['text'], labelpad=8)
    ax.tick_params(colors=t['text'], labelsize=9)
    ax.grid(axis='y', color=t['grid'], linestyle='--', linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)


def add_legend(ax, theme, **kwargs):
    t = THEMES[theme]
    defaults = dict(
        fontsize=8.5,
        frameon=True,
        facecolor=t['legend_frame'],
        edgecolor='none',
        labelcolor=t['text'],
    )
    defaults.update(kwargs)
    ax.legend(**defaults)


def generate_all(draw_fn, fig_num):
    """Run draw_fn(lang, theme) for all 4 combinations."""
    for lang in ('zh', 'en'):
        for theme in ('light', 'dark'):
            draw_fn(lang, theme)
    print(f'Fig{fig_num}: all variants generated.')
