"""MkDocs hook: generate index page figures before build.

Skips scripts whose output SVGs are all newer than the source files,
so that `mkdocs serve` doesn't enter an infinite rebuild loop.
"""

import subprocess
import sys
import os
import re

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")
LANGS = ("zh", "en")
THEMES = ("light", "dark")


def _output_svgs(script_name):
    """Return expected output paths for a script like 'index.fig1.py'."""
    m = re.search(r"index\.fig(\d+)\.py", script_name)
    if not m:
        return []
    fig_num = m.group(1)
    return [
        os.path.join(ASSETS_DIR, f"index.fig{fig_num}.{lang}.{theme}.svg")
        for lang in LANGS
        for theme in THEMES
    ]


def _needs_rebuild(script_path, common_path, svg_paths):
    """Check if any output SVG is missing or older than sources."""
    src_mtime = max(os.path.getmtime(script_path), os.path.getmtime(common_path))
    for svg in svg_paths:
        try:
            if os.path.getmtime(svg) < src_mtime:
                return True
        except OSError:
            return True
    return False


def on_pre_build(config, **kwargs):
    draw_dir = os.path.dirname(__file__)
    common_path = os.path.join(draw_dir, "common.py")
    scripts = sorted(
        f
        for f in os.listdir(draw_dir)
        if f.startswith("index.fig") and f.endswith(".py")
    )
    for f in scripts:
        script = os.path.join(draw_dir, f)
        svgs = _output_svgs(f)
        if not svgs or not _needs_rebuild(script, common_path, svgs):
            continue
        print(f"  [draw] Generating {f} ...")
        subprocess.check_call([sys.executable, script])
