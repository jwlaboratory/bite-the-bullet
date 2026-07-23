#!/usr/bin/env python3
"""Single-panel burst-audit figure: largest synchronized deep-prefix fan-out per
dataset, with a production-scale reference so the reader sees that no public
trace clears even a 100-way fan-out — while a real data-labeling / sub-agent
sweep (our Bursted-ART) is a 500-way burst.

Renders results/burst_audit_bars.{png,svg} from results/burst_audit.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter

HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / "results" / "burst_audit.json").read_text())

# --- validated default categorical palette (light mode) ---------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e1"
BLUE, AQUA, VIOLET, RED = "#2a78d6", "#1baf7a", "#4a3aa7", "#e34948"
AMBER = "#e08a1e"          # "ours" / production-scale highlight

res = {r["name"]: r for r in DATA["results"]}


def deep(name):
    return res[name]["headline"]["deep_sync"]["max_fanout"]


# Bursted-ART burst size (requests per synthetic same-prefix fan-out job).
BURSTED_ART_FANOUT = 500
PROD_THRESHOLD = 100       # a "real workload" starts here; no public trace reaches it

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": GRID,
    "xtick.color": INK2, "ytick.color": INK2,
})

fig, ax = plt.subplots(figsize=(10.8, 5.2))

# rows, bottom -> top
bars = [
    ("Mooncake (conversation)", deep("Mooncake-conversation"), AQUA, False),
    ("Mooncake (tool-agent)", deep("Mooncake-toolagent"), AQUA, False),
    ("Mooncake (arxiv)", deep("Mooncake-arxiv"), AQUA, False),
    ("ART-Chat-2.5M", deep("ART-Chat-2.5M"), BLUE, False),
    ("Bursted-ART  (ours)", BURSTED_ART_FANOUT, AMBER, True),
]
labels = [b[0] for b in bars]
vals = [b[1] for b in bars]
colors = [b[2] for b in bars]
y = list(range(len(bars)))

# production-scale band (>= 100-way) + threshold line
ax.axvspan(PROD_THRESHOLD, 2000, color=AMBER, alpha=0.07, zorder=0)
ax.axvline(PROD_THRESHOLD, color=AMBER, lw=1.4, ls=(0, (5, 3)), zorder=2)
ax.text(PROD_THRESHOLD * 1.12, 1.55,
        "production scale\n(real labeling / sub-agent\nfan-out — hundreds to\nthousands of requests)",
        fontsize=9.5, color="#9a6410", va="center", ha="left", fontweight="bold")

# 20-way "does it even qualify as a burst" threshold
ax.axvline(20, color=INK2, lw=1, ls=(0, (3, 3)), zorder=2)
ax.text(20, -0.66, "20-way\nthreshold", fontsize=8.5, color=INK2,
        va="top", ha="center")

bar_h = 0.6
ax.barh(y, vals, height=bar_h, color=colors, zorder=3,
        edgecolor=[INK if b[3] else "none" for b in bars],
        linewidth=[1.2 if b[3] else 0 for b in bars])
for yi, v, b in zip(y, vals, bars):
    ax.text(v * 1.12, yi, f"{v}", va="center", ha="left", fontsize=12,
            color=INK, fontweight="bold")

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=10.5)
ax.set_xscale("log")
ax.set_xlim(1, 2000)
ax.set_ylim(-0.9, len(bars) - 0.2)
ax.xaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 25, 100, 500, 1000]))
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))
ax.set_xlabel("Largest synchronized deep-prefix fan-out\n(≥16 blocks ≈ 8k tokens shared, ≤10 s window)  ·  log scale")
ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)

fig.text(0.035, 0.955, "No public trace reaches production burst scale",
         fontsize=16, color=INK, ha="left", va="top", fontweight="bold")
fig.text(0.035, 0.895,
         "Every public serving trace tops out at a 25-way fan-out — none clears the 100-way line a real "
         "data-labeling or\nsub-agent sweep sits well above. BurstGPT (no prefix hashes) and LMSYS-Chat-1M / "
         "ShareGPT (no timestamps)\ncan't express the pattern at all.",
         fontsize=9.5, color=INK2, ha="left", va="top")

fig.subplots_adjust(left=0.205, right=0.975, top=0.775, bottom=0.185)

out_png = HERE / "results" / "burst_audit_bars.png"
out_svg = HERE / "results" / "burst_audit_bars.svg"
fig.savefig(out_png, dpi=190, facecolor=SURFACE)
fig.savefig(out_svg, facecolor=SURFACE)
print("wrote", out_png)
print("wrote", out_svg)
