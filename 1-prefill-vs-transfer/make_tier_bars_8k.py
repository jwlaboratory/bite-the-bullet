#!/usr/bin/env python3
"""Bare bars: prefilled-KV load cost for 8k tokens across memory tiers.

    python3 1-prefill-vs-transfer/make_tier_bars_8k.py
        -> charts/tier_bars_8k.{png,svg}

No title, no caption — just tiers on x, milliseconds on y. Same visual
language as 2-bite-the-bullet/charts/ttft.png.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / "results.json").read_text())
OUT = HERE / "charts"
OUT.mkdir(exist_ok=True)

# validated default categorical palette (light) + text/surface tokens
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE = "#2a78d6"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": GRID,
    "xtick.color": INK2, "ytick.color": INK2,
})

row = next(r for r in DATA["rows_ms"] if r["tokens"] == 8000)
tiers = [("Pre-filled\nfrom scratch", "prefill"), ("Disk", "disk"),
         ("RDMA", "rdma"), ("RAM", "ram"), ("HBM", "hbm")]
labels = [t[0] for t in tiers]
vals = [row[t[1]] for t in tiers]
x = range(len(tiers))

fig, ax = plt.subplots(figsize=(8.2, 4.8))
ax.bar(x, vals, width=0.62, color=BLUE, zorder=3)


def fmt_ms(v):
    return f"{v:.0f} ms" if v >= 10 else f"{v:.2g} ms"


for xi, v in zip(x, vals):
    ax.text(xi, v * 1.12, fmt_ms(v), va="bottom", ha="center",
            fontsize=9.5, color=INK, fontweight="bold")

ax.set_yscale("log")
ax.set_ylim(0.1, max(vals) * 2.2)
ax.set_xticks(list(x))
ax.set_xticklabels(labels, fontsize=10.5)
ax.set_ylabel("milliseconds (log scale)")
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax.grid(axis="y", color=GRID, lw=0.8, zorder=0, which="major")
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(length=0)

fig.tight_layout()
fig.savefig(OUT / "tier_bars_8k.png", dpi=170, bbox_inches="tight", facecolor=SURFACE)
fig.savefig(OUT / "tier_bars_8k.svg", bbox_inches="tight", facecolor=SURFACE)
print("wrote", OUT / "tier_bars_8k.png")
print("wrote", OUT / "tier_bars_8k.svg")
