#!/usr/bin/env python3
"""Render charts from results.json (run bite_the_bullet.py first).

    python3 2-bite-the-bullet/make_charts.py   # -> charts/speedup.png, charts/ttft.png
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
BLUE, GREEN, MUTED = "#2a78d6", "#1baf7a", "#b9b8b4"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": GRID,
    "xtick.color": INK2, "ytick.color": INK2,
})

rows = DATA["datasets"]["Bursted-ART"]
rows = sorted(rows, key=lambda r: r["speedup_mean_pct"])   # ascending -> best on top in barh
labels = [r["setup"].replace("_", " ") for r in rows]
y = range(len(rows))
H = 0.38


def style(ax, spines=("top", "right")):
    for s in spines:
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)


# ------------------------------------------------------------- speedup.png ---
fig, ax = plt.subplots(figsize=(9, 4.6))
mean = [r["speedup_mean_pct"] for r in rows]
p95 = [r["speedup_p95_pct"] for r in rows]
ax.barh([i + H / 2 for i in y], mean, height=H, color=BLUE, zorder=3, label="mean TTFT")
ax.barh([i - H / 2 for i in y], p95, height=H, color=GREEN, zorder=3, label="p95 TTFT")
for i, (m, p) in enumerate(zip(mean, p95)):
    ax.text(m + 1.2, i + H / 2, f"{m:+.0f}%", va="center", ha="left", fontsize=9.5, color=INK, fontweight="bold")
    ax.text(p + 1.2, i - H / 2, f"{p:+.0f}%", va="center", ha="left", fontsize=9.5, color=INK)
ax.set_yticks(list(y))
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlim(0, max(max(mean), max(p95)) * 1.18)
ax.set_xlabel("TTFT reduction vs cache_aware (SGLang default router)")
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))
ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
ax.legend(loc="lower right", frameon=False, fontsize=10)
style(ax, ("top", "right", "left"))
ax.set_title("early_rdma cuts TTFT on Bursted-ART, across model × hardware setups",
             fontsize=12.5, color=INK, loc="left", pad=12, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "speedup.png", dpi=170, bbox_inches="tight", facecolor=SURFACE)
print("wrote", OUT / "speedup.png")

# ---------------------------------------------------------------- ttft.png ---
fig, ax = plt.subplots(figsize=(9, 4.6))
ca = [r["cache_aware_mean_ttft"] for r in rows]
bt = [r["early_rdma_mean_ttft"] for r in rows]
ax.barh([i + H / 2 for i in y], ca, height=H, color=MUTED, zorder=3, label="cache_aware")
ax.barh([i - H / 2 for i in y], bt, height=H, color=BLUE, zorder=3, label="early_rdma")


def fmt_s(v):
    return f"{v:.0f}s" if v >= 10 else f"{v:.3g}s"


for i, (c, b) in enumerate(zip(ca, bt)):
    ax.text(c * 1.18, i + H / 2, fmt_s(c), va="center", ha="left", fontsize=9, color=INK2)
    ax.text(b * 1.18, i - H / 2, fmt_s(b), va="center", ha="left", fontsize=9, color=INK, fontweight="bold")
ax.set_xscale("log")
ax.set_yticks(list(y))
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlim(min(bt) * 0.5, max(ca) * 12)
ax.set_xlabel("mean time-to-first-token (s, log scale)")
ax.grid(axis="x", color=GRID, lw=0.8, zorder=0, which="major")
ax.legend(loc="upper right", frameon=False, fontsize=10)
style(ax, ("top", "right", "left"))
ax.set_title("Mean TTFT: cache_aware vs early_rdma (Bursted-ART)",
             fontsize=12.5, color=INK, loc="left", pad=12, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "ttft.png", dpi=170, bbox_inches="tight", facecolor=SURFACE)
print("wrote", OUT / "ttft.png")
