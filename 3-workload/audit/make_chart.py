#!/usr/bin/env python3
"""Render the burst-audit figure from results/burst_audit.json.

Two panels:
  A. Largest synchronized deep-prefix fan-out (>=16 blocks ~= 8k tokens, <=10s)
     per dataset -- the headline magnitude comparison, log scale.
  B. Same-prefix fan-out vs required shared-prefix depth (60s window) -- the
     collapse curve: real traces fall off as you demand a longer shared prefix.
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

res = {r["name"]: r for r in DATA["results"]}


def deep(name):
    return res[name]["headline"]["deep_sync"]["max_fanout"]


def decay(name):
    d = res[name]["depth_stats"]
    return [d[str(k)]["windows"]["60s"]["max_fanout"] for k in (1, 2, 4, 8, 16, 32)]


plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": GRID,
    "xtick.color": INK2, "ytick.color": INK2,
})

fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.6, 5.1), gridspec_kw={"wspace": 0.28})

# ---------------------------------------------------------------- Panel A ----
# Measurable datasets (have prefix + timing).
bars = [
    ("Mooncake\n(conversation)", deep("Mooncake-conversation"), AQUA),
    ("Mooncake\n(tool-agent)", deep("Mooncake-toolagent"), AQUA),
    ("Mooncake\n(arxiv)", deep("Mooncake-arxiv"), AQUA),
    ("ART-Chat-2.5M", deep("ART-Chat-2.5M"), BLUE),
]
labels = [b[0] for b in bars]
vals = [b[1] for b in bars]
colors = [b[2] for b in bars]
y = list(range(len(bars)))[::-1]

axA.barh(y, vals, height=0.62, color=colors, zorder=3)
for yi, v in zip(y, vals):
    axA.text(v * 1.15, yi, f"{v}", va="center", ha="left", fontsize=11,
             color=INK, fontweight="bold")
axA.set_yticks(y)
axA.set_yticklabels(labels, fontsize=10)
axA.set_xscale("log")
axA.set_xlim(1, 100)
axA.xaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 25, 100]))
axA.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))
axA.set_xlabel("Largest synchronized deep-prefix fan-out\n(≥16 blocks ≈ 8k tokens, ≤10 s window)")
axA.axvline(20, color=INK2, lw=1, ls=(0, (4, 3)), zorder=2)
axA.text(20, len(bars) - 0.35, " 20-way\n threshold", fontsize=8.5, color=INK2, va="top")
axA.grid(axis="x", color=GRID, lw=0.8, zorder=0)
for s in ("top", "right", "left"):
    axA.spines[s].set_visible(False)
axA.set_title("A · What size fan-out actually occurs", fontsize=12, color=INK,
              loc="left", pad=12, fontweight="bold")

# ---------------------------------------------------------------- Panel B ----
depths = [1, 2, 4, 8, 16, 32]
xpos = list(range(len(depths)))
# (label anchor: (x_index, y_value, va) so the near-identical Mooncake tails
# don't collide -- tool-agent is labelled on its depth-8 plateau instead.)
series = [
    ("ART-Chat-2.5M", decay("ART-Chat-2.5M"), BLUE, (5, None, "center")),
    ("Mooncake (tool-agent)", decay("Mooncake-toolagent"), AQUA, (3, 206, "bottom")),
    ("Mooncake (conversation)", decay("Mooncake-conversation"), VIOLET, (5, 4, "center")),
]
for name, ys, c, (lx, ly, lva) in series:
    axB.plot(xpos, ys, color=c, lw=2, marker="o", ms=6, zorder=3,
             markeredgecolor=SURFACE, markeredgewidth=1.4)
    yval = ys[lx] if ly is None else ly
    dx = 0.12 if lx == len(depths) - 1 else 0.15
    axB.text(lx + dx, yval, name, color=c, fontsize=9.5,
             va=lva, ha="left", fontweight="bold")
axB.set_yscale("log")
axB.set_ylim(1, 600)
axB.set_xlim(-0.3, len(depths) - 0.3 + 2.9)
axB.set_xticks(xpos)
axB.set_xticklabels([f"{d}\n({d*512//1000 or '½'}k tok)" for d in depths], fontsize=9)
axB.set_xlabel("Required shared-prefix depth (blocks ≈ tokens)")
axB.set_ylabel("Max same-prefix fan-out (60 s window)")
axB.axvspan(3.5, len(depths) - 0.3 + 2.4, color=BLUE, alpha=0.045, zorder=0)
axB.text(4, 450, "“deep” regime\n(long shared context)", fontsize=8.5, color=INK2, va="top")
axB.grid(color=GRID, lw=0.8, zorder=0)
for s in ("top", "right"):
    axB.spines[s].set_visible(False)
axB.set_title("B · The bursts vanish once you require a long shared prefix",
              fontsize=12, color=INK, loc="left", pad=12, fontweight="bold")

fig.subplots_adjust(bottom=0.30, top=0.90)
fig.text(0.5, 0.055,
         "BurstGPT has no prefix hashes, and LMSYS-Chat-1M / ShareGPT have no arrival timestamps — "
         "the synchronized fan-out cannot even be expressed in those traces.",
         ha="center", fontsize=9, color=INK)
fig.text(0.5, 0.015,
         "Public traces vs. a synthetic data-labeling workload. Contiguous 300k-row slices "
         "(Mooncake read in full). Source: results/burst_audit.json",
         ha="center", fontsize=8.4, color=INK2)

out_png = HERE / "results" / "burst_audit_chart.png"
out_svg = HERE / "results" / "burst_audit_chart.svg"
fig.savefig(out_png, dpi=170, bbox_inches="tight", facecolor=SURFACE)
fig.savefig(out_svg, bbox_inches="tight", facecolor=SURFACE)
print("wrote", out_png)
print("wrote", out_svg)
