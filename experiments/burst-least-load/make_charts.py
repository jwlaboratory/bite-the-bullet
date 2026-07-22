"""Render the big-burst charts from results/burst_results.json.

Charts (written to charts/):
  1. burst_timeline.png   -- cluster queue depth over time: the burst builds a
                             deep backlog and drains (least_load vs cache_aware).
  2. ttft_gap.png         -- the punchline: least_load vs cache_aware TTFT in a
                             cluster WITH vs WITHOUT a shared KV cache.
  3. metrics_by_condition.png -- small multiples of every headline metric across
                             all four policies and both conditions.
  4. cache_tier.png       -- where prefix reuse comes from per policy: cache_aware
                             keeps ~2x more of it in free local HBM.

Palette: the data-viz reference instance (validated categorical slots).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

import run_burst as rb

HERE = os.path.dirname(os.path.abspath(__file__))
CHARTS = os.path.join(HERE, "charts")
os.makedirs(CHARTS, exist_ok=True)

# ---- validated reference palette (light mode) ------------------------------
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8984"
SURFACE = "#fcfcfb"
GRID = "#e6e5e1"
POLICY_COLOR = {          # categorical slots, fixed order
    "least_load": "#2a78d6",   # blue
    "cache_aware": "#eb6834",  # orange
    "round_robin": "#1baf7a",  # aqua
    "random": "#4a3aa7",       # violet
}
POLICY_LABEL = {p: p for p in POLICY_COLOR}
TIER_COLOR = {"hbm": "#2a78d6", "rdma": "#1baf7a", "ram": "#eda100",
              "disk": "#e87ba4", "miss": "#e34948"}
TIER_LABEL = {"hbm": "local HBM (free)", "rdma": "peer RDMA", "ram": "host RAM",
              "disk": "disk", "miss": "recompute (cold)"}
COND_LABEL = {"shared": "shared, contention-free",
              "congested": "shared, fabric contends",
              "isolated": "isolated replicas"}
COND_SHORT = {"shared": "shared", "congested": "congested", "isolated": "isolated"}
COND_HATCH = {"shared": "", "congested": "", "isolated": "///"}
COND_ALPHA = {"shared": 1.0, "congested": 0.62, "isolated": 0.4}
CONDITIONS = ["shared", "congested", "isolated"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 11,
    "font.family": "DejaVu Sans", "text.color": INK,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1,
})


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def load():
    with open(os.path.join(HERE, "results", "burst_results.json")) as f:
        return json.load(f)


def summ(data, condition, policy):
    for s in data["conditions"][condition]["summaries"]:
        if s["policy"] == policy:
            return s
    raise KeyError(policy)


# ----------------------------------------------------------------- chart 1
def chart_timeline(data):
    ev = data["conditions"]["shared"]["events"]
    nodes = data["conditions"]["shared"]["nodes"]
    cfg = data["config"]
    fig, ax = plt.subplots(figsize=(10, 5.2))

    t0 = min(e["arrival"] for pol in ev.values() for e in pol)
    t1 = max(e["finish"] for pol in ev.values() for e in pol) + 2
    peak = 0
    for policy in ("least_load", "cache_aware"):
        grid, total, _ = rb.queue_depth_timeline(ev[policy], nodes, t0, t1, step=0.25)
        ax.plot(grid - t0, total, color=POLICY_COLOR[policy], lw=2,
                label=POLICY_LABEL[policy])
        peak = max(peak, int(total.max()))

    ax.set_ylim(0, peak * 1.12)
    ax.set_xlim(0, t1 - t0)
    # shade the arrival window of the burst
    ax.axvspan(0, cfg["burst_window_s"], color="#2a78d6", alpha=0.07, lw=0)
    ax.annotate(f"peak backlog ≈ {peak:,} requests",
                (cfg["burst_window_s"], peak),
                textcoords="offset points", xytext=(10, -2), va="center",
                color=INK, fontsize=10, fontweight="bold")
    ax.annotate(f"{cfg['burst_size']:,} requests arrive in "
                f"{cfg['burst_window_s']:.0f}s",
                (cfg["burst_window_s"], peak * 0.62),
                textcoords="offset points", xytext=(12, 0), va="center",
                color=MUTED, fontsize=9.5)
    ax.annotate("", xy=(cfg["burst_window_s"], peak * 0.62),
                xytext=(cfg["burst_window_s"] + (t1 - t0) * 0.16, peak * 0.62),
                arrowprops=dict(arrowstyle="-", color=GRID, lw=1))

    style(ax)
    ax.set_xlabel("time since first arrival (s)")
    ax.set_ylabel("requests waiting in queue (cluster-wide)")
    ax.set_title("A big burst builds a deep backlog that takes minutes to drain",
                 color=INK, fontsize=13, fontweight="bold", loc="left", pad=12)
    ax.legend(frameon=False, loc="upper right")
    fig.text(0.125, 0.005,
             f"{cfg['num_families']} agent prefixes x {cfg['prefix_tokens']} tok  |  "
             f"4x H100 nodes  |  RDMA-shared cache condition",
             color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(CHARTS, "burst_timeline.png"), dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------- chart 2
def chart_ttft_gap(data):
    """The punchline: least_load vs cache_aware TTFT, WITH vs WITHOUT a shared
    KV cache. Sharing makes the routing choice almost free; without it, cache
    affinity is worth ~23%."""
    conds = CONDITIONS
    pols = ["least_load", "cache_aware"]
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    x = np.arange(len(conds))
    w = 0.34
    for k, p in enumerate(pols):
        off = (k - 0.5) * w
        vals = [summ(data, c, p)["mean_ttft"] for c in conds]
        ax.bar(x + off, vals, w, color=POLICY_COLOR[p], edgecolor=SURFACE,
               linewidth=1.5, label=POLICY_LABEL[p], zorder=3)
        for xi, v in zip(x + off, vals):
            ax.text(xi, v + 1.5, f"{v:.0f}s", ha="center", va="bottom",
                    fontsize=10.5, color=INK, fontweight="bold")
    # gap annotation per condition
    for xi, c in zip(x, conds):
        ll = summ(data, c, "least_load")["mean_ttft"]
        ca = summ(data, c, "cache_aware")["mean_ttft"]
        gap = (ll - ca) / ll * 100
        top = max(ll, ca)
        msg = "routing ≈ even" if abs(gap) < 2 else f"cache_aware {gap:.0f}% faster"
        ax.text(xi, top + 9, msg, ha="center", va="bottom", fontsize=10,
                color=INK2, style="italic")
    style(ax)
    ax.set_xticks(x)
    ax.set_xticklabels([COND_LABEL[c].replace(", ", ",\n") for c in conds],
                       fontsize=10.5)
    ax.set_ylabel("mean time to first token (s)")
    ax.set_ylim(0, max(summ(data, "isolated", "least_load")["mean_ttft"],
                       summ(data, "isolated", "cache_aware")["mean_ttft"]) * 1.28)
    ax.set_title("The cheaper KV sharing is, the less routing matters (4-node cluster)",
                 color=INK, fontsize=13, fontweight="bold", loc="left", pad=12)
    ax.legend(frameon=False, loc="upper left")
    fig.text(0.125, 0.005,
             "Contention (middle) is a mild 4-node penalty that grows with cluster "
             "size (see scaling.png); isolated replicas (right) recompute cold prefixes.",
             color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(CHARTS, "ttft_gap.png"), dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------- chart 3
def chart_metrics(data):
    metrics = [
        ("mean_queue", "mean queue wait (s)", "{:.0f}"),
        ("peak_queue", "peak queue wait (s)", "{:.0f}"),
        ("mean_ttft", "mean TTFT (s)", "{:.0f}"),
        ("mean_prefill", "mean prefix+prefill (s)", "{:.2f}"),
        ("max_depth_node", "max queue depth (1 node)", "{:.0f}"),
        ("throughput", "throughput (tok/s)", "{:.0f}"),
    ]
    policies = rb.POLICIES
    conds = CONDITIONS
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.6))
    x = np.arange(len(policies))
    w = 0.27
    for ax, (key, label, fmt) in zip(axes.flat, metrics):
        for j, cond in enumerate(conds):
            off = (j - 1) * w
            vals = [summ(data, cond, p)[key] for p in policies]
            ax.bar(x + off, vals, w,
                   color=[POLICY_COLOR[p] for p in policies],
                   alpha=COND_ALPHA[cond],
                   edgecolor=SURFACE, linewidth=1.0, hatch=COND_HATCH[cond])
            for xi, v in zip(x + off, vals):
                ax.text(xi, v, fmt.format(v), ha="center", va="bottom",
                        fontsize=6.3, color=INK2, rotation=90)
        style(ax)
        ax.set_xticks(x)
        ax.set_xticklabels([p.replace("_", "\n") for p in policies], fontsize=8)
        ax.set_title(label, color=INK, fontsize=10.5, fontweight="bold", loc="left")
        ax.margins(y=0.24)
    handles = [Patch(facecolor=MUTED, alpha=COND_ALPHA[c], hatch=COND_HATCH[c],
                     label=COND_LABEL[c]) for c in conds]
    fig.legend(handles=handles, frameon=False, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, -0.005), fontsize=10)
    fig.suptitle("Congestion is a mild tax on 4 nodes; isolated replicas is where "
                 "cache_aware clearly wins", color=INK, fontsize=13,
                 fontweight="bold", x=0.008, ha="left", y=0.99)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(os.path.join(CHARTS, "metrics_by_condition.png"), dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------- chart 5
def chart_scaling(data):
    """RDMA incast fan-in grows with node count, so the congestion penalty on
    cache-blind routing should widen with cluster size."""
    rows = data.get("scaling", [])
    if not rows:
        return
    nodes = [r["nodes"] for r in rows]
    x = np.arange(len(nodes))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 5.2),
                                  gridspec_kw={"width_ratios": [1.35, 1]})

    ll = [r["ll_ttft"] for r in rows]
    ca = [r["ca_ttft"] for r in rows]
    ax.plot(x, ll, "-o", color=POLICY_COLOR["least_load"], lw=2.2, ms=8,
            label="least_load")
    ax.plot(x, ca, "-o", color=POLICY_COLOR["cache_aware"], lw=2.2, ms=8,
            label="cache_aware")
    for xi, a, b in zip(x, ll, ca):
        ax.annotate(f"{a:.0f}s", (xi, a), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9,
                    color=POLICY_COLOR["least_load"], fontweight="bold")
        ax.annotate(f"{b:.0f}s", (xi, b), textcoords="offset points",
                    xytext=(0, -16), ha="center", fontsize=9,
                    color=POLICY_COLOR["cache_aware"], fontweight="bold")
    style(ax)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n} nodes" for n in nodes])
    ax.set_ylabel("mean time to first token (s)")
    ax.set_ylim(bottom=0)
    ax.set_title("Congested fabric: cache_aware pulls away at scale",
                 color=INK, fontsize=12, fontweight="bold", loc="left", pad=10)
    ax.legend(frameon=False, loc="lower left")

    gap = [r["gap_pct"] for r in rows]
    bars = ax2.bar(x, gap, 0.6, color=POLICY_COLOR["cache_aware"],
                   edgecolor=SURFACE, linewidth=1.2)
    for xi, g in zip(x, gap):
        ax2.text(xi, g + 0.6, f"{g:.0f}%", ha="center", va="bottom",
                 fontsize=10, color=INK, fontweight="bold")
    style(ax2)
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(n) for n in nodes])
    ax2.set_xlabel("cluster size (nodes)")
    ax2.set_ylabel("cache_aware TTFT advantage (%)")
    ax2.set_ylim(0, max(gap) * 1.25)
    ax2.set_title("Gap widens with fan-in", color=INK, fontsize=12,
                  fontweight="bold", loc="left", pad=10)
    fig.text(0.5, 0.005,
             "Burst scaled with the cluster to hold per-node load constant; "
             "RDMA congestion on. Fewer nodes = less incast = smaller gap.",
             color=MUTED, fontsize=9, ha="center")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(CHARTS, "scaling.png"), dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------- chart 4
def chart_cache_tier(data):
    """Per-policy reuse-source composition in the shared-cache cluster. The
    honest signal is BETWEEN policies (cache_aware keeps ~2x more reuse in free
    local HBM); the isolated condition is omitted because the simulator records
    the source it *picked*, not whether slow bandwidth forced a recompute."""
    policies = rb.POLICIES
    order = ["hbm", "rdma", "ram", "disk", "miss"]
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    seen = set()
    for i, p in enumerate(policies):
        tiers = summ(data, "shared", p)["tiers"]
        bottom = 0.0
        for t in order:
            v = tiers.get(t, 0.0) * 100
            if v <= 0:
                continue
            seen.add(t)
            ax.bar(i, v, 0.62, bottom=bottom, color=TIER_COLOR[t],
                   edgecolor=SURFACE, linewidth=1.4)
            if t == "hbm":
                ax.text(i, v / 2, f"{v:.0f}%", ha="center", va="center",
                        color="white", fontsize=11, fontweight="bold")
            bottom += v
    style(ax)
    ax.set_xticks(range(len(policies)))
    ax.set_xticklabels([POLICY_LABEL[p] for p in policies], fontsize=11)
    ax.set_ylabel("share of requests (%)")
    ax.set_ylim(0, 100)
    handles = [Patch(facecolor=TIER_COLOR[t], label=TIER_LABEL[t])
               for t in order if t in seen]
    ax.legend(handles=handles, frameon=False, loc="lower center",
              bbox_to_anchor=(0.5, 1.01), ncol=len(handles), fontsize=9.5)
    ax.set_title("cache_aware keeps ~2x more prefix reuse in free local HBM",
                 color=INK, fontsize=13, fontweight="bold", loc="left", pad=30)
    fig.text(0.125, 0.005,
             "RDMA-shared cluster. White number = local-HBM (free) share; "
             "least_load pushes the rest onto peer RDMA / host RAM.",
             color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(CHARTS, "cache_tier.png"), dpi=150)
    plt.close(fig)


def main():
    data = load()
    chart_timeline(data)
    chart_ttft_gap(data)
    chart_metrics(data)
    chart_cache_tier(data)
    chart_scaling(data)
    print("wrote charts/: burst_timeline.png, ttft_gap.png, "
          "metrics_by_condition.png, cache_tier.png, scaling.png")


if __name__ == "__main__":
    main()
