#!/usr/bin/env python3
"""Four routing-policy explainer GIFs for the BLOG "routers break" section.

Each GIF shows requests flowing left -> router -> a column of GPU nodes, with:
  * dot colour  = request PREFIX identity (same hue = same prefix)
  * node glow   = per-request OUTCOME  (green = warm cache hit, red = cold
                  prefill, blue = short/unrelated prefill)
  * a pile of little squares beside a node = requests QUEUED behind each other

Scenarios (policy x case):
  1. cache_aware_best   steady trickle, same prefix, all reuse one warm node
  2. cache_aware_worst  same-prefix burst funnels onto ONE node -> serial queue
  3. least_load_best    unrelated stream spread evenly, no hot-spot
  4. least_load_worst   same-prefix burst scattered cold -> every node prefills

Renders assets/routing/<scenario>.gif
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrow
from matplotlib.animation import FuncAnimation, PillowWriter

HERE = Path(__file__).resolve().parent

# --- palette (matches 3-workload/audit chart) --------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e1"
BLUE, AQUA, VIOLET, RED = "#2a78d6", "#1baf7a", "#4a3aa7", "#e34948"
AMBER = "#e08a1e"
IDLE = "#d9d8d3"

# prefix hues for the "unrelated stream" case
PREFIX_HUES = [BLUE, VIOLET, AMBER, AQUA, RED, "#0f9bd0"]

# outcome -> (glow colour, serve frames, short label)
OUTCOME = {
    "hit":     (AQUA, 16, "cache hit"),
    "cold":    (RED,  42, "prefill"),
    "short":   (BLUE, 15, "prefill"),
}

N_NODES = 5
FPS = 18
TRAVEL = 20          # frames for a request to fly from spawn -> node

# geometry (xlim 0-16, ylim 0-9)
SPAWN = (0.7, 4.5)
ROUTER = (6.4, 4.5)
NODE_X = 12.9
NODE_YS = [1.15 + i * 1.66 for i in range(N_NODES)]   # bottom -> top
NODE_W, NODE_H = 2.9, 1.12


def lerp(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def path_pos(spawn, node_c, t):
    """Piecewise: spawn -> router -> node, t in [0,1]."""
    if t < 0.5:
        return lerp(spawn, ROUTER, t / 0.5)
    return lerp(ROUTER, node_c, (t - 0.5) / 0.5)


def simulate(requests):
    """Given requests [{spawn, hue, node, outcome}], return per-frame render
    state plus the total frame count.

    Each node serves one request at a time; late arrivals wait in a queue.
    """
    for r in requests:
        r["arrive"] = r["spawn"] + TRAVEL
        r["serve"] = OUTCOME[r["outcome"]][1]

    # per-node discrete-event sim
    node_free = [0] * N_NODES          # frame at which node becomes free
    for r in sorted(requests, key=lambda x: x["arrive"]):
        start = max(r["arrive"], node_free[r["node"]])
        r["serve_start"] = start
        r["serve_end"] = start + r["serve"]
        node_free[r["node"]] = r["serve_end"]

    total = max(r["serve_end"] for r in requests) + 26

    frames = []
    for f in range(total):
        inflight = []           # (x, y, hue)
        node_state = [None] * N_NODES   # dict per active node
        node_queue = [[] for _ in range(N_NODES)]  # queued hues (waiting)
        for r in requests:
            if r["spawn"] <= f < r["arrive"]:
                t = (f - r["spawn"]) / TRAVEL
                x, y = path_pos(SPAWN, (NODE_X, NODE_YS[r["node"]]), t)
                inflight.append((x, y, r["hue"]))
            elif r["arrive"] <= f < r["serve_start"]:
                node_queue[r["node"]].append(r["hue"])
            elif r["serve_start"] <= f < r["serve_end"]:
                prog = (f - r["serve_start"]) / r["serve"]
                node_state[r["node"]] = {
                    "hue": r["hue"], "outcome": r["outcome"], "prog": prog,
                }
        frames.append((inflight, node_state, node_queue))
    return frames, total


def build(scenario):
    """Return (requests, title, policy, subtitle, verdict)."""
    if scenario == "cache_aware_best":
        # a few distinct prefixes, each pinned to its own already-warm node
        prefixes = [(BLUE, 2), (AMBER, 4), (VIOLET, 0)]
        reqs = []
        for w in range(4):                       # waves of near-simultaneous arrivals
            for k, (hue, node) in enumerate(prefixes):
                reqs.append(dict(spawn=w * 34 + k * 4, hue=hue, node=node,
                                 outcome="hit"))
        return (reqs, "Cache-Aware  —  best case", "CACHE-AWARE",
                "", "")
    if scenario == "cache_aware_worst":
        reqs = [dict(spawn=i * 4, hue=BLUE, node=2, outcome="hit")
                for i in range(9)]
        return (reqs, "Cache-Aware  —  worst case", "CACHE-AWARE",
                "a same-prefix BURST — affinity funnels it all onto the one node holding the prefix",
                "cache hits, but serialized — they queue behind each other while the cluster sits idle")
    if scenario == "least_load_best":
        reqs = []
        for i in range(12):
            reqs.append(dict(spawn=i * 8, hue=PREFIX_HUES[i % len(PREFIX_HUES)],
                             node=i % N_NODES, outcome="short"))
        return (reqs, "Least-Load  —  best case", "LEAST-LOAD",
                "a stream of unrelated requests — little prefix to reuse anyway",
                "spread evenly, no hot-spot — cheap short prefills in parallel")
    if scenario == "least_load_worst":
        reqs = []
        for i in range(10):
            reqs.append(dict(spawn=i * 5, hue=BLUE, node=i % N_NODES,
                             outcome="cold"))
        return (reqs, "Least-Load  —  worst case", "LEAST-LOAD",
                "a same-prefix burst SCATTERED across cold nodes — almost none hit cached KV",
                "every node pays the full prefill from scratch — the ≈44× expensive path")
    raise ValueError(scenario)


def render(scenario):
    reqs, title, policy, subtitle, verdict = build(scenario)
    frames, total = simulate(reqs)

    fig, ax = plt.subplots(figsize=(11.0, 6.2))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    fig.patch.set_facecolor(SURFACE)

    def draw(f):
        ax.clear()
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 9)
        ax.axis("off")
        inflight, node_state, node_queue = frames[f]

        # header
        ax.text(0.55, 8.55, title, fontsize=20, fontweight="bold", color=INK,
                fontfamily="DejaVu Sans", ha="left", va="center")

        # faint fabric lines router -> each node
        for y in NODE_YS:
            ax.plot([ROUTER[0] + 0.9, NODE_X - NODE_W / 2],
                    [ROUTER[1], y], color=GRID, lw=1.4, zorder=1)

        # spawn -> router feeder line
        ax.plot([SPAWN[0], ROUTER[0] - 0.9], [SPAWN[1], ROUTER[1]],
                color=GRID, lw=1.4, zorder=1)

        # router box
        rb = FancyBboxPatch((ROUTER[0] - 0.95, ROUTER[1] - 0.85), 1.9, 1.7,
                            boxstyle="round,pad=0.02,rounding_size=0.18",
                            linewidth=2, edgecolor=INK, facecolor="#ffffff",
                            zorder=3)
        ax.add_patch(rb)
        ax.text(ROUTER[0], ROUTER[1] + 0.28, "router", fontsize=12,
                fontweight="bold", ha="center", va="center", color=INK, zorder=4)
        ax.text(ROUTER[0], ROUTER[1] - 0.32, policy, fontsize=8.5,
                ha="center", va="center", color=INK2, zorder=4,
                fontweight="bold")

        # nodes
        for i, y in enumerate(NODE_YS):
            st = node_state[i]
            q = node_queue[i]
            busy = st is not None
            if busy:
                glow, _, _ = OUTCOME[st["outcome"]]
                face = glow
                alpha = 0.20 + 0.20 * (1 - abs(st["prog"] - 0.5) * 2)
                edge = glow
                lw = 2.4
            else:
                face, alpha, edge, lw = "#ffffff", 1.0, IDLE, 1.6

            nb = FancyBboxPatch((NODE_X - NODE_W / 2, y - NODE_H / 2),
                                NODE_W, NODE_H,
                                boxstyle="round,pad=0.02,rounding_size=0.14",
                                linewidth=lw, edgecolor=edge,
                                facecolor="#ffffff", zorder=3)
            ax.add_patch(nb)
            if busy:
                gb = FancyBboxPatch((NODE_X - NODE_W / 2, y - NODE_H / 2),
                                    NODE_W, NODE_H,
                                    boxstyle="round,pad=0.02,rounding_size=0.14",
                                    linewidth=0, edgecolor="none",
                                    facecolor=face, alpha=alpha, zorder=3)
                ax.add_patch(gb)

            ax.text(NODE_X - NODE_W / 2 + 0.28, y + NODE_H / 2 - 0.28,
                    f"GPU {i}", fontsize=9, color=INK2, ha="left", va="center",
                    zorder=5, fontweight="bold")

            if busy:
                _, _, lbl = OUTCOME[st["outcome"]]
                glow = OUTCOME[st["outcome"]][0]
                ax.text(NODE_X + 0.35, y - 0.02, lbl, fontsize=11,
                        color=glow, ha="center", va="center", zorder=5,
                        fontweight="bold")
                # progress bar
                bx0 = NODE_X - NODE_W / 2 + 0.28
                bw = NODE_W - 0.56
                ax.add_patch(plt.Rectangle((bx0, y - NODE_H / 2 + 0.16),
                             bw, 0.12, color=GRID, zorder=5))
                ax.add_patch(plt.Rectangle((bx0, y - NODE_H / 2 + 0.16),
                             bw * st["prog"], 0.12, color=glow, zorder=6))
            else:
                ax.text(NODE_X + 0.35, y - 0.02, "idle", fontsize=10,
                        color=IDLE, ha="center", va="center", zorder=5)

            # queued requests piling to the LEFT of the node
            for j, hue in enumerate(q):
                qx = NODE_X - NODE_W / 2 - 0.42 - j * 0.42
                ax.add_patch(Circle((qx, y), 0.17, facecolor=hue,
                             edgecolor="#ffffff", linewidth=1.2, zorder=6))
            if len(q) >= 2:
                ax.text(NODE_X - NODE_W / 2 - 0.42 - len(q) * 0.42 - 0.15, y,
                        f"queue ×{len(q)}", fontsize=9.5, color=RED,
                        ha="right", va="center", zorder=6, fontweight="bold")

        # in-flight request dots
        for (x, y, hue) in inflight:
            ax.add_patch(Circle((x, y), 0.19, facecolor=hue,
                         edgecolor="#ffffff", linewidth=1.3, zorder=7))

        # legend chips (top-right)
        chips = [(AQUA, "warm hit"), (RED, "cold prefill"), (IDLE, "idle")]
        cx = 10.4
        for col, txt in chips:
            ax.add_patch(Circle((cx, 8.5), 0.14, facecolor=col,
                         edgecolor="none", zorder=5))
            ax.text(cx + 0.22, 8.5, txt, fontsize=9.5, color=INK2, ha="left",
                    va="center", zorder=5)
            cx += 1.7
        return []

    anim = FuncAnimation(fig, draw, frames=total, interval=1000 / FPS,
                         blit=False)
    out = HERE / f"{scenario}.gif"
    anim.save(out, writer=PillowWriter(fps=FPS))
    plt.close(fig)
    print(f"wrote {out}  ({total} frames)")


# =====================================================================
#  Biting-the-Bullet (early_rdma):  detect -> pre-warm M copies -> spread
# =====================================================================
SPINE_X = NODE_X - NODE_W / 2 - 0.95   # RDMA fabric spine, left of the nodes
NODE_LEFT = NODE_X - NODE_W / 2
PULSE_TRAVEL = 26           # frames for a KV copy to cross the fabric
HIT_SERVE = OUTCOME["hit"][1]


def _seg_pos(pts, t):
    """Piecewise-linear position along waypoints pts at t in [0,1]."""
    segs = [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    lens = [((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5 for a, b in segs]
    total = sum(lens) or 1.0
    d = t * total
    for (a, b), L in zip(segs, lens):
        if L == 0:
            continue
        if d <= L:
            fr = d / L
            return (a[0] + (b[0] - a[0]) * fr, a[1] + (b[1] - a[1]) * fr)
        d -= L
    return pts[-1]


def _node_box(ax, y, edge, lw, fill=None, alpha=1.0):
    ax.add_patch(FancyBboxPatch(
        (NODE_X - NODE_W / 2, y - NODE_H / 2), NODE_W, NODE_H,
        boxstyle="round,pad=0.02,rounding_size=0.14", linewidth=lw,
        edgecolor=edge, facecolor="#ffffff", zorder=3))
    if fill is not None:
        ax.add_patch(FancyBboxPatch(
            (NODE_X - NODE_W / 2, y - NODE_H / 2), NODE_W, NODE_H,
            boxstyle="round,pad=0.02,rounding_size=0.14", linewidth=0,
            edgecolor="none", facecolor=fill, alpha=alpha, zorder=3))


def render_btb():
    prefix_node = 2                       # already holds the prefix
    warm_targets = [0, 1, 3, 4]           # M=4 replicas BTB warms
    warm_fire = 46                        # BTB fires after the 2nd repeat
    ready_frame = warm_fire + 6 + PULSE_TRAVEL   # copies resident
    burst_start = ready_frame + 6

    # requests: 2 detection repeats (the trigger) then the spread burst
    reqs = [dict(spawn=4, hue=BLUE, node=prefix_node),
            dict(spawn=22, hue=BLUE, node=prefix_node)]
    spread = warm_targets + [prefix_node]
    for i in range(10):
        reqs.append(dict(spawn=burst_start + i * 4, hue=BLUE,
                         node=spread[i % len(spread)]))

    # per-node serve scheduling (spread burst -> ~no queue)
    for r in reqs:
        r["arrive"] = r["spawn"] + TRAVEL
    node_free = [0] * N_NODES
    for r in sorted(reqs, key=lambda x: x["arrive"]):
        s = max(r["arrive"], node_free[r["node"]])
        r["serve_start"], r["serve_end"] = s, s + HIT_SERVE
        node_free[r["node"]] = r["serve_end"]
    det_ready = reqs[0]["serve_start"] + 2      # prefix node warm after 1st hit

    # warm-copy pulses RDMA -> each target
    pulses = [dict(node=n, t0=warm_fire + p * 6)
              for n in warm_targets for p in range(2)]

    total = max(r["serve_end"] for r in reqs) + 26

    fig, ax = plt.subplots(figsize=(11.0, 6.2))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.patch.set_facecolor(SURFACE)

    def draw(f):
        ax.clear()
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 9)
        ax.axis("off")

        ax.text(0.55, 8.55, "Biting the Bullet  —  predictive warming",
                fontsize=20, fontweight="bold", color=INK, ha="left",
                va="center")

        # phase-driven router status
        if f < warm_fire:
            status, scol = "watching prefixes…", INK2
        elif f < burst_start:
            status, scol = "burst predicted → replicate ×4", AMBER
        else:
            status, scol = "burst spread → all warm", AQUA

        # fabric lines router -> nodes
        for y in NODE_YS:
            ax.plot([ROUTER[0] + 0.9, NODE_X - NODE_W / 2],
                    [ROUTER[1], y], color=GRID, lw=1.4, zorder=1)
        ax.plot([SPAWN[0], ROUTER[0] - 0.9], [SPAWN[1], ROUTER[1]],
                color=GRID, lw=1.4, zorder=1)

        # RDMA fabric: a spine left of the node column + per-node stubs.
        # The prefix-holder (GPU2) pushes its KV out over this fabric to peers.
        warming_live = warm_fire <= f < ready_frame
        ys = [NODE_YS[i] for i in range(N_NODES)]
        ax.plot([SPINE_X, SPINE_X], [min(ys), max(ys)],
                color=AMBER if warming_live else GRID,
                lw=2.2 if warming_live else 1.2,
                alpha=0.7 if warming_live else 0.45, zorder=1)
        for i in range(N_NODES):
            hot = warming_live and (i in warm_targets or i == prefix_node)
            ax.plot([SPINE_X, NODE_LEFT], [NODE_YS[i], NODE_YS[i]],
                    color=AMBER if hot else GRID,
                    lw=1.8 if hot else 1.0,
                    alpha=0.6 if hot else 0.45, zorder=1)
        if warming_live:
            ax.text(SPINE_X - 0.18, (min(ys) + max(ys)) / 2, "RDMA fabric",
                    fontsize=9, color=AMBER, rotation=90, ha="center",
                    va="center", fontweight="bold", zorder=2)

        # router
        ax.add_patch(FancyBboxPatch(
            (ROUTER[0] - 0.95, ROUTER[1] - 0.85), 1.9, 1.7,
            boxstyle="round,pad=0.02,rounding_size=0.18", linewidth=2,
            edgecolor=INK, facecolor="#ffffff", zorder=3))
        ax.text(ROUTER[0], ROUTER[1] + 0.34, "router", fontsize=12,
                fontweight="bold", ha="center", va="center", color=INK,
                zorder=4)
        ax.text(ROUTER[0], ROUTER[1] - 0.02, "BTB", fontsize=8.5,
                ha="center", va="center", color=INK2, fontweight="bold",
                zorder=4)
        ax.text(ROUTER[0], ROUTER[1] - 0.52, status, fontsize=8.2,
                ha="center", va="center", color=scol, fontweight="bold",
                zorder=4)

        # nodes
        for i, y in enumerate(NODE_YS):
            serving = next((r for r in reqs
                            if r["node"] == i
                            and r["serve_start"] <= f < r["serve_end"]), None)
            warming = (i in warm_targets) and (warm_fire <= f < ready_frame)
            is_source = (i == prefix_node and warm_fire <= f < ready_frame
                         and serving is None)
            ready = ((i in warm_targets and f >= ready_frame)
                     or (i == prefix_node and f >= det_ready))

            if serving is not None:
                prog = (f - serving["serve_start"]) / HIT_SERVE
                a = 0.20 + 0.20 * (1 - abs(prog - 0.5) * 2)
                _node_box(ax, y, AQUA, 2.4, fill=AQUA, alpha=a)
                lbl, lcol = "cache hit", AQUA
                bx0 = NODE_X - NODE_W / 2 + 0.28
                bw = NODE_W - 0.56
                ax.add_patch(plt.Rectangle((bx0, y - NODE_H / 2 + 0.16),
                             bw, 0.12, color=GRID, zorder=5))
                ax.add_patch(plt.Rectangle((bx0, y - NODE_H / 2 + 0.16),
                             bw * prog, 0.12, color=AQUA, zorder=6))
            elif is_source:
                _node_box(ax, y, AMBER, 2.4, fill=AQUA, alpha=0.10)
                lbl, lcol = "sending KV →", AMBER
            elif warming:
                prog = (f - warm_fire) / (ready_frame - warm_fire)
                a = 0.16 + 0.16 * (1 - abs(prog - 0.5) * 2)
                _node_box(ax, y, AMBER, 2.2, fill=AMBER, alpha=a)
                lbl, lcol = "warming…", AMBER
            elif ready:
                _node_box(ax, y, AQUA, 1.8, fill=AQUA, alpha=0.07)
                lbl, lcol = "warm ✓", AQUA
            else:
                _node_box(ax, y, IDLE, 1.6)
                lbl, lcol = "idle", IDLE

            ax.text(NODE_X - NODE_W / 2 + 0.28, y + NODE_H / 2 - 0.28,
                    f"GPU {i}", fontsize=9, color=INK2, ha="left",
                    va="center", zorder=5, fontweight="bold")
            ax.text(NODE_X + 0.35, y - 0.02, lbl, fontsize=11, color=lcol,
                    ha="center", va="center", zorder=5, fontweight="bold")

        # KV-copy pulses: prefix-holder (GPU2) -> fabric spine -> peer HBM
        y_src = NODE_YS[prefix_node]
        for p in pulses:
            if p["t0"] <= f < p["t0"] + PULSE_TRAVEL:
                t = (f - p["t0"]) / PULSE_TRAVEL
                pts = [(NODE_LEFT, y_src), (SPINE_X, y_src),
                       (SPINE_X, NODE_YS[p["node"]]),
                       (NODE_LEFT, NODE_YS[p["node"]])]
                x, y = _seg_pos(pts, t)
                ax.add_patch(Circle((x, y), 0.13, facecolor=AMBER,
                             edgecolor="#ffffff", linewidth=1.0, zorder=6))

        # in-flight request dots
        for r in reqs:
            if r["spawn"] <= f < r["arrive"]:
                t = (f - r["spawn"]) / TRAVEL
                x, y = path_pos(SPAWN, (NODE_X, NODE_YS[r["node"]]), t)
                ax.add_patch(Circle((x, y), 0.19, facecolor=r["hue"],
                             edgecolor="#ffffff", linewidth=1.3, zorder=7))

        # legend chips
        chips = [(AQUA, "warm hit"), (AMBER, "warming copy"), (IDLE, "idle")]
        cx = 9.9
        for col, txt in chips:
            ax.add_patch(Circle((cx, 8.5), 0.14, facecolor=col,
                         edgecolor="none", zorder=5))
            ax.text(cx + 0.22, 8.5, txt, fontsize=9.5, color=INK2,
                    ha="left", va="center", zorder=5)
            cx += 1.95
        return []

    anim = FuncAnimation(fig, draw, frames=total, interval=1000 / FPS,
                         blit=False)
    out = HERE / "biting_the_bullet.gif"
    anim.save(out, writer=PillowWriter(fps=FPS))
    plt.close(fig)
    print(f"wrote {out}  ({total} frames)")


if __name__ == "__main__":
    for s in ("cache_aware_best", "cache_aware_worst",
              "least_load_best", "least_load_worst"):
        render(s)
    render_btb()
