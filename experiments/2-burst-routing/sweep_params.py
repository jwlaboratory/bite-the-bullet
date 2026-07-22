#!/usr/bin/env python3
"""Sweep the four early_rdma constants (X, Y, Z, M) for the best fixed values.

Regime: a long shared prefix (prefill-dominated), a fast burst, and a REALISTIC
baseline -- ADMIT_RDMA=False, so a node that misses the prefix recomputes it
rather than opportunistically stealing a peer's KV (real SGLang behavior; there
is no automatic peer-to-peer pull on the least-load / cache-aware path). BTB's
warming push still runs at full RDMA bandwidth -- only the baseline reuse path is
gated. Without warming, cache_aware piles the hot family on the source node and
least_load scatters it onto cold nodes that must recompute; early_rdma pre-warms
M copies so later requests land warm.

    X = THRESHOLD      repeats to fire        grid: 2, 4, 8
    Y = PREFIX_BLOCKS  prefix matched+copied  grid: 32, 64, 128 (of a 128-block prefix)
    Z = WINDOW_S       detection window (s)   grid: 1, 2, 4
    M = WARM_COPIES    HBM copies to warm     grid: 2, 4, 6, 8

Objective: mean-TTFT improvement over the best baseline (recompute-on-miss
cache_aware / least_load), on an 8-node cluster.

Run:  INFERENCE_SIM_ROOT=../../../inference-sim python3 sweep_params.py
"""
from __future__ import annotations

import itertools
import json
import math
import os
import random
import sys
from types import SimpleNamespace

SIM_ROOT = os.environ.get(
    "INFERENCE_SIM_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../inference-sim")),
)
if not os.path.isdir(SIM_ROOT):
    sys.exit(f"inference-sim not found at {SIM_ROOT} (set INFERENCE_SIM_ROOT)")
sys.path.insert(0, SIM_ROOT)

import config                       # noqa: E402
from simulate import run            # noqa: E402
from workload import Request        # noqa: E402
import btb_policy                   # noqa: E402
btb_policy.register()

HERE = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------- workload
BLOCK = 256                 # tokens/block (must match cfg.BLOCK_TOKENS)
PREFIX_BLOCKS = 128         # shared prefix: 128 * 256 = 32,768 tokens (prefill-heavy)
USER_BLOCKS = 1             # unique per-request suffix
OUTPUT_TOKENS = 1           # decode-negligible -> prefill dominates TTFT
BURST_SIZE = 400            # hot-family requests
BURST_WINDOW = 8.0          # arrive over 8 s -> copies have lead time to pay back
SEED = 42
NODES = 8                   # so M can range up to 8

# ---------------------------------------------------------------------- grids
GRID_X = [2, 4, 8]
GRID_Y = [32, 64, 128]
GRID_Z = [1.0, 2.0, 4.0]
GRID_M = [2, 4, 6, 8]


def make_workload():
    """One hot family with a long shared prefix, bursting over BURST_WINDOW.
    Two calm warmup requests seed a source HBM copy before the burst."""
    rng = random.Random(SEED)
    prefix = [f"sys#{b}" for b in range(PREFIX_BLOCKS)]
    reqs = []

    def add(arrival, uid):
        blocks = prefix + [f"u{uid}#{b}" for b in range(USER_BLOCKS)]
        inp = len(blocks) * BLOCK
        extra = math.ceil((inp + OUTPUT_TOKENS) / BLOCK) - len(blocks)
        cache_blocks = blocks + [f"u{uid}#o{j}" for j in range(max(0, extra))]
        reqs.append(Request(0, arrival, "hot", inp, inp, OUTPUT_TOKENS, blocks, cache_blocks))

    add(0.0, 0)                       # warmup: seed the source node's HBM
    add(1.0, 1)
    for i in range(BURST_SIZE):       # the burst
        add(4.0 + rng.uniform(0.0, BURST_WINDOW), 100 + i)

    reqs.sort(key=lambda r: r.arrival)
    for i, r in enumerate(reqs):
        r.id = i
    return reqs


def base_cfg():
    cfg = SimpleNamespace(**config.as_dict())
    cfg.BLOCK_TOKENS = BLOCK
    cfg.RDMA_CONGESTION = False
    # Realistic SGLang model: a miss recomputes -- no opportunistic peer-to-peer
    # KV steal (ADMIT_RDMA off) and no shared pool (DISK_CACHE off). BTB's warming
    # push still uses full RDMA bandwidth; only the baseline reuse path is gated.
    cfg.ADMIT_RDMA = False
    cfg.DISK_CACHE = False
    spec, gpus = cfg.CLUSTER[0][1], cfg.CLUSTER[0][2]
    cfg.CLUSTER = [(f"node{i}", spec, gpus) for i in range(NODES)]
    return cfg


def run_seeded(policy, requests, cfg):
    random.seed(SEED)
    return run(policy, requests, cfg)


def main():
    requests = make_workload()
    print(f"workload: {len(requests)} reqs, {PREFIX_BLOCKS * BLOCK}-tok shared prefix, "
          f"{OUTPUT_TOKENS}-tok output, burst over {BURST_WINDOW}s, {NODES} nodes\n")

    ca = run_seeded("cache_aware", requests, base_cfg())["metrics"]
    ll = run_seeded("least_load", requests, base_cfg())["metrics"]
    # BTB must beat the *best* non-BTB policy, not just the weaker one.
    base_ttft = min(ca["mean_ttft"], ll["mean_ttft"])
    base_name = "cache_aware" if ca["mean_ttft"] <= ll["mean_ttft"] else "least_load"
    print(f"baselines: cache_aware {ca['mean_ttft']:.3f}s  least_load {ll['mean_ttft']:.3f}s"
          f"  -> compare vs best ({base_name} {base_ttft:.3f}s)\n")

    rows = []
    for x, y, z, m in itertools.product(GRID_X, GRID_Y, GRID_Z, GRID_M):
        cfg = base_cfg()
        cfg.BTB_THRESHOLD = x
        cfg.BTB_PREFIX_BLOCKS = y
        cfg.BTB_WINDOW_S = z
        cfg.BTB_WARM_COPIES = m
        met = run_seeded("early_rdma", requests, cfg)["metrics"]
        impr = (base_ttft - met["mean_ttft"]) / base_ttft * 100.0
        rows.append({
            "X": x, "Y": y, "Z": z, "M": m,
            "mean_ttft": met["mean_ttft"], "improvement_pct": impr,
            "p95_lat": met.get("p95_lat"),
        })

    rows.sort(key=lambda r: r["improvement_pct"], reverse=True)
    print(f"{'rank':>4}  {'X':>2} {'Y':>4} {'Z':>4} {'M':>2}  "
          f"{'mean_ttft':>10} {'vs base':>9}")
    for i, r in enumerate(rows[:12], 1):
        print(f"{i:>4}  {r['X']:>2} {r['Y']:>4} {r['Z']:>4.0f} {r['M']:>2}  "
              f"{r['mean_ttft']:>9.3f}s {r['improvement_pct']:>+8.2f}%")

    best = rows[0]
    print(f"\nBEST: X={best['X']} Y={best['Y']} Z={best['Z']:.0f} M={best['M']}  "
          f"-> {best['improvement_pct']:+.2f}% mean TTFT vs {base_name}")

    out = {
        "regime": {"prefix_tokens": PREFIX_BLOCKS * BLOCK, "output_tokens": OUTPUT_TOKENS,
                   "burst_size": BURST_SIZE, "burst_window_s": BURST_WINDOW, "nodes": NODES,
                   "congestion": True,
                   "imbalance_abs": base_cfg().IMBALANCE_ABS, "imbalance_rel": base_cfg().IMBALANCE_REL},
        "baseline": {"cache_aware_mean_ttft": ca["mean_ttft"], "least_load_mean_ttft": ll["mean_ttft"],
                     "compared_vs": base_name},
        "grid": {"X": GRID_X, "Y": GRID_Y, "Z": GRID_Z, "M": GRID_M},
        "results": rows,
        "best": best,
    }
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "sweep_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote results/sweep_results.json")
    return out


if __name__ == "__main__":
    main()
