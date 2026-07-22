#!/usr/bin/env python3
"""Bite The Bullet (early_rdma) — the routing algorithm and a runnable demo.

THE ALGORITHM — four constants, no per-model learning:

    If the same Y-block prefix arrives X times within Z seconds, replicate its
    KV to the M least-busy replicas, then route later same-prefix requests to a
    warm copy.

    X = THRESHOLD      repeats needed to fire        (2)
    Y = PREFIX_BLOCKS  shared prefix, matched + copied (the whole prefix)
    Z = WINDOW_S       detection window in seconds     (1)
    M = WARM_COPIES    HBM copies to warm              (4)

A prefix a node does not hold is recomputed, so warming a replica before its
first same-prefix request lands saves that recompute. This file contains the
algorithm and a single head-to-head run: least_load and cache_aware (neither
pre-warms) vs early_rdma, on a prefill-heavy shared-prefix burst.

RUN (clone inference-sim next to this repo, then):

    python3 2-bite-the-bullet/bite_the_bullet.py

or point at the simulator explicitly:

    INFERENCE_SIM_ROOT=/path/to/inference-sim python3 2-bite-the-bullet/bite_the_bullet.py
"""
from __future__ import annotations

import json
import os
import random
import math
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim_path import add_inference_sim_to_path  # noqa: E402

add_inference_sim_to_path(ROOT)
import config                       # noqa: E402
from simulate import run            # noqa: E402
from workload import Request        # noqa: E402
import router                       # noqa: E402

HERE = Path(__file__).resolve().parent


# ============================ THE ALGORITHM ============================ #
def _load(node):
    return len(node.running) + len(node.waiting)


def _resident(node, blocks):
    if not blocks:
        return False
    hbm_n, _ = node.match(blocks)
    return hbm_n >= len(blocks)


def _pick(nodes, key):
    best = min(key(nd) for nd in nodes)
    return random.choice([nd for nd in nodes if key(nd) == best])


def _cache_aware(req, nodes, cfg):
    """Route to the longest local prefix match, then lightest load; fall back to
    least-load when the cluster gets skewed (SGLang imbalance thresholds)."""
    loads = [_load(nd) for nd in nodes]
    if max(loads) > cfg.IMBALANCE_ABS and max(loads) > cfg.IMBALANCE_REL * min(loads):
        return _pick(nodes, lambda nd: _load(nd))
    return _pick(nodes, lambda nd: (-sum(nd.match(req.blocks)), _load(nd)))


class _PendingWarm:
    __slots__ = ("ready", "node", "blocks", "key")

    def __init__(self, ready, node, blocks, key):
        self.ready, self.node, self.blocks, self.key = ready, node, blocks, key


class EarlyRdma:
    """The four-constant rule (X/Y/Z/M). A prefix is active while its trailing
    Z-second count is >= X; while active it warms up to M copies and routes to
    the least-loaded warm replica, else falls back to cache_aware."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.prefix_blocks = int(getattr(cfg, "BTB_PREFIX_BLOCKS", 128))  # Y
        self.threshold = int(getattr(cfg, "BTB_THRESHOLD", 2))            # X
        self.window = float(getattr(cfg, "BTB_WINDOW_S", 1.0))            # Z
        self.copies = int(getattr(cfg, "BTB_WARM_COPIES", 4))            # M
        self.hist = {}
        self.pending = []
        self.planned = set()
        self.stats = {"warm_count": 0, "warm_bytes": 0.0, "warm_busy_s": 0.0}

    def _key(self, req):
        k = tuple(req.blocks[: self.prefix_blocks])
        return k if len(k) == self.prefix_blocks else None

    def _active(self, key, now):
        h = self.hist.setdefault(key, [])
        h.append(now)
        cut = now - self.window
        while h and h[0] < cut:
            h.pop(0)
        return len(h) >= self.threshold

    def _apply_ready(self, now):
        keep = []
        for w in self.pending:
            if w.ready <= now + 1e-12:
                w.node.insert(w.blocks)
                self.planned.discard((w.key, w.node.name))
            else:
                keep.append(w)
        self.pending = keep

    def _schedule(self, nodes, now, key, blocks):
        if not blocks or not any(_resident(nd, blocks) for nd in nodes):
            return
        have = sum(1 for nd in nodes if _resident(nd, blocks)) \
            + sum(1 for (k, _) in self.planned if k == key)
        need = self.copies - have
        if need <= 0:
            return
        targets = sorted((nd for nd in nodes
                          if not _resident(nd, blocks) and (key, nd.name) not in self.planned),
                         key=_load)
        nbytes = len(blocks) * nodes[0].block_bytes
        for nd in targets[:need]:
            dur = nbytes / nd.tier_bw["rdma"]        # structured warming push at full RDMA bandwidth
            start = max(now, nd.now)
            nd.now = start + dur
            nd.busy += dur
            self.pending.append(_PendingWarm(start + dur, nd, blocks, key))
            self.planned.add((key, nd.name))
            self.stats["warm_count"] += 1
            self.stats["warm_bytes"] += nbytes
            self.stats["warm_busy_s"] += dur

    def route(self, req, nodes, now):
        self._apply_ready(now)
        key = self._key(req)
        if key is not None and self._active(key, now):
            blocks = req.blocks[: self.prefix_blocks]
            self._schedule(nodes, now, key, blocks)
            warm = [nd for nd in nodes if _resident(nd, blocks)]
            if warm:
                return _pick(warm, _load)
        return _cache_aware(req, nodes, self.cfg)


def register():
    router.POLICIES["early_rdma"] = EarlyRdma


# ============================ THE DEMO ============================ #
BLOCK = 256                 # tokens/block (must match cfg.BLOCK_TOKENS)
PREFIX_BLOCKS = 128         # shared prefix: 128 * 256 = 32,768 tokens (prefill-heavy)
OUTPUT_TOKENS = 1           # decode-negligible -> prefill dominates TTFT
BURST_SIZE = 400            # requests sharing the hot prefix
BURST_WINDOW = 8.0          # arrive over 8 s
NODES = 8
SEED = 42

# The four constants, hard-coded.
X_THRESHOLD, Y_PREFIX_BLOCKS, Z_WINDOW_S, M_WARM_COPIES = 2, PREFIX_BLOCKS, 1.0, 4


def make_workload():
    rng = random.Random(SEED)
    prefix = [f"sys#{b}" for b in range(PREFIX_BLOCKS)]
    reqs = []

    def add(arrival, uid):
        blocks = prefix + [f"u{uid}#0"]
        inp = len(blocks) * BLOCK
        extra = math.ceil((inp + OUTPUT_TOKENS) / BLOCK) - len(blocks)
        cache_blocks = blocks + [f"u{uid}#o{j}" for j in range(max(0, extra))]
        reqs.append(Request(0, arrival, "hot", inp, inp, OUTPUT_TOKENS, blocks, cache_blocks))

    add(0.0, 0)                       # seed a source HBM copy
    add(1.0, 1)
    for i in range(BURST_SIZE):
        add(4.0 + rng.uniform(0.0, BURST_WINDOW), 100 + i)

    reqs.sort(key=lambda r: r.arrival)
    for i, r in enumerate(reqs):
        r.id = i
    return reqs


def base_cfg():
    cfg = SimpleNamespace(**config.as_dict())
    cfg.BLOCK_TOKENS = BLOCK
    cfg.DISK_CACHE = False            # a prefix a node lacks is recomputed
    spec, gpus = cfg.CLUSTER[0][1], cfg.CLUSTER[0][2]
    cfg.CLUSTER = [(f"node{i}", spec, gpus) for i in range(NODES)]
    cfg.BTB_THRESHOLD = X_THRESHOLD
    cfg.BTB_PREFIX_BLOCKS = Y_PREFIX_BLOCKS
    cfg.BTB_WINDOW_S = Z_WINDOW_S
    cfg.BTB_WARM_COPIES = M_WARM_COPIES
    return cfg


def run_seeded(policy, reqs, cfg):
    random.seed(SEED)
    return run(policy, reqs, cfg)["metrics"]


def main():
    register()
    reqs = make_workload()
    print(f"workload: {len(reqs)} reqs, {PREFIX_BLOCKS * BLOCK}-tok shared prefix, "
          f"{OUTPUT_TOKENS}-tok output, burst over {BURST_WINDOW}s, {NODES} nodes")
    print(f"algorithm constants: X={X_THRESHOLD} Y={Y_PREFIX_BLOCKS} "
          f"Z={Z_WINDOW_S:.0f}s M={M_WARM_COPIES}\n")

    ll = run_seeded("least_load", reqs, base_cfg())
    ca = run_seeded("cache_aware", reqs, base_cfg())
    btb = run_seeded("early_rdma", reqs, base_cfg())

    base = min(ll["mean_ttft"], ca["mean_ttft"])
    base_name = "least_load" if ll["mean_ttft"] <= ca["mean_ttft"] else "cache_aware"
    impr = (base - btb["mean_ttft"]) / base * 100.0

    print(f"{'policy':<14}{'mean_ttft':>12}")
    for name, m in (("least_load", ll), ("cache_aware", ca), ("early_rdma", btb)):
        print(f"{name:<14}{m['mean_ttft']:>11.3f}s")
    print(f"\nearly_rdma cuts mean TTFT {impr:+.1f}% vs the best baseline "
          f"({base_name} {base:.3f}s -> {btb['mean_ttft']:.3f}s)")

    out = {
        "constants": {"X": X_THRESHOLD, "Y": Y_PREFIX_BLOCKS, "Z": Z_WINDOW_S, "M": M_WARM_COPIES},
        "workload": {"prefix_tokens": PREFIX_BLOCKS * BLOCK, "output_tokens": OUTPUT_TOKENS,
                     "burst_size": BURST_SIZE, "burst_window_s": BURST_WINDOW, "nodes": NODES},
        "mean_ttft": {"least_load": ll["mean_ttft"], "cache_aware": ca["mean_ttft"],
                      "early_rdma": btb["mean_ttft"]},
        "best_baseline": base_name,
        "improvement_pct": impr,
    }
    (HERE / "results.json").write_text(json.dumps(out, indent=2) + "\n")
    print("\nwrote results.json")
    return out


if __name__ == "__main__":
    main()
