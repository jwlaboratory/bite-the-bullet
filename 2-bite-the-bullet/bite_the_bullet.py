#!/usr/bin/env python3
"""Bite the Bullet (early_rdma): the algorithm, replayed on Bursted-ART.

THE ALGORITHM — four constants, no per-model learning:

    If the same Y-block prefix arrives X times within Z seconds, replicate its
    KV to the M least-busy replicas, then route later same-prefix requests to a
    warm copy.

    X = THRESHOLD      repeats needed to fire        (2)
    Y = PREFIX_BLOCKS  shared prefix, matched + copied (256 blocks = the Bursted-ART burst prefix)
    Z = WINDOW_S       detection window in seconds     (1)
    M = WARM_COPIES    HBM copies to warm              (4)

This replays the real Bursted-ART test set (3-workload/generate/out/Bursted-ART),
window by window, across several model x hardware setups. For each setup it runs
cache_aware (the SGLang default router, no warming) and early_rdma (cache_aware +
warming), and reports how much early_rdma cuts mean TTFT on the bursty
(synthetic) windows and on the full mixed set.

RUN (clone inference-sim next to this repo, and generate/download the dataset
     into 3-workload/generate/out/Bursted-ART first):

    python3 2-bite-the-bullet/bite_the_bullet.py
"""
from __future__ import annotations

import json
import math
import random
import sys
from dataclasses import replace
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
DATA = ROOT / "3-workload" / "generate" / "out" / "Bursted-ART" / "test.jsonl"

# The four constants, hard-coded.
X_THRESHOLD, Y_PREFIX_BLOCKS, Z_WINDOW_S, M_WARM_COPIES = 2, 256, 1.0, 4
BLOCK = 256
SEED = 42


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
    loads = [_load(nd) for nd in nodes]
    if max(loads) > cfg.IMBALANCE_ABS and max(loads) > cfg.IMBALANCE_REL * min(loads):
        return _pick(nodes, lambda nd: _load(nd))
    return _pick(nodes, lambda nd: (-sum(nd.match(req.blocks)), _load(nd)))


class _PendingWarm:
    __slots__ = ("ready", "node", "blocks", "key")

    def __init__(self, ready, node, blocks, key):
        self.ready, self.node, self.blocks, self.key = ready, node, blocks, key


class EarlyRdma:
    """The four-constant rule (X/Y/Z/M): a prefix is active while its trailing
    Z-second count is >= X; while active it warms up to M copies and routes to
    the least-loaded warm replica, else falls back to cache_aware."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.prefix_blocks = int(getattr(cfg, "BTB_PREFIX_BLOCKS", 256))  # Y
        self.threshold = int(getattr(cfg, "BTB_THRESHOLD", 2))            # X
        self.window = float(getattr(cfg, "BTB_WINDOW_S", 1.0))            # Z
        self.copies = int(getattr(cfg, "BTB_WARM_COPIES", 4))            # M
        self.hist = {}
        self.pending = []
        self.planned = set()

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


# ============================ MODEL x HARDWARE SETUPS ============================ #
MODEL_PRESETS = {
    "70b":      {"PARAMS": 70.6e9, "ACTIVE_PARAMS": 70.6e9, "DTYPE_BYTES": 2.0, "LAYERS": 80, "KV_HEADS": 8, "HEAD_DIM": 128},
    "glm45":    {"PARAMS": 355e9,  "ACTIVE_PARAMS": 32e9,   "DTYPE_BYTES": 0.5, "LAYERS": 92, "KV_HEADS": 4, "HEAD_DIM": 128},
    "glm52":    {"PARAMS": 744e9,  "ACTIVE_PARAMS": 40e9,   "DTYPE_BYTES": 0.5, "LAYERS": 78, "KV_HEADS": 1, "HEAD_DIM": 288},
    "qwen3-8b": {"PARAMS": 8.19e9, "ACTIVE_PARAMS": 8.19e9, "DTYPE_BYTES": 2.0, "LAYERS": 36, "KV_HEADS": 8, "HEAD_DIM": 128},
    "kimi-k2":  {"PARAMS": 1e12,   "ACTIVE_PARAMS": 32e9,   "DTYPE_BYTES": 0.5, "LAYERS": 61, "KV_HEADS": 8, "HEAD_DIM": 128},
    "dense1t":  {"PARAMS": 1e12,   "ACTIVE_PARAMS": 1e12,   "DTYPE_BYTES": 1.0, "LAYERS": 120, "KV_HEADS": 8, "HEAD_DIM": 128},
}

# (label, preset, GPU, gpus_per_replica, num_replicas)
SETUPS = [
    ("70b_h100x4",    "70b",      "H100", 4, 4),
    ("qwen3_8b_h100x4", "qwen3-8b", "H100", 4, 4),
    ("glm45_h100x4",  "glm45",    "H100", 4, 4),
    ("glm52_h100x8",  "glm52",    "H100", 8, 4),
    ("kimi_k2_h100x8", "kimi-k2", "H100", 8, 4),
    ("dense1t_b300x4", "dense1t", "B300", 4, 4),
]


def setup_cfg(preset, gpu, gpr, nrep):
    cfg = SimpleNamespace(**config.as_dict())
    cfg.BLOCK_TOKENS = BLOCK
    cfg.DISK_CACHE = False                     # a prefix a node lacks is recomputed
    for k, v in MODEL_PRESETS[preset].items():
        setattr(cfg, k, v)
    cfg.BTB_PREFIX_BLOCKS = Y_PREFIX_BLOCKS
    cfg.BTB_THRESHOLD = X_THRESHOLD
    cfg.BTB_WINDOW_S = Z_WINDOW_S
    cfg.BTB_WARM_COPIES = M_WARM_COPIES
    spec = getattr(config, gpu)
    cfg.CLUSTER = [(f"node{i}", spec, gpr) for i in range(nrep)]
    return cfg


# ============================ DATASET REPLAY ============================ #
def load_windows():
    """Group Bursted-ART rows by trace window; build sim Requests per window.
    Returns [(source, requests), ...] where source is 'art' or 'synthetic'."""
    by_win = {}
    with open(DATA) as f:
        for line in f:
            r = json.loads(line)
            by_win.setdefault(r["trace_id"], []).append(r)
    windows = []
    for tid, rows in by_win.items():
        rows.sort(key=lambda r: r.get("arrival_s", 0.0))
        reqs = []
        for i, r in enumerate(rows):
            blocks = [str(x) for x in (r.get("hash_ids") or [])]
            inp = max(1, int(r.get("input_length") or 1))
            out = max(1, int(r.get("output_length") or 1))
            extra = math.ceil((inp + out) / BLOCK) - len(blocks)
            cache_blocks = blocks + [f"{tid}:{i}#o{j}" for j in range(max(0, extra))]
            reqs.append(Request(
                id=i, arrival=float(r.get("arrival_s", 0.0)), group=tid,
                prefix_tokens=min(len(blocks) * BLOCK, inp),
                input_tokens=inp, output_tokens=out,
                blocks=blocks, cache_blocks=cache_blocks))
        windows.append((rows[0]["source"], reqs))
    return windows


def replay_mean_ttft(policy, windows, cfg):
    """Run every window under `policy`; return request-weighted mean TTFT for the
    synthetic windows and for the full (mixed) set."""
    syn_sum = syn_n = mix_sum = mix_n = 0.0
    for source, reqs in windows:
        random.seed(SEED)
        m = run(policy, reqs, cfg)["metrics"]
        s, n = m["mean_ttft"] * len(reqs), len(reqs)
        mix_sum += s; mix_n += n
        if source == "synthetic":
            syn_sum += s; syn_n += n
    return (syn_sum / syn_n if syn_n else 0.0), (mix_sum / mix_n if mix_n else 0.0)


def main():
    if not DATA.exists():
        sys.exit(f"dataset not found: {DATA}\n"
                 f"generate it: python3 3-workload/generate/generate_combined_dataset.py "
                 f"--synthetic-burst-window-s 60 --out-dir 3-workload/generate/out/Bursted-ART")
    register()
    windows = load_windows()
    n_syn = sum(1 for s, _ in windows if s == "synthetic")
    print(f"Bursted-ART test set: {len(windows)} windows ({n_syn} synthetic, "
          f"{len(windows) - n_syn} ART), {sum(len(r) for _, r in windows):,} requests")
    print(f"algorithm: X={X_THRESHOLD} Y={Y_PREFIX_BLOCKS} Z={Z_WINDOW_S:.0f}s M={M_WARM_COPIES}\n")

    print(f"{'setup':<18}{'model':<9}{'hw':<9}{'synthetic':>11}{'mixed':>9}")
    print("-" * 56)
    rows = []
    for label, preset, gpu, gpr, nrep in SETUPS:
        cfg = setup_cfg(preset, gpu, gpr, nrep)
        try:
            ca_syn, ca_mix = replay_mean_ttft("cache_aware", windows, cfg)
            bt_syn, bt_mix = replay_mean_ttft("early_rdma", windows, cfg)
        except Exception as e:
            print(f"{label:<18}{preset:<9}{gpu+'x'+str(gpr):<9}  skipped ({type(e).__name__})")
            continue
        syn = (ca_syn - bt_syn) / ca_syn * 100 if ca_syn else 0.0
        mix = (ca_mix - bt_mix) / ca_mix * 100 if ca_mix else 0.0
        print(f"{label:<18}{preset:<9}{gpu+'x'+str(gpr):<9}{syn:>+10.1f}%{mix:>+8.1f}%")
        rows.append({"setup": label, "model": preset, "gpu": gpu, "gpus_per_replica": gpr,
                     "num_replicas": nrep, "synthetic_improvement_pct": syn, "mixed_improvement_pct": mix,
                     "cache_aware_mean_ttft_synthetic": ca_syn, "early_rdma_mean_ttft_synthetic": bt_syn})
    print("\n(positive = early_rdma lower mean TTFT than cache_aware, the SGLang default router)")

    out = {
        "constants": {
            "X": X_THRESHOLD, "Y": Y_PREFIX_BLOCKS, "Z": Z_WINDOW_S, "M": M_WARM_COPIES,
            "_definitions": {
                "X": "BTB_THRESHOLD: same-prefix arrivals within Z needed to fire",
                "Y": "BTB_PREFIX_BLOCKS: shared-prefix length in blocks, matched on AND copied (256 blocks x 256 tok = the 65,536-tok burst prefix)",
                "Z": "BTB_WINDOW_S: detection window in seconds",
                "M": "BTB_WARM_COPIES: number of REPLICAS (nodes) to warm; each replica is n_gpus GPUs tensor-parallel, so a copy is sharded across that node's GPUs",
            },
        },
        "dataset": "Bursted-ART/test.jsonl", "baseline": "cache_aware (SGLang default router, no warming)",
        "field_definitions": {
            "synthetic_improvement_pct": "(cache_aware - early_rdma) / cache_aware * 100, mean TTFT over the synthetic (bursty) windows only; positive = early_rdma faster",
            "mixed_improvement_pct": "same ratio over ALL windows (synthetic + real ART); the realistic average (BTB is inert on non-bursty ART windows)",
            "cache_aware_mean_ttft_synthetic": "baseline mean time-to-first-token (s), request-weighted, synthetic windows",
            "early_rdma_mean_ttft_synthetic": "early_rdma mean time-to-first-token (s), request-weighted, synthetic windows",
        },
        "setups": rows,
    }
    (HERE / "results.json").write_text(json.dumps(out, indent=2) + "\n")
    print("wrote results.json")
    return out


if __name__ == "__main__":
    main()
