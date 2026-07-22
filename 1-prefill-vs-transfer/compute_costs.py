"""Prefill (recompute) vs. KV-cache transfer cost, derived from the simulator.

For a fixed prompt of N tokens, there are two ways to get its KV cache resident
on a node so decoding can start:

  1. Recompute it   -> prefill, compute-bound: gpu.Node.prefill_time(N)
  2. Move it in     -> stream the already-computed KV from a memory tier
                       (hbm / ram / rdma / disk), bandwidth-bound:
                       gpu.Node.load_time(N * kv_per_tok, tier)

Both are linear in N, so the prefill/transfer ratio is constant and is the
whole argument for warming from a cheaper tier instead of prefilling again.

Every number comes from the real sim constants (inference-sim/config.py, gpu.py)
via the sim_path shim — nothing is hardcoded here. Run it to regenerate
results.json and print the table.

    python 1-prefill-vs-transfer/compute_costs.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from sim_path import add_inference_sim_to_path

add_inference_sim_to_path(ROOT)
import config
from gpu import Node

# The standard bite-the-bullet setup: one node = 4x H100 tensor-parallel,
# serving Llama-3.3-70B fp16 (config.CLUSTER[0]).
NAME, SPEC, N_GPUS = config.CLUSTER[0]
node = Node(NAME, SPEC, N_GPUS, config)

TOKENS = [500, 1000, 2000, 8000, 16000, 32000]
TIERS = ["hbm", "ram", "rdma", "disk"]


def costs_ms(n: int) -> dict:
    """All ways to make N tokens of KV resident, in milliseconds."""
    kv_bytes = n * node.kv_per_tok
    out = {"prefill": node.prefill_time(n) * 1e3}
    for tier in TIERS:
        out[tier] = node.load_time(kv_bytes, tier) * 1e3
    return out


def main() -> None:
    setup = {
        "node": f"{N_GPUS}x{SPEC.name}",
        "model": f"Llama-3.3-70B fp{int(config.DTYPE_BYTES * 8)}",
        "n_gpus": N_GPUS,
        "mfu": config.MFU,
        "kv_bytes_per_token": node.kv_per_tok,
        "prefill_us_per_token": node.prefill_time(1) * 1e6,
        # node-aggregate bandwidths (per-GPU spec x n_gpus, except shared NVMe)
        "node_bandwidth_bytes_per_s": {t: node.tier_bw[t] for t in TIERS},
        "per_gpu_bandwidth_bytes_per_s": {
            "hbm": SPEC.hbm_bw, "ram": SPEC.ram_bw,
            "rdma": SPEC.rdma_bw, "disk": SPEC.disk_bw,
        },
        "node_flops_effective": node.flops * config.MFU,
    }

    rows = [{"tokens": n, **costs_ms(n)} for n in TOKENS]

    # constant ratios (linear in N, so evaluate at N=1)
    unit = costs_ms(1)
    ratios = {t: unit["prefill"] / unit[t] for t in TIERS}

    results = {"setup": setup, "tokens": TOKENS, "rows_ms": rows,
               "prefill_over_tier": ratios}

    out_path = Path(__file__).with_name("results.json")
    out_path.write_text(json.dumps(results, indent=2))

    # pretty print
    kv = node.kv_per_tok
    print(f"node = {setup['node']}  model = {setup['model']}  "
          f"KV = {kv} B/tok ({kv/1024:.0f} KiB)")
    print(f"prefill = {setup['prefill_us_per_token']:.1f} us/tok  "
          f"(node {setup['node_flops_effective']/1e12:.0f} eff TFLOP/s, MFU {config.MFU})")
    bw = setup["node_bandwidth_bytes_per_s"]
    print("node bandwidth: " + "  ".join(
        f"{t}={bw[t]/1e9:.1f} GB/s" for t in TIERS))
    print()
    hdr = "{:>7} | " + "  ".join("{:>10}" for _ in range(5))
    print(hdr.format("tokens", "prefill", *TIERS) + "   (ms)")
    for r in rows:
        print("{:>7} | {:>10.2f}  {:>10.4f}  {:>10.3f}  {:>10.3f}  {:>10.2f}".format(
            r["tokens"], r["prefill"], r["hbm"], r["ram"], r["rdma"], r["disk"]))
    print()
    print("prefill / tier (constant):  " + "  ".join(
        f"{t}={ratios[t]:.1f}x" for t in TIERS))
    print(f"\nwrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
