# Bite the Bullet — the algorithm

`bite_the_bullet.py` is the whole thing: the routing algorithm and a single
runnable head-to-head that reports the result.

## The algorithm (four constants)

> If the same **Y**-block prefix arrives **X** times within **Z** seconds,
> replicate its KV to the **M** least-busy replicas, then route later
> same-prefix requests to a warm copy.

| Constant | Meaning | Value |
|----------|---------|-------|
| **X** | repeats needed to fire | 2 |
| **Y** | shared-prefix length — matched on **and** copied | 256 blocks (the 65,536-tok burst prefix) |
| **Z** | detection window (seconds) | 1 |
| **M** | **replicas (nodes)** to warm | 4 |

**M is replicas, not GPUs.** A replica is `n_gpus` GPUs serving tensor-parallel —
weights and KV are *sharded* across them, so one node holds one sharded copy of
the prefix KV. M=4 replicates that copy onto 4 separate nodes. (Every setup here
runs 4 replicas, so M=4 warms the whole cluster.)

A prefix a node does not hold is recomputed, so putting the KV on a replica
*before* its first same-prefix request lands saves that recompute. `least_load`
and `cache_aware` never pre-warm — one scatters the burst onto cold nodes that
recompute, the other piles it on the single node that holds the prefix.
`early_rdma` warms M copies and spreads the burst across them.

## Run

```bash
# clone inference-sim next to this repo, then:
python3 2-bite-the-bullet/bite_the_bullet.py

# or point at the simulator explicitly:
INFERENCE_SIM_ROOT=/path/to/inference-sim python3 2-bite-the-bullet/bite_the_bullet.py
```

## What it runs on

It **replays two datasets as real traces**, window by window, across six
model×hardware setups ([`../3-workload/generate/`](../3-workload/generate/) —
generate or download first):

- **Bursted-ART** — the full test set (real ART traffic + synchronized bursts).
- **normal-ART** — the plain-ART windows only, no bursts (a control: BTB should
  be inert).

TTFT is pooled over all requests; both **mean** and **p95** are reported.

## Result — Bursted-ART

early_rdma vs `cache_aware` (SGLang default router, no warming):

| setup | CA mean | CA p95 | BTB mean | BTB p95 | mean speedup | p95 speedup |
|---|--:|--:|--:|--:|--:|--:|
| 70b_h100x4 | 1.373s | 4.697s | 0.632s | 4.697s | **+54.0%** | +0.0% |
| qwen3_8b_h100x4 | 0.034s | 0.325s | 0.023s | 0.059s | **+33.3%** | +81.8% |
| glm45_h100x4 | 0.288s | 1.905s | 0.115s | 0.780s | **+60.0%** | +59.0% |
| glm52_h100x8 | 0.134s | 1.101s | 0.062s | 0.243s | **+53.5%** | +78.0% |
| kimi_k2_h100x8 | 0.093s | 0.832s | 0.048s | 0.167s | **+48.1%** | +79.9% |
| dense1t_b300x4 | 436.8s | 926.1s | 392.0s | 857.9s | **+10.3%** | +7.4% |

## Result — normal-ART (control)

| setup | CA mean | BTB mean | mean speedup | p95 speedup |
|---|--:|--:|--:|--:|
| 70b_h100x4 | 0.927s | 0.919s | +0.9% | +1.3% |
| qwen3_8b_h100x4 | 0.036s | 0.036s | −0.5% | −0.2% |
| glm45_h100x4 | 0.220s | 0.218s | +0.6% | +1.5% |
| glm52_h100x8 | 0.111s | 0.112s | −0.6% | +1.0% |
| kimi_k2_h100x8 | 0.084s | 0.084s | −0.4% | −0.4% |
| dense1t_b300x4 | 204.8s | 198.6s | +3.0% | +2.9% |

On plain ART the prefix never repeats deeply enough to fire, so early_rdma ≈
cache_aware — it neither helps nor hurts ordinary traffic. Positive = lower TTFT.
Raw numbers + field definitions in [`results.json`](results.json).
