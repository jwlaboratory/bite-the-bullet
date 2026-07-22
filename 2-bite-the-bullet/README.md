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
| **Y** | shared-prefix length — matched on **and** copied | the whole prefix |
| **Z** | detection window (seconds) | 1 |
| **M** | HBM copies to warm | 4 |

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

It **replays the real Bursted-ART test set**
([`../3-workload/generate/`](../3-workload/generate/)) — all 30 windows /
76,800 requests — window by window, across six model×hardware setups. Generate
or download the dataset first (see folder 3).

## Result

early_rdma vs `cache_aware` (the SGLang default router, no warming), mean-TTFT
reduction on Bursted-ART:

| setup | model | hw | synthetic | mixed |
|---|---|---|--:|--:|
| 70b_h100x4 | Llama-70B | H100×4 | +62.0% | +54.0% |
| qwen3_8b_h100x4 | Qwen3-8B | H100×4 | +41.9% | +33.3% |
| glm45_h100x4 | GLM-4.5 | H100×4 | +70.3% | +60.0% |
| glm52_h100x8 | GLM-4.6 | H100×8 | +64.0% | +53.5% |
| kimi_k2_h100x8 | Kimi K2 | H100×8 | +58.5% | +48.1% |
| dense1t_b300x4 | dense-1T | B300×4 | +11.0% | +10.3% |

**synthetic** = the bursty windows the mechanism targets; **mixed** = the full
test set including real ART traffic (where early_rdma stays inert, so the gain
dilutes but stays large). Positive = lower TTFT. Raw numbers in
[`results.json`](results.json).
