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

## Result

On a prefill-heavy shared-prefix burst (32,768-token prefix, 1-token output,
400 requests over 8 s, 8 nodes):

| policy | mean TTFT |
|--------|----------:|
| cache_aware | 0.176 s |
| least_load  | 0.103 s |
| **early_rdma** | **0.030 s** |

**early_rdma cuts mean TTFT ~71% vs the best baseline.** Raw numbers in
[`results.json`](results.json).
