# Experiments

The studies that motivate the `early_rdma` claim, **numbered in reading order**.
Each folder is self-contained: a `README.md`, its code, and its own results /
figures alongside.

| # | Folder | Role | Punchline |
|---|--------|------|-----------|
| 1 | [`1-prefill-vs-transfer/`](1-prefill-vs-transfer/) | motivation — the physics | moving already-computed KV costs **44–48×** less than recomputing it |
| 2 | [`2-burst-routing/`](2-burst-routing/) | motivation — when locality matters | cache-aware routing only wins when KV sharing is off or the fabric congests |
| 3 | [`3-early-rdma/`](3-early-rdma/) ★ | **the headline claim** | predict sustained reuse, RDMA-copy KV into HBM *before* the burst |

The short paper memo and the current filtered result dump live inside the
headline experiment: [`3-early-rdma/PAPER.md`](3-early-rdma/PAPER.md) and
[`3-early-rdma/results/`](3-early-rdma/results/).

Every live script expects the sibling `inference-sim` checkout (see the root
[`README.md`](../README.md)) and must stay exactly two directories deep so its
`parents[2]` / `../../../inference-sim` path resolution keeps working.

Superseded experiment families are under
[`../archive/superseded-experiments/`](../archive/superseded-experiments/).
