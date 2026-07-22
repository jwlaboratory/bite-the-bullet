# Bite the Bullet

Research repo for **predictive KV movement in bursty LLM inference**.

## The claim

Detect sustained shared-prefix reuse, then RDMA-copy existing KV into HBM on
less-busy replicas *before* later requests arrive — while the fabric is still
quiet — instead of recomputing the prefix or contending for it reactively during
the spike. The active policy is `early_rdma`.

## The argument, in reading order

The repo is laid out as the argument that motivates that claim. Read it
top to bottom:

| Step | Question | Where |
|------|----------|-------|
| 0 | Does this workload even exist in public traces? | [`workload/audit/`](workload/audit/) — mostly **no**, so we synthesize it |
| 1 | Build the synthetic workload | [`workload/generate/`](workload/generate/) — the Bursted-ART dataset |
| 2 | Why would moving KV ever beat recomputing it? | [`experiments/1-prefill-vs-transfer/`](experiments/1-prefill-vs-transfer/) — transfer is 44–48× cheaper |
| 3 | When does locality/routing actually matter under a burst? | [`experiments/2-burst-routing/`](experiments/2-burst-routing/) — only when sharing is off or the fabric congests |
| 4 | **The policy: move KV predictively, ahead of the burst** | [`experiments/3-early-rdma/`](experiments/3-early-rdma/) — ★ the headline claim |

The write-up lives in [`BLOG.md`](BLOG.md) (long form) and
[`experiments/3-early-rdma/PAPER.md`](experiments/3-early-rdma/PAPER.md) (short
paper memo).

## Layout

```
workload/            build and interrogate the target workload
  generate/            Bursted-ART dataset generation (-> HF)
  audit/               is the same-prefix fan-out burst present in real traces?
experiments/         the studies, numbered in argument order
  1-prefill-vs-transfer/   cost model: transfer vs recompute
  2-burst-routing/         least_load vs cache_aware under a flash crowd
  3-early-rdma/            THE claim; owns its results/ and PAPER.md
BLOG.md              long-form write-up
FUTURE_IDEAS.md      backlog kept for later
sim_path.py          shared shim that locates the inference-sim checkout
archive/             history, not the current claim (see archive/README.md)
```

## Simulator dependency

Every live experiment runs on a sibling `inference-sim` checkout:

```text
../inference-sim
```

Override with:

```bash
export INFERENCE_SIM_ROOT=/path/to/inference-sim
```

Live experiment scripts assume they sit exactly two directories deep
(`experiments/<name>/script.py`) and that `sim_path.py` stays at the repo
root — keep that shape when adding experiments.
