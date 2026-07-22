# 3 · early_rdma — the headline claim

Detect a sustained shared-prefix burst, then RDMA-copy existing KV into HBM on
less-busy replicas **before** later requests arrive.

## The rule (this is the whole method)

> If the same **Y-block prefix** is seen **X times within Z seconds**, don't wait
> for a queue to build — **bite the bullet and replicate** its KV to the
> least-busy replicas, then route later same-prefix requests across those warm
> copies.

Defaults: **X = 4** repeats, **Y = 4** blocks, **Z = 2 s**, replicate to **4**
HBM copies. It is a fixed rule, not a per-model learned gate — the same
thresholds run on every model/hardware setup.

The mechanism is one small file: [`../2-burst-routing/btb_policy.py`](../2-burst-routing/btb_policy.py)
(`EarlyRdma`). Experiment 2 runs it head-to-head against `least_load` /
`cache_aware` on the real simulator; this experiment reports how it does on the
Bursted-ART trace across model/hardware setups.

No seed-prefill / fake-prefill policy is part of the claim; those live in
[`../../archive/superseded-experiments/`](../../archive/superseded-experiments/).
An earlier *learned* per-model utility gate was explored and set aside — it is
not the claim — and is archived under
[`../../archive/learned-utility-gate/`](../../archive/learned-utility-gate/).

## Files

| File | What |
|------|------|
| `PAPER.md` | short paper memo — **the current claim** |
| `results/results.json` | raw filtered `early_rdma` result dump (the headline run) |

The policy itself and a runnable demo live in experiment 2:

| File | What |
|------|------|
| [`../2-burst-routing/btb_policy.py`](../2-burst-routing/btb_policy.py) | the `early_rdma` rule (X/Y/Z + replicate) |
| [`../2-burst-routing/run_burst.py`](../2-burst-routing/run_burst.py) | runnable demo on the simulator |

## Run the mechanism

```bash
# needs the sibling inference-sim checkout (see repo root README)
cd ../2-burst-routing
python3 run_burst.py      # drives early_rdma vs least_load vs cache_aware
```

`results/results.json` is a frozen artifact from the Bursted-ART model/hardware
sweep; the script that produced that exact sweep is not kept in the active tree.

## Result

Read [`PAPER.md`](PAPER.md); raw numbers in [`results/results.json`](results/results.json).
