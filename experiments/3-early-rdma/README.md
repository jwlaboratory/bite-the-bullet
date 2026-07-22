# 3 · early_rdma — the headline claim

Predict sustained shared-prefix reuse, then RDMA-copy existing KV into HBM on
less-busy replicas **before** later requests arrive.

The gate (the "utility gate") decides when biting the bullet — paying to move KV
now — beats recomputing it later:

1. detect repeated prefix reuse;
2. require an existing source KV copy in HBM;
3. RDMA-copy that KV to less-busy replicas, ahead of the burst.

No seed-prefill / fake-prefill policy is part of the active claim; those are in
[`../../archive/superseded-experiments/`](../../archive/superseded-experiments/).

## Files

| File | What |
|------|------|
| `btb_utility_gate.py` | main simulator harness (the policy) |
| `btb_art_model_hardware_sweep.py` | optional model × hardware sweep runner |
| `btb_result_summary.py` | summarize result JSON files |
| `PAPER.md` | short paper memo — **the current claim** |
| `results/` | raw filtered `early_rdma` result dumps (`results.json` is the headline run) |

Older utility-gate result dumps and sweeps are archived under
[`../../archive/old-results/`](../../archive/old-results/).

## Run

```bash
# needs the sibling inference-sim checkout (see repo root README)
python3 experiments/3-early-rdma/btb_utility_gate.py      # writes results/btb_utility_gate_results.json
python3 experiments/3-early-rdma/btb_result_summary.py    # summarize results/*.json
```

## Result

Read [`PAPER.md`](PAPER.md); raw numbers in [`results/results.json`](results/results.json).
