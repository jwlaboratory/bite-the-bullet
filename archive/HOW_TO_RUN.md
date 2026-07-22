# How to run

Bite The Bullet (BTB) studies one mechanism — `early_rdma`: detect a sustained
shared-prefix burst and RDMA-copy the KV to less-busy replicas **before** later
requests arrive. This page is how to run the simulations and what you can tune.

## What was done (short version)

BTB was stripped to its working core — the `early_rdma` rule with **four
constants (X/Y/Z/M)** — by archiving the ~2,600-line learned utility gate,
collapsing to one 60-second `Bursted-ART` dataset, and matching the cache-aware
router to SGLang's real defaults. A parameter sweep plus a realistic
recompute-on-miss baseline (`ADMIT_RDMA`) show `early_rdma` cutting mean TTFT
~72–84% in the favorable single-prefix regime — the microbenchmark ceiling
behind the paper's ~15% realistic-average claim.

## Prerequisites

The simulations import the **Infer-Sim** engine. Clone it next to this repo:

```
jwlabs/
├── bite-the-bullet/     # this repo
└── inference-sim/       # the simulator (needs config.py, gpu.py, simulate.py, router.py)
```

Or point at it explicitly: `export INFERENCE_SIM_ROOT=/path/to/inference-sim`.
Python 3.11+; `numpy` and `matplotlib` for the charts.

## Run a simulation

```bash
# A) Policy comparison + cluster-size study
#    least_load vs cache_aware vs early_rdma, across 3 fabric conditions
cd experiments/2-burst-routing
python3 run_burst.py        # -> results/burst_results.json
python3 make_charts.py      # -> charts/*.png

# B) Parameter sweep — find the best X/Y/Z/M on a prefill-heavy regime
python3 sweep_params.py     # -> results/sweep_results.json ; prints a ranked table + BEST

# C) Regenerate / publish the dataset
python3 workload/generate/generate_combined_dataset.py \
    --synthetic-burst-window-s 60 --out-dir workload/generate/out/Bursted-ART
python3 workload/generate/upload_to_hf.py \
    --dataset-dir workload/generate/out/Bursted-ART --repo-id shreybirmiwal/Bursted-ART

# D) Audit — does this burst pattern exist in public traces? (downloads Mooncake/BurstGPT/ART)
python3 workload/audit/audit_burst_absence.py --out-dir workload/audit/results
python3 workload/audit/make_chart.py
```

The frozen paper numbers are in `experiments/3-early-rdma/PAPER.md` +
`results/results.json` (an artifact; the sweep harness that produced them is not
in the active tree).

## Tunables

### The policy — 4 core constants (`experiments/2-burst-routing/btb_policy.py`)

Set per run via `cfg.BTB_*`; `run_burst.py` / `sweep_params.py` set them for you.

| Var | cfg name | Meaning | Default / how set |
|-----|----------|---------|-------------------|
| **X** | `BTB_THRESHOLD`     | repeats needed to fire            | 2 (sweep: insensitive) |
| **Y** | `BTB_PREFIX_BLOCKS` | shared-prefix length — matched on **and** copied | the workload's prefix (sweep: use the *whole* prefix) |
| **Z** | `BTB_WINDOW_S`      | detection window (seconds)        | 1.0 (sweep: insensitive) |
| **M** | `BTB_WARM_COPIES`   | how many HBM copies to warm       | the one real lever (more = more spread) |

The rule: *if the same Y-block prefix arrives X times within Z seconds,
replicate its KV to the M least-busy replicas and route later same-prefix
requests across them.*

### The simulator / fabric model (`inference-sim/config.py`, override per run)

| Knob | Meaning | Note |
|------|---------|------|
| `ADMIT_RDMA` | baseline may opportunistically peer-pull KV on a miss | **`False` = realistic** (miss recomputes, as in real SGLang); `True` = shared-fabric. BTB's own warming push always runs at full RDMA speed. |
| `RDMA_CONGESTION` | fabric contends under incast | mean-field congestion model |
| `DISK_CACHE` | shared disk / remote KV pool | HiCache-style shared pool |
| `IMBALANCE_ABS` / `IMBALANCE_REL` | cache_aware → least-load fallback thresholds | 64 / 1.5 (SGLang router defaults) |
| `CLUSTER` | nodes × GPU spec | e.g. 8× H100 |

### The workload (constants at the top of `run_burst.py` / `sweep_params.py`)

`PREFIX_BLOCKS` (shared-prefix length), `OUTPUT_TOKENS`, `BURST_SIZE`,
`BURST_WINDOW` (arrival spread), `NODES`, plus `NUM_FAMILIES` / `ZIPF` (skew) in
the multi-tenant demo.

### The dataset (`workload/generate/`)

- **Generation:** `--synthetic-burst-window-s` (60), `--synthetic-prefix-tokens`
  (65536), `--synthetic-burst-size` (500), `--train-windows` / `--test-windows`
  (10 / 30).
- **Slice:** `train` vs `test` (split by whole window, not by row).
- **Scope:** `synthetic` (bursts only) vs `mixed` (bursts + real ART traffic).

## The one thing to internalize

The size of the gain depends entirely on **baseline realism**. Against a baseline
that recomputes on a miss (real SGLang) or piles requests on one node,
`early_rdma` wins big. If the simulator instead hands the baseline a free
peer-to-peer KV steal (`ADMIT_RDMA=True`, which real systems don't do on the
serving path), that baseline gets warming's benefit for free and BTB looks
break-even. `ADMIT_RDMA=False` is the honest setting.
