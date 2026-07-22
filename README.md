# bite-the-bullet

Research artifacts for **Biting the Bullet**: predicting synchronized
same-prefix inference bursts and speculatively warming prefix KV before the
queue forms.

This repo contains the experiment scripts, saved results, and blog draft. The
simulator engine itself lives separately in `inference-sim`.

## Layout

- `bite-the-bullet/`: predictive KV warming, ART/Mooncake predictor, and
  end-to-end warming evaluators.
- `partial-prefill/`: partial-prefix and adaptive idle-only warming sweeps.
- `experiments/utility-gate/`: real-trace utility-gate harness, model/hardware
  sweeps, and saved utility-gate results.
- `clean-experiments/`: clean experiment specs, result JSONs, cross-system
  matrix, and the main blog draft.
- `data-generation/`: scripts for building BTB JSONL datasets; generated
  outputs under `data-generation/out/` are ignored.
- `archive/`: older exploratory runs and paper notes that are useful history
  but not the current runnable path.
- `BLOG.md`: current long-form research blog draft.

## Dependency On The Simulator

The experiment scripts use the simulator modules from a sibling checkout:

```text
../inference-sim
```

You can override that path with:

```bash
export INFERENCE_SIM_ROOT=/path/to/inference-sim
```

Recommended local layout:

```text
jwlabs/
  inference-sim/
  bite-the-bullet/
```

## Reproduce The Main Result

From this repo root:

```bash
python3 partial-prefill/sweep_partial_prefill.py \
  --imbalance-abs 8 \
  --model-preset glm52-int4 \
  --num-replicas 8 \
  --gpus-per-replica 8 \
  --gpu H100 \
  --max-batch 256 \
  --block-tokens 256 \
  --num-bursts 8 \
  --burst-size 500 \
  --prefix-tokens 65536 \
  --suffix-tokens 256 \
  --output-tokens 1 \
  --first-burst-s 20 \
  --burst-spacing-s 40 \
  --burst-window-s 1 \
  --num-decoys 0 \
  --background-requests 0 \
  --lead-s 6 \
  --precision-sweep 1 \
  --recall 1 \
  --trials 5 \
  --warm-tokens 16384 32768 65536 \
  --out clean-experiments/results/experiment_01_target_burst.json
```

The saved result summary is in:

```text
clean-experiments/results/SUMMARY.md
```

Cross-system percentages are in:

```text
clean-experiments/results/standard-systems/SUMMARY.md
```

## Main Claim

For synchronized same-prefix bursts, moving prefix prefill off the request
critical path can reduce p95 TTFT by **67-99%** on H100-class simulated
deployments.

The claim is deliberately scoped. Predictive warming is not a universal
replacement for cache-aware routing or reactive RDMA. It helps when the burst is
predictable, the prefix is shared, and the system has enough lead time/slack to
finish warming before the burst lands.
