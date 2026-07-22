# Data Generation

Builds **Bursted-ART**: real ART replay windows plus synthetic same-prefix
bursts. The synthetic bursts span **60 seconds** — the sustained-reuse regime
where predictive KV warming has time to repay the copy (this is the regime the
paper's headline result comes from).

Hugging Face:
[shreybirmiwal/Bursted-ART](https://huggingface.co/datasets/shreybirmiwal/Bursted-ART)

Canonical output: `out/Bursted-ART/` (gitignored — regenerate as below).

Generate:

```bash
python3 workload/generate/generate_combined_dataset.py \
  --synthetic-burst-window-s 60 \
  --out-dir workload/generate/out/Bursted-ART
```

Upload:

```bash
python3 workload/generate/upload_to_hf.py \
  --dataset-dir workload/generate/out/Bursted-ART \
  --repo-id shreybirmiwal/Bursted-ART
```

The earlier 1-second (fast-burst / adversarial) variant is archived; see
[`../../archive/datasets/README.md`](../../archive/datasets/README.md) to
regenerate it if needed.
