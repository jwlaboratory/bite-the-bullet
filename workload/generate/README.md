# Data Generation

Builds Bursted-ART: ART windows plus synthetic same-prefix bursts.

Hugging Face:
[shreybirmiwal/Bursted-ART](https://huggingface.co/datasets/shreybirmiwal/Bursted-ART)

Canonical outputs:

- `out/Bursted-ART`
- `out/Bursted-ART-60s`

Generate the uploaded dataset:

```bash
python3 data-generation/generate_combined_dataset.py \
  --out-dir data-generation/out/Bursted-ART
```

Generate the 60-second burst variant:

```bash
python3 data-generation/generate_combined_dataset.py \
  --synthetic-burst-window-s 60 \
  --out-dir data-generation/out/Bursted-ART-60s
```

Upload:

```bash
python3 data-generation/upload_to_hf.py \
  --dataset-dir data-generation/out/Bursted-ART \
  --repo-id shreybirmiwal/Bursted-ART
```
