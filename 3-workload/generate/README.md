# generate — Bursted-ART

Builds **Bursted-ART**: real ART replay windows plus synthetic same-prefix
bursts. Each synthetic burst is 500 requests sharing a 65,536-token prefix,
arriving over a 60-second window — the sustained-reuse pattern `early_rdma`
targets.

Published at
[shreybirmiwal/Bursted-ART](https://huggingface.co/datasets/shreybirmiwal/Bursted-ART).
The generated `out/Bursted-ART/` is gitignored (large, regenerable).

```bash
# generate
python3 3-workload/generate/generate_combined_dataset.py \
    --synthetic-burst-window-s 60 --out-dir 3-workload/generate/out/Bursted-ART

# upload
python3 3-workload/generate/upload_to_hf.py \
    --dataset-dir 3-workload/generate/out/Bursted-ART --repo-id shreybirmiwal/Bursted-ART
```
