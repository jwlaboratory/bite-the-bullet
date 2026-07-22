# Archived datasets

Big generated datasets are gitignored (they are large and fully regenerable), so
the folders here exist only on a local checkout. This note is the tracked record.

## `Bursted-ART-1s/`

The **1-second burst** variant of Bursted-ART — the hard/adversarial case where
the synthetic burst arrives too fast for early RDMA to repay the copy (mostly
break-even, with the tail regression in the paper's failure row).

Superseded by the **60-second** variant, which is now the single active dataset
at `workload/generate/out/Bursted-ART/` and the one published to
[shreybirmiwal/Bursted-ART](https://huggingface.co/datasets/shreybirmiwal/Bursted-ART).

Regenerate the 1s variant if ever needed:

```bash
python3 workload/generate/generate_combined_dataset.py \
  --synthetic-burst-window-s 1 \
  --out-dir archive/datasets/Bursted-ART-1s
```
