# Prefill vs. KV-cache transfer: the cost that warming spends

Why is prefilling expensive? Because **prefill is compute-bound while reuse is
bandwidth-bound.** For a prompt of N tokens there are two ways to get its KV
cache resident so decoding can start:

1. **Recompute it** — prefill, `2 × active_params` FLOPs per token
   (`gpu.Node.prefill_time`).
2. **Move it in** — stream the already-computed KV from a memory tier
   (HBM / RAM / RDMA / disk), `bytes / bandwidth` (`gpu.Node.load_time`).

Both are linear in N, so the prefill/transfer **ratio is constant** — it holds at
any prompt length and is the entire budget the bite-the-bullet warming policy is
spending when it fills a cheaper tier instead of prefilling again.

Every number is derived from the real simulator constants
(`../../inference-sim/config.py`, `gpu.py`) via the `sim_path` shim — nothing is hardcoded.

## The standard setup (per node)

One node = **4× H100 tensor-parallel**, serving **Llama-3.3-70B fp16**
(`config.CLUSTER[0]`). Compute, HBM, PCIe and RDMA aggregate across the 4 GPUs;
the NVMe pool is shared (not multiplied).

| Tier | Per-GPU (datasheet) | **Per-node (×4)** | Role |
|------|--------------------:|------------------:|------|
| **HBM** | 3.35 TB/s | **13.4 TB/s** | local GPU memory (bandwidth floor / local hit) |
| **RAM (PCIe)** | 55 GB/s | **220 GB/s** | KV offloaded to host DRAM |
| **RDMA** | 50 GB/s (400G NIC) | **200 GB/s** | a peer node's KV over the fabric |
| **Disk / NVMe** | 7 GB/s | **7 GB/s** (shared) | local SSD prefix cache |
| **Prefill** | 989 TFLOP/s peak | **1.98 PFLOP/s eff** (MFU 0.5) | recompute |

- **KV size:** `2 × 80 layers × 8 KV heads × 128 dim × 2 B` = **320 KiB/token**.
- **Prefill:** `2 × 70.6e9 / 1.98e15` = **71.4 µs/token**.

> Bandwidths above are node-aggregate (the honest unit here — a KV load streams
> across all 4 GPUs at once). For per-GPU datasheet numbers, divide bandwidth and
> multiply time by 4; the ratios below are unchanged.

## Result

Time to make N tokens of KV resident (milliseconds):

| Source | 500 | 1k | 2k | 8k | 16k | 32k | vs. prefill |
|--------|----:|---:|---:|---:|----:|----:|------------:|
| **Prefill (recompute)** | 35.7 | 71.4 | 142.8 | 571 | 1142 | 2284 | 1× |
| Disk / NVMe | 23.4 | 46.8 | 93.6 | 374 | 749 | 1498 | **1.5×** faster |
| RDMA (remote GPU) | 0.82 | 1.64 | 3.28 | 13.1 | 26.2 | 52.4 | **44×** faster |
| RAM (host, PCIe) | 0.75 | 1.49 | 2.98 | 11.9 | 23.8 | 47.7 | **48×** faster |
| HBM (local floor) | 0.012 | 0.024 | 0.049 | 0.20 | 0.39 | 0.78 | **2919×** faster |

**Takeaways for the paper:**

- Anywhere KV already exists in HBM / RAM / RDMA, moving it costs **1–2 orders of
  magnitude less** than the FLOPs to regenerate it (≈48× vs RAM, ≈44× vs RDMA).
- A **local HBM prefix hit is essentially free** relative to prefill (~2900×).
- **Disk is the exception:** NVMe (7 GB/s, unaggregated) is only **~1.5×** faster
  than recomputing — which is why pulling KV off local SSD barely beats prefill.

*Caveat:* these are steady-state bandwidth/compute models; they exclude fixed
per-transfer setup latency (RDMA handshake, PCIe/kernel launch), negligible at
these sizes but dominant for a single-token move.

## Files

| File | What |
|------|------|
| `compute_costs.py` | Derives all numbers from the sim; writes `results.json`, prints the table. |
| `results.json` | Machine-readable setup + per-token-count costs + ratios. |
| `chart.html` | Standalone log-log chart (self-contained SVG). |
| `chart-full.html` | Chart + data table + setup-spec tiles (theme-aware). |
| `figures/prefill-vs-transfer-chart.png` | Chart only, 2856×1560 @3× — the figure for LaTeX. |
| `figures/prefill-vs-transfer-full.png` | Full page (chart + table + specs) @2×. |

## Reproduce

```bash
# numbers + results.json (needs the sibling inference-sim checkout)
python 1-prefill-vs-transfer/compute_costs.py

# re-render the PNGs from the HTML (headless Chrome, no deps)
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
cd 1-prefill-vs-transfer
"$CHROME" --headless --disable-gpu --hide-scrollbars --force-device-scale-factor=3 \
  --default-background-color=FFFFFFFF --window-size=952,520 \
  --screenshot=figures/prefill-vs-transfer-chart.png "file://$PWD/chart.html"
```
