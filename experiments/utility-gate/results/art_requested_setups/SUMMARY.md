# ART Utility-Gate Requested Setups

Run date: 2026-07-21.

Protocol: ART-Chat-2.5M parquet replay, 12 windows x 500 rows, first 8
windows for training/selection, last 4 held out for evaluation. Threshold/top-k
selection used the final 3 training windows. The trained objective was:

```text
mean_ttft + 0.0001 * warm_gb + 0.001 * warm_busy_s
```

All runs used 8-block / 2,048-token speculative fake-prefill warming, cache-aware
HBM-only baseline routing, and up to 4 warmed replicas.

| Setup | Hardware | Threshold/top-k | Held-out objective delta | Mean TTFT delta | p95 TTFT delta | Triggers/window | Warm GB/window | Warm busy/window | Verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `kimi-code-1t-int4` | 8 x 8 H100 | 0.45 / all | +0.0021s | +0.0016s | -0.0173s | 7.8 | 0.558 | 0.514s | Mixed; do not ship as-is |
| `glm52-int4` | 8 x 8 H100 | 0.05 / all | -0.0027s | -0.0038s | -0.0076s | 13.0 | 1.139 | 1.025s | Accept as weak-positive |
| `qwen3-8b` | 8 x 1 H100 | 0.00 / 1 | +0.0246s | +0.0243s | -0.0424s | 0.8 | 0.679 | 0.153s | Reject for mean/objective |
| `qwen3-8b-int4` | 8 x 1 H100 | 0.20 / all | -0.0230s | -0.0249s | -0.0954s | 10.8 | 1.906 | 1.713s | Accept as positive, but high-fire |

Oracle-greedy replay remained better than the trained gate in every setup,
which means the opportunity exists but the current gate is still leaving
selection quality on the table.

## Artifacts

- `btb_utility_gate_art_kimi_code_1t_int4_w12.json`
- `btb_utility_gate_art_glm52_int4_w12.json`
- `btb_utility_gate_art_qwen3_8b_w12.json`
- `btb_utility_gate_art_qwen3_8b_int4_w12.json`

