# audit — is the burst pattern in public traces?

`audit_burst_absence.py` streams public request traces (ART-Chat, Mooncake,
BurstGPT) and measures the pattern `early_rdma` targets: a **deep-sync fan-out**
— ≥ 20 requests sharing a prefix of ≥ 16 blocks (~8k tokens), all arriving inside
a ≤ 10-second window.

The answer is mostly **no**: the largest deep-sync fan-out is 2 in Mooncake and
25 in ART-Chat (vs the hundreds-to-thousands of a real data-labeling sweep), and
the chat/arena dumps have no arrival timestamps at all — which is why the
workload has to be synthesized ([`../generate/`](../generate/)).

```bash
python3 3-workload/audit/audit_burst_absence.py --out-dir 3-workload/audit/results
python3 3-workload/audit/make_chart.py     # -> results/burst_audit_chart.png
```

Outputs land in `results/`: `burst_audit.json` (full numbers), `burst_audit.md`
(readable report), and the chart.
