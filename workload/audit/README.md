# Dataset Audit: are synchronized same-prefix fan-out bursts in public traces?

`audit_burst_absence.py` tests the claim behind Bite-The-Bullet's `early_rdma`
policy — that public LLM inference traces do **not** contain the workload it
targets:

> **K+ requests sharing a long, job-unique prefix, arriving together**
> (a data-labeling sweep, or an agent fan-out to ~20 sub-agents that all inherit
> the same big system prompt / context).

It streams each public trace, measures how large a *same-prefix, co-arriving*
fan-out actually occurs, and writes a JSON + Markdown report.

## Run

```bash
python3 dataset-audit/audit_burst_absence.py --max-rows 300000
# subset / faster:
python3 dataset-audit/audit_burst_absence.py --datasets mooncake control
```

Outputs land in `dataset-audit/results/`:
- `burst_audit.json` — full per-depth / per-window numbers.
- `burst_audit.md`   — the readable report (headline table + verdict).

No credentials needed for the datasets it downloads (ART is a public HF parquet;
Mooncake + BurstGPT are public GitHub files). LMSYS-Chat-1M is gated and ShareGPT
is a timestamp-less conversation dump, so both are handled *structurally* (see
below) rather than downloaded.

## What it measures

For every trace that carries prefix hashes, and for each required shared-prefix
depth `d` (in prefix blocks; Mooncake blocks ≈ 512 tokens):

- `prefix_key = first d block hashes` (only for requests with ≥ d blocks).
- `fanout(key, W)` = max requests sharing that key inside any `W`-second sliding
  window (exact, two-pointer).

Two headline numbers per trace:

| Number | Definition | What it tells you |
| --- | --- | --- |
| **Deep-sync fan-out** | max fan-out at depth ≥ 16 blocks (~8k tok) within ≤ 10 s, **sessions excluded** | the target pattern; ≥ 20 ⇒ present |
| **Shallow-wide fan-out** | max fan-out at depth ≥ 8 blocks within 60 s | the number that *looks* like a burst |

Two discriminators keep the result honest:

1. **Depth.** A global system-prompt template is shared shallowly by everyone;
   a real labeling/agent job shares a *long* prefix. Requiring depth ≥ 16 blocks
   removes the "everyone shares the system prompt" artifact.
2. **Session vs fan-out.** A single agent session's sequential turns also share a
   deep prefix, but its `input_length` grows monotonically as context
   accumulates. `classify_cluster()` labels a co-arriving cluster `session`
   (monotone-growing, wide input-length spread) vs `fanout` (independent
   requests, similar input lengths). Only `fanout` clusters count toward the
   target.

The **positive control** is the local synthetic `Bursted-ART` trace: it *does*
contain the pattern, so it must light up (deep-sync fan-out = 500 in 1 s). That
proves a null result on the real traces is a real absence, not a broken detector.

## Datasets

| Trace | Source | Has prefix hashes? | Has arrival times? |
| --- | --- | :---: | :---: |
| ART-Chat-2.5M | HF `alessiotoniolo/ART-Chat-2.5M` | yes | yes |
| Mooncake conversation / toolagent / arxiv | GitHub `kvcache-ai/Mooncake` FAST25-release | yes | yes |
| BurstGPT | GitHub `HPMLL/BurstGPT` | **no** | yes |
| Bursted-ART (control) | local `data-generation/out/` | yes | yes |
| LMSYS-Chat-1M | HF `lmsys/lmsys-chat-1m` (gated) | no | **no** |
| ShareGPT | scraped conversation dump | no | **no** |

- **BurstGPT** has timestamps + token counts but no prompt content or prefix
  hashes, so same-prefix fan-out is *structurally impossible to express*. Only
  aggregate arrival burstiness is measurable.
- **LMSYS / ShareGPT** are chat-UI conversation dumps with no per-request arrival
  timestamps, so a synchronized arrival burst cannot exist in them by
  construction. Reported as structural absences without a download.

## Interpreting the result

The intended article point: these traces are captured on demo/arena/chat
endpoints and short-lived keys. They contain conversational multi-turn prefix
reuse (small fan-out, spread over minutes) and shared system-prompt templates —
**not** the synchronized long-prefix fan-out of a production data-labeling or
multi-agent job. Read the generated `results/burst_audit.md` for the exact
numbers, including any trace that partially exhibits the pattern; the script does
not suppress a positive finding.
