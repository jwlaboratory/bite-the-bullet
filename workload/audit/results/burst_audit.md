# Audit: are synchronized same-prefix fan-out bursts present in public traces?

_Generated 2026-07-22T17:03:38.665904+00:00 • max_rows/trace=300000 • windows=[1.0, 10.0, 60.0]s • depths=[1, 2, 4, 8, 16, 32] blocks_

**Target pattern:** K+ requests sharing a *long, job-unique* prefix, arriving *together*. Concretely: fan-out ≥ 20 at shared-prefix depth ≥ 16 blocks (~8k tokens) inside a ≤ 10s window. A shallow prefix shared by everyone (a global system-prompt template) or a reuse spread over minutes does **not** count — those are ordinary chat/agent traffic, not a data-labeling sweep or a synchronized sub-agent fan-out.

## Headline

Two numbers per trace. **Deep-sync fan-out** is the target metric (≥16 blocks, ≤10s). **Shallow-wide fan-out** (≥8 blocks, 60s) is the number that *looks* like a burst — its arrival span tells you whether it is a synchronized job (span ≪ window) or a steady shared-template stream (span ≈ window).

| Trace | Rows | Span | **Deep-sync fan-out** (≥16blk,≤10s, sessions excluded) | Largest deep cluster incl. sessions | Shallow fan-out (≥8blk,60s) | Target present? |
| --- | ---: | ---: | ---: | ---: | ---: | :---: |
| ART-Chat-2.5M | 300,000 | 1483.6 min | **25** | 25 (fanout) | 103 (59.636s, steady) | ✅ YES |
| Mooncake-conversation | 12,031 | 58.9 min | **2** | 2 (fanout) | 4 (57s, steady) | ❌ no |
| Mooncake-toolagent | 23,608 | 58.9 min | **2** | 2 (fanout) | 206 (59.999s, steady) | ❌ no |
| Mooncake-arxiv | 23,608 | 60.0 min | **2** | 2 (fanout) | 195 (58.014s, steady) | ❌ no |
| BurstGPT | 300,000 | 21005.4 min | n/a (no prefix) | n/a | raw 884/60s | ❌ n/a |
| Bursted-ART-synthetic (CONTROL) | 76,800 | 7.9 min | **500** | 500 (fanout) | 500 (1s) | ✅ YES |

## Structural absence (no download needed)

- **LMSYS-Chat-1M** (`hf:lmsys/lmsys-chat-1m (gated)`): Chat-UI conversation dump. Gated; and it carries no per-request arrival timestamps -> a synchronized arrival burst cannot exist by construction. It records what users typed into a demo arena, not a production serving endpoint.
- **ShareGPT** (`web-scraped ShareGPT conversations`): Conversation dump scraped from the ShareGPT sharing site. No timestamps and no arrival order -> no burst structure exists to measure. These are shared chat transcripts, not an inference endpoint's request log.

## Per-trace detail

### ART-Chat-2.5M

- source: `hf:alessiotoniolo/ART-Chat-2.5M`
- rows read: 300,000  (with prefix: 300,000)
- time span: 89019.0s (1483.6 min), mean rate 3.37 req/s
- raw arrival burst (any prefix): {'1s': 98, '10s': 343, '60s': 1656}

  Same-prefix fan-out by required shared-prefix depth (window = 60s):

  | depth (blocks) | prefix groups | max fan-out | span(s) | groups ≥20 |
  | ---: | ---: | ---: | ---: | ---: |
  | 1 | 4,167 | 337 | 59.83 | 76 |
  | 2 | 6,956 | 177 | 59.808 | 114 |
  | 4 | 4,087 | 108 | 59.808 | 120 |
  | 8 | 3,925 | 103 | 59.636 | 166 |
  | 16 | 4,888 | 82 | 59.227 | 171 |
  | 32 | 7,365 | 82 | 59.227 | 143 |

  → **Target pattern PRESENT.** Fan-out-like cluster of 25 independent requests at depth ≥16 blocks within 2.809s.


### Mooncake-conversation

- source: `https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/conversation_trace.jsonl`
- rows read: 12,031  (with prefix: 12,031)
- time span: 3537.0s (58.9 min), mean rate 3.40 req/s
- raw arrival burst (any prefix): {'1s': 28, '10s': 73, '60s': 263}

  Same-prefix fan-out by required shared-prefix depth (window = 60s):

  | depth (blocks) | prefix groups | max fan-out | span(s) | groups ≥20 |
  | ---: | ---: | ---: | ---: | ---: |
  | 1 | 1 | 263 | 60.0 | 1 |
  | 2 | 7,373 | 5 | 24.0 | 0 |
  | 4 | 6,015 | 5 | 24.0 | 0 |
  | 8 | 5,277 | 4 | 57.0 | 0 |
  | 16 | 3,775 | 4 | 57.0 | 0 |
  | 32 | 1,945 | 4 | 57.0 | 0 |

  → **Target pattern ABSENT.** At a genuinely long shared prefix (≥16 blocks ≈ 8k tokens) the largest *fan-out* of independent requests in any 10s window is only **2**. The largest *shallow* (≥8-block) fan-out is 4 over 57.0s (steady shared-template stream (not a synchronized job)) — a shared system-prompt template, not a labeling/fan-out job.


### Mooncake-toolagent

- source: `https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/toolagent_trace.jsonl`
- rows read: 23,608  (with prefix: 23,608)
- time span: 3537.0s (58.9 min), mean rate 6.67 req/s
- raw arrival burst (any prefix): {'1s': 47, '10s': 132, '60s': 497}

  Same-prefix fan-out by required shared-prefix depth (window = 60s):

  | depth (blocks) | prefix groups | max fan-out | span(s) | groups ≥20 |
  | ---: | ---: | ---: | ---: | ---: |
  | 1 | 4 | 243 | 60.0 | 3 |
  | 2 | 6,554 | 206 | 59.999 | 2 |
  | 4 | 5,349 | 206 | 59.999 | 2 |
  | 8 | 5,128 | 206 | 59.999 | 1 |
  | 16 | 3,825 | 5 | 48.0 | 0 |
  | 32 | 1,711 | 4 | 57.0 | 0 |

  → **Target pattern ABSENT.** At a genuinely long shared prefix (≥16 blocks ≈ 8k tokens) the largest *fan-out* of independent requests in any 10s window is only **2**. The largest *shallow* (≥8-block) fan-out is 206 over 59.999s (steady shared-template stream (not a synchronized job)) — a shared system-prompt template, not a labeling/fan-out job.


### Mooncake-arxiv

- source: `https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/arxiv-trace/mooncake_trace.jsonl`
- rows read: 23,608  (with prefix: 23,608)
- time span: 3600.0s (60.0 min), mean rate 6.56 req/s
- raw arrival burst (any prefix): {'1s': 47, '10s': 132, '60s': 472}

  Same-prefix fan-out by required shared-prefix depth (window = 60s):

  | depth (blocks) | prefix groups | max fan-out | span(s) | groups ≥20 |
  | ---: | ---: | ---: | ---: | ---: |
  | 1 | 4 | 236 | 58.018 | 3 |
  | 2 | 6,557 | 195 | 58.014 | 2 |
  | 4 | 5,348 | 195 | 58.014 | 2 |
  | 8 | 5,128 | 195 | 58.014 | 1 |
  | 16 | 3,825 | 5 | 48.856 | 0 |
  | 32 | 1,707 | 4 | 58.017 | 0 |

  → **Target pattern ABSENT.** At a genuinely long shared prefix (≥16 blocks ≈ 8k tokens) the largest *fan-out* of independent requests in any 10s window is only **2**. The largest *shallow* (≥8-block) fan-out is 195 over 58.014s (steady shared-template stream (not a synchronized job)) — a shared system-prompt template, not a labeling/fan-out job.


### BurstGPT

- source: `https://raw.githubusercontent.com/HPMLL/BurstGPT/main/data/BurstGPT_1.csv`
- rows read: 300,000  (with prefix: 0)
- time span: 1260322.0s (21005.4 min), mean rate 0.24 req/s
- raw arrival burst (any prefix): {'1s': 52, '10s': 216, '60s': 884}
- note: No prefix/content in this trace -> same-prefix fan-out is structurally impossible to express. Only aggregate arrival burstiness is measurable.

### Bursted-ART-synthetic (CONTROL)

- source: `data-generation/out/Bursted-ART/test.jsonl`
- rows read: 76,800  (with prefix: 76,800)
- time span: 476.1s (7.9 min), mean rate 161.32 req/s
- raw arrival burst (any prefix): {'1s': 3670, '10s': 8242, '60s': 19274}

  Same-prefix fan-out by required shared-prefix depth (window = 60s):

  | depth (blocks) | prefix groups | max fan-out | span(s) | groups ≥20 |
  | ---: | ---: | ---: | ---: | ---: |
  | 1 | 2,465 | 1389 | 59.92 | 156 |
  | 2 | 2,730 | 500 | 1.0 | 167 |
  | 4 | 2,526 | 500 | 1.0 | 165 |
  | 8 | 2,710 | 500 | 1.0 | 150 |
  | 16 | 2,854 | 500 | 1.0 | 143 |
  | 32 | 2,883 | 500 | 1.0 | 138 |

  → **Target pattern PRESENT.** Fan-out-like cluster of 500 independent requests at depth ≥16 blocks within 1.0s.


## Verdict

**1. Chat / arena dumps (LMSYS-Chat-1M, ShareGPT):** no per-request arrival timestamps → a synchronized burst cannot exist by construction. Not a serving log.

**2. Aggregate arrival traces (BurstGPT):** timestamps and token counts, but **no prefix/content at all** → same-prefix fan-out is structurally impossible to express, regardless of how bursty the raw arrival rate is (peak 884 req/60s here).

**3. Prefix-bearing serving traces (Mooncake-conversation, Mooncake-toolagent, Mooncake-arxiv):** carry both prefix hashes and timing, yet the largest synchronized fan-out of independent requests over a long shared prefix is **2** — no data-labeling or sub-agent fan-out job. Their big-looking numbers are shallow, steady system-prompt templates, or single growing-context sessions.

**4. ART-Chat-2.5M:** the *only* trace that contains the pattern at all — but marginally. Max fan-out **25-way** (vs the control's 500-way, and vs the hundreds-to-thousands of a real production labeling sweep), and only **3 such events in 300,000 requests** (~1.0 per 100k, ≈0.1/hour). ART is a specialized decoded-LLM-response / agent-eval corpus (request IDs are batch-timestamped `decoded_llm_responses_…`), i.e. already the most batch-like public dataset — which is exactly why BTB builds on ART and then *adds* synthetic bursts to reach production scale.

**Positive control (Bursted-ART-synthetic (CONTROL)):** deep-sync fan-out **500** in 1.0s — the detector fires hard when the pattern is present, so the absences above are real, not a broken detector.

### Bottom line

> The synchronized long-prefix fan-out that KV-warming targets is **absent from every general-endpoint public trace** and **structurally impossible in the chat/arena dumps**. The one trace that shows it (ART) is a specialized agent/eval decode corpus, and even there it is rare and an order of magnitude below production scale. These datasets were captured on demo/arena/chat endpoints and short-lived keys — not on production endpoints running data-labeling or multi-agent jobs — so none of them represent that regime.
