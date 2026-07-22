# Bite The Bullet: Early RDMA For Shared-Prefix Bursts

This repo studies one narrow mechanism:

> Detect sustained reuse of a long prompt prefix, then RDMA-copy that prefix KV
> into the HBM of less-busy replicas before later requests arrive.

The active policy is `early_rdma`. It does not speculate by recomputing prefill
on idle GPUs. It only moves KV that already exists in HBM.

The short paper memo is `experiments/3-early-rdma/PAPER.md`. This blog is the longer
explanation: what the method is, when it should work, when it should fail, how
the dataset is generated, and where the artifacts live.

The experiments run in
[Infer-Sim](https://jwlabs.vercel.app/post/infer-sim), our open-source
inference simulator for trace replay, routing, queueing, batching, and
prefix-cache policy experiments. These are simulator results, not hardware
measurements.

## The Problem

Some inference workloads send many requests that share a long prefix:

- data-labeling jobs scoring many records against the same document;
- batch extraction over one shared context;
- agent fanout where many subagents inherit the same system prompt;
- evaluation workloads that sample many answers from the same prompt.

For those requests, the expensive part is the shared prefill. Once one replica
has computed KV for the prefix, later requests can be much cheaper if they land
where that KV already lives.

But pure cache affinity creates another problem. If every request goes to the
same replica, that replica can form a queue while other replicas sit less busy.
Least-load routing fixes the queue, but it may throw away KV locality.

BTB is trying to get both:

1. keep KV locality;
2. spread later requests across multiple replicas;
3. pay the KV movement cost early enough that request TTFT improves.

## The Policy

We study one policy: `early_rdma`.

When requests arrive, the router watches prefix-block hashes. If the same prefix
appears enough times inside a short window, that prefix becomes active. Later
requests with that active prefix trigger a prefetch attempt.

The prefetch rule is simple:

1. Find a source replica that already has the prefix KV in HBM.
2. Pick less-busy target replicas.
3. RDMA-copy the KV into target HBM until the prefix has the configured number
   of resident copies.
4. Route future same-prefix requests across those warmed replicas.

In the current experiments, the target is 4 total HBM copies of the hot prefix.
So yes, BTB can send KV to multiple replicas. It is not just copying to one GPU.
A replica may itself contain multiple GPUs; the simulator models the serving
replica as the scheduling target.

## How It Knows When To Fire

The current detector is deliberately simple:

```text
if the same prefix appears enough times inside a short time window:
    mark that prefix active
```

This is not a trained model yet. It is a gate around an observable signal:
repeated same-prefix arrivals. That makes the result easier to reason about.
We are testing whether the mechanism pays off when the burst signal is present,
not claiming we have solved every predictor problem.

The policy should fire when future reuse is likely large enough to repay the
copy. It should not fire for isolated repeats, tiny prefixes, or bursts that
finish before the copied KV can be reused.

## Cost Model

The simulator does account for RDMA transfer cost.

For each prefetch:

```text
bytes_moved = warm_blocks * block_bytes
copy_time = bytes_moved / rdma_bandwidth
```

That copy time advances the target replica's local busy timeline. Results track:

- `warm_gb`: total KV moved;
- `warm_busy_s`: simulated busy time caused by prefetch;
- `warm_count`: number of prefetch actions.

This captures per-replica bandwidth cost and queue interference. It does not
yet model a full shared network fabric, source-NIC contention, switch-level
congestion, or multi-tenant RDMA interference. So the current claim should be
read as "bandwidth-cost-aware," not "full-network-contention-complete."

## Why We Had To Build A Dataset

Before generating anything, we checked whether the workload already exists in the
public traces these papers usually reach for. The learned-prefix-caching work and
"Not All Tokens Are Worth Caching," for example, lean on LMSYS-Chat-1M and
ShareGPT. The problem is that none of the standard datasets actually contain the
pattern `early_rdma` targets: a **synchronized deep-prefix fan-out** — many
requests that share a long, job-unique prefix and arrive together, the way a
data-labeling sweep or a multi-agent job hits a serving endpoint.

We audited them (`workload/audit/`). We define a **deep** burst precisely: ≥ 20
requests sharing a prefix of **≥ 16 blocks (~8k tokens)**, all arriving inside a
**≤ 10-second window**. The results split cleanly into three failure modes:

- **LMSYS-Chat-1M and ShareGPT don't have arrival timestamps at all.** They are
  chat-arena / share-link conversation dumps, so a burst cannot exist by
  construction — there is no arrival process to be bursty. They record what users
  typed into a demo, not the request log of a production endpoint.
- **BurstGPT has timestamps but no prompts or prefix hashes.** You can see the
  arrival rate spike, but with no content there is no way to tell whether a spike
  is many requests sharing a prefix or just unrelated traffic. The one thing the
  policy keys on — repeated shared prefixes — simply isn't recorded.
- **Mooncake and ART-Chat have both, and still barely show it.** In the Mooncake
  traces (conversation, tool-agent, arxiv), the largest synchronized deep-prefix
  fan-out is **2** — essentially none. The big-looking numbers there (~200
  requests sharing the first few blocks) are a single system-prompt *template*
  shared by a steady trickle of unrelated requests across the whole hour; require
  a genuinely long shared prefix and it collapses to single digits. ART-Chat-2.5M
  is the only public trace that contains the pattern at all, and only marginally:
  **3 qualifying events in 300,000 requests**, the largest a **25-way** fan-out
  (within ~2.8 s) — versus the hundreds-to-thousands of a real production labeling
  job. Tellingly, ART is already a specialized decoded-LLM-response / agent-eval
  corpus, which is exactly why we build on it.

![Public LLM traces measured for the pattern early_rdma targets. Left: the largest
synchronized deep-prefix fan-out per dataset (log scale) — Mooncake 2, ART 25, both
at or below the 20-way threshold. Right: fan-out collapses as you demand a longer
shared prefix in the real traces.](workload/audit/results/burst_audit_chart.png)

The takeaway is simple — these datasets were
captured on demo, arena, and short-lived chat/coding endpoints, not on production
endpoints running data-labeling or multi-agent fan-out jobs, so none of them
represent the regime this mechanism is for. That is why we built one.

The full report and per-dataset numbers are in
`workload/audit/results/burst_audit.md`; regenerate everything with
`python3 workload/audit/audit_burst_absence.py`.

## Dataset: Bursted-ART

The current dataset work lives outside the experiment harness in
`workload/generate/`.

Hugging Face repo:
[shreybirmiwal/Bursted-ART](https://huggingface.co/datasets/shreybirmiwal/Bursted-ART)

Local generated dataset: `workload/generate/out/Bursted-ART/` (gitignored —
regenerate with the command below). The folder contains:

- `train.jsonl`
- `test.jsonl`
- `dataset_info.json`
- `README.md`

`Bursted-ART` is a mixed dataset. "Mixed" means real ART replay windows plus
synthetic same-prefix fanout windows. "Synthetic" means only the generated
fanout windows.

The ART rows come from `alessiotoniolo/ART-Chat-2.5M`. We keep the trace shape:
timestamps, prompt lengths, output lengths, request IDs, session IDs, group IDs,
and prefix hashes. Raw ART messages are omitted; message byte counts are kept in
metadata.

The synthetic rows are added because public traces do not reliably contain the
large synchronized same-prefix bursts that this mechanism is designed for. A
synthetic window models a batch or agent-fanout job:

- 8 burst jobs per synthetic window;
- 500 requests per burst;
- 65,536 shared prefix tokens;
- 256 unique suffix tokens per request;
- 1 output token;
- 120 one-request decoy jobs;
- 6 seconds of predictor lead time.

The default generated dataset has:

- 40 complete trace windows;
- 20 ART windows and 20 synthetic windows;
- 10 train windows and 30 test windows;
- 102,400 request rows total;
- 25,600 train rows and 76,800 test rows;
- 20,000 ART rows and 82,400 synthetic rows.

The split is by complete trace window, not individual request row. That is
important because requests inside one burst are correlated. Row-level splitting
would leak burst structure from train to test.

The synthetic bursts span **60 seconds**: reuse is sustained long enough that
early movement can pay back, which is the regime the paper's headline result
comes from. (An earlier 1-second fast-burst variant — the hard case where the
burst is over before the copy lands — is archived under
`archive/datasets/`; regenerate it with `--synthetic-burst-window-s 1` if
needed.)

Generate the uploaded dataset:

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

The exact generation args and window boundaries are recorded in each
`dataset_info.json`.

## Experiment Setup

The current paper result uses four model/hardware setups:

- `dense1t_b300x4`
- `70b_h100x4_base`
- `kimi_k2_h100x8`
- `glm52_h100x8`

Each setup is evaluated on both datasets:

- synthetic-only held-out windows;
- mixed held-out windows with ART plus synthetic traffic.

The active results are stored in:

- `experiments/3-early-rdma/PAPER.md`
- `experiments/3-early-rdma/results/results.json`

Older speculative-prefill, partial-prefix, Mooncake, and model-sweep result
dumps are preserved under `archive/`.

## Results

Positive numbers mean lower TTFT than the baseline. Negative numbers mean BTB
made TTFT worse.

| Case | Mean TTFT | P95 TTFT | Speedup |
| --- | ---: | ---: | ---: |
| Best: 60s synthetic `kimi_k2_h100x8` | +15.01% | +3.93% | 1.1766x |
| Mixed: 60s mixed `kimi_k2_h100x8` | +8.60% | +3.37% | 1.0941x |
| Medium: 1s mixed `70b_h100x4_base` | +2.26% | +1.59% | 1.0231x |
| Bad tail: 1s synthetic `dense1t_b300x4` | -3.05% | -16.91% | 0.9704x |

The clean takeaway:

> BTB shines when a long shared prefix is reused over a sustained window, and
> RDMA can create extra HBM copies before the future requests need them.

The 60-second synthetic case is the ideal paper case. It represents a workload
where requests keep arriving for the same prefix long enough that one early KV
copy can serve many later requests.

The 1-second case is much harder. If the burst is mostly over before the copy
pays back, extra movement can hurt the tail.

## When It Is Good

BTB is a good idea when:

- the prefix is long;
- many future requests reuse that prefix;
- the source KV already exists in HBM;
- the burst lasts long enough for copies to be reused;
- target replicas have enough HBM and queue slack;
- RDMA is cheaper than repeated prefill or remote KV access.

That is why the 60-second same-prefix workload is the best case. The movement
cost is paid once, then amortized over many future requests.

## When It Is Bad

BTB is bad when:

- the burst is too short;
- the prefix is small;
- reuse is weak or noisy;
- target replicas are already busy;
- HBM pressure evicts something more useful;
- RDMA traffic competes with more important work;
- the workload is tail-sensitive and copied KV is not reused quickly.

This is why the paper should not claim "always prefetch." The right claim is
more precise:

> Predictive, tail-aware KV movement can improve TTFT for sustained
> shared-prefix bursts, but should be gated by reuse, lead time, HBM pressure,
> and bandwidth cost.

## Paper Shape

The paper should show three regimes:

1. Best case: sustained same-prefix reuse, where BTB clearly helps.
2. Medium case: mixed ART plus synthetic traffic, where BTB helps modestly.
3. Worst case: short or tail-sensitive bursts, where BTB can regress.

That gives the story a spine. We are not trying to hide the failure case. The
failure case tells us exactly what the production gate must avoid.

## Next Experiment

The next clean ideal-case experiment is a 10-minute rising-load trace:

- one hot long prefix;
- request rate rises over time;
- the detector catches the burst early;
- `early_rdma` creates more HBM copies;
- future requests spread across warmed replicas.

That experiment should plot TTFT, p95 TTFT, HBM hit rate, RDMA GB, warm busy
time, and time-to-payback. It is the clearest version of the bet: use early
signal to move KV before the queue forms.
