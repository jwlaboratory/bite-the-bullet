# Bite the Bullet

## One-Line Claim

BTB helps when we detect a sustained shared-prefix burst early enough to RDMA-copy existing KV into HBM before later requests arrive.

These experiments were run in
[Infer-Sim](https://jwlabs.vercel.app/post/infer-sim), our open-source
inference simulator for trace replay, routing, queueing, batching, and
prefix-cache policy experiments.

## Policy

We study one policy: `early_rdma`.

1. Detect repeated use of the same prefix.
2. Check that a source KV copy already exists in HBM.
3. RDMA-copy that KV to other replicas.

No speculative prefill. No seed policy. Just move existing KV.

## How It Decides

BTB triggers on a simple burst rule:

> if the same prefix appears enough times inside a short time window, mark that prefix active.

After a prefix is active, later requests with that prefix try to prefetch its KV.

## Where It Sends KV

BTB can send KV to multiple replicas, not just one GPU.

- A **replica** is a serving node. Each replica may contain multiple GPUs.
- We copy KV into target replicas until the prefix has the configured number of HBM copies.
- In these experiments, the target was 4 total HBM copies.
- Targets are chosen by current load, so less-busy replicas get warmed first.

## Cost Model

We do account for RDMA transfer cost.

- Each copy moves `warm_blocks * block_bytes`.
- Copy time is `bytes / rdma_bandwidth`.
- The copy advances the target replica's local busy timeline.
- Results track `warm_gb`, `warm_busy_s`, and `warm_count`.

This models per-replica bandwidth cost and queue interference. It does **not** model a full shared network fabric, source-NIC contention, or switch-level congestion.

## Dataset

We created **Bursted-ART** by combining real ART traffic with synthetic hot-prefix bursts.

- **ART windows:** real ART-Chat request windows. We keep timestamps, prompt lengths, output lengths, and prefix hashes.
- **Synthetic windows:** controlled same-prefix fanout. Each synthetic window has 8 burst jobs, 500 requests per burst, a 65,536-token shared prefix, a 256-token unique suffix, 1 output token, and 120 decoy jobs.
- **Split:** 40 complete windows total: 10 train, 30 test. The split is by whole window, not by row.
- **Size:** 102,400 rows total: 25,600 train and 76,800 test.

We use two versions:

- **Bursted-ART:** synthetic bursts span 1 second.
- **Bursted-ART-60s:** same generation, but synthetic bursts span 60 seconds.

In the tables:

- **Synthetic** means we evaluate only held-out synthetic burst windows.
- **Mixed** means we evaluate held-out ART and synthetic windows together.

## When It Works

BTB works when:

- the prefix is long;
- many future requests reuse it;
- the burst lasts long enough for prefetch to pay back;
- HBM has room;
- RDMA is cheaper than repeated prefill or remote KV access.

It fails when the copied KV is not reused soon enough or when extra movement hurts tail latency.

## Results

Positive numbers mean lower TTFT than baseline.

| Case | Mean TTFT | P95 TTFT | Speedup |
| --- | ---: | ---: | ---: |
| Best: 60s synthetic `kimi_k2_h100x8` | +15.01% | +3.93% | 1.1766x |
| Mixed: 60s mixed `kimi_k2_h100x8` | +8.60% | +3.37% | 1.0941x |
| Medium: 1s mixed `70b_h100x4_base` | +2.26% | +1.59% | 1.0231x |
| Bad tail: 1s synthetic `dense1t_b300x4` | -3.05% | -16.91% | 0.9704x |

## Paper Shape

The paper should be honest and simple:

- **Best case:** long sustained reuse.
- **Realistic case:** mixed traces still improve.
- **Failure case:** dense/tail-sensitive workloads can regress.

The right claim is not "always prefetch." It is:

> Predictive, tail-aware KV movement under a break-even utility model.

## Next Experiment

Run a 10-minute rising-load trace:

- one hot long prefix;
- request rate rises over time;
- compare baseline vs `early_rdma`;
- plot TTFT, p95, HBM hit rate, RDMA GB, and time-to-payback.

This should show the ideal BTB regime: fixed movement cost, rising future reuse.

## Artifact

Raw filtered results: `results.json`.
