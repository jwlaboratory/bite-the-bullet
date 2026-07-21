# Literature review: LLM inference routing, scheduling, and caching

Scope: the systems research that this simulator models — cluster-level request
routing, node-level batch scheduling, multi-tier KV caching, parallelism
placement, and autoscaling — with reported numbers from each paper/system and
notes on how each maps onto `inference-sim`. Findings from our own experiments
on the ART-Chat-2.5M trace (see `docs/learned-routing-and-predictive-scaling.md`
for the two open research questions) are folded in at the end.

---

## 1. Cluster-level routing (what `router.py` models)

**SGLang cache-aware load balancer** ([v0.4 blog](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/)).
The scheme `router.py`'s `CacheAware` is modeled on: route to the worker with
the longest prefix match in an approximate radix tree mirrored at the router,
fall back to shortest-queue when imbalance thresholds (`balance_abs`,
`balance_rel`) trip. Reported up to **1.9× throughput and 3.8× cache-hit-rate**
vs. round-robin, approaching 96% of optimal hit rate. Key limitation (confirmed
in our experiments): the fallback is binary — the router is always ignoring
either the cache signal or the load signal.

**NVIDIA Dynamo KV Router**
([design](https://docs.nvidia.com/dynamo/latest/router/README.html),
[guide](https://docs.nvidia.com/dynamo/latest/user-guides/kv-cache-aware-routing)).
The successor design: a *continuous* cost function,
`logit = kv_overlap_score_weight × potential_prefill_blocks + potential_active_blocks`,
scoring every worker on cache-adjusted prefill cost plus current load — no
threshold switch. NVIDIA reports **~50% TTFT and ~34% TPOT reductions** on
average; [Baseten measured ~2× faster inference](https://www.baseten.co/blog/how-baseten-achieved-2x-faster-inference-with-nvidia-dynamo/)
after adopting it. Our sim experiments reproduced the qualitative result: a
continuous `work + prefill_after_cache_credit` score was the only policy near
the top on every metric simultaneously.

**Preble** ([arXiv:2407.00023](https://arxiv.org/abs/2407.00023)).
Distributed *prompt* scheduling: a global request-level scheduler (E2
algorithm) explicitly trades KV reuse (exploitation) against load spreading
(exploration), and — notably — **replicates hot prefixes across GPUs** when
demand for one prefix exceeds a single GPU. **1.5–14.5× average and 2–10× p99
latency** improvement over SGLang/vLLM baselines. Relevant gap for the sim:
we model prefix *fetching* over RDMA but not deliberate prefix *replication*.

**TrueFoundry's practitioner survey of KV-cache routing** ([blog](https://www.truefoundry.com/blog/kv-cache-routing-why-standard-load-balancers-break-prefix-caching-and-how-to-fix-it))
is a good short overview of why classic load balancers (round-robin,
least-connections) break prefix caching in production.

## 2. Node-level scheduling (what `simulate.advance` models)

**Orca** ([OSDI '22](https://www.usenix.org/conference/osdi22/presentation/yu))
introduced iteration-level (continuous) batching — the admission/decode loop
`advance()` implements. **vLLM / PagedAttention**
([arXiv:2309.06180](https://arxiv.org/abs/2309.06180)) made KV memory block-
granular; the sim's block-hash caches inherit this framing, with SGLang's
RadixAttention ([arXiv:2312.07104](https://arxiv.org/abs/2312.07104)) as the
prefix-tree equivalent.

**The FCFS / head-of-line-blocking literature.** Most engines admit FCFS; a
long request at the head starves short ones. A cluster of papers attacks this
with *predicted output length*:

- [S3 (NeurIPS '23)](https://arxiv.org/abs/2306.06000) — predict output length
  to pack batches; up to **6.49× throughput**.
- [SSJF](https://arxiv.org/pdf/2404.08509) — BERT-base proxy predicts
  verbosity, speculative shortest-job-first.
- [Learning to Rank (NeurIPS '24)](https://arxiv.org/pdf/2408.15792) — don't
  predict lengths, predict their *ranking* within a batch; **2.8× lower
  chatbot latency, 6.5× throughput** on synthetic-data workloads.
- [PARS](https://arxiv.org/html/2510.03243v2) — pairwise-ranking SJF
  approximation with margin loss.
- [ELIS](https://arxiv.org/pdf/2505.09142) — iterative shortest-remaining-
  time-first with a response-length predictor; up to **19.6% lower JCT**.
- [Don't Stop Me Now](https://arxiv.org/pdf/2410.01035) — embedding-based
  scheduling; [Multi-Bin Batching](https://arxiv.org/pdf/2412.04504) — bin
  requests by predicted length so batches finish together.
- [TetriInfer](https://arxiv.org/abs/2401.11181) — length prediction to avoid
  decode hotspots when packing instances.

**Our measured counterpoint:** on this sim's trace, output-length prediction
adds ~nothing to *routing* (oracle ≈ no-predictor; a σ=1.5 noisy predictor was
*worse* than count-based least-load) — the literature's gains come from
*ordering within a node*, a mechanism the sim does not yet expose (admission
is strictly FCFS, `simulate.py:81`). Prediction accuracy is a cliff, not a
slope: mediocre predictors actively hurt.

## 3. KV cache tiering (what `gpu.py`'s hbm/ram/rdma/disk models)

**Mooncake** ([arXiv:2407.00079](https://arxiv.org/abs/2407.00079), FAST '25
best paper) — KV-cache-centric disaggregated serving at Moonshot/Kimi:
cluster-wide prefix cache pool spanning GPU HBM, CPU DRAM, and SSD, plus
SLO-aware early rejection under overload. The trace format this sim replays
(256-token block hashes) is Mooncake's.

**LMCache** ([lmcache.ai](https://lmcache.ai)) — the open-source KV cache
layer for vLLM: chunk-hashed (256-token default) KV moved across
HBM → CPU DRAM → local disk → remote/P2P backends, LRU per tier, "load only
when faster than recompute." The sim's tier ladder and `min(load, prefill)`
rule mirror this; the sim simplifies by making RAM and disk infinite (only
HBM has real LRU eviction — see `gpu.py:93-102`).

**SGLang HiCache** ([blog](https://www.lmsys.org/blog/2025-09-10-sglang-hicache/))
— hierarchical KV caching with pluggable storage backends.
**TokenLake** ([arXiv:2508.17219](https://arxiv.org/pdf/2508.17219)) — a
*pooled* segment-level prefix cache shared by all serving instances, directly
attacking the cache-fragmentation problem our pooled-node experiment exposed.

## 4. Parallelism placement and disaggregation

**AlpaServe** ([OSDI '23](https://arxiv.org/abs/2302.11665)) — model
parallelism as *statistical multiplexing*: splitting a model across GPUs even
when it fits on one is worth the communication overhead **when workloads are
bursty**, because a burst can then borrow the whole cluster's compute.
**10× higher sustainable rate / 6× more burstiness** within SLO on production
traces. This is the direct ancestor of the "split when a batch is coming"
idea; see the companion doc.

**DistServe** ([arXiv:2401.09670](https://arxiv.org/abs/2401.09670)) and
**Splitwise** ([arXiv:2311.18677](https://arxiv.org/abs/2311.18677)) —
prefill/decode disaggregation on separate GPU pools (goodput-optimized
placement; phase-specific hardware). The sim currently co-locates phases with
prefill-priority, like vLLM/SGLang defaults.

**LoongServe** ([arXiv:2404.09526](https://arxiv.org/pdf/2404.09526)) —
*elastic sequence parallelism*: the parallelism degree of a single request
changes dynamically between phases. **SpotServe**
([arXiv:2311.15566](https://arxiv.org/abs/2311.15566)) — live
re-parallelization (data/tensor/pipeline degrees) as spot instances come and
go, with KM-matching migration plans.

## 5. Autoscaling and workload forecasting

**SageServe** ([arXiv:2502.14617](https://arxiv.org/pdf/2502.14617)) —
forecast-aware autoscaling across regions at Microsoft scale (**~25% GPU-hour
savings**). **SuperServe** ([arXiv:2312.16733](https://arxiv.org/pdf/2312.16733))
— fine-grained serving for unpredictable workloads.
Surveyed systems ([LLM scheduling survey](https://www.techrxiv.org/users/994660/articles/1355915/master/file/data/LLM_Scheduling_Survey_Arxiv_06Oct2025/LLM_Scheduling_Survey_Arxiv_06Oct2025.pdf?inline=true))
include burst-detection scalers (BAScaler) and prewarming predictors
(WarmServe, reported 93%-accurate 5-minute demand windows). Consistent theme:
reactive scaling is too slow for LLM burst patterns; prediction buys the
provisioning lead time.

## 6. Where `inference-sim` sits, and what our experiments showed

The sim composes: Mooncake-format trace replay → SGLang-style gateway routing
→ Orca-style continuous batching → LMCache-style tiered prefix cache, with
closed-form roofline timing (compute-bound prefill, bandwidth-bound decode).

Measured on 3 real trace windows (Llama-70B, 4×4 H100, details in
`scratchpad/router_lab.py`):

| finding | evidence |
|---|---|
| Continuous cost routing ≈ best of cache_aware + least_load | matches Dynamo/Baseten reports |
| Output-length prediction ≈ worthless for placement, harmful when noisy | contrasts with its large *scheduling* gains (S3, LtR) |
| Work-based load metric (known input lengths) beats request counting | ~10% TTFT, ~15% peak-queue improvement, no ML |
| KV head-of-line blocking absent at 4-GPU nodes | skip-ahead admission changed nothing |
| Partitioning is the dominant loss: pooled 16-GPU node −16% mean lat, −35% TTFT, −2.3× peak queue | motivates TokenLake/Preble-style pooling & replication |
