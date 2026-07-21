# Research questions: self-evolving routers & burst-predictive GPU splitting

Two specific questions, researched July 2026. Verdict up front:

1. **"Has anyone trained a model that continuously evolves with the workload
   and learns the best routing?"** — *Partially.* Offline-trained RL routers
   for LLM replica routing exist (Microsoft's Intelligent Router is the
   closest match), and *online*-adapting bandit routers exist in the adjacent
   "which model should answer this?" problem. A router that keeps
   *continuously* learning in production against your live workload — the
   thing you described — does not appear to have been published. The pieces
   all exist; the combination is open.

2. **"Has anyone predicted that a batch is coming and preemptively split the
   model across two GPUs?"** — *The two halves exist separately, not
   together.* AlpaServe proved that splitting-for-burst-absorption is worth
   the overhead; forecast-aware autoscalers predict the bursts. Nobody we
   could find closes the loop into "forecast → proactively re-shard."

---

## Q1 — Continuously-learning routing

### Closest existing work

**Intelligent Router for LLM workloads (Microsoft, [arXiv:2408.13510](https://arxiv.org/pdf/2408.13510))**
— the nearest published system to the question. A workload-aware **RL-trained
router** for a pool of LLM instances: a lightweight predictor estimates each
request's decode time from its characteristics, and the learned policy routes
so heavy (long-decode) and light requests don't share an instance — precisely
the interference pattern that count-based least-load can't see. Trained on a
simulator of instance-level continuous batching, then deployed. Limitation
vs. your idea: **trained offline, frozen at deployment** — it does not keep
evolving with the live workload.

**Efficient routing across LLM instances in cloud-edge settings
([arXiv:2507.15553](https://arxiv.org/pdf/2507.15553))** — same family:
learned routing across heterogeneous instances.

**Decima ([SIGCOMM '19](https://arxiv.org/pdf/1810.01963))** — the lineage
ancestor: RL over a graph neural network learns *workload-specific* cluster
scheduling from scratch, **21% better JCT than tuned heuristics, up to 2×
under high load**. Decima's core claim is exactly your premise: hand-tuning a
policy per workload is infeasible, so let the policy learn the workload. But
again: train-then-freeze, with periodic retraining — not continuous.

**Online adaptation exists next door.** In the *model-selection* routing
problem (route a query to GPT-4 vs. a cheap model — different from replica
routing but structurally similar), routers that adapt online are published:
[Adaptive LLM Routing under Budget Constraints](https://arxiv.org/html/2508.21141v1)
trains the router **online with bandit feedback** as query distribution
drifts, formulated as a multi-choice knapsack. Surveys
([Dynamic Model Routing and Cascading](https://arxiv.org/pdf/2603.04445),
[Awesome-Routing-LLMs](https://github.com/MilkThink-Lab/Awesome-Routing-LLMs))
show a fast-moving space, including RL-trained routers like
[Router-R1](https://arxiv.org/pdf/2506.09033) — but all target model choice,
not replica/GPU placement.

### The gap (i.e., the opportunity)

No published system combines: (a) replica-level routing, (b) a learned policy,
and (c) **continual online updates** from live reward (observed TTFT/latency
vs. SLO). Reasons cited across the literature: exploration is expensive when
every bad routing decision is a real user's latency; reward is delayed
(a routing decision's cost lands seconds later); and distribution shift cuts
both ways (the reason to learn online is also what makes it risky). The
practical recipe the pieces suggest:

- **State:** per-node outstanding work, cache-overlap score for the incoming
  request, arrival-rate estimate — all things `inference-sim` already exposes.
- **Reward:** negative request latency (or SLO-violation indicator), delayed.
- **Safe exploration:** shadow the heuristic (Dynamo-style cost model) and
  ε-explore only when queues are short; or train off-policy from logged
  decisions.
- **This sim is the natural gym.** Trace replay is deterministic,
  a policy is ~10 lines (`router.py`), and 400-request episodes run in
  milliseconds — an RL loop (even a bandit over the 4 nodes with the cost
  model's features) could be trained and evaluated here directly, then the
  learned policy compared against `cost_model`/`cache_aware` in the UI's
  comparison table.

## Q2 — Predict a burst, bite the bullet, split across GPUs

### The two halves that exist

**Half 1 — splitting pays off under bursts: AlpaServe
([OSDI '23](https://arxiv.org/abs/2302.11665)).** The foundational result.
Even when a model *fits on one GPU*, sharding it via model parallelism across
many GPUs lets bursts statistically multiplex the whole cluster's compute:
the parallelism overhead ("the bullet") is repaid because no single replica's
queue can pile up while neighbors idle. **10× higher sustainable request rate
/ 6× more burstiness within SLO** on production traces. Crucially though,
AlpaServe chooses the placement **offline with a planner** over historical
arrival traces — it does not watch for an incoming batch and re-shard in
response.

**Half 1b — re-sharding at runtime is feasible: SpotServe
([arXiv:2311.15566](https://arxiv.org/abs/2311.15566))** live-migrates between
(data, pipeline, tensor) parallel configs when spot instances are preempted
or acquired, using bipartite matching to minimize KV/weight movement — proof
that the mechanism (cheap dynamic re-parallelization) exists. **LoongServe
([arXiv:2404.09526](https://arxiv.org/pdf/2404.09526))** does elastic
*sequence* parallelism per-request. Both are triggered by resource
availability or request shape — **not by traffic forecasts**.

**Half 2 — bursts are predictable: forecast-aware autoscalers.**
[SageServe](https://arxiv.org/pdf/2502.14617) (forecast-driven GPU allocation,
~25% GPU-hour savings at Microsoft scale), WarmServe (93%-accurate 5-minute
demand prediction driving model prewarming, ~50× TTFT vs. reactive scaling),
BAScaler (burst detection → preemptive allocation) — surveyed in the
[LLM scheduling survey](https://www.techrxiv.org/users/994660/articles/1355915/master/file/data/LLM_Scheduling_Survey_Arxiv_06Oct2025/LLM_Scheduling_Survey_Arxiv_06Oct2025.pdf?inline=true).
These predict *when* to add capacity — but the action space is "spin up more
replicas," never "change the parallelism degree of existing ones."

### The gap

"Arrival forecast → proactively increase TP/PP degree (accepting per-token
overhead) → shrink back when the burst passes" appears unpublished. It is a
genuinely interesting control problem because the trade is time-varying:
wider sharding lowers queueing delay but raises per-token cost and evicts KV
cache during the transition (SpotServe's migration cost model applies).

### How this sim could test it first

The sim already errors when a model doesn't fit and models per-node
aggregation of compute/bandwidth, so the experiment is cheap:

1. Add an arrival-rate estimator (EWMA over recent inter-arrival gaps —
   the trace has real diurnal/bursty structure).
2. Add a "reconfigure" event: at time t, merge two 4-GPU nodes into one
   8-GPU node (or split back), charging a configurable migration penalty
   (weights reload at RDMA bandwidth + prefix-cache loss).
3. Compare static 4×4 vs. static 2×8 vs. predictive-switching on the same
   trace windows. Our pooled-node experiment already bounds the prize:
   pooling all 16 GPUs cut mean latency 16% and peak queue 2.3× — the
   question the experiment answers is how much of that a *part-time,
   forecast-triggered* pooling captures, net of migration cost.
