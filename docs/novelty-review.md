# Novelty review: our ideas vs. prior art

Five parallel research agents each stress-tested one of our ideas against the
literature (July 2026). Verdict summary, then per-idea detail with the closest
prior work and our honest differentiation. Companion docs: `lit-review.md`
(the field overall), `learned-routing-and-predictive-scaling.md` (the original
open-question scans).

| # | Idea | Verdict | Our edge, in one line |
|---|------|---------|----------------------|
| 1 | Simulator as a training gym for the router | Partially done | CEM over *interpretable linear* weights + production-trace fidelity + holdout selection; prior work is neural RL on synthetic loads |
| 2 | Burst-momentum × cache-hit learned routing | Partially done | Nobody conditions on *arrival dynamics of a prefix group*; nobody *learned* the concentrate-vs-spread tradeoff offline |
| 3 | Config-adaptive router (survives GPU/model changes) | Partially done | Domain randomization for *zero-shot* config transfer is unclaimed in LLM routing; prior art adapts online with warm-up cost |
| 4 | Prompt content as a routing signal | Partially done | All ingredients exist in silos; fusing content features with cache/load features in one replica-router is unclaimed |
| 5 | Forecast bursts → proactively re-shard parallelism | Not done (gap confirmed) | Both halves mature, loop never closed — but our own experiment says the loop may not be worth closing (see below) |

**The one system to watch: [Lodestar](https://arxiv.org/abs/2606.00946)
(June 2026)** — an online-learning LLM inference router that beats
prefix-cache- and load-aware heuristics by 1.4–4.4× TTFT on real clusters. It
surfaced independently in three of the five searches and is the closest
overall competitor to our whole program. Its training regime is the mirror
image of ours: purely online against the live cluster (≈5 min convergence, no
simulator), an opaque learned reward predictor rather than fixed interpretable
weights. Every claim we make should be phrased relative to Lodestar.

---

## 1. Simulator-as-gym for router training

**Prior art.** [Microsoft's Intelligent Router](https://arxiv.org/abs/2408.13510)
literally trains an RL routing policy in an LLM-serving simulator (profiled
batch-latency model, DistilBERT length predictor, >11% E2E latency gain — but
vs. weak baselines like round-robin, with no KV-cache modeling, on synthetic
task mixes). [Decima](https://web.mit.edu/decima/) (SIGCOMM '19) established
the whole sim-train-then-deploy paradigm for cluster scheduling.
[RLScheduler](https://arxiv.org/pdf/1910.08925) did trace-driven gym training
for HPC jobs. [Vidur](https://arxiv.org/pdf/2405.05465) (MLSys '24) is a
high-fidelity LLM serving simulator (<9% error) but used for config *search*,
not policy training. [llm-d's predicted-latency scheduling](https://llm-d.ai/blog/predicted-latency-based-scheduling-for-llms)
attacks the same hand-tuned-weights pain point with a learned latency
predictor (~43% P50 gain). [AgentServeSim](https://arxiv.org/abs/2606.09613)
(2026) confirms simulators still treat routing policies as things to
*evaluate*, not *train*.

**Our differentiation.** (1) Policy class: CEM over an 11-feature
interpretable linear scorer — zero serving-time inference cost, auditable
signed weights, same form factor as SGLang's hand-tuned scorer but with
learned weights. Prior learned routers are opaque neural policies. (2)
Training-signal fidelity: replay of a real Mooncake-format production trace
with block-hash prefix-cache hits, tiered KV, continuous batching — vs.
synthetic mixes with no cache model. (3) Holdout-window selection — none of
the found works control for overfitting to a workload segment. (4) Baselines:
we beat modern cache-aware/least-load routers, not round-robin.

**Honest weakness.** Our ~6% is smaller than Microsoft's 11% or Lodestar's
1.4× (different baselines, not directly comparable), and we have not shown
sim-to-real transfer — that is the claim prior work would attack first, and
closing it (even on a 2-replica vLLM cluster) is the highest-value next
experiment.

## 2. Burst-aware cache-affinity routing (the momentum × cache feature)

**Prior art.** The concentrate-vs-spread tension is hand-coded everywhere:
[Preble](https://arxiv.org/abs/2407.00023) replicates hot prefixes via
thresholds, [DualMap](https://arxiv.org/abs/2602.06502) (Feb 2026) uses
deterministic SLO rules, [SGLang's balancer](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)
uses imbalance thresholds, [Dynamo's KV router](https://docs.nvidia.com/dynamo/latest/router/README.html)
exposes a single human-tuned scalar (their docs tell operators to tune it by
watching TTFT), Ray Serve's PrefixCacheAffinityRouter likewise. Lodestar
*learns* the tradeoff but from KV-hit-ratio and load features only.

**Our differentiation — the strongest novelty claim we have.** No prior
router, learned or heuristic, conditions on the *arrival dynamics of a prefix
group* (our momentum feature), and none learns a momentum × cache-hit
interaction. Heuristic systems react *after* load accumulates; our feature
anticipates the burst. And the result is citable as a mechanistic finding,
not just a benchmark: the trained weight (+0.73) *rediscovers Preble's
hand-coded hot-prefix spreading as the data-optimal policy*.

**Supporting evidence from our experiments.** The burst signal is genuinely
predictive (≥2 same-group arrivals in 10s predicts an incoming batch with
F1≈0.86 on 3,200 requests), and the follow-up feature-augmentation experiment
showed the incumbent momentum × hit feature already captures most of its
routable value (5 added metadata features won on the selection holdout but
lost on fresh windows — selection noise, honestly diagnosed).

## 3. Config-adaptive / continually-learning routing

**Prior art.** [Lodestar](https://arxiv.org/abs/2606.00946) adapts to
infrastructure change by online retraining (with a ~5-min warm-up of degraded
decisions after each change). Decima's interarrival feature and workload
mixing are proto-domain-randomization. [TayMAML](https://www.sciencedirect.com/science/article/abs/pii/S0957417426001661)
does meta-RL for edge scheduling; [RAN RL work](https://arxiv.org/abs/2507.06602)
validates domain randomization in a systems domain. Helix/HexGen-2 re-run
solvers per config — re-optimization, not transfer. Nobody trains an LLM
router with domain randomization over GPU types/counts/model sizes for
zero-shot transfer.

**Our differentiation + our measured evidence.** Zero-shot at the moment of
change, no online exploration on production traffic. Our experiment: the
70B-trained specialist already transfers to an 8B deployment (0.975 vs LL
1.000) and a heterogeneous H100+H200 cluster (0.928), but *fails on a 2-node
topology* (1.021); a domain-randomized generalist fixes most of this
(overall 0.948 vs specialist 0.966, hetero 0.862) at a ~2.5pt giveback on the
native config. Naive online-CEM is actively harmful (+0.307 regret vs +0.115
frozen); guarded updates (accept only if better on recent data) win (+0.092).
That guarded-online result mirrors Lodestar's motivation while avoiding its
warm-up cost — a genuinely publishable comparison point.

## 4. Prompt content as a routing signal

**Prior art — thoroughly siloed.** Prompt→length drives *queue ordering*
([SSJF](https://arxiv.org/abs/2404.08509), [TRAIL](https://arxiv.org/abs/2410.01035),
[learning-to-rank](https://arxiv.org/pdf/2408.15792)); prompt→semantic class
drives *model selection* ([vLLM Semantic Router](https://blog.vllm.ai/2025/09/11/semantic-router.html),
RouteLLM); learned replica routing uses *structural* features only
(Lodestar); session awareness is *declared by the app*, not inferred
([Autellix](https://arxiv.org/abs/2502.13965), Parrot);
[semantic caching](https://arxiv.org/pdf/2508.07675) reuses responses
reactively. Nobody fuses content features with cache/load state in one
learned replica-router, and *semantic prediction of future arrivals*
(anticipating a batch from what requests say) appears entirely unexplored.

**Our differentiation.** The fusion itself, plus the proactive framing. Our
trace lacks raw text, so we proxy content with the system-prompt group hash —
a real deployment could add embeddings. Measured so far: batch arrival is
highly predictable from metadata (F1≈0.86), but converting the extra signals
into routing gains beyond the momentum feature did not survive honest
evaluation. The next step with real leverage is per-group output-length
estimation feeding the *scheduler* (SJF ordering), not the router.

## 5. Forecast-driven re-sharding

**Prior art — the gap is real, and freshly confirmed.**
[Amoeba](https://arxiv.org/abs/2509.19729) (2025) is the exact merge/split
mechanism and *explicitly disclaims prediction* ("we do not involve...
mechanisms for predicting the arrival"); [Flying Serving](https://arxiv.org/pdf/2602.22593)
(2026) switches DP↔TP on observed load; SpotServe reacts to preemptions;
LoongServe regroups greedily per batch. On the forecast side,
[SageServe](https://arxiv.org/abs/2502.14617) (SIGMETRICS '26) and
[WarmServe](https://arxiv.org/pdf/2512.09472) predict minutes ahead but only
add/prewarm replicas. Nobody closes forecast → reshard.

**But our own experiment argues the gap may be a dead end** — an honest
negative result worth as much as the novelty: on 5 held-out windows, an
EWMA-forecast switcher *lost to static shapes on every window* (+5% to +20%
mean latency), because switch costs land exactly when load is highest. Even
an oracle switcher only wins at a 2s migration penalty (−19%); at 10s the
oracle chooses "never switch." Meanwhile static 2×8 captures −11% all by
itself. If we publish anything here, it is: *"just use bigger static nodes;
part-time pooling needs ~2s migrations plus near-oracle forecasts to beat
them"* — a useful negative that directly qualifies Amoeba-style systems.

---

## What we can defensibly claim (ranked)

1. **Momentum-conditioned cache routing, learned and interpretable** — no
   prior work; supported by a mechanistic finding (rediscovers Preble's
   replication) and a measured predictor (F1≈0.86 burst detection).
2. **Domain-randomized zero-shot config transfer for LLM routing** — no prior
   work; our generalist-vs-specialist-vs-online table is the evidence, and it
   contrasts cleanly with Lodestar's warm-up-based adaptation.
3. **CEM-over-interpretable-features in a production-trace gym** — paradigm
   exists (Microsoft, Decima), but our policy class, trace fidelity, and
   holdout methodology are a distinct, defensible recipe.
4. **Forecast→reshard doesn't pay (negative result)** — the gap everyone left
   open, we measured why it stays open.
5. **Prompt-content fusion routing** — the *idea space* is open, but our own
   fresh-window evidence says metadata-beyond-momentum adds nothing for
   routing on this trace; the honest claim is the batch-predictability
   measurement, not a routing win.

Known weaknesses reviewers will raise: single trace dataset; small windows
(400 requests); no sim-to-real validation (Lodestar and Decima both deployed);
~6% headline gain is modest. The cheapest high-value additions: a second
public trace, longer windows, and a 2-replica real-cluster replay.
