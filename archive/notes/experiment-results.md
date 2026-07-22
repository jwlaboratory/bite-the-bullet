# Experiment results: five routes to a better router

Five agent-run experiments on the simulator (July 2026), each evaluated on
real ART-Chat-2.5M trace windows against the same baselines (least_load,
cache_aware, and the CEM-trained `learned` router) with identical seeding.
Cost metric unless noted: `0.7·mean_lat + 0.3·p95_lat`, normalized per-window
to least_load (lower is better). Scripts and raw results live in the session
scratchpad (`exp1-continual/ … exp5-mlp-v2/`).

## Scoreboard

| experiment | verdict | headline number |
|---|---|---|
| 1. Config-adaptive router | **Adopt** (generalist training) | 0.948 overall vs specialist 0.966 across 4 deployments; hetero cluster 0.862 |
| 2. Prompt-metadata features | Don't adopt (predictor is real, routing gain isn't) | batch prediction F1≈0.86; but fresh-window cost 0.992 vs incumbent 0.976 |
| 3. SJF admission ordering | **Adopt if SLO is mean/TTFT** (with aging) | −14% mean, −38% TTFT; max lat +10–26% (aging r=0.2 → +7–8%) |
| 4. Forecast-driven re-sharding | Don't adopt (clean negative) | forecast switcher lost on every window; static 2×8 alone gives −11% |
| 5. Nonlinear router (MLP / interactions) | Don't ship at this data scale | holdout 0.927–0.933 vs 0.941, within selection noise on 3 windows |

## 1. Config-adaptive / continually-learning router

- **Zero-shot transfer mostly works.** The specialist weights (trained on
  70B / 4×4×H100) beat least_load unchanged on an 8B/4×1×H100 deployment
  (0.975) and a heterogeneous H100+H200 cluster (0.928) — the time/fraction
  feature units transfer — but fail on a 2-node 8×H100 topology (1.021).
- **Domain-randomized generalist wins overall**: CEM scored across 4
  (deployment, window) pairs → 0.948 overall vs specialist 0.966,
  cache_aware 0.985; biggest gains on hetero (0.862) and 8B (0.944), giving
  back ~2.5pts on the native config. ~1.5k sims, ~4 min to train.
- **Naive online learning is actively harmful**: one unguarded CEM update per
  window gives +0.307 cumulative regret vs +0.115 for frozen weights
  (single-window fitness is too noisy). A guarded update (accept only if it
  beats the incumbent on the window just seen) achieves the best regret
  (+0.092) — but only marginally beats frozen, because zero-shot transfer
  already captures ~97% of the per-config oracle.
- Open issue: no learned variant wins on 2-node topologies; include them in
  the randomization mix or fall back to least_load there.

## 2. Prompt-metadata routing features

- **Batch arrival is highly predictable** from arrival-time-observable
  metadata (3,200 requests, 8 windows): ≥2 same-group arrivals in the last
  10s predicts ≥3 more same-group/high-overlap arrivals in the next 30s with
  P=0.89 / R=0.84 / F1=0.86 (base rate 0.325). Correlations: 10s group count
  +0.77, block-overlap +0.73, group novelty −0.50.
- **But feature-augmented routing did not survive honest evaluation.** A
  16-feature CEM retrain won on the 3-window selection holdout (0.9322 vs
  0.9410) yet lost on 4 never-touched fresh windows (0.9924 vs 0.9762) and on
  the train windows — the holdout win was selection noise. The incumbent's
  momentum × local-hit feature already captures most of the burst signal's
  routable value.
- Most promising surviving mechanism (worth re-testing with more windows):
  per-group expected-output-length × KV-pressure (weight −0.489) — steering
  long-decode groups toward already-loaded nodes to join large amortized
  decode batches.

## 3. SJF / SRPT admission ordering (scheduler, not router)

- **Contention is window-dependent**: window 42's waiting queue never exceeds
  1 at admission time (order can't matter — this is why a first attempt saw
  bit-identical results); congested windows bind hard (63–68% of admissions
  pick a non-head request).
- **SJF helps means, hurts tails** (5-window averages vs FCFS, both routers):
  mean −14–15%, p50 −18–21%, mean TTFT −38%; but max latency +10–26% and
  peak queueing delay up to +82% (starvation). Aging (rank = work −
  0.2·age) keeps −11% mean / −30% TTFT at only +7–8% max.
- **Predictor quality barely matters**: oracle 0.855 vs constant-52 0.856
  normalized mean. On this trace SJF is effectively *shortest-prompt-first* —
  known input length (up to ~171k tokens) dominates predicted work; a noisy
  output predictor is slightly worse than no predictor.
- Effect stacks with the learned router and is larger than the ~6% routing
  win on means — but routing was Pareto; SJF is an SLO-dependent trade.

## 4. Forecast-driven dynamic re-sharding

- **Negative on every window.** EWMA-forecast switching (4×4×H100 ↔ 2×8×H100)
  lost to the best static shape everywhere: +5% mean latency at a 2s
  migration penalty, +20% at 5s, +18% at 10s — switch costs (admission
  freeze + cold prefix caches + drain) land exactly when load is highest.
- **The pooling prize is real but static**: permanent 2×8 gives −11% average
  mean latency (−27% on calm windows), consistent with the earlier −16%
  pooled-16-GPU bound.
- **Even an oracle switcher needs ~2s migrations**: −19% at P=2s (short
  ~22s mid-burst merges), −7% at P=5s, ~0 at P=10s (oracle chooses "never
  switch" on 3/5 windows). The EWMA forecaster captured none of it.
- Recommendation: bigger static nodes, not dynamic re-sharding — revisit only
  with ~2s migrations, near-oracle forecasts, and longer burst cycles.
  (Caveat: the drain mechanic transiently double-counts hardware, which
  *flatters* switching — the real verdict is more negative.)

## 5. Nonlinear router (interaction-linear and tiny MLP)

- Both nominally beat the incumbent on holdout (interaction-linear 0.9332,
  105-param MLP 0.9267 vs 0.9410) — but margins (0.8–1.4%) sit inside
  per-window variance, both lose to the incumbent on its best window, the
  16-param model retains only ~18% of its train gain out-of-sample, and
  min-over-iterates selection on 3 holdout windows is optimistically biased.
- **Verdict: 8 windows × 400 requests cannot support the capacity; keep the
  11-weight linear incumbent.** The interaction-term *signs* are the reusable
  insight — congestion-conditional cache affinity (busy node forgiven a local
  hit; decode-bound node forgiven a remote prefix) and
  wait_prefill × kv_used congestion compounding — cheap to re-test with 5
  extra weights once 5–10× more training windows exist.

## Cross-cutting conclusions

1. The biggest available wins are **not in the router**: SJF admission
   (means/TTFT) and bigger static nodes (pooling) both out-size the ~6%
   routing improvement.
2. Where learning helps routing, **robustness beats capacity**: the
   domain-randomized generalist is the only clear upgrade; more features and
   more parameters both failed out-of-sample at this data scale.
3. **Output-length prediction keeps not mattering** in this sim — not for
   routing (earlier oracle test), not for SJF (input length dominates). The
   length-prediction literature's wins live in workloads with long, variable
   decodes; this trace's 52-token mean decode isn't that workload.
4. More trace data is the binding constraint on every learned improvement:
   every failed idea failed on fresh-window generalization, not on train fit.
