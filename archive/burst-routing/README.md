# Big-burst study: least_load vs cache_aware

What happens to a cluster when a **big burst** hits, and how does **least-load**
routing hold up against **cache-aware** routing? This experiment drives the exact
same flash crowd through every routing policy on the real simulator
(`../../../inference-sim`) and charts the queueing behavior.

## The scenario

A multi-tenant flash crowd: **2,600 requests arrive inside 3 seconds** on a
4× H100 cluster (`run_burst.py` knobs).

- **300 distinct agent prefixes** (each a 6,144-token system prompt), request
  volume drawn Zipf-skewed across them — one "viral" agent plus a long tail.
- Caches start **cold**; the burst is the first traffic.
- Every request routed, queued, prefilled, and decoded by the simulator's
  continuous-batching model.

Run through three cluster conditions on the identical burst:

| Condition | Meaning |
|-----------|---------|
| **shared, contention-free** | Nodes borrow each other's KV over RDMA with unlimited fabric — the original model, and this repo's premise. |
| **shared, fabric contends** | Same, but the RDMA fabric is a finite shared resource: concurrent peer transfers during the burst slow each other down (`RDMA_CONGESTION`, added to the sim in this pass). |
| **isolated replicas** | Cross-node RDMA sharing is off; a request routed to a node that never cached its prefix must **recompute** it. |

### The congestion model (new)

Real RDMA fabrics congest under exactly the incast a burst creates — many nodes
pulling KV from the few holding a hot prefix at once (see the web-sourced
evidence in the memory note `sim-rdma-no-contention`). The sim previously gave
every transfer full bandwidth in isolation. We added `RDMA_CONGESTION` to
`../../../inference-sim` (`config.py` + `simulate.advance`): a peer transfer is
stretched by up to the **node count** (the fabric fan-in bound) when the puller
is backlogged — a mean-field proxy sensed locally, so it needs no global event
clock. A single transfer on an idle fabric is unchanged; only concurrency slows
things down, and when contention makes a local recompute cheaper the request
recomputes instead. `RDMA_CONGESTION=False` is byte-identical to the old model.

## How to reproduce

```bash
export INFERENCE_SIM_ROOT=/path/to/inference-sim   # optional; defaults to ../../../inference-sim
python3 run_burst.py      # runs all three conditions + a cluster-size sweep
python3 make_charts.py    # renders charts/*.png from results/burst_results.json
python3 sweep_params.py   # sweeps the early_rdma constants X/Y/Z/M -> results/sweep_results.json
```

## The early_rdma rule: four constants (X/Y/Z/M)

`btb_policy.py` is the whole method — a fixed rule, no per-model learning:

> if the same **Y**-block prefix arrives **X** times within **Z** seconds,
> replicate its KV to the **M** least-busy replicas and route later same-prefix
> requests across them.

`sweep_params.py` grids these on a prefill-heavy single-hot-prefix regime (long
shared prefix, 1-token output, 8 nodes) against a **realistic** baseline:
`ADMIT_RDMA=False`, so a miss *recomputes* the prefix rather than opportunistically
stealing a peer's KV — which is what SGLang actually does (there is no automatic
peer-to-peer KV pull on the least-load / cache-aware path; cross-worker KV moves
only via structured PD handoff or a shared pool). BTB's own warming push still
runs at full RDMA bandwidth. What it found:

- **Y — copy the *whole* shared prefix.** Every top-ranked config uses the full
  prefix; warming a fraction always loses (the request still prefills the rest).
  So Y is not a free dial — it is set to the workload's prefix length.
- **X and Z barely matter.** Across X∈{2,4,8} and Z∈{1,2,4}s the result moves
  <1%, so they are frozen as small constants (X=2, Z=1 s) rather than tuned.
- **M is the one real lever** — more copies = more spread, bounded by node count
  and warming cost.
- **Result:** early_rdma cuts mean TTFT **~72% vs recompute-on-miss `least_load`**
  and **~84% vs cache-sticky `cache_aware`** — pre-warming avoids the cold-node
  prefill that both baselines otherwise eat. (Note: if the simulator is instead
  allowed to hand the baseline a free peer-to-peer KV steal — `ADMIT_RDMA=True`,
  which real SGLang does not do — that baseline gets warming's benefit for free
  and BTB looks break-even; see finding 7 on why that regime flatters the
  baseline.)

## What the charts show

**`burst_timeline.png`** — The burst instantly builds a backlog of **~2,568
waiting requests** that drains over **~180 s**. least_load and cache_aware trace
almost the same curve when the cache is shared.

**`ttft_gap.png`** — The punchline (4-node cluster). Contention-free the routing
choice is **even** (TTFT 81 vs 80 s); a contending fabric is a **mild ~3% tax**;
isolated replicas is where cache_aware clearly wins (**~24%**, 93 vs 122 s).

**`scaling.png`** — The three-way head-to-head under a congested fabric across
cluster sizes: **least_load vs cache_aware vs early_rdma (BTB)**. Congestion's
effect grows with cluster size — cache_aware's TTFT advantage over least_load
widens **3% → 8% → 20% → 35%** (4 → 8 → 16 → 32 nodes) as incast fan-in scales.
**BTB tracks cache_aware but never beats it** (−1% / 5% / 19% / 34%): on tiny
clusters its warming overhead is a slight net loss; at scale it converges to
cache_aware. See finding 7 for why.

**`metrics_by_condition.png`** — Every headline metric (mean & peak queue wait,
mean TTFT, mean prefix+prefill, max queue depth, throughput) across all four
policies and all three conditions (opacity: solid = contention-free, mid =
congested, light hatched = isolated).

**`cache_tier.png`** — Where prefix reuse comes from. cache_aware keeps **68%**
of reuse in **free local HBM** vs least_load's **37%** — it pushes the rest onto
peer RDMA / host RAM. That's why congestion (which taxes the RDMA tier) barely
touches cache_aware but slows the cache-blind policies.

## Findings

1. **The discriminator is the cluster's KV-sharing topology, not the load.**
   Sweeping burst size 200 → 2,600 (shared condition), the least_load-vs-
   cache_aware TTFT gap stays **0–2% at every load level**. It jumps to
   **21–28%** only when cross-node sharing is off — and stays there across the
   same load sweep. Request volume is not what makes the policies differ;
   whether nodes can cheaply borrow each other's KV is.

2. **cache_aware routes by prefix ~100% of the time — it does *not* fall back.**
   (Measured: 0% fallback on this burst.) The imbalance guard (`IMBALANCE_ABS=64`,
   `IMBALANCE_REL=1.5`, the SGLang router defaults) never trips, because cache_aware breaks prefix-match ties
   by load and so stays balanced on its own. It genuinely earns **2× the free
   local-HBM hits** of least_load (68% vs 37%, `cache_tier.png`).

3. **RDMA sharing makes that locality edge worth almost nothing.** When least_load
   scatters a hot prefix onto a node that lacks it, that node grabs it from a peer
   over RDMA (~0.01 s for a 6k-token prefix) instead of recomputing (~0.44 s) — so
   the "mistake" is nearly free and TTFT matches cache_aware. Turn sharing off and
   the same scatter forces a full recompute: TTFT inflates ~23% and throughput
   drops from 5.5k to 4.3k tok/s.

4. **A big burst is also queue-bound**, which hides the last sliver: TTFT is
   ~99% queue wait, so even the residual prefill difference is invisible against
   ~80 s of waiting. Real, but secondary to reason 1.

5. **Fabric contention doesn't change the 4-node story but it scales.** Adding a
   realistic RDMA-congestion model, the least_load-vs-cache_aware gap on 4 nodes
   grows only from ~1% to ~3% — the fabric can only be shared ~4 ways, so a peer
   pull (~10 ms → ~40 ms) is still far cheaper than a recompute (~440 ms). But the
   incast fan-in scales with cluster size: at 8 / 16 / 32 nodes the gap is
   8% / 20% / 35%. Congestion is what turns "shared ≈ isolated is a corner case"
   into "cache-aware matters on any real (large) cluster."

6. **This is the motivation for predictive KV movement.** The whole gap between
   "routing barely matters" and "cache_aware wins" is the value of getting the
   right KV onto the right node cheaply — which is exactly what `early_rdma` (BTB)
   tries to do *ahead* of the burst (while the fabric is quiet) rather than
   contending for it reactively during the spike.

7. **BTB matches cache_aware here but does not beat it — because in this sim
   reactive caching already replicates hot prefixes.** `early_rdma` (ported to
   the sim in `btb_policy.py`) detects a sustained shared-prefix burst, RDMA-copies
   the prefix onto the least-busy replicas ahead of demand, and routes active-prefix
   requests to the least-loaded warm replica. Under the congested/scaling sweep it
   tracks cache_aware (−1% / 5% / 19% / 34% vs least_load at 4/8/16/32 nodes) but
   never overtakes it. Why: the simulator warms a node whenever it *serves* a
   request (`node.insert`), so a hot prefix naturally replicates across every node
   that touches it, and cache_aware balances across those replicas for free. BTB's
   only structural edge is warming a replica *before* its first-touch — real, but
   small once first-touch is amortized, and offset by warming interference (a warm
   copy serializes with the target's compute, the sim's own convention for RDMA).
   BTB's *positive* result in `../results/PAPER.md` comes from a different regime
   (65,536-token prefixes, 1 output token — prefill-dominated, so first-touch cost
   dwarfs everything) **and** a harness with no fabric congestion. The honest read:
   under a strong reactive cache_aware baseline plus a congestion model, BTB's
   marginal value over cache_aware is small in this simulator — it wins clearly
   only when first-touch is catastrophic and there is genuine lead time to prewarm.

### Caveats

- The congestion model is **mean-field**: it senses burst pressure from the
  admitting node's own backlog rather than a true global fabric schedule (the sim
  advances each node on its own clock, so there is no global event ordering to
  hang a shared-link queue on). It captures the first-order effect — peer
  transfers slow under concurrency, more so at scale — not exact per-transfer
  contention. Bounded by the node count (fabric fan-in).
- The per-request `tier` field records the reuse *source picked*. Under
  contention the sim now relabels to `recompute` when a local recompute wins, but
  in the `isolated` condition the crippled-bandwidth fallback is still visible in
  the prefill/throughput panels rather than the tier mix — so `cache_tier.png`
  is shown for the shared condition only.
- BTB's warm copy serializes with the target replica's compute clock (the sim's
  convention for RDMA reuse). Real RDMA GPUDirect DMA overlaps decode more than
  that, so this **under**-credits BTB; conversely, treating warming as fully free
  would over-credit it. The mean-field congestion model can't cleanly represent
  the fabric as a shared resource, so BTB's exact margin over cache_aware is
  inside the model's error bars — the robust conclusion is only "BTB ≈ cache_aware,
  both ≫ least_load at scale," not a precise BTB-vs-cache_aware number.
