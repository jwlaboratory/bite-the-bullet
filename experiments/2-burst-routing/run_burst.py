"""Big-burst study: how least_load behaves under a flash crowd, vs cache_aware.

Scenario
--------
A cluster of 4x H100 nodes is hit by a *burst*: a large number of requests that
all share the same long system-prompt prefix arrive inside a short window (a
viral moment / flash crowd on one agent). We replay the exact same burst through
each routing policy on the real simulator (../inference-sim) and record, per
policy:

  - mean / peak queue time      (how long requests wait before admission)
  - mean time-to-first-token    (queue + prefix-load + prefill)
  - mean / total prefill time   (compute spent turning prompts into KV)
  - max queue depth             (largest waiting backlog on any node)
  - throughput, cache hit, util

The per-request event stream also lets us reconstruct the *queue-depth timeline*
(how the backlog builds and drains) without touching the simulator.

The workload is synthetic and fully seeded, so the sim needs no network access.

Run:  python3 run_burst.py
Out:  results/burst_results.json  and  charts/*.png
"""
import json
import os
import random
import sys
from types import SimpleNamespace

# --- locate the sibling simulator -------------------------------------------
SIM_ROOT = os.environ.get(
    "INFERENCE_SIM_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../inference-sim")),
)
if not os.path.isdir(SIM_ROOT):
    sys.exit(f"inference-sim not found at {SIM_ROOT} (set INFERENCE_SIM_ROOT)")
sys.path.insert(0, SIM_ROOT)

import config                       # noqa: E402
from simulate import run            # noqa: E402
from workload import Request        # noqa: E402

import btb_policy                   # noqa: E402
btb_policy.register()              # installs "early_rdma" into the sim's POLICIES

HERE = os.path.dirname(os.path.abspath(__file__))


def set_btb_knobs(cfg, warm_copies):
    """BTB early_rdma parameters (see btb_policy). warm_blocks = the shared
    family prefix; key = its first few blocks; a prefix warms once it is seen
    >= THRESHOLD times in WINDOW."""
    cfg.BTB_WARM_BLOCKS = PREFIX_BLOCKS
    cfg.BTB_KEY_BLOCKS = 4
    cfg.BTB_THRESHOLD = 4
    cfg.BTB_WINDOW_S = 2.0
    cfg.BTB_HORIZON_S = 120.0
    cfg.BTB_WARM_COPIES = warm_copies


def run_seeded(policy, requests, cfg):
    """The sim's router breaks routing ties with the global RNG, which the sim
    never seeds. Seed it here so results are reproducible and every policy sees
    the same tie-break stream (a fair A/B)."""
    random.seed(SEED)
    return run(policy, requests, cfg)


def run_seeded(policy, requests, cfg):
    """The sim's router breaks routing ties with the global RNG, which the sim
    never seeds. Seed it here so results are reproducible and every policy sees
    the same tie-break stream (a fair A/B)."""
    random.seed(SEED)
    return run(policy, requests, cfg)

# --------------------------------------------------------------------- knobs
# A multi-tenant flash crowd: NUM_FAMILIES distinct agents (each its own long
# system-prompt prefix) all burst at once. Family sizes are heavily skewed
# (Zipf) so one "viral" agent dominates -- the case where prefix-affinity and
# load-balancing pull in different directions.
BLOCK = 256                 # tokens per prefix block (must match config.BLOCK_TOKENS)
PREFIX_BLOCKS = 24          # per-family system prompt: 24 * 256 = 6144 cached tokens
USER_BLOCKS = 2             # per-request unique user turn (~512 tokens)
OUTPUT_TOKENS = 400         # decode length per request
NUM_FAMILIES = 300          # distinct agent prefixes in the burst (many => a
                            # scattered request often hits a node that never
                            # cached its family)
ZIPF = 0.6                  # family-size skew (higher => more concentrated)
BURST_SIZE = 2600           # total requests in the burst
BURST_WINDOW = 3.0          # seconds the burst arrives over (tight => big backlog)
WARMUP = 0                  # cold caches: the burst is the first traffic
SEED = 42

POLICIES = ["least_load", "cache_aware", "round_robin", "random"]

# Cluster conditions, run on the identical burst:
#   shared    -- nodes share prefix caches over RDMA, contention-free (the ideal)
#   congested -- same, but the shared RDMA fabric contends under the burst
#                (RDMA_CONGESTION on) -- the realistic case
#   isolated  -- replicas cannot borrow a peer's KV at all (RDMA ~ off, no disk
#                cache), so a scattered request must recompute a cold prefix
CONDITIONS = ["shared", "congested", "isolated"]

# (cripple_rdma, rdma_congestion) per condition
COND_SPEC = {
    "shared":    (False, False),
    "congested": (False, True),
    "isolated":  (True,  False),
}


def _family_blocks(fam):
    return [f"sys{fam}#{b}" for b in range(PREFIX_BLOCKS)]


def make_burst():
    """A deterministic multi-tenant flash crowd. Returns simulator Request
    objects (id/arrival assigned) covering a calm warmup that seeds each
    family's prefix, then a large burst whose requests are drawn across the
    families with a Zipf skew."""
    import math
    import random
    rng = random.Random(SEED)

    # Zipf family weights: family 0 is the "viral" one, then a long tail.
    weights = [1.0 / ((f + 1) ** ZIPF) for f in range(NUM_FAMILIES)]
    wsum = sum(weights)
    weights = [w / wsum for w in weights]
    reqs = []
    uid = 0

    def add(arrival, fam):
        nonlocal uid
        user = [f"u{uid}#{b}" for b in range(USER_BLOCKS)]
        blocks = _family_blocks(fam) + user
        inp = len(blocks) * BLOCK
        out = OUTPUT_TOKENS
        extra = math.ceil((inp + out) / BLOCK) - len(blocks)
        cache_blocks = blocks + [f"u{uid}#o{j}" for j in range(max(0, extra))]
        reqs.append(Request(0, arrival, f"fam{fam}", len(blocks) * BLOCK,
                            inp, out, blocks, cache_blocks))
        uid += 1

    # warmup: at least two calm requests per family, spread over ~6s, so every
    # family's prefix already exists somewhere before the burst.
    for i in range(WARMUP):
        add(rng.uniform(0.0, 6.0), i % NUM_FAMILIES)
    # the burst: BURST_SIZE requests over BURST_WINDOW, family drawn by weight.
    fams = rng.choices(range(NUM_FAMILIES), weights=weights, k=BURST_SIZE)
    for fam in fams:
        add(8.0 + rng.uniform(0.0, BURST_WINDOW), fam)

    reqs.sort(key=lambda r: r.arrival)
    for i, r in enumerate(reqs):
        r.id = i
    return reqs


def queue_depth_timeline(events, nodes, t0, t1, step=0.1):
    """Reconstruct waiting-queue depth over time from the event stream.

    A request waits on its node from `arrival` until `start` (admission). We
    sweep a fine time grid and count, per node and cluster-wide, how many
    requests are waiting at each instant. Returns (grid, total, per_node)."""
    import numpy as np
    grid = np.arange(t0, t1, step)
    per_node = {n["name"]: np.zeros_like(grid) for n in nodes}
    for e in events:
        a, s = e["arrival"], e["start"]
        mask = (grid >= a) & (grid < s)
        per_node[e["node"]] += mask
    total = np.sum([v for v in per_node.values()], axis=0)
    return grid, total, per_node


def summarize(policy, res):
    m = res["metrics"]
    ev = res["events"]
    total_prefill = sum(e["reuse"] + e["prefill"] for e in ev)
    # max queue depth on any node, any instant
    nodes = res["nodes"]
    t0 = min(e["arrival"] for e in ev)
    t1 = max(e["finish"] for e in ev) + 0.5
    _, total, per_node = queue_depth_timeline(ev, nodes, t0, t1)
    max_depth_cluster = int(total.max())
    max_depth_node = int(max(v.max() for v in per_node.values()))
    # cache reuse tier composition (share of requests served from each tier)
    tiers = {}
    for e in ev:
        tiers[e["tier"]] = tiers.get(e["tier"], 0) + 1
    tiers = {k: v / len(ev) for k, v in tiers.items()}
    # how the burst was spread across nodes (request count per node)
    per_node_count = {}
    for e in ev:
        per_node_count[e["node"]] = per_node_count.get(e["node"], 0) + 1
    return {
        "policy": policy,
        "mean_queue": m["mean_queue"],
        "peak_queue": m["peak_queue"],
        "mean_ttft": m["mean_ttft"],
        "mean_prefill": m["mean_prefill"],
        "total_prefill": total_prefill,
        "mean_lat": m["mean_lat"],
        "p95_lat": m["p95_lat"],
        "max_depth_cluster": max_depth_cluster,
        "max_depth_node": max_depth_node,
        "throughput": m["throughput"],
        "cache_hit": m["cache_hit"],
        "util": m["util"],
        "span": res["span"],
        "tiers": tiers,
        "per_node_count": per_node_count,
    }


def apply_condition(cfg, condition, base_cluster):
    """Mutate cfg for the given condition: RDMA congestion on/off, and whether
    the cross-node RDMA path is crippled (isolated replicas)."""
    from dataclasses import replace
    cripple, congestion = COND_SPEC[condition]
    cfg.RDMA_CONGESTION = congestion
    if cripple:
        cfg.CLUSTER = [(n, replace(s, rdma_bw=1e6), g) for n, s, g in base_cluster]
        cfg.DISK_CACHE = False
    else:
        cfg.CLUSTER = list(base_cluster)
        cfg.DISK_CACHE = True


def main():
    base = SimpleNamespace(**config.as_dict())
    base.BLOCK_TOKENS = BLOCK
    requests = make_burst()
    span_in = requests[-1].arrival - requests[0].arrival
    print(f"burst workload: {len(requests)} requests, arrivals over {span_in:.1f}s, "
          f"{NUM_FAMILIES} families x {PREFIX_BLOCKS * BLOCK} tok prefix, "
          f"output {OUTPUT_TOKENS} tok")
    print(f"cluster: {', '.join(f'{n}x{s.name}' for _, s, n in base.CLUSTER)}\n")

    out = {"config": {
        "prefix_tokens": PREFIX_BLOCKS * BLOCK,
        "num_families": NUM_FAMILIES, "zipf": ZIPF,
        "user_tokens": USER_BLOCKS * BLOCK,
        "output_tokens": OUTPUT_TOKENS,
        "burst_size": BURST_SIZE, "burst_window_s": BURST_WINDOW,
        "warmup": WARMUP, "seed": SEED, "num_requests": len(requests),
        "cluster": [f"{n}x{s.name}" for _, s, n in base.CLUSTER],
        "imbalance_abs": base.IMBALANCE_ABS, "imbalance_rel": base.IMBALANCE_REL,
    }, "conditions": {}}

    desc = {"shared": "RDMA-shared, contention-free",
            "congested": "RDMA-shared, fabric contends",
            "isolated": "isolated replicas"}
    for condition in CONDITIONS:
        cfg = SimpleNamespace(**config.as_dict())
        cfg.BLOCK_TOKENS = BLOCK
        apply_condition(cfg, condition, base.CLUSTER)
        cond = {"summaries": [], "events": {}}
        print(f"=== condition: {condition} ({desc[condition]}) ===")
        print(f"{'policy':<12} {'mean_q':>8} {'peak_q':>8} {'ttft':>8} "
              f"{'prefill':>8} {'maxdepth':>9} {'hit':>6} {'tok/s':>8}")
        for policy in POLICIES:
            res = run_seeded(policy, requests, cfg)
            s = summarize(policy, res)
            cond["summaries"].append(s)
            # store the event stream only for the two headline policies of the
            # shared/congested conditions -- all the timeline charts need
            if condition in ("shared", "congested") and policy in ("least_load", "cache_aware"):
                cond["events"][policy] = res["events"]
                cond["nodes"] = res["nodes"]
            print(f"{policy:<12} {s['mean_queue']:>7.2f}s {s['peak_queue']:>7.2f}s "
                  f"{s['mean_ttft']:>7.2f}s {s['mean_prefill']:>7.2f}s "
                  f"{s['max_depth_node']:>9d} {s['cache_hit']:>5.0%} {s['throughput']:>8.0f}")
        out["conditions"][condition] = cond
        print()

    out["scaling"] = scaling_sweep(base.CLUSTER)

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "burst_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("wrote results/burst_results.json")
    return out


def scaling_sweep(base_cluster):
    """Head-to-head under a congested fabric across cluster sizes: least_load vs
    cache_aware vs early_rdma (BTB). Scale the burst with the cluster to hold
    per-node load ~constant. Incast fan-in grows with node count, so the routing
    that avoids cross-node transfers (cache_aware / BTB) should pull away from
    least_load as the cluster grows."""
    global BURST_SIZE
    spec = base_cluster[0][1]
    gpus = base_cluster[0][2]
    saved = BURST_SIZE
    per_node = saved // len(base_cluster)
    print("=== scaling sweep: least_load vs cache_aware vs early_rdma "
          "(congested fabric) ===")
    print(f"{'nodes':>6} {'burst':>7} {'LL ttft':>8} {'CA ttft':>8} "
          f"{'BTB ttft':>9} {'CA gap':>7} {'BTB gap':>8}")
    rows = []
    for n_nodes in (4, 8, 16, 32):
        BURST_SIZE = per_node * n_nodes          # hold per-node load constant
        reqs = make_burst()
        cfg = SimpleNamespace(**config.as_dict())
        cfg.BLOCK_TOKENS = BLOCK
        cfg.RDMA_CONGESTION = True
        cfg.CLUSTER = [(f"node{i}", spec, gpus) for i in range(n_nodes)]
        cfg.DISK_CACHE = True
        set_btb_knobs(cfg, warm_copies=4)
        ll = run_seeded("least_load", reqs, cfg)["metrics"]
        ca = run_seeded("cache_aware", reqs, cfg)["metrics"]
        btb = run_seeded("early_rdma", reqs, cfg)["metrics"]
        ca_gap = (ll["mean_ttft"] - ca["mean_ttft"]) / ll["mean_ttft"] * 100
        btb_gap = (ll["mean_ttft"] - btb["mean_ttft"]) / ll["mean_ttft"] * 100
        rows.append({"nodes": n_nodes, "burst": BURST_SIZE,
                     "ll_ttft": ll["mean_ttft"], "ca_ttft": ca["mean_ttft"],
                     "btb_ttft": btb["mean_ttft"],
                     "gap_pct": ca_gap, "btb_gap_pct": btb_gap})
        print(f"{n_nodes:>6} {BURST_SIZE:>7} {ll['mean_ttft']:>7.1f}s "
              f"{ca['mean_ttft']:>7.1f}s {btb['mean_ttft']:>8.1f}s "
              f"{ca_gap:>6.0f}% {btb_gap:>7.0f}%")
    BURST_SIZE = saved
    print()
    return rows


if __name__ == "__main__":
    main()
