"""BTB `early_rdma` -- the whole method, as an inference-sim routing policy.

The rule is deliberately simple and fixed (no per-model learning):

    If the same Y-block prefix is seen X times within Z seconds, don't wait for
    a queue to build -- bite the bullet and replicate its KV to the least-busy
    replicas, then route later same-prefix requests across those warm copies.

    X = THRESHOLD   (repeats needed to fire)      default 4
    Y = KEY_BLOCKS  (prefix length that must match) default 4 blocks
    Z = WINDOW_S    (detection window)             default 2 s
    replicate to WARM_COPIES HBM copies            default 4

Mechanics:

  1. Detect the burst: a prefix seen >= X times inside a Z-second window becomes
     "active" for a HORIZON (so later requests keep routing to the warm copies).
  2. Once active, and once a source copy exists in some node's HBM, RDMA-copy the
     prefix KV onto the least-busy replicas until it has WARM_COPIES HBM copies.
  3. Route active-prefix requests to the least-loaded *warm* replica; fall back to
     cache_aware otherwise.

The warm copies pay RDMA cost on the target's timeline and are subject to the
same fabric contention as ordinary reuse (cfg.RDMA_CONGESTION): a copy to an
idle target is cheap, which is the point -- warming rides the quiet pre-burst
fabric instead of the congested spike.

register() installs it into the sim's POLICIES registry (the same dict object
simulate.py imported), so `run("early_rdma", ...)` works without touching the sim.
"""
import random


def _load(node):
    return len(node.running) + len(node.waiting)


def _resident(node, blocks):
    """True if the whole block run is resident in this node's HBM."""
    if not blocks:
        return False
    hbm_n, _ = node.match(blocks)
    return hbm_n >= len(blocks)


def _cache_aware(req, nodes, cfg, now):
    """Identical to router.CacheAware: imbalance-fallback, else longest local
    prefix then lightest load, ties broken randomly."""
    loads = [_load(nd) for nd in nodes]
    if max(loads) > cfg.IMBALANCE_ABS and max(loads) > cfg.IMBALANCE_REL * min(loads):
        return _pick(nodes, lambda nd: _load(nd))
    return _pick(nodes, lambda nd: (-sum(nd.match(req.blocks)), _load(nd)))


def _pick(nodes, key):
    best = min(key(nd) for nd in nodes)
    return random.choice([nd for nd in nodes if key(nd) == best])


class PendingWarm:
    __slots__ = ("ready", "node", "blocks", "key")

    def __init__(self, ready, node, blocks, key):
        self.ready, self.node, self.blocks, self.key = ready, node, blocks, key


class EarlyRdma:
    def __init__(self, cfg):
        self.cfg = cfg
        self.key_blocks = int(getattr(cfg, "BTB_KEY_BLOCKS", 4))    # Y: prefix length to match on
        self.threshold = int(getattr(cfg, "BTB_THRESHOLD", 4))      # X: repeats to fire
        self.window = float(getattr(cfg, "BTB_WINDOW_S", 2.0))      # Z: detection window (s)
        self.copies = int(getattr(cfg, "BTB_WARM_COPIES", 4))       # how many HBM copies to warm
        self.warm_blocks = int(getattr(cfg, "BTB_WARM_BLOCKS", 24)) # how much of the prefix KV to copy
        self.horizon = float(getattr(cfg, "BTB_HORIZON_S", 120.0))  # how long a fired prefix stays active
        self.hist = {}          # key -> list of recent arrival times
        self.active = {}        # key -> time the prefix stops being "active"
        self.pending = []       # scheduled warm copies not yet resident
        self.planned = set()    # (key, node.name) copies in flight, to dedupe
        self.stats = {"warm_count": 0, "warm_bytes": 0.0, "warm_busy_s": 0.0}

    # --- burst detection ---
    def _key(self, req):
        k = tuple(req.blocks[: self.key_blocks])
        return k if len(k) == self.key_blocks else None

    def _detect(self, key, now):
        h = self.hist.setdefault(key, [])
        h.append(now)
        cut = now - self.window
        while h and h[0] < cut:
            h.pop(0)
        if len(h) >= self.threshold:
            self.active[key] = now + self.horizon

    # --- warm bookkeeping ---
    def _apply_ready(self, now):
        keep = []
        for w in self.pending:
            if w.ready <= now + 1e-12:
                w.node.insert(w.blocks)          # prefix now resident in target HBM
                self.planned.discard((w.key, w.node.name))
            else:
                keep.append(w)
        self.pending = keep

    def _schedule(self, req, nodes, now, key):
        blocks = req.blocks[: self.warm_blocks]
        if not blocks:
            return
        # need a source copy in some node's HBM before we can RDMA it anywhere
        if not any(_resident(nd, blocks) for nd in nodes):
            return
        have = sum(1 for nd in nodes if _resident(nd, blocks)) \
            + sum(1 for (k, _) in self.planned if k == key)
        need = self.copies - have
        if need <= 0:
            return
        # warm the least-busy replicas that don't have it and aren't already queued
        targets = sorted((nd for nd in nodes
                          if not _resident(nd, blocks)
                          and (key, nd.name) not in self.planned),
                         key=_load)
        nbytes = len(blocks) * nodes[0].block_bytes
        for nd in targets[:need]:
            dur = nbytes / nd.tier_bw["rdma"]
            if getattr(self.cfg, "RDMA_CONGESTION", False):
                dur *= min(len(nodes), max(1, len(nd.waiting)))
            start = max(now, nd.now)
            nd.now = start + dur                 # copy occupies the target
            nd.busy += dur
            self.pending.append(PendingWarm(start + dur, nd, blocks, key))
            self.planned.add((key, nd.name))
            self.stats["warm_count"] += 1
            self.stats["warm_bytes"] += nbytes
            self.stats["warm_busy_s"] += dur

    # --- routing ---
    def route(self, req, nodes, now):
        self._apply_ready(now)
        key = self._key(req)
        if key is not None:
            self._detect(key, now)
            if self.active.get(key, -1.0) >= now:
                self._schedule(req, nodes, now, key)
                blocks = req.blocks[: self.warm_blocks]
                warm = [nd for nd in nodes if _resident(nd, blocks)]
                if warm:
                    return _pick(warm, _load)   # least-loaded warm replica
        return _cache_aware(req, nodes, self.cfg, now)


def register():
    """Install early_rdma into the sim's shared POLICIES dict."""
    import router
    router.POLICIES["early_rdma"] = EarlyRdma
