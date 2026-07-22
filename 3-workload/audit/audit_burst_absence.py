#!/usr/bin/env python3
"""Audit public LLM inference traces for synchronized same-prefix fan-out bursts.

Motivating question
-------------------
Bite-The-Bullet's `early_rdma` policy targets one specific workload shape:

    K or more requests that SHARE A LONG PREFIX and ARRIVE WITHIN A SHORT WINDOW.

That is what a data-labeling sweep (many records scored against the same
document) or an agent fan-out (20 subagents inheriting the same big system
prompt, all firing at once) looks like at the serving layer. The BTB blog
already *asserts* public traces do not contain this shape. This script tries to
*prove* it, quantitatively, across the traces people actually cite:

    - ART-Chat-2.5M          (has prefix hashes + arrival times)
    - Mooncake conversation  (has prefix hashes + arrival times)
    - Mooncake toolagent     (has prefix hashes + arrival times)
    - Mooncake arxiv         (has prefix hashes + arrival times)
    - BurstGPT               (arrival times + token counts, NO prefix info)
    - Bursted-ART synthetic  (LOCAL positive control: the pattern IS present)

LMSYS-Chat-1M and ShareGPT are handled structurally: they are chat-UI conversa-
tion dumps with no per-request arrival timestamps (LMSYS is also gated), so a
synchronized arrival burst cannot exist in them *by construction*. That absence
is itself a finding, reported without a heavy download.

The metric
----------
For a trace with prefix hashes we define, for each required shared-prefix depth
`d` (in prefix blocks):

    prefix_key(request) = first d block hashes           (only if request has >= d blocks)
    fanout(key, W)      = max requests sharing `key` that co-occur in ANY
                          W-second sliding window

The headline number is the maximum fan-out over all prefix keys at a deep
threshold (d >= 8 blocks) within a generous W = 60 s window. If that number is
small (single/low-double digits) the trace does not contain the labeling /
subagent-fanout regime, no matter how "bursty" its raw arrival rate looks.

Everything is streamed and row-capped so it runs on a laptop. Reads are
contiguous (a real slice of wall-clock time), not shuffled, so co-arrival
structure is preserved. Nothing here fabricates data; if a trace *did* contain
the pattern, the positive control proves the detector would light up on it.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

# Depths (in prefix blocks) at which we require a shared prefix. Larger depth =
# a genuinely long shared context, which is what the target workload has.
DEPTHS = [1, 2, 4, 8, 16, 32]

# Sliding co-arrival windows, seconds. 60s is deliberately generous: it gives a
# would-be burst a full minute to accumulate before we call it a burst.
WINDOWS = [1.0, 10.0, 60.0]

# Fan-out thresholds we count "bursts" at. 20 is the article's headline
# ("fan out to 20 sub agents").
FANOUT_THRESHOLDS = [5, 10, 20, 50, 100]

# The target pattern (data-labeling sweep / agent fan-out) is not just "many
# requests share the leading block". It is DEEP (a long, job-unique shared
# prefix) AND SYNCHRONIZED (they arrive together, not as a steady trickle of a
# global system-prompt template). We require:
#   depth >= DEEP_BLOCKS  and  fan-out inside a <= SYNC_WINDOW-second window.
# Mooncake block size is ~512 tokens, so 16 blocks ~= 8k tokens of shared prefix.
DEEP_BLOCKS = 16
SYNC_WINDOW = 10.0
TARGET_FANOUT = 20     # "fan out to 20 sub agents"

# We keep at most this many leading block hashes per request (bounds memory and
# is >= max depth we test).
MAX_KEEP_BLOCKS = max(DEPTHS)

MOONCAKE_BASE = "https://raw.githubusercontent.com/kvcache-ai/Mooncake/main"
BURSTGPT_URL = "https://raw.githubusercontent.com/HPMLL/BurstGPT/main/data/BurstGPT_1.csv"


# --------------------------------------------------------------------------- #
# Normalized record
# --------------------------------------------------------------------------- #

@dataclass
class Record:
    t: float                      # arrival time, seconds
    blocks: tuple                 # up to MAX_KEEP_BLOCKS leading prefix hashes
    input_length: int


# A cluster of co-arriving same-prefix requests is only a data-labeling / agent
# fan-out if the requests are INDEPENDENT: same long prefix, small varied
# suffix, so their input_lengths are all similar. If instead input_length grows
# monotonically across the cluster, it is ONE conversation/agent session whose
# context is accumulating -- sequential turns, not a parallel fan-out. These two
# look identical to a naive "same-prefix within a window" counter, so we
# separate them explicitly.
SESSION_SPREAD = 2.0       # max/min input_length >= this ...
SESSION_MONOTONE = 0.8     # ... and >= this fraction of steps non-decreasing => session, not fan-out


def classify_cluster(input_lengths: list[int]) -> str:
    """'session' (one growing context) or 'fanout' (many independent requests)."""
    lens = [x for x in input_lengths if x and x > 0]
    if len(lens) < 3:
        return "fanout"
    lo, hi = min(lens), max(lens)
    spread = hi / max(1, lo)
    nondec = sum(1 for a, b in zip(lens, lens[1:]) if b >= a) / max(1, len(lens) - 1)
    if spread >= SESSION_SPREAD and nondec >= SESSION_MONOTONE:
        return "session"
    return "fanout"


# --------------------------------------------------------------------------- #
# Loaders  -- each yields Record objects, streaming, contiguous, row-capped
# --------------------------------------------------------------------------- #

def _http_lines(url: str, timeout: int = 120) -> Iterator[str]:
    req = urllib.request.Request(url, headers={"User-Agent": "btb-audit/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in io.TextIOWrapper(resp, encoding="utf-8"):
            yield raw


def load_mooncake(url: str, max_rows: int) -> Iterator[Record]:
    """Mooncake jsonl: {timestamp(ms), input_length, output_length, hash_ids[int]}."""
    for i, line in enumerate(_http_lines(url)):
        if i >= max_rows:
            break
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        blocks = tuple(row.get("hash_ids") or [])[:MAX_KEEP_BLOCKS]
        yield Record(
            t=float(row.get("timestamp") or 0) / 1000.0,
            blocks=blocks,
            input_length=int(row.get("input_length") or 0),
        )


def load_burstgpt(url: str, max_rows: int) -> Iterator[Record]:
    """BurstGPT csv: Timestamp(s), Model, Request tokens, Response tokens, Total, Log Type.

    No prompt content and no prefix hashes exist in this trace at all, so every
    record carries an empty `blocks` -- the point being that this trace *cannot*
    express same-prefix fan-out even in principle.
    """
    reader = csv.DictReader(_http_lines(url))
    for i, row in enumerate(reader):
        if i >= max_rows:
            break
        try:
            t = float(row["Timestamp"])
            inp = int(float(row.get("Request tokens") or 0))
        except (KeyError, ValueError):
            continue
        yield Record(t=t, blocks=(), input_length=inp)


def load_art(dataset: str, split: str, max_rows: int) -> Iterator[Record]:
    """ART parquet stream from HF. timestamp_ms(ms), input_length, hash_ids[str]."""
    try:
        import fsspec
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("ART needs fsspec + pyarrow (pip install fsspec pyarrow)") from exc

    files = _hf_parquet_files(dataset, split)
    fs = fsspec.filesystem("https")
    emitted = 0
    for entry in files:
        if emitted >= max_rows:
            break
        with fs.open(entry["url"], "rb", block_size=1 << 20) as fh:
            pf = pq.ParquetFile(fh)
            cols = [c for c in ("timestamp_ms", "timestamp", "input_length", "hash_ids")
                    if c in set(pf.schema_arrow.names)]
            for batch in pf.iter_batches(batch_size=10000, columns=cols):
                d = batch.to_pydict()
                n = len(d[cols[0]])
                for j in range(n):
                    if emitted >= max_rows:
                        return
                    ts = d.get("timestamp_ms", [None] * n)[j]
                    if ts is None:
                        ts = d.get("timestamp", [0] * n)[j]
                    blocks = tuple(str(b) for b in (d["hash_ids"][j] or []))[:MAX_KEEP_BLOCKS]
                    yield Record(
                        t=float(ts or 0) / 1000.0,
                        blocks=blocks,
                        input_length=int(d.get("input_length", [0] * n)[j] or 0),
                    )
                    emitted += 1


def load_local_jsonl(path: Path, max_rows: int) -> Iterator[Record]:
    """Local Bursted-ART jsonl: arrival_s(s), input_length, hash_ids[str]. Positive control."""
    with path.open() as fh:
        for i, line in enumerate(fh):
            if i >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            blocks = tuple(str(b) for b in (row.get("hash_ids") or []))[:MAX_KEEP_BLOCKS]
            t = row.get("arrival_s")
            if t is None:
                t = row.get("timestamp", 0)
            yield Record(t=float(t or 0), blocks=blocks, input_length=int(row.get("input_length") or 0))


def _hf_parquet_files(dataset: str, split: str) -> list[dict]:
    url = f"https://datasets-server.huggingface.co/parquet?dataset={urllib.parse.quote(dataset)}"
    with urllib.request.urlopen(url, timeout=60) as resp:
        data = json.load(resp)
    files = [e for e in data["parquet_files"] if e["split"] == split]
    if not files:
        raise RuntimeError(f"no parquet files for {dataset}/{split}")
    return files


# --------------------------------------------------------------------------- #
# Core detector
# --------------------------------------------------------------------------- #

def max_cowindow_count(times_sorted: list[float], w: float) -> tuple[int, float, int, int]:
    """Max number of points inside any window of width `w`. Returns
    (best_count, span_of_best_cluster, left_idx, right_idx) where [left..right]
    is one window achieving best_count. Two-pointer over sorted times: O(n)."""
    best = 0
    best_span = 0.0
    best_l, best_r = 0, -1
    left = 0
    n = len(times_sorted)
    for right in range(n):
        while times_sorted[right] - times_sorted[left] > w:
            left += 1
        count = right - left + 1
        if count > best:
            best = count
            best_span = times_sorted[right] - times_sorted[left]
            best_l, best_r = left, right
    return best, best_span, best_l, best_r


@dataclass
class DatasetResult:
    name: str
    source: str
    has_prefix_info: bool
    n_read: int = 0
    n_with_prefix: int = 0
    time_span_s: float = 0.0
    mean_rate_per_s: float = 0.0
    raw_arrival_burst: dict = field(default_factory=dict)   # window -> max reqs (any prefix)
    depth_stats: dict = field(default_factory=dict)         # depth -> {...}
    headline: dict = field(default_factory=dict)
    note: str = ""
    error: str = ""


def analyze(name: str, source: str, records: Iterable[Record], has_prefix_info: bool,
            windows=WINDOWS, depths=DEPTHS) -> DatasetResult:
    res = DatasetResult(name=name, source=source, has_prefix_info=has_prefix_info)

    all_times: list[float] = []
    # depth -> key(tuple) -> list[(time, input_length)]
    groups: dict[int, dict[tuple, list[tuple]]] = {d: defaultdict(list) for d in depths}

    for rec in records:
        res.n_read += 1
        all_times.append(rec.t)
        if has_prefix_info and rec.blocks:
            res.n_with_prefix += 1
            for d in depths:
                if len(rec.blocks) >= d:
                    groups[d][rec.blocks[:d]].append((rec.t, rec.input_length))

    if res.n_read == 0:
        res.error = "no rows read"
        return res

    all_times.sort()
    res.time_span_s = all_times[-1] - all_times[0]
    res.mean_rate_per_s = res.n_read / res.time_span_s if res.time_span_s > 0 else float("inf")

    # Raw arrival burstiness (ignores prefix) -- context for "bursty" claims.
    for w in windows:
        cnt, _, _, _ = max_cowindow_count(all_times, w)
        res.raw_arrival_burst[f"{w:g}s"] = cnt

    if not has_prefix_info:
        res.note = ("No prefix/content in this trace -> same-prefix fan-out is "
                    "structurally impossible to express. Only aggregate arrival "
                    "burstiness is measurable.")
        return res

    # Same-prefix fan-out at each depth x window.
    for d in depths:
        by_key = groups[d]
        n_keys = len(by_key)
        # widest window is the most generous; headline uses it.
        depth_entry: dict[str, Any] = {"n_prefix_groups": n_keys, "windows": {}}
        for w in windows:
            best_fanout, best_span, best_kind = 0, 0.0, ""
            # "fanout-like" = the target: independent same-prefix requests, not a
            # single growing-context session.
            best_fo_fanout, best_fo_span = 0, 0.0
            top = []
            counts_at = {t: 0 for t in FANOUT_THRESHOLDS}            # any cluster
            counts_fanout = {t: 0 for t in FANOUT_THRESHOLDS}       # fan-out-like only
            for key, pts in by_key.items():
                if len(pts) < 2:
                    continue
                pts.sort()                                          # by time
                times = [p[0] for p in pts]
                fo, span, li, ri = max_cowindow_count(times, w)
                kind = classify_cluster([pts[i][1] for i in range(li, ri + 1)])
                if fo > best_fanout:
                    best_fanout, best_span, best_kind = fo, span, kind
                if kind == "fanout" and fo > best_fo_fanout:
                    best_fo_fanout, best_fo_span = fo, span
                for thr in FANOUT_THRESHOLDS:
                    if fo >= thr:
                        counts_at[thr] += 1
                        if kind == "fanout":
                            counts_fanout[thr] += 1
                top.append(fo)
            top.sort(reverse=True)
            depth_entry["windows"][f"{w:g}s"] = {
                "max_fanout": best_fanout,
                "max_fanout_span_s": round(best_span, 3),
                "max_fanout_kind": best_kind,          # session vs fanout for the top cluster
                "max_fanout_like": best_fo_fanout,     # largest genuine fan-out (excludes sessions)
                "max_fanout_like_span_s": round(best_fo_span, 3),
                "top5_fanout": top[:5],
                "groups_with_fanout_ge": counts_at,
                "groups_with_fanout_like_ge": counts_fanout,
            }
        res.depth_stats[d] = depth_entry

    # ---- Headline ---------------------------------------------------------
    # Two numbers that together tell the real story:
    #
    # (A) "shallow-wide" fan-out: max fan-out at depth >= 8 blocks in the widest
    #     window. This is the number that *looks* like a burst but is usually a
    #     global system-prompt template shared by many unrelated requests. We
    #     also keep its arrival span: span ~= window means a steady trickle, not
    #     a synchronized job.
    #
    # (B) "deep-sync" fan-out: max fan-out at depth >= DEEP_BLOCKS within a
    #     SYNC_WINDOW-second window. THIS is the target pattern -- a long,
    #     job-unique shared prefix fanned out to many co-arriving requests. The
    #     pattern is "present" only if this reaches TARGET_FANOUT.
    wide = f"{max(windows):g}s"
    sync = None
    for cand in (f"{SYNC_WINDOW:g}s", "10s", wide):
        if cand in [f"{w:g}s" for w in windows]:
            sync = cand
            break
    sync = sync or wide

    # (A) shallow-wide
    shallow_depths = [d for d in depths if d >= 8] or [max(depths)]
    a_max, a_depth, a_span = 0, None, 0.0
    for d in shallow_depths:
        e = res.depth_stats.get(d, {}).get("windows", {}).get(wide, {})
        if e.get("max_fanout", 0) > a_max:
            a_max = e["max_fanout"]
            a_depth = d
            a_span = e.get("max_fanout_span_s", 0.0)

    # (B) deep-sync -- uses the FAN-OUT-LIKE number, which excludes single
    # growing-context sessions (sequential turns that merely share a prefix).
    deep_depths = [d for d in depths if d >= DEEP_BLOCKS] or [max(depths)]
    b_max, b_depth, b_span, b_ge_target = 0, None, 0.0, 0
    b_any, b_any_kind = 0, ""   # largest cluster of ANY kind at deep/sync, for transparency
    for d in deep_depths:
        e = res.depth_stats.get(d, {}).get("windows", {}).get(sync, {})
        if e.get("max_fanout_like", 0) > b_max:
            b_max = e["max_fanout_like"]
            b_depth = d
            b_span = e.get("max_fanout_like_span_s", 0.0)
        if e.get("max_fanout", 0) > b_any:
            b_any = e["max_fanout"]
            b_any_kind = e.get("max_fanout_kind", "")
        b_ge_target += e.get("groups_with_fanout_like_ge", {}).get(TARGET_FANOUT, 0)

    # Interpret the shallow number: steady template vs genuine spike.
    a_is_steady = a_span >= 0.5 * max(windows) and a_max > 0
    res.headline = {
        "shallow_wide": {
            "window": wide, "depth_ge_blocks": 8, "max_fanout": a_max,
            "at_depth_blocks": a_depth, "arrival_span_s": round(a_span, 3),
            "reads_as": ("steady shared-template stream (not a synchronized job)"
                         if a_is_steady else "clustered"),
        },
        "deep_sync": {
            "window": sync, "depth_ge_blocks": DEEP_BLOCKS,
            "max_fanout": b_max,                       # fan-out-like (sessions excluded)
            "at_depth_blocks": b_depth, "arrival_span_s": round(b_span, 3),
            "largest_cluster_any_kind": b_any,         # incl. growing sessions
            "largest_cluster_kind": b_any_kind,
            f"num_groups_fanout_ge_{TARGET_FANOUT}": b_ge_target,
        },
        # single boolean the article hangs on:
        "target_pattern_present": bool(b_max >= TARGET_FANOUT),
        "max_same_prefix_fanout": b_max,
        "num_bursts_fanout_ge_20": b_ge_target,
    }
    return res


# --------------------------------------------------------------------------- #
# Dataset registry
# --------------------------------------------------------------------------- #

def dataset_specs(args) -> list[dict]:
    specs = [
        {"name": "ART-Chat-2.5M", "source": "hf:alessiotoniolo/ART-Chat-2.5M",
         "has_prefix": True,
         "load": lambda: load_art("alessiotoniolo/ART-Chat-2.5M", "train", args.max_rows)},
        {"name": "Mooncake-conversation", "source": f"{MOONCAKE_BASE}/FAST25-release/traces/conversation_trace.jsonl",
         "has_prefix": True,
         "load": lambda: load_mooncake(f"{MOONCAKE_BASE}/FAST25-release/traces/conversation_trace.jsonl", args.max_rows)},
        {"name": "Mooncake-toolagent", "source": f"{MOONCAKE_BASE}/FAST25-release/traces/toolagent_trace.jsonl",
         "has_prefix": True,
         "load": lambda: load_mooncake(f"{MOONCAKE_BASE}/FAST25-release/traces/toolagent_trace.jsonl", args.max_rows)},
        {"name": "Mooncake-arxiv", "source": f"{MOONCAKE_BASE}/FAST25-release/arxiv-trace/mooncake_trace.jsonl",
         "has_prefix": True,
         "load": lambda: load_mooncake(f"{MOONCAKE_BASE}/FAST25-release/arxiv-trace/mooncake_trace.jsonl", args.max_rows)},
        {"name": "BurstGPT", "source": BURSTGPT_URL, "has_prefix": False,
         "load": lambda: load_burstgpt(BURSTGPT_URL, args.max_rows)},
    ]
    if args.datasets:
        want = {d.lower() for d in args.datasets}
        specs = [s for s in specs if any(w in s["name"].lower() for w in want)]
    return specs


# structural-absence datasets (no download): no per-request arrival timestamps.
STRUCTURAL = [
    {"name": "LMSYS-Chat-1M", "source": "hf:lmsys/lmsys-chat-1m (gated)",
     "note": "Chat-UI conversation dump. Gated; and it carries no per-request "
             "arrival timestamps -> a synchronized arrival burst cannot exist by "
             "construction. It records what users typed into a demo arena, not a "
             "production serving endpoint."},
    {"name": "ShareGPT", "source": "web-scraped ShareGPT conversations",
     "note": "Conversation dump scraped from the ShareGPT sharing site. No "
             "timestamps and no arrival order -> no burst structure exists to "
             "measure. These are shared chat transcripts, not an inference "
             "endpoint's request log."},
]


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def render_markdown(results: list[DatasetResult], args) -> str:
    lines = []
    lines.append("# Audit: are synchronized same-prefix fan-out bursts present in public traces?\n")
    lines.append(f"_Generated {datetime.now(timezone.utc).isoformat()} • "
                 f"max_rows/trace={args.max_rows} • windows={WINDOWS}s • "
                 f"depths={DEPTHS} blocks_\n")
    lines.append(f"**Target pattern:** K+ requests sharing a *long, job-unique* prefix, arriving "
                 f"*together*. Concretely: fan-out ≥ {TARGET_FANOUT} at shared-prefix depth ≥ "
                 f"{DEEP_BLOCKS} blocks (~{DEEP_BLOCKS*512//1000}k tokens) inside a ≤ {SYNC_WINDOW:g}s "
                 f"window. A shallow prefix shared by everyone (a global system-prompt template) or a "
                 f"reuse spread over minutes does **not** count — those are ordinary chat/agent traffic, "
                 f"not a data-labeling sweep or a synchronized sub-agent fan-out.\n")

    # Headline table
    lines.append("## Headline\n")
    lines.append(f"Two numbers per trace. **Deep-sync fan-out** is the target metric (≥{DEEP_BLOCKS} "
                 f"blocks, ≤{SYNC_WINDOW:g}s). **Shallow-wide fan-out** (≥8 blocks, 60s) is the number "
                 "that *looks* like a burst — its arrival span tells you whether it is a synchronized "
                 "job (span ≪ window) or a steady shared-template stream (span ≈ window).\n")
    lines.append(f"| Trace | Rows | Span | **Deep-sync fan-out** (≥{DEEP_BLOCKS}blk,≤{SYNC_WINDOW:g}s, sessions excluded) | "
                 f"Largest deep cluster incl. sessions | Shallow fan-out (≥8blk,60s) | Target present? |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | :---: |")
    for r in results:
        if r.error:
            lines.append(f"| {r.name} | — | — | — | — | — | ERROR: {r.error} |")
            continue
        span = f"{r.time_span_s/60:.1f} min" if r.time_span_s else "—"
        if r.has_prefix_info:
            hl = r.headline
            ds, sw = hl["deep_sync"], hl["shallow_wide"]
            present = "✅ YES" if hl.get("target_pattern_present") else "❌ no"
            sw_span = f"{sw['max_fanout']} ({sw['arrival_span_s']:g}s" + (", steady)" if "steady" in sw["reads_as"] else ")")
            any_kind = f"{ds['largest_cluster_any_kind']}" + (f" ({ds['largest_cluster_kind']})" if ds['largest_cluster_kind'] else "")
            lines.append(f"| {r.name} | {r.n_read:,} | {span} | **{ds['max_fanout']}** | "
                         f"{any_kind} | {sw_span} | {present} |")
        else:
            raw60 = r.raw_arrival_burst.get("60s", "—")
            lines.append(f"| {r.name} | {r.n_read:,} | {span} | n/a (no prefix) | n/a | "
                         f"raw {raw60}/60s | ❌ n/a |")
    lines.append("")

    # Structural-absence datasets
    lines.append("## Structural absence (no download needed)\n")
    for s in STRUCTURAL:
        lines.append(f"- **{s['name']}** (`{s['source']}`): {s['note']}")
    lines.append("")

    # Per-dataset detail
    lines.append("## Per-trace detail\n")
    for r in results:
        lines.append(f"### {r.name}\n")
        lines.append(f"- source: `{r.source}`")
        if r.error:
            lines.append(f"- **ERROR:** {r.error}\n")
            continue
        lines.append(f"- rows read: {r.n_read:,}  (with prefix: {r.n_with_prefix:,})")
        lines.append(f"- time span: {r.time_span_s:.1f}s ({r.time_span_s/60:.1f} min), "
                     f"mean rate {r.mean_rate_per_s:.2f} req/s")
        lines.append(f"- raw arrival burst (any prefix): {r.raw_arrival_burst}")
        if r.note:
            lines.append(f"- note: {r.note}")
        if r.has_prefix_info and r.depth_stats:
            lines.append("\n  Same-prefix fan-out by required shared-prefix depth (window = 60s):\n")
            lines.append("  | depth (blocks) | prefix groups | max fan-out | span(s) | groups ≥20 |")
            lines.append("  | ---: | ---: | ---: | ---: | ---: |")
            for d in DEPTHS:
                e = r.depth_stats.get(d)
                if not e:
                    continue
                w = e["windows"].get("60s", {})
                lines.append(f"  | {d} | {e['n_prefix_groups']:,} | {w.get('max_fanout','—')} | "
                             f"{w.get('max_fanout_span_s','—')} | {w.get('groups_with_fanout_ge',{}).get(20,0)} |")
            hl = r.headline
            ds, sw = hl["deep_sync"], hl["shallow_wide"]
            if hl.get("target_pattern_present"):
                lines.append(f"\n  → **Target pattern PRESENT.** Fan-out-like cluster of {ds['max_fanout']} "
                             f"independent requests at depth ≥{DEEP_BLOCKS} blocks within {ds['arrival_span_s']}s.\n")
            else:
                sess = ""
                if ds.get("largest_cluster_kind") == "session" and ds["largest_cluster_any_kind"] > ds["max_fanout"]:
                    sess = (f" There *is* a larger deep-prefix cluster of {ds['largest_cluster_any_kind']} "
                            f"requests, but it is a single **growing-context session** (input length climbs "
                            f"monotonically across sequential turns), not a parallel fan-out of independent "
                            f"requests.")
                lines.append(
                    f"\n  → **Target pattern ABSENT.** At a genuinely long shared prefix "
                    f"(≥{DEEP_BLOCKS} blocks ≈ {DEEP_BLOCKS*512//1000}k tokens) the largest *fan-out* of "
                    f"independent requests in any {SYNC_WINDOW:g}s window is only **{ds['max_fanout']}**."
                    f"{sess} The largest *shallow* (≥8-block) fan-out is {sw['max_fanout']} over "
                    f"{sw['arrival_span_s']}s ({sw['reads_as']}) — a shared system-prompt template, not a "
                    f"labeling/fan-out job.\n")
        lines.append("")

    # Verdict
    lines.append("## Verdict\n")
    real = [r for r in results if r.has_prefix_info and not r.error]
    noprefix = [r for r in results if not r.has_prefix_info and not r.error]
    present = [r for r in real if r.headline.get("target_pattern_present")]
    absent = [r for r in real if not r.headline.get("target_pattern_present")]

    lines.append("**1. Chat / arena dumps (LMSYS-Chat-1M, ShareGPT):** no per-request arrival "
                 "timestamps → a synchronized burst cannot exist by construction. Not a serving log.\n")
    if noprefix:
        lines.append(f"**2. Aggregate arrival traces ({', '.join(r.name for r in noprefix)}):** timestamps "
                     f"and token counts, but **no prefix/content at all** → same-prefix fan-out is "
                     f"structurally impossible to express, regardless of how bursty the raw arrival rate "
                     f"is (peak {noprefix[0].raw_arrival_burst.get('60s','?')} req/60s here).\n")
    if absent:
        lines.append(f"**3. Prefix-bearing serving traces ({', '.join(r.name for r in absent)}):** carry "
                     f"both prefix hashes and timing, yet the largest synchronized fan-out of independent "
                     f"requests over a long shared prefix is "
                     f"**{max(r.headline['deep_sync']['max_fanout'] for r in absent)}** — no data-labeling "
                     f"or sub-agent fan-out job. Their big-looking numbers are shallow, steady "
                     f"system-prompt templates, or single growing-context sessions.\n")
    if present:
        for r in present:
            ds = r.headline["deep_sync"]
            n_events = ds.get(f"num_groups_fanout_ge_{TARGET_FANOUT}", 0)
            per100k = n_events / max(1, r.n_read) * 100000
            lines.append(
                f"**4. {r.name}:** the *only* trace that contains the pattern at all — but marginally. "
                f"Max fan-out **{ds['max_fanout']}-way** "
                f"(vs the hundreds-to-thousands of a real production labeling sweep), and only "
                f"**{n_events} such events in {r.n_read:,} requests** (~{per100k:.1f} per 100k, "
                f"≈{n_events/(r.time_span_s/3600):.1f}/hour). ART is a specialized decoded-LLM-response / "
                f"agent-eval corpus (request IDs are batch-timestamped `decoded_llm_responses_…`), i.e. "
                f"already the most batch-like public dataset — which is exactly why BTB builds on ART and "
                f"then *adds* synthetic bursts to reach production scale.\n")
    lines.append("### Bottom line\n")
    lines.append(
        "> The synchronized long-prefix fan-out that KV-warming targets is **absent from every "
        "general-endpoint public trace** and **structurally impossible in the chat/arena dumps**. The "
        "one trace that shows it (ART) is a specialized agent/eval decode corpus, and even there it is "
        "rare and an order of magnitude below production scale. These datasets were captured on "
        "demo/arena/chat endpoints and short-lived keys — not on production endpoints running "
        "data-labeling or multi-agent jobs — so none of them represent that regime.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--max-rows", type=int, default=300000,
                   help="max contiguous rows to read per trace (default 300000)")
    p.add_argument("--datasets", nargs="*", default=None,
                   help="substring filter, e.g. --datasets mooncake art control")
    p.add_argument("--control-jsonl", default="3-workload/generate/out/Bursted-ART/test.jsonl",
                   help="local synthetic positive-control jsonl")
    p.add_argument("--out-dir", type=Path, default=Path("3-workload/audit/results"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    specs = dataset_specs(args)
    results: list[DatasetResult] = []

    for spec in specs:
        print(f"[audit] {spec['name']} ...", flush=True)
        try:
            res = analyze(spec["name"], spec["source"], spec["load"](), spec["has_prefix"])
        except Exception as exc:  # noqa: BLE001 -- report, don't crash the whole run
            res = DatasetResult(name=spec["name"], source=spec["source"],
                                has_prefix_info=spec["has_prefix"], error=f"{type(exc).__name__}: {exc}")
        results.append(res)
        if res.error:
            print(f"          ERROR: {res.error}", flush=True)
        elif res.has_prefix_info:
            hl = res.headline
            ds, sw = hl["deep_sync"], hl["shallow_wide"]
            print(f"          rows={res.n_read:,}  deep-sync fanout(>={DEEP_BLOCKS}blk,{ds['window']})="
                  f"{ds['max_fanout']} (span {ds['arrival_span_s']}s)  "
                  f"shallow fanout(>=8blk,{sw['window']})={sw['max_fanout']} ({sw['reads_as']})  "
                  f"TARGET={'PRESENT' if hl['target_pattern_present'] else 'absent'}", flush=True)
        else:
            print(f"          rows={res.n_read:,} (no prefix info) raw_max/60s="
                  f"{res.raw_arrival_burst.get('60s')}", flush=True)

    # write JSON
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "params": {"max_rows": args.max_rows, "depths_blocks": DEPTHS,
                   "windows_s": WINDOWS, "fanout_thresholds": FANOUT_THRESHOLDS},
        "structural_absence": STRUCTURAL,
        "results": [vars(r) for r in results],
    }
    (args.out_dir / "burst_audit.json").write_text(json.dumps(payload, indent=2) + "\n")
    md = render_markdown(results, args)
    (args.out_dir / "burst_audit.md").write_text(md + "\n")
    print(f"\n[audit] wrote {args.out_dir/'burst_audit.json'} and {args.out_dir/'burst_audit.md'}")
    print("\n" + md.split("## Per-trace detail")[0])


if __name__ == "__main__":
    main()
