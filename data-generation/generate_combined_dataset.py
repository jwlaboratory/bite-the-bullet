#!/usr/bin/env python3
"""Generate a mixed ART + synthetic BTB trace dataset.

The output is request-level JSONL plus a manifest. Windows are split as whole
units, so train/test leakage cannot happen through neighboring rows from the
same trace window.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VERSION = "btb-mixed-trace-v1"
ART_COLUMNS = [
    "request_id",
    "token_hash",
    "system_prompt_hash",
    "timestamp",
    "timestamp_ms",
    "input_length",
    "output_length",
    "hash_ids",
    "messages",
]


@dataclass
class TraceWindow:
    trace_id: str
    source: str
    rows: list[dict[str, Any]]
    metadata: dict[str, Any]
    split: str = ""


def ceil_blocks(tokens: int, block_tokens: int) -> int:
    return math.ceil(tokens / block_tokens)


def key_for(blocks: list[str], key_blocks: int) -> str:
    if len(blocks) < key_blocks:
        return ""
    return "|".join(blocks[:key_blocks])


def parquet_files(dataset: str, split: str, config_name: str | None) -> list[dict[str, Any]]:
    url = f"https://datasets-server.huggingface.co/parquet?dataset={urllib.parse.quote(dataset)}"
    if config_name:
        url += f"&config={urllib.parse.quote(config_name)}"
    with urllib.request.urlopen(url, timeout=60) as resp:
        data = json.load(resp)
    files = [entry for entry in data["parquet_files"] if entry["split"] == split]
    if config_name:
        files = [entry for entry in files if entry.get("config") == config_name]
    if not files:
        raise RuntimeError(f"no parquet files found for {dataset}/{split}")
    return files


def timestamp_seconds(row: dict[str, Any], t0: float) -> float:
    if row.get("timestamp_ms") is not None:
        return (float(row["timestamp_ms"]) - t0) / 1000.0
    raw = float(row.get("timestamp") or 0.0)
    if raw > 1e9:
        return (raw - t0) / 1000.0
    return raw - t0


def normalize_art_rows(
    rows: list[dict[str, Any]],
    *,
    trace_id: str,
    source_window_index: int,
    key_blocks: int,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    rows = sorted(rows, key=lambda row: row.get("timestamp_ms", row.get("timestamp", 0)))
    first_raw = rows[0].get("timestamp_ms")
    if first_raw is None:
        first_raw = rows[0].get("timestamp", 0)
    t0 = float(first_raw or 0.0)

    normalized = []
    for i, row in enumerate(rows):
        blocks = [str(block) for block in list(row.get("hash_ids") or [])]
        group = key_for(blocks, key_blocks)
        request_id = str(row.get("request_id", f"{trace_id}:{i}"))
        messages = row.get("messages") or ""
        normalized.append(
            {
                "dataset_version": VERSION,
                "source": "art",
                "trace_id": trace_id,
                "source_window_index": source_window_index,
                "row_in_window": i,
                "request_id": request_id,
                "timestamp": timestamp_seconds(row, t0),
                "arrival_s": timestamp_seconds(row, t0),
                "input_length": max(1, int(row.get("input_length") or 1)),
                "output_length": max(1, int(row.get("output_length") or 1)),
                "hash_ids": blocks,
                "session_id": group,
                "group_id": group,
                "system_prompt_hash": str(row.get("system_prompt_hash") or ""),
                "messages": "",
                "metadata": {
                    "family": "art_replay",
                    "message_bytes": len(str(messages)),
                    "token_hash": str(row.get("token_hash") or ""),
                },
            }
        )
    return normalized


def load_art_windows(args: argparse.Namespace, rng: random.Random) -> list[TraceWindow]:
    if args.art_windows <= 0:
        return []

    try:
        import fsspec
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit(
            "ART generation needs fsspec and pyarrow. Install the simulator requirements first."
        ) from exc

    files = parquet_files(args.art_dataset, args.art_split, args.art_config_name)[: args.art_max_parquet_files]
    fs = fsspec.filesystem("https")
    candidates: list[tuple[int, int, int]] = []
    metadata: dict[int, dict[str, Any]] = {}

    for file_idx, entry in enumerate(files):
        with fs.open(entry["url"], "rb", block_size=1 << 20) as fh:
            pf = pq.ParquetFile(fh)
            row_groups = [pf.metadata.row_group(rg).num_rows for rg in range(pf.metadata.num_row_groups)]
        metadata[file_idx] = {"entry": entry, "row_groups": row_groups}
        for rg, nrows in enumerate(row_groups):
            if nrows >= args.art_rows_per_window:
                candidates.append((file_idx, rg, nrows))

    if not candidates:
        raise RuntimeError(f"no ART row groups can supply {args.art_rows_per_window} rows")

    rng.shuffle(candidates)
    chosen = [rng.choice(candidates) for _ in range(args.art_windows)]
    if len(candidates) >= args.art_windows:
        chosen = candidates[: args.art_windows]

    windows = []
    for w, (file_idx, rg, nrows) in enumerate(chosen):
        entry = metadata[file_idx]["entry"]
        start = rng.randrange(0, nrows - args.art_rows_per_window + 1)
        with fs.open(entry["url"], "rb", block_size=1 << 20) as fh:
            pf = pq.ParquetFile(fh)
            available = set(pf.schema_arrow.names)
            columns = [name for name in ART_COLUMNS if name in available]
            raw_rows = pf.read_row_group(rg, columns=columns).slice(start, args.art_rows_per_window).to_pylist()

        trace_id = f"art_{w:05d}"
        rows = normalize_art_rows(raw_rows, trace_id=trace_id, source_window_index=w, key_blocks=args.key_blocks)
        ident = f"{entry['filename']}:rg{rg}:start{start}"
        windows.append(
            TraceWindow(
                trace_id=trace_id,
                source="art",
                rows=rows,
                metadata={
                    "hf_dataset": args.art_dataset,
                    "hf_split": args.art_split,
                    "hf_config": args.art_config_name or "",
                    "source_file": entry["filename"],
                    "source_ident": ident,
                    "row_group": rg,
                    "start": start,
                },
            )
        )
    return windows


def make_blocks(prefix: str, count: int) -> list[str]:
    return [f"{prefix}:b{i}" for i in range(count)]


def synthetic_request(
    *,
    trace_id: str,
    source_window_index: int,
    row_in_window: int,
    request_id: str,
    arrival_s: float,
    group_id: str,
    shared_blocks: list[str],
    suffix_name: str,
    suffix_tokens: int,
    output_tokens: int,
    block_tokens: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    suffix_blocks = [
        f"{suffix_name}:s{i}"
        for i in range(ceil_blocks(suffix_tokens, block_tokens))
    ]
    blocks = shared_blocks + suffix_blocks
    input_length = len(blocks) * block_tokens
    return {
        "dataset_version": VERSION,
        "source": "synthetic",
        "trace_id": trace_id,
        "source_window_index": source_window_index,
        "row_in_window": row_in_window,
        "request_id": request_id,
        "timestamp": arrival_s,
        "arrival_s": arrival_s,
        "input_length": input_length,
        "output_length": output_tokens,
        "hash_ids": blocks,
        "session_id": group_id,
        "group_id": group_id,
        "system_prompt_hash": group_id,
        "messages": "",
        "metadata": metadata,
    }


def build_synthetic_window(args: argparse.Namespace, window_index: int, rng: random.Random) -> TraceWindow:
    trace_id = f"synthetic_{window_index:05d}"
    prefix_blocks = ceil_blocks(args.synthetic_prefix_tokens, args.block_tokens)
    rows: list[dict[str, Any]] = []

    def add_row(row: dict[str, Any]) -> None:
        row["row_in_window"] = len(rows)
        rows.append(row)

    for job_idx in range(args.synthetic_num_bursts):
        start = (
            args.synthetic_first_burst_s
            + job_idx * args.synthetic_burst_spacing_s
            + rng.uniform(-args.synthetic_start_jitter_s, args.synthetic_start_jitter_s)
        )
        start = max(0.0, start)
        group_id = f"{trace_id}:burst:{job_idx}"
        shared = make_blocks(group_id, prefix_blocks)
        for i in range(args.synthetic_burst_size):
            if args.synthetic_burst_size == 1:
                arrival = start
            else:
                arrival = start + args.synthetic_burst_window_s * i / (args.synthetic_burst_size - 1)
            add_row(
                synthetic_request(
                    trace_id=trace_id,
                    source_window_index=window_index,
                    row_in_window=len(rows),
                    request_id=f"{group_id}:r{i}",
                    arrival_s=arrival,
                    group_id=group_id,
                    shared_blocks=shared,
                    suffix_name=f"{group_id}:r{i}",
                    suffix_tokens=args.synthetic_suffix_tokens,
                    output_tokens=args.synthetic_output_tokens,
                    block_tokens=args.block_tokens,
                    metadata={
                        "family": "synthetic_same_prefix_fanout",
                        "job_kind": "burst",
                        "job_index": job_idx,
                        "job_size": args.synthetic_burst_size,
                        "lead_s": args.synthetic_lead_s,
                    },
                )
            )

    end_s = args.synthetic_first_burst_s + max(1, args.synthetic_num_bursts) * args.synthetic_burst_spacing_s
    for job_idx in range(args.synthetic_num_decoys):
        start = rng.uniform(max(0.0, args.synthetic_first_burst_s - args.synthetic_lead_s), end_s)
        group_id = f"{trace_id}:decoy:{job_idx}"
        shared = make_blocks(group_id, prefix_blocks)
        for i in range(args.synthetic_decoy_size):
            if args.synthetic_decoy_size == 1:
                arrival = start
            else:
                arrival = start + args.synthetic_decoy_window_s * i / (args.synthetic_decoy_size - 1)
            add_row(
                synthetic_request(
                    trace_id=trace_id,
                    source_window_index=window_index,
                    row_in_window=len(rows),
                    request_id=f"{group_id}:r{i}",
                    arrival_s=arrival,
                    group_id=group_id,
                    shared_blocks=shared,
                    suffix_name=f"{group_id}:r{i}",
                    suffix_tokens=args.synthetic_suffix_tokens,
                    output_tokens=args.synthetic_output_tokens,
                    block_tokens=args.block_tokens,
                    metadata={
                        "family": "synthetic_same_prefix_fanout",
                        "job_kind": "decoy",
                        "job_index": job_idx,
                        "job_size": args.synthetic_decoy_size,
                        "lead_s": args.synthetic_lead_s,
                    },
                )
            )

    for bg_idx in range(args.synthetic_background_requests):
        group_id = f"{trace_id}:background:{bg_idx}"
        shared = make_blocks(group_id, ceil_blocks(args.synthetic_background_prefix_tokens, args.block_tokens))
        add_row(
            synthetic_request(
                trace_id=trace_id,
                source_window_index=window_index,
                row_in_window=len(rows),
                request_id=f"{group_id}:r0",
                arrival_s=rng.uniform(0.0, end_s),
                group_id=group_id,
                shared_blocks=shared,
                suffix_name=f"{group_id}:r0",
                suffix_tokens=args.synthetic_suffix_tokens,
                output_tokens=args.synthetic_background_output_tokens,
                block_tokens=args.block_tokens,
                metadata={
                    "family": "synthetic_same_prefix_fanout",
                    "job_kind": "background",
                    "job_index": bg_idx,
                    "job_size": 1,
                },
            )
        )

    rows.sort(key=lambda row: (row["arrival_s"], row["request_id"]))
    for i, row in enumerate(rows):
        row["row_in_window"] = i
    return TraceWindow(
        trace_id=trace_id,
        source="synthetic",
        rows=rows,
        metadata={
            "family": "synthetic_same_prefix_fanout",
            "num_bursts": args.synthetic_num_bursts,
            "burst_size": args.synthetic_burst_size,
            "num_decoys": args.synthetic_num_decoys,
            "background_requests": args.synthetic_background_requests,
            "prefix_tokens": args.synthetic_prefix_tokens,
            "suffix_tokens": args.synthetic_suffix_tokens,
            "output_tokens": args.synthetic_output_tokens,
            "lead_s": args.synthetic_lead_s,
        },
    )


def build_synthetic_windows(args: argparse.Namespace, rng: random.Random) -> list[TraceWindow]:
    return [build_synthetic_window(args, i, rng) for i in range(args.synthetic_windows)]


def assign_ratio_splits(windows: list[TraceWindow], train_ratio: float, rng: random.Random) -> None:
    by_source: dict[str, list[TraceWindow]] = defaultdict(list)
    for window in windows:
        by_source[window.source].append(window)

    for source_windows in by_source.values():
        rng.shuffle(source_windows)
        train_count = int(round(len(source_windows) * train_ratio))
        if len(source_windows) > 1:
            train_count = min(len(source_windows) - 1, max(1, train_count))
        for i, window in enumerate(source_windows):
            window.split = "train" if i < train_count else "test"
            for row in window.rows:
                row["split"] = window.split

    if len(windows) > 1 and not any(window.split == "test" for window in windows):
        moved = sorted(windows, key=lambda window: (window.source, window.trace_id))[-1]
        moved.split = "test"
        for row in moved.rows:
            row["split"] = moved.split
    if len(windows) > 1 and not any(window.split == "train" for window in windows):
        moved = sorted(windows, key=lambda window: (window.source, window.trace_id))[0]
        moved.split = "train"
        for row in moved.rows:
            row["split"] = moved.split


def proportional_train_counts(
    by_source: dict[str, list[TraceWindow]],
    train_windows: int,
) -> dict[str, int]:
    total = sum(len(items) for items in by_source.values())
    raw = {
        source: train_windows * len(items) / total
        for source, items in by_source.items()
    }
    counts = {
        source: min(len(by_source[source]), int(math.floor(value)))
        for source, value in raw.items()
    }
    remainder = train_windows - sum(counts.values())
    order = sorted(
        by_source,
        key=lambda source: (raw[source] - counts[source], len(by_source[source])),
        reverse=True,
    )
    while remainder > 0:
        changed = False
        for source in order:
            if counts[source] < len(by_source[source]):
                counts[source] += 1
                remainder -= 1
                changed = True
                if remainder == 0:
                    break
        if not changed:
            break
    while remainder < 0:
        changed = False
        for source in reversed(order):
            if counts[source] > 0:
                counts[source] -= 1
                remainder += 1
                changed = True
                if remainder == 0:
                    break
        if not changed:
            break
    return counts


def assign_exact_splits(
    windows: list[TraceWindow],
    train_windows: int,
    test_windows: int,
    rng: random.Random,
) -> None:
    if train_windows + test_windows != len(windows):
        raise SystemExit(
            f"exact split requires train_windows + test_windows == generated windows "
            f"({train_windows} + {test_windows} != {len(windows)})"
        )
    by_source: dict[str, list[TraceWindow]] = defaultdict(list)
    for window in windows:
        by_source[window.source].append(window)
    for source_windows in by_source.values():
        rng.shuffle(source_windows)

    train_counts = proportional_train_counts(by_source, train_windows)
    for source, source_windows in by_source.items():
        cutoff = train_counts[source]
        for i, window in enumerate(source_windows):
            window.split = "train" if i < cutoff else "test"
            for row in window.rows:
                row["split"] = window.split


def assign_splits(windows: list[TraceWindow], args: argparse.Namespace, rng: random.Random) -> None:
    if args.train_windows is not None or args.test_windows is not None:
        if args.train_windows is None or args.test_windows is None:
            raise SystemExit("--train-windows and --test-windows must be provided together")
        assign_exact_splits(windows, args.train_windows, args.test_windows, rng)
    else:
        assign_ratio_splits(windows, args.train_ratio, rng)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, separators=(",", ":")) + "\n")


def counts_for(windows: list[TraceWindow]) -> dict[str, Any]:
    by_split = Counter(window.split for window in windows)
    by_source = Counter(window.source for window in windows)
    rows_by_split = Counter()
    rows_by_source = Counter()
    for window in windows:
        rows_by_split[window.split] += len(window.rows)
        rows_by_source[window.source] += len(window.rows)
    return {
        "windows_by_split": dict(by_split),
        "windows_by_source": dict(by_source),
        "rows_by_split": dict(rows_by_split),
        "rows_by_source": dict(rows_by_source),
        "total_windows": len(windows),
        "total_rows": sum(len(window.rows) for window in windows),
    }


def jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    out = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def write_dataset_card(out_dir: Path, info: dict[str, Any]) -> None:
    counts = info["counts"]
    args = info["args"]
    card = f"""---
license: mit
task_categories:
- text-generation
pretty_name: Bursted-ART
---

# Bursted-ART

Request-level trace dataset for Bite The Bullet utility-gate experiments.

## Why This Exists

Public ART-style traces are useful for realistic arrivals, prompt lengths, and
prefix hashes, but they do not reliably contain the large synchronized
same-prefix fanout pattern that predictive KV warming is designed to exploit.
Bursted-ART adds that missing regime explicitly while keeping ordinary ART
traffic in the same corpus.

Sources:

- ART-Chat replay windows from `{args['art_dataset']}`.
- Synthetic same-prefix fanout windows generated by the local BTB benchmark.

## What Was Added

The synthetic windows model data-labeling, batch, and agent/subagent fanout
jobs:

- `{args['synthetic_num_bursts']}` burst jobs per synthetic window.
- `{args['synthetic_burst_size']}` requests per burst job.
- `{args['synthetic_prefix_tokens']}` shared prefix tokens per burst job.
- `{args['synthetic_suffix_tokens']}` unique suffix tokens per request.
- `{args['synthetic_output_tokens']}` output token(s) per request.
- `{args['synthetic_num_decoys']}` one/few-request decoy jobs per synthetic window.

Each synthetic burst has generated `hash_ids` that make the shared-prefix
structure explicit. ART rows keep their original `hash_ids`; raw ART messages
are omitted, while `message_bytes` is retained in `metadata`.

## Splitting

Splits are by complete trace window, not by individual request row, to avoid
neighboring-request leakage.

Rows:

- train: {counts['rows_by_split'].get('train', 0)}
- test: {counts['rows_by_split'].get('test', 0)}

Windows:

- train: {counts['windows_by_split'].get('train', 0)}
- test: {counts['windows_by_split'].get('test', 0)}

Each row contains `timestamp`, `input_length`, `output_length`, `hash_ids`,
`request_id`, `session_id`, `group_id`, `source`, `trace_id`, and `metadata`.
Window boundaries are recorded in `dataset_info.json`.
"""
    (out_dir / "README.md").write_text(card, encoding="utf-8")


def write_outputs(out_dir: Path, windows: list[TraceWindow], args: argparse.Namespace) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    manifest_windows = []
    line_offsets = {"train": 0, "test": 0}

    for window in sorted(windows, key=lambda w: (w.split, w.source, w.trace_id)):
        target = train_rows if window.split == "train" else test_rows
        start = line_offsets[window.split]
        target.extend(window.rows)
        line_offsets[window.split] += len(window.rows)
        manifest_windows.append(
            {
                "trace_id": window.trace_id,
                "source": window.source,
                "split": window.split,
                "rows": len(window.rows),
                "line_start": start,
                "line_end": line_offsets[window.split],
                "metadata": window.metadata,
            }
        )

    write_jsonl(out_dir / "train.jsonl", train_rows)
    write_jsonl(out_dir / "test.jsonl", test_rows)

    info = {
        "dataset_version": VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "args": jsonable_args(args),
        "counts": counts_for(windows),
        "files": {
            "train": "train.jsonl",
            "test": "test.jsonl",
        },
        "windows": manifest_windows,
        "schema": {
            "timestamp": "arrival time in seconds relative to its trace window",
            "input_length": "prompt/input tokens",
            "output_length": "generated tokens",
            "hash_ids": "ordered prefix/suffix block ids",
            "source": "art or synthetic",
            "trace_id": "window id for leakage-safe splitting",
        },
    }
    (out_dir / "dataset_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    write_dataset_card(out_dir, info)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data-generation/out/btb-art-synthetic-mixed"))
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--train-windows", type=int)
    parser.add_argument("--test-windows", type=int)
    parser.add_argument("--key-blocks", type=int, default=8)
    parser.add_argument("--block-tokens", type=int, default=256)

    parser.add_argument("--art-dataset", default="alessiotoniolo/ART-Chat-2.5M")
    parser.add_argument("--art-config-name", default="")
    parser.add_argument("--art-split", default="train")
    parser.add_argument("--art-windows", type=int, default=48)
    parser.add_argument("--art-rows-per-window", type=int, default=1000)
    parser.add_argument("--art-max-parquet-files", type=int, default=1)

    parser.add_argument("--synthetic-windows", type=int, default=48)
    parser.add_argument("--synthetic-num-bursts", type=int, default=8)
    parser.add_argument("--synthetic-burst-size", type=int, default=500)
    parser.add_argument("--synthetic-prefix-tokens", type=int, default=65536)
    parser.add_argument("--synthetic-suffix-tokens", type=int, default=256)
    parser.add_argument("--synthetic-output-tokens", type=int, default=1)
    parser.add_argument("--synthetic-first-burst-s", type=float, default=20.0)
    parser.add_argument("--synthetic-burst-spacing-s", type=float, default=40.0)
    parser.add_argument("--synthetic-burst-window-s", type=float, default=1.0)
    parser.add_argument("--synthetic-start-jitter-s", type=float, default=2.0)
    parser.add_argument("--synthetic-num-decoys", type=int, default=120)
    parser.add_argument("--synthetic-decoy-size", type=int, default=1)
    parser.add_argument("--synthetic-decoy-window-s", type=float, default=0.1)
    parser.add_argument("--synthetic-background-requests", type=int, default=0)
    parser.add_argument("--synthetic-background-prefix-tokens", type=int, default=2048)
    parser.add_argument("--synthetic-background-output-tokens", type=int, default=1)
    parser.add_argument("--synthetic-lead-s", type=float, default=6.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.train_ratio < 1.0:
        raise SystemExit("--train-ratio must be between 0 and 1")

    rng = random.Random(args.seed)
    windows = []
    windows.extend(load_art_windows(args, rng))
    windows.extend(build_synthetic_windows(args, rng))
    if not windows:
        raise SystemExit("no windows generated")

    assign_splits(windows, args, rng)
    write_outputs(args.out_dir, windows, args)
    counts = counts_for(windows)
    print(f"wrote {args.out_dir}")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
