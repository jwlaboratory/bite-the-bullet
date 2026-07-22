#!/usr/bin/env python3
"""Upload a generated BTB dataset folder to Hugging Face Datasets."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--repo-id", required=True, help="Example: username/btb-art-synthetic-mixed")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--commit-message", default="Upload BTB ART + synthetic trace mix")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dataset_dir.exists():
        raise SystemExit(f"dataset directory not found: {args.dataset_dir}")
    for required in ["train.jsonl", "test.jsonl", "dataset_info.json", "README.md"]:
        if not (args.dataset_dir / required).exists():
            raise SystemExit(f"missing {required} in {args.dataset_dir}")

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit("install huggingface_hub first: pip install huggingface_hub") from exc

    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )
    api.upload_folder(
        folder_path=str(args.dataset_dir),
        repo_id=args.repo_id,
        repo_type="dataset",
        path_in_repo=".",
        commit_message=args.commit_message,
    )
    print(f"uploaded {args.dataset_dir} to dataset repo {args.repo_id}")


if __name__ == "__main__":
    main()

