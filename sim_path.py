"""Locate the sibling inference-sim checkout used by the experiments."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def add_inference_sim_to_path(repo_root: Path) -> Path:
    candidates: list[Path] = []
    env_root = os.environ.get("INFERENCE_SIM_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())
    candidates.extend(
        [
            repo_root.parent / "inference-sim",
            repo_root / "inference-sim",
            repo_root,
        ]
    )

    for candidate in candidates:
        candidate = candidate.resolve()
        if (candidate / "config.py").exists() and (candidate / "gpu.py").exists():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return candidate

    searched = "\n".join(f"- {path}" for path in candidates)
    raise RuntimeError(
        "Could not find inference-sim. Set INFERENCE_SIM_ROOT or clone "
        f"inference-sim next to this repo.\nSearched:\n{searched}"
    )
