#!/usr/bin/env python3
"""Load and inspect one locally recorded FR3 LeRobot dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


STUDY_ROOT = Path(__file__).resolve().parents[3]
LEROBOT_SRC = STUDY_ROOT / "lerobot" / "src"
for path in (STUDY_ROOT, LEROBOT_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _describe(value) -> str:
    shape = tuple(value.shape) if hasattr(value, "shape") else None
    dtype = getattr(value, "dtype", type(value).__name__)
    return f"shape={shape}, dtype={dtype}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_root",
        type=Path,
        nargs="?",
        default=STUDY_ROOT / "datasets" / "fr3_block_push",
    )
    parser.add_argument("--repo-id", default="local/fr3_block_push")
    args = parser.parse_args()

    from lerobot.datasets import LeRobotDataset

    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=args.dataset_root.expanduser().resolve(),
    )
    print(f"root: {dataset.root}")
    print(f"episodes: {dataset.num_episodes}")
    print(f"frames: {dataset.num_frames}")
    print(f"fps: {dataset.fps}")
    print("features:")
    for key, feature in dataset.features.items():
        print(f"  {key}: shape={tuple(feature['shape'])}, dtype={feature['dtype']}")
    if dataset.num_frames == 0:
        print("first sample: unavailable (dataset is empty)")
        return 0
    sample = dataset[0]
    print("first sample:")
    for key in (
        "observation.images.top",
        "observation.state",
        "action",
        "timestamp",
        "episode_index",
        "frame_index",
    ):
        print(f"  {key}: {_describe(sample[key])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
