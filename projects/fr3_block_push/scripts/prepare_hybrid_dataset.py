#!/usr/bin/env python3
"""Build a joint+EE-state LeRobot dataset from the enriched FR3 dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch


STUDY_ROOT = Path(__file__).resolve().parents[3]
LEROBOT_SRC = STUDY_ROOT / "lerobot" / "src"
for path in (STUDY_ROOT, LEROBOT_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lerobot.datasets import LeRobotDataset


IMAGE_KEY = "observation.images.top"
STATE_KEY = "observation.state"
ACTION_KEY = "action"
EE_POSE_KEY = "observation.ee_pose"
GOAL_DISTANCE_KEY = "observation.goal_distance"
GOAL_DELTA_KEY = "observation.goal_delta"
JOINT_NAMES = [f"fr3_joint{i}" for i in range(1, 8)]
EE_NAMES = ["qw", "qx", "qy", "qz", "x", "y", "z"]
HYBRID_NAMES = JOINT_NAMES + EE_NAMES + ["xy_distance", "dx", "dy"]


def make_features() -> dict[str, dict]:
    return {
        IMAGE_KEY: {
            "dtype": "video",
            "shape": (96, 96, 3),
            "names": ["height", "width", "channel"],
        },
        STATE_KEY: {
            "dtype": "float32",
            "shape": (17,),
            "names": HYBRID_NAMES,
        },
        ACTION_KEY: {
            "dtype": "float32",
            "shape": (7,),
            "names": JOINT_NAMES,
        },
    }


def as_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def convert(source_root: Path, target_root: Path) -> None:
    if target_root.exists():
        raise FileExistsError(f"target dataset already exists: {target_root}")
    source = LeRobotDataset(
        "local/fr3_block_push",
        root=source_root,
        download_videos=True,
    )
    target = LeRobotDataset.create(
        repo_id="local/fr3_block_push_hybrid",
        root=target_root,
        fps=source.fps,
        features=make_features(),
        use_videos=True,
        image_writer_threads=2,
    )

    previous_episode: int | None = None
    for index in range(source.num_frames):
        item = source[index]
        episode = int(as_numpy(item["episode_index"]))
        if previous_episode is not None and episode != previous_episode:
            target.save_episode()

        image = as_numpy(item[IMAGE_KEY])
        if image.shape == (3, 96, 96):
            image = np.transpose(image, (1, 2, 0))
        if image.dtype != np.uint8:
            image = np.clip(image * 255.0, 0, 255).astype(np.uint8)

        state = np.concatenate(
            [
                as_numpy(item[STATE_KEY]).reshape(7),
                as_numpy(item[EE_POSE_KEY]).reshape(7),
                np.asarray([float(as_numpy(item[GOAL_DISTANCE_KEY]))], dtype=np.float32),
                as_numpy(item[GOAL_DELTA_KEY]).reshape(2),
            ]
        ).astype(np.float32)
        action = as_numpy(item[ACTION_KEY]).reshape(7).astype(np.float32)
        target.add_frame(
            {
                IMAGE_KEY: image,
                STATE_KEY: state,
                ACTION_KEY: action,
                "task": item.get("task", "Push the block into the goal region."),
            }
        )
        previous_episode = episode

    if previous_episode is not None:
        target.save_episode()
    target.finalize()
    print(f"Created hybrid dataset: {target_root}")
    print(f"Episodes: {source.num_episodes} | Frames: {source.num_frames}")
    print(f"Hybrid state dimension: {len(HYBRID_NAMES)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=STUDY_ROOT / "datasets/data_success_50_v3",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=STUDY_ROOT / "datasets/data_success_50_v3_hybrid",
    )
    args = parser.parse_args()
    convert(args.source_root.expanduser().resolve(), args.target_root.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
