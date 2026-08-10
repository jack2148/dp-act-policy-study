#!/usr/bin/env python3
"""Merge FR3 demonstrations and attach approach-direction episode labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
for p in (ROOT, ROOT / "lerobot" / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from lerobot.datasets import LeRobotDataset

IMAGE = "observation.images.top"
STATE = "observation.state"
ACTION = "action"
JOINTS = [f"fr3_joint{i}" for i in range(1, 8)]
TASK = "Push the block into the goal region."


def features() -> dict[str, dict]:
    return {
        IMAGE: {"dtype": "video", "shape": (96, 96, 3), "names": ["height", "width", "channel"]},
        STATE: {"dtype": "float32", "shape": (7,), "names": JOINTS},
        ACTION: {"dtype": "float32", "shape": (7,), "names": JOINTS},
    }


def npv(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def direction(root: Path) -> str:
    name = root.name.lower()
    for key in ("right", "center", "left"):
        if key in name:
            return key
    return "base"


def convert(sources: list[Path], target_root: Path) -> None:
    if target_root.exists():
        raise FileExistsError(f"target dataset already exists: {target_root}")
    target = LeRobotDataset.create(
        repo_id="local/fr3_block_push_act_balanced",
        root=target_root,
        fps=20,
        features=features(),
        use_videos=True,
        image_writer_threads=2,
    )
    labels: list[dict[str, object]] = []
    out_episode = 0
    total_frames = 0
    for source_root in sources:
        source = LeRobotDataset("local/source", root=source_root, download_videos=True)
        current_ep = None
        frame_count = 0
        label = direction(source_root)
        for index in range(source.num_frames):
            item = source[index]
            ep = int(npv(item["episode_index"]))
            if current_ep is not None and ep != current_ep:
                target.save_episode()
                labels.append({"episode_index": out_episode, "source": str(source_root), "source_episode": current_ep, "approach_direction": label, "frames": frame_count})
                out_episode += 1
                total_frames += frame_count
                frame_count = 0
            image = npv(item[IMAGE])
            if image.shape == (3, 96, 96):
                image = np.transpose(image, (1, 2, 0))
            if image.dtype != np.uint8:
                image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
            target.add_frame({
                IMAGE: image,
                STATE: npv(item[STATE]).reshape(7).astype(np.float32),
                ACTION: npv(item[ACTION]).reshape(7).astype(np.float32),
                "task": TASK,
            })
            current_ep = ep
            frame_count += 1
        if current_ep is not None:
            target.save_episode()
            labels.append({"episode_index": out_episode, "source": str(source_root), "source_episode": current_ep, "approach_direction": label, "frames": frame_count})
            out_episode += 1
            total_frames += frame_count
    target.finalize()
    label_path = target_root / "meta" / "approach_labels.jsonl"
    with label_path.open("w") as f:
        for label in labels:
            f.write(json.dumps(label) + "\n")
    counts = {k: sum(x["approach_direction"] == k for x in labels) for k in ("base", "right", "center", "left")}
    print(f"Created ACT dataset: {target_root}")
    print(f"Episodes: {len(labels)} | Frames: {total_frames} | Labels: {counts}")
    print(f"Approach labels: {label_path}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-root", type=Path, default=ROOT / "datasets/data_act_balanced_140")
    p.add_argument("--source-root", type=Path, action="append", dest="sources")
    args = p.parse_args()
    sources = args.sources or [
        str(ROOT / "datasets/data_success_50_v3"),
        str(ROOT / "datasets/data_right_30_v2"),
        str(ROOT / "datasets/data_center_30_v2"),
        str(ROOT / "datasets/data_left_30_v2"),
    ]
    convert([Path(x).expanduser().resolve() for x in sources], args.target_root.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
