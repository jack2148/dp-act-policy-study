#!/usr/bin/env python3
"""Print the schema and first sample of the local LeRobot Push-T dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import typing


STUDY_ROOT = Path(__file__).resolve().parent
LEROBOT_SRC = STUDY_ROOT / "lerobot" / "src"
DEFAULT_DATASET_ROOT = Path.home() / ".cache/huggingface/lerobot/lerobot/pusht"

if sys.version_info < (3, 12):
    raise SystemExit(
        "This LeRobot checkout requires Python 3.12 or newer; "
        f"found Python {sys.version_info.major}.{sys.version_info.minor}."
    )

if str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

# This checkout uses ``typing.Self`` while some existing LeRobot environments
# still run Python 3.10, where Self is provided by typing_extensions.
if not hasattr(typing, "Self"):
    from typing_extensions import Self

    typing.Self = Self  # type: ignore[attr-defined]

from lerobot.datasets import LeRobotDataset  # noqa: E402


def _shape(value) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        return ()
    return tuple(int(size) for size in shape)


def _dtype(value) -> str:
    dtype = getattr(value, "dtype", None)
    return str(dtype) if dtype is not None else type(value).__name__


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Local LeRobot Push-T dataset directory.",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_root}")

    dataset = LeRobotDataset(
        repo_id="lerobot/pusht",
        root=dataset_root,
    )
    sample = dataset[0]
    image_key = next(
        key for key in dataset.features if key.startswith("observation.image")
    )

    print("Dataset features:")
    for key, feature in dataset.features.items():
        print(
            f"  {key}: shape={tuple(feature['shape'])}, "
            f"dtype={feature['dtype']}"
        )

    print(f"First sample keys: {list(sample.keys())}")
    print(
        f"observation image: key={image_key}, "
        f"shape={_shape(sample[image_key])}, dtype={_dtype(sample[image_key])}"
    )

    for key in ("observation.state", "action", "timestamp", "episode_index", "frame_index"):
        value = sample[key]
        print(f"{key}: shape={_shape(value)}, dtype={_dtype(value)}")


if __name__ == "__main__":
    main()
