#!/usr/bin/env python3
"""Run OMY-controlled FR3 block pushing in one MuJoCo simulation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


STUDY_ROOT = Path(__file__).resolve().parents[3]
if str(STUDY_ROOT) not in sys.path:
    sys.path.insert(0, str(STUDY_ROOT))

from projects.fr3_block_push.control.teleop_backend import resolve_teleop_root
from projects.fr3_block_push.control.teleop_runner import run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teleop-root",
        type=Path,
        help="path to the OMY_FRANKA_TELEOP repository",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="run without the passive MuJoCo viewer",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        help="optional bounded runtime for smoke checks",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        teleop_root = resolve_teleop_root(
            args.teleop_root,
            study_root=STUDY_ROOT,
        )
        run(
            teleop_root,
            viewer_enabled=not args.no_viewer,
            max_seconds=args.max_seconds,
        )
    except (FileNotFoundError, RuntimeError) as error:
        print(f"TELEOP STARTUP FAILED: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
