#!/usr/bin/env python3
"""Interactively collect FR3 block-push demonstrations in LeRobot format."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import time


STUDY_ROOT = Path(__file__).resolve().parents[3]
if str(STUDY_ROOT) not in sys.path:
    sys.path.insert(0, str(STUDY_ROOT))
LEROBOT_SRC = STUDY_ROOT / "lerobot" / "src"
if str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))


@dataclass
class CollectionKeys:
    start: bool = False
    save: bool = False
    discard: bool = False
    reset: bool = False
    status: bool = False
    camera: bool = False
    quit: bool = False

    def callback(self, keycode: int) -> None:
        if keycode in (ord("Z"), ord("z")):
            self.start = True
        elif keycode in (ord("S"), ord("s")):
            self.save = True
        elif keycode in (ord("X"), ord("x")):
            self.discard = True
        elif keycode in (ord("R"), ord("r")):
            self.reset = True
        elif keycode == ord(" "):
            self.status = True
        elif keycode in (ord("V"), ord("v")):
            self.camera = True
        elif keycode in (ord("Q"), ord("q"), 256):
            self.quit = True


def _print_collection_status(runner, recorder, fail_recorder, phase: str) -> None:
    print(
        f"collection={phase} buffered_frames={recorder.frame_count} "
        f"saved_success={recorder.saved_episodes} "
        f"saved_fail={fail_recorder.saved_episodes} | {runner.status_line()}",
        flush=True,
    )


def collect(args: argparse.Namespace) -> None:
    from projects.fr3_block_push.control.teleop_backend import resolve_teleop_root
    from projects.fr3_block_push.control.teleop_runner import TeleopRunner
    from projects.fr3_block_push.dataset.frame_extractor import FrameExtractor
    from projects.fr3_block_push.dataset.recorder import DatasetRecorder, RecorderConfig

    teleop_root = resolve_teleop_root(args.teleop_root, study_root=STUDY_ROOT)
    config = RecorderConfig(
        root=args.dataset_root.expanduser().resolve(),
        repo_id=args.repo_id,
    )
    fail_config = RecorderConfig(
        root=args.fail_dataset_root.expanduser().resolve(),
        repo_id=args.fail_repo_id,
    )
    recorder = DatasetRecorder.create(config)
    fail_recorder = DatasetRecorder.create(fail_config)
    extractor = FrameExtractor(image_size=config.image_size)
    keys = CollectionKeys()
    phase = "idle"
    terminal_reason = None
    terminal_reported = False
    next_tick = time.perf_counter()

    try:
        with TeleopRunner(
            teleop_root,
            viewer_enabled=True,
            key_callback=keys.callback,
            image_size=config.image_size,
            input_backend=args.input_backend,
            omy_endpoint=args.omy_endpoint,
            omy_stale_timeout=args.omy_stale_timeout,
            success_margin=args.success_margin,
        ) as runner:
            print(f"Dataset root: {config.root}")
            print("Sampling: 20 Hz from MuJoCo simulation time")
            print(
                f"OMY input: {args.input_backend} "
                f"endpoint={args.omy_endpoint} timeout={args.omy_stale_timeout:.3f}s"
            )
            print(
                "Controls: Z start | S save success | X save fail | "
                "R discard+reset | V camera | SPACE status | Q/ESC quit"
            )
            print("Press Z to reset the environment and begin episode 0.")

            while runner.is_running() and not keys.quit:
                if keys.reset:
                    recorder.discard_episode()
                    fail_recorder.discard_episode()
                    runner.reset()
                    phase = "idle"
                    terminal_reason = None
                    terminal_reported = False
                    keys.reset = False
                    print("Episode discarded; environment reset to FR3 home.")

                if keys.discard:
                    keys.discard = False
                    try:
                        fail_recorder.save_episode()
                    except ValueError as error:
                        print(f"FAIL SAVE REJECTED: {error}", flush=True)
                    else:
                        print(
                            f"Saved failed episode {fail_recorder.saved_episodes - 1}",
                            flush=True,
                        )
                    recorder.discard_episode()
                    runner.reset()
                    recorder.start_episode(runner.simulation_time)
                    fail_recorder.start_episode(runner.simulation_time)
                    phase = "recording"
                    terminal_reason = None
                    terminal_reported = False
                    print("Failed episode stored; reset complete and recording restarted.")

                if keys.save:
                    keys.save = False
                    try:
                        recorder.save_episode()
                    except ValueError as error:
                        print(f"SAVE REJECTED: {error}", flush=True)
                    else:
                        print(
                            f"Saved successful episode {recorder.saved_episodes - 1} "
                            f"({terminal_reason or 'operator stop'}).",
                            flush=True,
                        )
                        if recorder.saved_episodes >= args.episodes:
                            break
                        fail_recorder.discard_episode()
                        runner.reset()
                        phase = "idle"
                        terminal_reason = None
                        terminal_reported = False
                        print("Reset complete. Press Z for the next episode.")

                if keys.start:
                    keys.start = False
                    if phase != "idle":
                        print("Z ignored: save/discard the current episode first.")
                    else:
                        runner.reset()
                        recorder.start_episode(runner.simulation_time)
                        fail_recorder.start_episode(runner.simulation_time)
                        phase = "recording"
                        terminal_reason = None
                        terminal_reported = False
                        next_tick = time.perf_counter()
                        print("Recording started after reset.")

                if keys.status:
                    _print_collection_status(runner, recorder, fail_recorder, phase)
                    keys.status = False

                if keys.camera:
                    view = runner.toggle_viewer_camera()
                    keys.camera = False
                    print(f"Viewer camera: {view}", flush=True)

                capture_sample = phase == "recording" and recorder.is_sample_due(
                    runner.simulation_time
                )
                result = runner.step(capture_sample=capture_sample)
                if result.reset_occurred:
                    recorder.discard_episode()
                    fail_recorder.discard_episode()
                    phase = "idle"
                    terminal_reason = None
                    terminal_reported = False
                    print(
                        "MuJoCo UI reset detected: buffered episode discarded; "
                        "FR3 returned home. Press Z to restart.",
                        flush=True,
                    )
                elif phase == "recording":
                    if result.sample is not None:
                        frame = extractor.extract(result.sample)
                        # LeRobot may consume/pop metadata fields such as
                        # ``task`` while adding a frame. Give each parallel
                        # recorder an independent dictionary.
                        recorder.add_frame(dict(frame))
                        fail_recorder.add_frame(dict(frame))
                    if result.task.terminated and not terminal_reported:
                        terminal_reason = result.task.reason
                        terminal_reported = True
                        if result.task.success:
                            message = "Success criterion reached"
                        else:
                            message = "Task failure condition reached"
                        print(
                            f"{message} ({terminal_reason}); recording continues. "
                            "Press S to save success or X to save fail.",
                            flush=True,
                        )

                runner.sync_viewer()
                next_tick += runner.timestep
                delay = next_tick - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_tick = time.perf_counter()
    finally:
        recorder.finalize()
        fail_recorder.finalize()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teleop-root", type=Path)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=STUDY_ROOT / "datasets" / "fr3_block_push",
    )
    parser.add_argument("--repo-id", default="local/fr3_block_push")
    parser.add_argument("--fail-repo-id", default="local/fr3_block_push_fail")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument(
        "--input-backend",
        choices=("ros", "zmq"),
        default="ros",
    )
    parser.add_argument("--fail-dataset-root", type=Path)
    parser.add_argument(
        "--omy-endpoint",
        default="tcp://127.0.0.1:5557",
    )
    parser.add_argument("--omy-stale-timeout", type=float, default=0.2)
    parser.add_argument(
        "--success-margin",
        type=float,
        default=0.025,
        help="Collection-only extra margin; 0.025 gives about +/-1.5 cm center tolerance.",
    )
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.omy_stale_timeout <= 0:
        parser.error("--omy-stale-timeout must be positive")
    if args.success_margin < 0:
        parser.error("--success-margin must be non-negative")
    if args.fail_dataset_root is None:
        args.fail_dataset_root = args.dataset_root.parent / f"{args.dataset_root.name}_fail"

    try:
        collect(args)
    except (FileNotFoundError, RuntimeError, ValueError, FileExistsError) as error:
        print(f"DATASET COLLECTION FAILED: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
