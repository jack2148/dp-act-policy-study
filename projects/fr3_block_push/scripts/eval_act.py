#!/usr/bin/env python3
"""Run a trained LeRobot ACT policy in the FR3 MuJoCo block-push task."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch


STUDY_ROOT = Path(__file__).resolve().parents[3]
LEROBOT_SRC = STUDY_ROOT / "lerobot" / "src"
for path in (STUDY_ROOT, LEROBOT_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lerobot.datasets import LeRobotDatasetMetadata
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act import ACTPolicy
from lerobot.policies.utils import build_inference_frame

from projects.fr3_block_push.control.teleop_runner import TeleopRunner


def evaluate(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    checkpoint = args.checkpoint.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()

    print(f"Loading ACT checkpoint: {checkpoint}")
    model = ACTPolicy.from_pretrained(str(checkpoint))
    model.to(device)
    if args.n_action_steps is not None:
        if args.n_action_steps <= 0 or args.n_action_steps > model.config.chunk_size:
            raise ValueError(
                "n_action_steps must be between 1 and the checkpoint chunk_size "
                f"({model.config.chunk_size})"
            )
        model.config.n_action_steps = args.n_action_steps
        print(f"Override ACT n_action_steps={args.n_action_steps}", flush=True)
    model.eval()

    metadata = LeRobotDatasetMetadata(
        args.repo_id,
        root=dataset_root,
    )
    preprocess, postprocess = make_pre_post_processors(
        model.config,
        dataset_stats=metadata.stats,
    )

    successes = 0
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_path = output_dir / "debug_steps.jsonl"
    trajectory_rows: list[dict[str, float | int | str]] = []
    trajectories: list[tuple[int, list[tuple[float, float]]]] = []
    with TeleopRunner(
        args.teleop_root,
        viewer_enabled=not args.headless,
        image_size=(args.image_size, args.image_size),
        # The policy drives MuJoCo directly; no OMY/ROS input is consumed.
        # ZMQ keeps TeleopBackend in its ROS-free controller-only path.
        input_backend="zmq",
    ) as runner:
        # Dataset actions were sampled at 20 Hz. Keep each predicted action
        # target for the equivalent number of MuJoCo integration steps.
        sim_steps_per_action = max(1, round(1.0 / (20.0 * runner.timestep)))
        print(
            f"Policy rate: 20 Hz ({sim_steps_per_action} MuJoCo steps/action)",
            flush=True,
        )
        for episode in range(args.episodes):
            model.reset()
            runner.reset()
            print(f"Episode {episode} started")
            episode_path: list[tuple[float, float]] = []

            for step in range(args.max_steps):
                observation = {
                    # build_inference_frame expects raw camera names and
                    # named state values, not the dataset-prefixed keys.
                    "top": runner.render_policy_camera(),
                    **{
                        f"fr3_joint{index}": float(value)
                        for index, value in enumerate(
                            runner.get_fr3_joint_positions(), start=1
                        )
                    },
                    **{
                        name: float(value)
                        for name, value in zip(
                            ("qw", "qx", "qy", "qz", "x", "y", "z"),
                            runner.get_ee_pose(),
                        )
                    },
                    "xy_distance": float(runner.get_goal_distance_xy()[0]),
                    "dx": float(runner.get_goal_delta_xy()[0]),
                    "dy": float(runner.get_goal_delta_xy()[1]),
                }
                frame = build_inference_frame(
                    observation=observation,
                    ds_features=metadata.features,
                    device=device,
                )
                action = postprocess(model.select_action(preprocess(frame)))
                action_np = action.squeeze(0).detach().cpu().numpy()
                runner.apply_policy_action(action_np)
                result = runner.step_policy(sim_steps=sim_steps_per_action)
                runner.sync_viewer()

                block_xy, goal_xy = runner.get_block_goal_xy()
                distance = float(np.linalg.norm(block_xy - goal_xy))
                episode_path.append((float(block_xy[0]), float(block_xy[1])))
                debug_row = {
                    "episode": episode,
                    "step": step,
                    "sim_time": float(runner.simulation_time),
                    "block_x": float(block_xy[0]),
                    "block_y": float(block_xy[1]),
                    "goal_x": float(goal_xy[0]),
                    "goal_y": float(goal_xy[1]),
                    "goal_distance": distance,
                    "action_min": float(action_np.min()),
                    "action_max": float(action_np.max()),
                    "action_mean": float(action_np.mean()),
                    "joint1_action": float(action_np[0]),
                    "joint7_action": float(action_np[6]),
                    "reason": result.task.reason or "running",
                }
                trajectory_rows.append(debug_row)
                if step % args.debug_every == 0:
                    print(
                        f"[debug] ep={episode} step={step} "
                        f"block=({block_xy[0]:.3f},{block_xy[1]:.3f}) "
                        f"goal_dist={distance:.3f} "
                        f"action=[{action_np.min():.3f},{action_np.max():.3f}] "
                        f"qtarget={np.array2string(action_np, precision=3, separator=',')} "
                        f"task={result.task.reason or 'running'}",
                        flush=True,
                    )

                if result.task.terminated:
                    if result.task.success:
                        successes += 1
                    print(
                        f"Episode {episode}: {result.task.reason} "
                        f"at step {step + 1}",
                        flush=True,
                    )
                    break
            else:
                print(f"Episode {episode}: max_steps timeout", flush=True)
            trajectories.append((episode, episode_path))

    with debug_path.open("w", encoding="utf-8") as stream:
        for row in trajectory_rows:
            stream.write(json.dumps(row) + "\n")
    csv_path = output_dir / "debug_steps.csv"
    if trajectory_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(trajectory_rows[0]))
            writer.writeheader()
            writer.writerows(trajectory_rows)

    plot_path = output_dir / "trajectories_xy.png"
    try:
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(7, 6))
        for episode, path in trajectories:
            if not path:
                continue
            points = np.asarray(path)
            axis.plot(points[:, 0], points[:, 1], label=f"ep {episode}")
            axis.scatter(*points[0], marker="o", s=25)
            axis.scatter(*points[-1], marker="x", s=35)
        if trajectory_rows:
            axis.scatter(
                trajectory_rows[0]["goal_x"],
                trajectory_rows[0]["goal_y"],
                marker="*",
                s=140,
                label="goal",
            )
        axis.set_title("FR3 ACT block trajectory")
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
        axis.axis("equal")
        axis.grid(True, alpha=0.3)
        if len(trajectories) <= 10:
            axis.legend()
        figure.tight_layout()
        figure.savefig(plot_path, dpi=160)
        plt.close(figure)
    except Exception as error:
        print(f"Warning: could not create trajectory plot: {error}", flush=True)

    print(f"Debug JSONL: {debug_path}")
    print(f"Debug CSV: {csv_path}")
    print(f"Trajectory plot: {plot_path}")
    print(f"Success rate: {successes}/{args.episodes} ({successes / args.episodes:.1%})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=STUDY_ROOT / "datasets/fr3_block_push_20260804_140806",
    )
    parser.add_argument("--repo-id", default="local/fr3_block_push")
    parser.add_argument("--teleop-root", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=2400)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=None,
        help="Override the checkpoint action queue length for evaluation.",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=STUDY_ROOT / "outputs/eval/act_fr3_test_50",
    )
    parser.add_argument("--debug-every", type=int, default=20)
    args = parser.parse_args()
    if args.episodes <= 0 or args.max_steps <= 0 or args.debug_every <= 0:
        parser.error("episodes, max-steps, and debug-every must be positive")
    evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
