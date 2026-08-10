#!/usr/bin/env python3
"""Run a trained Diffusion Policy in the FR3 MuJoCo block-push task."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
import torch

STUDY_ROOT = Path(__file__).resolve().parents[3]
for path in (STUDY_ROOT, STUDY_ROOT / "lerobot" / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lerobot.datasets import LeRobotDatasetMetadata
from lerobot.policies import make_pre_post_processors
from lerobot.policies.diffusion import DiffusionPolicy
from lerobot.policies.utils import build_inference_frame
from projects.fr3_block_push.control.teleop_runner import TeleopRunner


def evaluate(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    checkpoint = args.checkpoint.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    print(f"Loading Diffusion Policy checkpoint: {checkpoint}", flush=True)
    model = DiffusionPolicy.from_pretrained(str(checkpoint))
    model.to(device)
    if args.n_action_steps is not None:
        if not 1 <= args.n_action_steps <= model.config.horizon:
            raise ValueError(f"n_action_steps must be in [1, {model.config.horizon}]")
        model.config.n_action_steps = args.n_action_steps
        print(f"Override DP n_action_steps={args.n_action_steps}", flush=True)
    model.eval()

    metadata = LeRobotDatasetMetadata(args.repo_id, root=dataset_root)
    preprocess, postprocess = make_pre_post_processors(
        model.config, dataset_stats=metadata.stats
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int | str]] = []
    successes = 0

    with TeleopRunner(
        args.teleop_root,
        viewer_enabled=not args.headless,
        image_size=(args.image_size, args.image_size),
        input_backend="zmq",
    ) as runner:
        sim_steps = max(1, round(1.0 / (20.0 * runner.timestep)))
        print(f"Policy rate: 20 Hz ({sim_steps} MuJoCo steps/action)", flush=True)
        for episode in range(args.episodes):
            model.reset()
            runner.reset()
            print(f"Episode {episode} started", flush=True)
            for step in range(args.max_steps):
                joints = runner.get_fr3_joint_positions()
                pose = runner.get_ee_pose()
                delta = runner.get_goal_delta_xy()
                observation = {
                    "top": runner.render_policy_camera(),
                    **{f"fr3_joint{i}": float(v) for i, v in enumerate(joints, 1)},
                    **{n: float(v) for n, v in zip(("qw", "qx", "qy", "qz", "x", "y", "z"), pose)},
                    "xy_distance": float(runner.get_goal_distance_xy()[0]),
                    "dx": float(delta[0]),
                    "dy": float(delta[1]),
                }
                frame = build_inference_frame(
                    observation=observation,
                    ds_features=metadata.features,
                    device=device,
                )
                with torch.inference_mode():
                    action = postprocess(model.select_action(preprocess(frame)))
                action_np = action.squeeze(0).detach().cpu().numpy()
                runner.apply_policy_action(action_np)
                runner.step_policy(sim_steps=sim_steps)
                runner.sync_viewer()
                block_xy, goal_xy = runner.get_block_goal_xy()
                dist = float(np.linalg.norm(block_xy - goal_xy))
                rows.append({
                    "episode": episode, "step": step,
                    "block_x": float(block_xy[0]), "block_y": float(block_xy[1]),
                    "goal_x": float(goal_xy[0]), "goal_y": float(goal_xy[1]),
                    "goal_distance": dist,
                    "action_min": float(action_np.min()),
                    "action_max": float(action_np.max()),
                })
                if step % args.debug_every == 0:
                    print(
                        f"[debug] ep={episode} step={step} "
                        f"block=({block_xy[0]:.3f},{block_xy[1]:.3f}) "
                        f"goal_dist={dist:.3f} action_range="
                        f"[{action_np.min():.3f},{action_np.max():.3f}]",
                        flush=True,
                    )
                if dist <= args.success_distance:
                    successes += 1
                    print(f"Episode {episode}: success at step {step}", flush=True)
                    break
            else:
                print(f"Episode {episode}: max_steps timeout", flush=True)

    csv_path = output_dir / "debug_steps.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["episode"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Debug CSV: {csv_path}")
    print(f"Success rate: {successes}/{args.episodes} ({100*successes/max(1,args.episodes):.1f}%)")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, default=STUDY_ROOT / "datasets/data_success_50_v3")
    p.add_argument("--repo-id", default="local/fr3_block_push_dp")
    p.add_argument("--teleop-root", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--image-size", type=int, default=96)
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-action-steps", type=int, default=None)
    p.add_argument("--success-distance", type=float, default=0.035)
    p.add_argument("--debug-every", type=int, default=20)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--output-dir", type=Path, default=STUDY_ROOT / "outputs/eval/dp_fr3_test_50")
    args = p.parse_args()
    evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
