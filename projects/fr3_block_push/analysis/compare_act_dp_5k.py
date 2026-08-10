#!/usr/bin/env python3
"""Compare ACT and Diffusion Policy evaluation logs without hard-coded metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED = ("episode", "step", "block_x", "block_y", "goal_distance")


def load_rows(directory: Path) -> tuple[list[dict[str, Any]], str]:
    csv_path = directory / "debug_steps.csv"
    jsonl_path = directory / "debug_steps.jsonl"
    if csv_path.exists():
        with csv_path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        source = str(csv_path)
    elif jsonl_path.exists():
        rows = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
        source = str(jsonl_path)
    else:
        raise FileNotFoundError(f"No debug_steps.csv or debug_steps.jsonl in {directory}")
    if not rows:
        raise ValueError(f"Evaluation log is empty: {source}")
    missing = sorted(set(REQUIRED) - set(rows[0]))
    if missing:
        raise ValueError(f"{source}: missing required columns {missing}")
    return rows, source


def number(row: dict[str, Any], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric field {key!r}: {row.get(key)!r}") from exc
    if not np.isfinite(value):
        raise ValueError(f"Non-finite numeric field {key!r}: {value}")
    return value


def episodes(rows: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["episode"]), []).append(row)
    return sorted(grouped.items(), key=lambda item: int(item[0]))


def metrics(rows: list[dict[str, Any]], success_distance: float, max_steps: int) -> list[dict[str, Any]]:
    output = []
    for episode, group in episodes(rows):
        group = sorted(group, key=lambda r: number(r, "step"))
        x = np.asarray([number(r, "block_x") for r in group])
        y = np.asarray([number(r, "block_y") for r in group])
        distance = np.asarray([number(r, "goal_distance") for r in group])
        success_indices = np.flatnonzero(distance <= success_distance)
        success_step = int(number(group[int(success_indices[0])], "step")) if len(success_indices) else None
        path_length = float(np.linalg.norm(np.diff(np.column_stack((x, y)), axis=0), axis=1).sum()) if len(group) > 1 else 0.0
        output.append({
            "episode": int(episode),
            "frames": len(group),
            "success": bool(len(success_indices)),
            "success_step": success_step,
            "success_time_s": None if success_step is None else success_step / 20.0,
            "final_block_x": float(x[-1]),
            "final_block_y": float(y[-1]),
            "final_goal_distance": float(distance[-1]),
            "min_goal_distance": float(distance.min()),
            "max_abs_lateral_deviation": float(np.max(np.abs(y - y[0]))),
            "block_path_length": path_length,
            "termination": "success" if len(success_indices) else ("timeout" if len(group) >= max_steps else "unknown"),
            "mean_action_difference": "N/A — current log does not contain 7D action columns",
            "max_action_difference": "N/A — current log does not contain 7D action columns",
        })
    return output


def find_nested(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        if key in value:
            return value[key]
        for child in value.values():
            found = find_nested(child, key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = find_nested(child, key)
            if found is not None:
                return found
    return None


def config_info(checkpoint: Path) -> dict[str, Any]:
    info: dict[str, Any] = {}
    for filename in ("config.json", "train_config.json"):
        path = checkpoint / filename
        if path.exists():
            data = json.loads(path.read_text())
            info[filename] = data
    return info


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--act-eval", type=Path, default=Path("outputs/eval/act_fr3_balanced_140"))
    parser.add_argument("--dp-eval", type=Path, default=Path("outputs/eval/dp_fr3_balanced_140"))
    parser.add_argument("--act-checkpoint", type=Path, default=Path("outputs/train/act_fr3_balanced_140/checkpoints/005000/pretrained_model"))
    parser.add_argument("--dp-checkpoint", type=Path, default=Path("outputs/train/dp_fr3_balanced_140/checkpoints/005000/pretrained_model"))
    parser.add_argument("--output-dir", type=Path, default=Path("projects/fr3_block_push/analysis/outputs"))
    parser.add_argument("--success-distance", type=float, default=0.035)
    parser.add_argument("--max-steps", type=int, default=600)
    args = parser.parse_args()

    experiments = [("ACT", args.act_eval, args.act_checkpoint), ("Diffusion Policy", args.dp_eval, args.dp_checkpoint)]
    summary: list[dict[str, Any]] = []
    all_episode_rows: list[dict[str, Any]] = []
    config_report: dict[str, Any] = {}
    for name, eval_dir, checkpoint in experiments:
        rows, log_path = load_rows(eval_dir)
        episode_rows = metrics(rows, args.success_distance, args.max_steps)
        for row in episode_rows:
            row["policy"] = name
        all_episode_rows.extend(episode_rows)
        success = [r for r in episode_rows if r["success"]]
        summary.append({
            "policy": name,
            "log_path": str(Path(log_path).resolve()),
            "checkpoint": str(checkpoint.resolve()),
            "episodes": len(episode_rows),
            "success_episodes": len(success),
            "success_rate": len(success) / len(episode_rows),
            "success_step_mean": float(np.mean([r["success_step"] for r in success])) if success else "N/A",
            "success_step_std": float(np.std([r["success_step"] for r in success])) if success else "N/A",
            "success_time_mean_s": float(np.mean([r["success_time_s"] for r in success])) if success else "N/A",
            "final_goal_distance_mean": float(np.mean([r["final_goal_distance"] for r in episode_rows])),
            "final_goal_distance_min": float(np.min([r["final_goal_distance"] for r in episode_rows])),
            "final_goal_distance_max": float(np.max([r["final_goal_distance"] for r in episode_rows])),
            "max_lateral_deviation_mean": float(np.mean([r["max_abs_lateral_deviation"] for r in episode_rows])),
            "block_path_length_mean": float(np.mean([r["block_path_length"] for r in episode_rows])),
            "action_smoothness": "N/A — current logs contain action_min/action_max only, not 7D actions",
        })
        config_report[name] = config_info(checkpoint)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing analysis outputs to {output_dir.resolve()}")
    write_csv(output_dir / "act_dp_5k_summary.csv", summary)
    write_csv(output_dir / "act_dp_5k_episode_metrics.csv", all_episode_rows)
    (output_dir / "act_dp_5k_configs.json").write_text(json.dumps(config_report, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
