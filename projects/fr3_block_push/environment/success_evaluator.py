"""Read-only success/failure evaluation for the block-push task."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .config import BlockPushConfig


@dataclass(frozen=True)
class TaskStatus:
    success: bool
    failed: bool
    terminated: bool
    reason: str | None
    block_goal_error_xy: float
    block_speed_xy: float
    hold_time: float
    elapsed_time: float


class SuccessEvaluator:
    def __init__(
        self,
        model: mujoco.MjModel,
        config: BlockPushConfig,
    ) -> None:
        config.validate()
        self.config = config
        self.block_body_id = model.body(config.block_body).id
        self.goal_body_id = model.body(config.goal_body).id
        block_joint = model.joint(config.block_joint)
        self.block_dof_address = int(model.jnt_dofadr[block_joint.id])

        block_half = model.geom(config.block_geom).size[:2].copy()
        goal_half = model.geom(config.goal_geom).size[:2].copy()
        self.allowed_xy = goal_half - block_half - config.success_margin
        if np.any(self.allowed_xy <= 0):
            raise ValueError(
                "goal must be larger than block plus success margin"
            )
        self.hold_time = 0.0
        self.elapsed_time = 0.0
        self.terminal_reason: str | None = None

    def reset(self) -> None:
        self.hold_time = 0.0
        self.elapsed_time = 0.0
        self.terminal_reason = None

    def evaluate(self, data: mujoco.MjData, dt: float) -> TaskStatus:
        if dt < 0:
            raise ValueError("dt must be non-negative")
        self.elapsed_time += dt

        block_xy = data.xpos[self.block_body_id, :2].copy()
        goal_xy = data.xpos[self.goal_body_id, :2].copy()
        delta = block_xy - goal_xy
        error = float(np.linalg.norm(delta))
        speed = float(
            np.linalg.norm(
                data.qvel[
                    self.block_dof_address : self.block_dof_address + 2
                ]
            )
        )
        inside = bool(np.all(np.abs(delta) <= self.allowed_xy))
        stopped = speed <= self.config.success_speed_threshold
        self.hold_time = self.hold_time + dt if inside and stopped else 0.0
        success = self.hold_time + 1e-12 >= self.config.success_hold_seconds

        x_min, x_max = self.config.workspace_x_limits
        y_min, y_max = self.config.workspace_y_limits
        outside = not (
            x_min <= block_xy[0] <= x_max
            and y_min <= block_xy[1] <= y_max
        )
        timeout = self.elapsed_time >= self.config.episode_timeout_seconds

        if self.terminal_reason is None:
            if success:
                self.terminal_reason = "success"
            elif outside:
                self.terminal_reason = "outside_workspace"
            elif timeout:
                self.terminal_reason = "episode_timeout"

        reason = self.terminal_reason
        success = reason == "success"
        failed = reason in {"outside_workspace", "episode_timeout"}

        return TaskStatus(
            success=success,
            failed=failed,
            terminated=success or failed,
            reason=reason,
            block_goal_error_xy=error,
            block_speed_xy=speed,
            hold_time=self.hold_time,
            elapsed_time=self.elapsed_time,
        )
