"""Central task names and evaluation/reset parameters."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BlockPushConfig:
    block_body: str = "push_block"
    block_joint: str = "push_block_joint"
    block_geom: str = "push_block_geom"
    goal_body: str = "push_goal"
    goal_geom: str = "push_goal_geom"
    table_body: str = "push_table"
    table_geom: str = "push_table_geom"
    tool_geom: str = "push_tool_geom"
    tool_tip_site: str = "push_tool_tip"
    camera: str = "policy_camera_top"

    success_margin: float = 0.005
    success_speed_threshold: float = 0.01
    success_hold_seconds: float = 0.5
    episode_timeout_seconds: float = 120.0
    workspace_x_limits: tuple[float, float] = (0.10, 1.00)
    workspace_y_limits: tuple[float, float] = (-0.40, 0.40)

    reset_randomization_enabled: bool = False
    reset_xy_range: tuple[float, float] = (0.015, 0.015)
    reset_seed: int = 0

    def validate(self) -> None:
        if self.success_margin < 0:
            raise ValueError("success_margin must be non-negative")
        if self.success_speed_threshold <= 0:
            raise ValueError("success_speed_threshold must be positive")
        if self.success_hold_seconds <= 0:
            raise ValueError("success_hold_seconds must be positive")
        if self.episode_timeout_seconds <= 0:
            raise ValueError("episode_timeout_seconds must be positive")
        if self.workspace_x_limits[0] >= self.workspace_x_limits[1]:
            raise ValueError("workspace_x_limits must be increasing")
        if self.workspace_y_limits[0] >= self.workspace_y_limits[1]:
            raise ValueError("workspace_y_limits must be increasing")
        if any(value < 0 for value in self.reset_xy_range):
            raise ValueError("reset_xy_range values must be non-negative")
