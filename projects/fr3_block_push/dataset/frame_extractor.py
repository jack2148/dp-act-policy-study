"""Extract a policy frame through the public teleoperation-runner API."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from .schema import (
    ACTION_EE_POSE_KEY,
    ACTION_KEY,
    EE_POSE_KEY,
    GOAL_DISTANCE_KEY,
    GOAL_DELTA_KEY,
    IMAGE_KEY,
    STATE_KEY,
    TASK_TEXT,
)
from .validation import validate_frame


class FrameSource(Protocol):
    image: np.ndarray
    state: np.ndarray
    action: np.ndarray
    ee_pose: np.ndarray
    goal_distance: np.ndarray
    goal_delta: np.ndarray
    action_ee_pose: np.ndarray


class FrameExtractor:
    """Read synchronized observation and action values from one runner."""

    def __init__(self, *, image_size: tuple[int, int]) -> None:
        self.image_size = image_size

    def extract(self, source: FrameSource) -> dict[str, Any]:
        frame = {
            IMAGE_KEY: np.asarray(source.image).copy(),
            STATE_KEY: np.asarray(source.state).copy(),
            ACTION_KEY: np.asarray(source.action).copy(),
            EE_POSE_KEY: np.asarray(source.ee_pose).copy(),
            GOAL_DISTANCE_KEY: np.asarray(source.goal_distance).copy(),
            GOAL_DELTA_KEY: np.asarray(source.goal_delta).copy(),
            ACTION_EE_POSE_KEY: np.asarray(source.action_ee_pose).copy(),
            "task": TASK_TEXT,
        }
        validate_frame(frame, image_size=self.image_size)
        return frame
