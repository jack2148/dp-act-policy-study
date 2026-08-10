"""Strict validation at the MuJoCo-to-LeRobot boundary."""

from __future__ import annotations

from typing import Any

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


def validate_frame(
    frame: dict[str, Any],
    *,
    image_size: tuple[int, int],
) -> None:
    required = {
        IMAGE_KEY,
        STATE_KEY,
        ACTION_KEY,
        EE_POSE_KEY,
        GOAL_DISTANCE_KEY,
        GOAL_DELTA_KEY,
        ACTION_EE_POSE_KEY,
        "task",
    }
    if set(frame) != required:
        raise ValueError(
            f"frame keys must be exactly {sorted(required)}, got {sorted(frame)}"
        )

    image = frame[IMAGE_KEY]
    state = frame[STATE_KEY]
    action = frame[ACTION_KEY]
    expected_image_shape = (*image_size, 3)
    if not isinstance(image, np.ndarray) or image.shape != expected_image_shape:
        raise ValueError(
            f"{IMAGE_KEY} must have shape {expected_image_shape}, "
            f"got {getattr(image, 'shape', None)}"
        )
    if image.dtype != np.uint8:
        raise TypeError(f"{IMAGE_KEY} must be uint8, got {image.dtype}")

    for key, value in ((STATE_KEY, state), (ACTION_KEY, action)):
        if not isinstance(value, np.ndarray) or value.shape != (7,):
            raise ValueError(
                f"{key} must have shape (7,), got {getattr(value, 'shape', None)}"
            )
        if value.dtype != np.float32:
            raise TypeError(f"{key} must be float32, got {value.dtype}")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{key} contains NaN or Inf")

    for key in (EE_POSE_KEY, ACTION_EE_POSE_KEY):
        value = frame[key]
        if not isinstance(value, np.ndarray) or value.shape != (7,):
            raise ValueError(f"{key} must have shape (7,)")
        if value.dtype != np.float32 or not np.all(np.isfinite(value)):
            raise ValueError(f"{key} must be finite float32")

    distance = frame[GOAL_DISTANCE_KEY]
    if (
        not isinstance(distance, np.ndarray)
        or distance.shape != (1,)
        or distance.dtype != np.float32
        or not np.all(np.isfinite(distance))
        or float(distance[0]) < 0
    ):
        raise ValueError(f"{GOAL_DISTANCE_KEY} must be a non-negative float32 scalar")

    if frame["task"] != TASK_TEXT:
        raise ValueError(f"task must be exactly {TASK_TEXT!r}")

    delta = frame[GOAL_DELTA_KEY]
    if (
        not isinstance(delta, np.ndarray)
        or delta.shape != (2,)
        or delta.dtype != np.float32
        or not np.all(np.isfinite(delta))
    ):
        raise ValueError(f"{GOAL_DELTA_KEY} must be a finite float32 vector of shape (2,)")


def validate_episode_frame_count(frame_count: int) -> None:
    if frame_count <= 0:
        raise ValueError("refusing to save an empty episode")
