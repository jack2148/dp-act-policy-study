"""Canonical LeRobot schema for FR3 block-push demonstrations."""

from __future__ import annotations


DATASET_FPS = 20
TASK_TEXT = "Push the block into the goal region."
IMAGE_KEY = "observation.images.top"
STATE_KEY = "observation.state"
ACTION_KEY = "action"
EE_POSE_KEY = "observation.ee_pose"
GOAL_DISTANCE_KEY = "observation.goal_distance"
GOAL_DELTA_KEY = "observation.goal_delta"
ACTION_EE_POSE_KEY = "action.ee_pose"
FR3_JOINT_NAMES = tuple(f"fr3_joint{i}" for i in range(1, 8))
DEFAULT_IMAGE_SIZE = (96, 96)


def make_features(
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> dict[str, dict]:
    """Return only policy inputs and the controller action.

    ``task`` is supplied to :meth:`LeRobotDataset.add_frame`; LeRobot adds its
    own timestamp, frame index, episode index, task index, and dataset index.
    """
    height, width = image_size
    if height <= 0 or width <= 0:
        raise ValueError("image height and width must be positive")
    joint_names = list(FR3_JOINT_NAMES)
    return {
        IMAGE_KEY: {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channel"],
        },
        STATE_KEY: {
            "dtype": "float32",
            "shape": (7,),
            "names": joint_names,
        },
        ACTION_KEY: {
            "dtype": "float32",
            "shape": (7,),
            "names": joint_names,
        },
        EE_POSE_KEY: {
            "dtype": "float32",
            "shape": (7,),
            "names": ["qw", "qx", "qy", "qz", "x", "y", "z"],
        },
        GOAL_DISTANCE_KEY: {
            "dtype": "float32",
            "shape": (1,),
            "names": ["xy_distance"],
        },
        GOAL_DELTA_KEY: {
            "dtype": "float32",
            "shape": (2,),
            "names": ["x", "y"],
        },
        ACTION_EE_POSE_KEY: {
            "dtype": "float32",
            "shape": (7,),
            "names": ["qw", "qx", "qy", "qz", "x", "y", "z"],
        },
    }
