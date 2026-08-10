"""Episode-safe wrapper around the current LeRobotDataset writer API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

from .schema import DATASET_FPS, DEFAULT_IMAGE_SIZE, make_features
from .validation import validate_episode_frame_count, validate_frame


@dataclass(frozen=True)
class RecorderConfig:
    root: Path
    repo_id: str = "local/fr3_block_push"
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE


class SimulationTimeSampler:
    """Select frames at a fixed cadence using MuJoCo simulation time."""

    def __init__(self, fps: int = DATASET_FPS) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.fps = fps
        self.interval = 1.0 / fps
        self._next_sample_time: float | None = None

    def reset(self, start_time: float) -> None:
        if start_time < 0:
            raise ValueError("simulation time must be non-negative")
        self._next_sample_time = float(start_time)

    def should_sample(self, simulation_time: float) -> bool:
        if self._next_sample_time is None:
            raise RuntimeError("sampler must be reset at episode start")
        if simulation_time + 1e-9 < self._next_sample_time:
            return False
        while self._next_sample_time <= simulation_time + 1e-9:
            self._next_sample_time += self.interval
        return True


class DatasetRecorder:
    """Own LeRobot creation, frame buffering, save, discard, and finalize."""

    def __init__(self, dataset: Any, config: RecorderConfig) -> None:
        self.dataset = dataset
        self.config = config
        self.sampler = SimulationTimeSampler(DATASET_FPS)
        self.recording = False
        self.frame_count = 0
        self.saved_episodes = 0

    @classmethod
    def create(cls, config: RecorderConfig) -> "DatasetRecorder":
        if config.root.exists():
            raise FileExistsError(
                f"dataset root already exists: {config.root}. "
                "Choose a new path; existing datasets are never overwritten."
            )
        if sys.version_info < (3, 12):
            raise RuntimeError(
                "this LeRobot checkout requires Python 3.12 or newer; "
                "run collection with a Python environment that can also import ROS 2"
            )
        from lerobot.datasets import LeRobotDataset

        dataset = LeRobotDataset.create(
            repo_id=config.repo_id,
            fps=DATASET_FPS,
            features=make_features(config.image_size),
            root=config.root,
            use_videos=True,
            image_writer_threads=2,
        )
        return cls(dataset, config)

    def start_episode(self, simulation_time: float) -> None:
        if self.dataset.has_pending_frames() or self.frame_count:
            raise RuntimeError("discard or save the current episode before starting")
        self.sampler.reset(simulation_time)
        self.recording = True

    def maybe_add_frame(
        self,
        frame: dict[str, Any],
        *,
        simulation_time: float,
    ) -> bool:
        if not self.is_sample_due(simulation_time):
            return False
        self.add_frame(frame)
        return True

    def is_sample_due(self, simulation_time: float) -> bool:
        """Consume one 20 Hz sampling slot when it is due."""
        return self.recording and self.sampler.should_sample(simulation_time)

    def add_frame(self, frame: dict[str, Any]) -> None:
        """Validate and append a frame for a sampling slot already selected."""
        if not self.recording:
            raise RuntimeError("cannot add a frame while recording is paused")
        validate_frame(frame, image_size=self.config.image_size)
        self.dataset.add_frame(frame)
        self.frame_count += 1

    def pause_episode(self) -> None:
        """Stop sampling while the operator decides to save or discard."""
        self.recording = False

    def save_episode(self) -> None:
        validate_episode_frame_count(self.frame_count)
        if not self.dataset.has_pending_frames():
            raise RuntimeError("frame count and LeRobot episode buffer disagree")
        self.dataset.save_episode()
        self.saved_episodes += 1
        self.frame_count = 0
        self.recording = False

    def discard_episode(self) -> None:
        if self.dataset.has_pending_frames():
            self.dataset.clear_episode_buffer()
        self.frame_count = 0
        self.recording = False

    def finalize(self) -> None:
        if self.dataset.has_pending_frames():
            self.dataset.clear_episode_buffer()
        self.frame_count = 0
        self.recording = False
        self.dataset.finalize()
