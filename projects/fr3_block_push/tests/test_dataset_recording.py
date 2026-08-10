"""Dataset boundary, sampling, episode lifecycle, and LeRobot round-trip tests."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from projects.fr3_block_push.dataset.frame_extractor import FrameExtractor
from projects.fr3_block_push.dataset.recorder import (
    DatasetRecorder,
    RecorderConfig,
    SimulationTimeSampler,
)
from projects.fr3_block_push.dataset.schema import (
    ACTION_EE_POSE_KEY,
    ACTION_KEY,
    DATASET_FPS,
    EE_POSE_KEY,
    GOAL_DISTANCE_KEY,
    GOAL_DELTA_KEY,
    IMAGE_KEY,
    STATE_KEY,
    TASK_TEXT,
    make_features,
)
from projects.fr3_block_push.dataset.validation import validate_frame


IMAGE_SIZE = (24, 32)


@dataclass(frozen=True)
class FakeSample:
    image: np.ndarray
    state: np.ndarray
    action: np.ndarray
    ee_pose: np.ndarray
    goal_distance: np.ndarray
    goal_delta: np.ndarray
    action_ee_pose: np.ndarray
    simulation_time: float = 0.0


def valid_frame(value: float = 0.0) -> dict:
    return {
        IMAGE_KEY: np.full((*IMAGE_SIZE, 3), 17, dtype=np.uint8),
        STATE_KEY: np.full(7, value, dtype=np.float32),
        ACTION_KEY: np.full(7, value + 1.0, dtype=np.float32),
        EE_POSE_KEY: np.full(7, value + 2.0, dtype=np.float32),
        GOAL_DISTANCE_KEY: np.asarray([0.1], dtype=np.float32),
        GOAL_DELTA_KEY: np.asarray([0.1, -0.02], dtype=np.float32),
        ACTION_EE_POSE_KEY: np.full(7, value + 3.0, dtype=np.float32),
        "task": TASK_TEXT,
    }


class FakeLeRobotDataset:
    def __init__(self) -> None:
        self.buffer = []
        self.saved = []
        self.finalized = False

    def add_frame(self, frame) -> None:
        self.buffer.append(frame)

    def has_pending_frames(self) -> bool:
        return bool(self.buffer)

    def save_episode(self) -> None:
        self.saved.append(list(self.buffer))
        self.buffer.clear()

    def clear_episode_buffer(self) -> None:
        self.buffer.clear()

    def finalize(self) -> None:
        self.finalized = True


class DatasetRecordingTest(unittest.TestCase):
    def make_recorder(self):
        dataset = FakeLeRobotDataset()
        recorder = DatasetRecorder(
            dataset,
            RecorderConfig(Path("unused"), image_size=IMAGE_SIZE),
        )
        return recorder, dataset

    def test_schema_feature_shapes(self) -> None:
        features = make_features(IMAGE_SIZE)
        self.assertEqual(
            set(features),
            {IMAGE_KEY, STATE_KEY, ACTION_KEY, EE_POSE_KEY, GOAL_DISTANCE_KEY, GOAL_DELTA_KEY, ACTION_EE_POSE_KEY},
        )
        self.assertEqual(features[IMAGE_KEY]["shape"], (*IMAGE_SIZE, 3))
        self.assertEqual(features[IMAGE_KEY]["dtype"], "video")
        self.assertEqual(features[STATE_KEY]["shape"], (7,))
        self.assertEqual(features[ACTION_KEY]["shape"], (7,))

    def test_frame_extractor_shapes_and_dtypes(self) -> None:
        sample = FakeSample(
            image=np.zeros((*IMAGE_SIZE, 3), dtype=np.uint8),
            state=np.arange(7, dtype=np.float32),
            action=np.arange(7, dtype=np.float32) + 0.5,
            ee_pose=np.arange(7, dtype=np.float32) + 1.0,
            goal_distance=np.asarray([0.1], dtype=np.float32),
            goal_delta=np.asarray([0.1, -0.02], dtype=np.float32),
            action_ee_pose=np.arange(7, dtype=np.float32) + 1.5,
        )
        frame = FrameExtractor(image_size=IMAGE_SIZE).extract(sample)
        self.assertEqual(frame[IMAGE_KEY].shape, (*IMAGE_SIZE, 3))
        self.assertEqual(frame[IMAGE_KEY].dtype, np.uint8)
        self.assertEqual(frame[STATE_KEY].shape, (7,))
        self.assertEqual(frame[STATE_KEY].dtype, np.float32)
        self.assertEqual(frame[ACTION_KEY].shape, (7,))
        self.assertEqual(frame[ACTION_KEY].dtype, np.float32)

    def test_nan_and_inf_are_rejected(self) -> None:
        for key, value in ((STATE_KEY, np.nan), (ACTION_KEY, np.inf)):
            frame = valid_frame()
            frame[key][3] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_frame(frame, image_size=IMAGE_SIZE)

    def test_sampling_is_20_hz_in_simulation_time(self) -> None:
        sampler = SimulationTimeSampler(DATASET_FPS)
        sampler.reset(1.0)
        decisions = [
            sampler.should_sample(t)
            for t in (1.0, 1.01, 1.049, 1.05, 1.099, 1.10)
        ]
        self.assertEqual(decisions, [True, False, False, True, False, True])

    def test_empty_episode_save_is_rejected(self) -> None:
        recorder, dataset = self.make_recorder()
        with self.assertRaisesRegex(ValueError, "empty episode"):
            recorder.save_episode()
        self.assertEqual(dataset.saved, [])

    def test_existing_dataset_root_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory(prefix="fr3-existing-") as directory:
            with self.assertRaises(FileExistsError):
                DatasetRecorder.create(RecorderConfig(root=Path(directory)))

    def test_discard_clears_lerobot_buffer(self) -> None:
        recorder, dataset = self.make_recorder()
        recorder.start_episode(0.0)
        recorder.maybe_add_frame(valid_frame(), simulation_time=0.0)
        self.assertTrue(dataset.has_pending_frames())
        recorder.discard_episode()
        self.assertFalse(dataset.has_pending_frames())
        self.assertEqual(recorder.frame_count, 0)

    def test_reset_boundary_does_not_mix_episode_frames(self) -> None:
        recorder, dataset = self.make_recorder()
        recorder.start_episode(0.0)
        recorder.maybe_add_frame(valid_frame(1.0), simulation_time=0.0)
        recorder.discard_episode()  # collection reset path
        recorder.start_episode(0.0)
        recorder.maybe_add_frame(valid_frame(2.0), simulation_time=0.0)
        recorder.save_episode()
        self.assertEqual(len(dataset.saved), 1)
        self.assertEqual(len(dataset.saved[0]), 1)
        np.testing.assert_array_equal(
            dataset.saved[0][0][STATE_KEY], np.full(7, 2.0, np.float32)
        )


@unittest.skipIf(sys.version_info < (3, 12), "current LeRobot requires Python >=3.12")
class LeRobotRoundTripTest(unittest.TestCase):
    def test_synthetic_one_episode_round_trip(self) -> None:
        lerobot_src = Path(__file__).resolve().parents[3] / "lerobot" / "src"
        if str(lerobot_src) not in sys.path:
            sys.path.insert(0, str(lerobot_src))
        with tempfile.TemporaryDirectory(prefix="fr3-lerobot-") as directory:
            os.environ["HF_HOME"] = str(Path(directory) / "huggingface")
            os.environ["HF_DATASETS_CACHE"] = str(
                Path(directory) / "huggingface" / "datasets"
            )
            from lerobot.datasets import LeRobotDataset

            root = Path(directory) / "dataset"
            config = RecorderConfig(
                root=root,
                repo_id="local/fr3_block_push_test",
                image_size=IMAGE_SIZE,
            )
            dataset = LeRobotDataset.create(
                repo_id=config.repo_id,
                fps=DATASET_FPS,
                features=make_features(IMAGE_SIZE),
                root=root,
                use_videos=True,
                image_writer_threads=1,
            )
            recorder = DatasetRecorder(dataset, config)
            recorder.start_episode(0.0)
            self.assertTrue(
                recorder.maybe_add_frame(valid_frame(), simulation_time=0.0)
            )
            recorder.save_episode()
            recorder.finalize()

            loaded = LeRobotDataset(repo_id=config.repo_id, root=root)
            self.assertEqual(loaded.num_episodes, 1)
            self.assertEqual(loaded.num_frames, 1)
            self.assertEqual(loaded.fps, DATASET_FPS)
            self.assertEqual(
                set(make_features(IMAGE_SIZE)),
            {IMAGE_KEY, STATE_KEY, ACTION_KEY, EE_POSE_KEY, GOAL_DISTANCE_KEY, GOAL_DELTA_KEY, ACTION_EE_POSE_KEY},
            )
            sample = loaded[0]
            self.assertEqual(tuple(sample[STATE_KEY].shape), (7,))
            self.assertEqual(tuple(sample[ACTION_KEY].shape), (7,))


if __name__ == "__main__":
    unittest.main()
