"""OMY IPC protocol, latest-state transport, and stale-policy tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
import zmq

from projects.fr3_block_push.control.teleop_backend import (
    TeleopBackend,
    _load_controller_only_bridge,
)
from projects.fr3_block_push.control.teleop_runner import TeleopRunner
from projects.fr3_block_push.ipc.omy_state_client import OmyStateClient
from projects.fr3_block_push.ipc.protocol import (
    OMY_JOINT_NAMES,
    OMY_ROS_JOINT_NAMES,
    OMY_TRIGGER_JOINT,
    ProtocolError,
    decode_message,
    encode_message,
    extract_joint_position,
    reorder_joint_positions,
)


TELEOP_ROOT = Path("/home/chan/omy_franka_teleop")


def payload(sequence: int, position=None) -> bytes:
    return encode_message(
        sequence=sequence,
        source_timestamp_ns=100 + sequence,
        sender_monotonic_ns=200 + sequence,
        joint_names=OMY_JOINT_NAMES,
        position=position if position is not None else range(6),
        trigger_position=-0.5,
    )


class ProtocolTest(unittest.TestCase):
    def test_encode_decode_round_trip(self) -> None:
        message = decode_message(payload(7))
        self.assertEqual(message.protocol_version, 2)
        self.assertEqual(message.sequence, 7)
        self.assertEqual(message.joint_names, OMY_JOINT_NAMES)
        self.assertEqual(message.position, tuple(float(i) for i in range(6)))
        self.assertEqual(message.trigger_position, -0.5)

    def test_joint_names_are_reordered(self) -> None:
        names = ("Joint4", "Joint1", "Joint6", "Joint2", "Joint5", "Joint3")
        values = (4.0, 1.0, 6.0, 2.0, 5.0, 3.0)
        self.assertEqual(
            reorder_joint_positions(names, values),
            (1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
        )

    def test_lowercase_ros_joint_names_are_reordered_for_transport(self) -> None:
        names = ("joint4", "joint1", "joint6", "joint2", "joint5", "joint3")
        values = (4.0, 1.0, 6.0, 2.0, 5.0, 3.0)
        self.assertEqual(
            reorder_joint_positions(
                names,
                values,
                required_joint_names=OMY_ROS_JOINT_NAMES,
            ),
            (1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
        )

    def test_trigger_position_is_extracted_from_ros_joint_state(self) -> None:
        names = (*OMY_ROS_JOINT_NAMES, OMY_TRIGGER_JOINT)
        positions = (1, 2, 3, 4, 5, 6, -0.95)
        self.assertEqual(
            extract_joint_position(names, positions, OMY_TRIGGER_JOINT),
            -0.95,
        )

    def test_missing_and_duplicate_joint_are_rejected(self) -> None:
        with self.assertRaisesRegex(ProtocolError, "missing"):
            reorder_joint_positions(OMY_JOINT_NAMES[:-1], range(5))
        with self.assertRaisesRegex(ProtocolError, "duplicate"):
            reorder_joint_positions(
                ("Joint1", "Joint1", "Joint2", "Joint3", "Joint4", "Joint5"),
                range(6),
            )

    def test_position_must_have_shape_six(self) -> None:
        with self.assertRaisesRegex(ProtocolError, "lengths differ"):
            reorder_joint_positions(OMY_JOINT_NAMES, range(5))

    def test_nan_and_inf_are_rejected(self) -> None:
        for invalid in (np.nan, np.inf, -np.inf):
            positions = np.arange(6, dtype=float)
            positions[2] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(ProtocolError):
                encode_message(
                    sequence=1,
                    source_timestamp_ns=1,
                    sender_monotonic_ns=1,
                    joint_names=OMY_JOINT_NAMES,
                    position=positions,
                    trigger_position=-0.5,
                )

    def test_sequence_regression_is_rejected(self) -> None:
        with self.assertRaisesRegex(ProtocolError, "sequence must increase"):
            decode_message(payload(4), last_sequence=4)
        with self.assertRaisesRegex(ProtocolError, "sequence must increase"):
            decode_message(payload(3), last_sequence=4)


class LatestClientTest(unittest.TestCase):
    def setUp(self) -> None:
        self.context = zmq.Context()
        self.sender = self.context.socket(zmq.PAIR)
        self.receiver = self.context.socket(zmq.PAIR)
        endpoint = "inproc://omy-ipc-test"
        self.sender.bind(endpoint)
        self.receiver.connect(endpoint)
        self.clock = [1_000_000_000]
        self.client = OmyStateClient(
            socket=self.receiver,
            context=self.context,
            stale_timeout=0.2,
            monotonic_ns=lambda: self.clock[0],
        )

    def tearDown(self) -> None:
        self.client.close()
        self.sender.close(linger=0)
        self.receiver.close(linger=0)
        self.context.term()

    def test_nonblocking_receive_uses_latest_valid_state(self) -> None:
        self.sender.send(payload(0))
        self.sender.send(b"not-json")
        self.sender.send(payload(1, [10, 11, 12, 13, 14, 15]))
        state = self.client.read()
        self.assertTrue(state.fresh)
        self.assertEqual(state.message.sequence, 1)
        self.assertEqual(state.message.position[0], 10.0)
        self.assertEqual(self.client.rejected_messages, 1)
        self.assertIsNone(self.client.poll())

    def test_reverse_sequence_is_ignored(self) -> None:
        self.sender.send(payload(5))
        self.client.read()
        self.sender.send(payload(4))
        state = self.client.read()
        self.assertEqual(state.message.sequence, 5)
        self.assertEqual(self.client.rejected_messages, 1)

    def test_stale_timeout(self) -> None:
        self.sender.send(payload(0))
        self.assertTrue(self.client.read().fresh)
        self.clock[0] += 199_000_000
        self.assertTrue(self.client.read().fresh)
        self.clock[0] += 2_000_000
        self.assertFalse(self.client.read().fresh)


class FakeBridge:
    TRIGGER_ON_THRESHOLD = -0.9
    TRIGGER_OFF_THRESHOLD = -0.7
    TELEOP_MODE = "full_pose"

    @staticmethod
    def read_site_pose(data, site):
        return np.asarray(data.qpos[:3], dtype=float).copy(), np.eye(3)

    @staticmethod
    def make_desired_target(
        omy_anchor_position,
        omy_anchor_rotation,
        omy_position,
        omy_rotation,
        fr3_anchor_position,
        fr3_anchor_rotation,
    ):
        return (
            fr3_anchor_position + omy_position - omy_anchor_position,
            fr3_anchor_rotation.copy(),
        )

    @staticmethod
    def condition_target(
        old_position,
        old_rotation,
        command_position,
        command_rotation,
        linear_velocity,
        angular_velocity,
        dt,
    ):
        return (
            command_position.copy(),
            command_rotation.copy(),
            None,
            None,
            None,
            np.zeros(3),
            np.zeros(3),
        )


def make_state_machine_backend(states):
    backend = TeleopBackend.__new__(TeleopBackend)
    backend.bridge = FakeBridge()
    backend.teleop_mode = "full_pose"
    backend.omy_data = SimpleNamespace(qpos=np.zeros(6))
    backend.omy_model = object()
    backend.omy_qpos_addresses = np.arange(6)
    backend.omy_ee_site_id = 0
    backend.fr3_command_position = np.zeros(3)
    backend.fr3_command_rotation = np.eye(3)
    backend.fr3_target_position = np.zeros(3)
    backend.fr3_target_rotation = np.eye(3)
    backend.target_linear_velocity = np.zeros(3)
    backend.target_angular_velocity = np.zeros(3)
    backend.teleop_active = False
    backend.clutch_count = 0
    backend.omy_anchor_position = None
    backend.omy_anchor_rotation = None
    backend.fr3_anchor_position = np.zeros(3)
    backend.fr3_anchor_rotation = np.eye(3)
    backend.input_fresh = False
    backend.ros_fresh = False
    backend.gripper_command = 0.0
    iterator = iter(states)
    backend.update_omy_state = lambda: next(iterator)
    return backend


class StalePolicyTest(unittest.TestCase):
    def test_zmq_state_preserves_physical_trigger_position(self) -> None:
        backend = TeleopBackend.__new__(TeleopBackend)
        backend.input_backend = "zmq"
        message = SimpleNamespace(
            position=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
            trigger_position=-0.95,
        )
        backend.omy_client = SimpleNamespace(
            read=lambda: SimpleNamespace(message=message, fresh=True)
        )
        backend.joint_positions = np.zeros(6)
        backend.gripper_command = 0.0

        joints, trigger, _, fresh = backend.update_omy_state()

        np.testing.assert_array_equal(joints, np.arange(1.0, 7.0))
        self.assertEqual(trigger, -0.95)
        self.assertTrue(fresh)

    @patch("projects.fr3_block_push.control.teleop_backend.mujoco.mj_forward")
    def test_stale_holds_and_reconnect_resynchronizes_reference(self, _forward) -> None:
        q0 = np.array([1, 2, 3, 4, 5, 6], dtype=float)
        q1 = q0.copy()
        q1[0] += 0.25
        q_reconnected = q0 + 20.0
        backend = make_state_machine_backend(
            [
                (q0, -1.0, 0.0, True),
                (q1, -1.0, 0.0, True),
                (q1, -0.7, 0.0, False),
                (q_reconnected, -1.0, 0.0, True),
            ]
        )

        backend.update_command(0.001)
        np.testing.assert_allclose(backend.fr3_target_position, 0.0)
        self.assertTrue(backend.teleop_active)

        backend.update_command(0.001)
        moved_target = backend.fr3_target_position.copy()
        self.assertAlmostEqual(moved_target[0], 0.25)

        backend.update_command(0.001)
        np.testing.assert_allclose(backend.fr3_command_position, moved_target)
        self.assertFalse(backend.teleop_active)
        self.assertFalse(backend.input_fresh)

        backend.update_command(0.001)
        np.testing.assert_allclose(backend.fr3_target_position, moved_target)
        self.assertTrue(backend.teleop_active)
        self.assertTrue(backend.input_fresh)

    @patch("projects.fr3_block_push.control.teleop_runner.mujoco.mj_step")
    def test_stale_input_suppresses_recording_capture(self, mj_step) -> None:
        runner = TeleopRunner.__new__(TeleopRunner)
        runner.viewer = None
        runner.model = SimpleNamespace(opt=SimpleNamespace(timestep=0.001))
        runner.data = SimpleNamespace(time=0.0)
        runner.last_sim_time = 0.0
        runner.backend = SimpleNamespace(
            input_fresh=False,
            update_command=lambda dt: None,
            apply_controller=lambda model, data, dt: None,
        )
        status = object()
        runner.evaluator = SimpleNamespace(evaluate=lambda data, dt: status)
        mj_step.side_effect = lambda model, data: setattr(data, "time", 0.001)
        result = runner.step(capture_sample=True)
        self.assertIsNone(result.sample)
        self.assertIs(result.task, status)


class ImportBoundaryTest(unittest.TestCase):
    def test_python310_protocol_import_does_not_import_mujoco_or_lerobot(self) -> None:
        code = """
import sys
import projects.fr3_block_push.ipc.protocol
assert 'mujoco' not in sys.modules
assert 'lerobot' not in sys.modules
print('publisher protocol import is simulation-free')
"""
        completed = subprocess.run(
            ["/usr/bin/python3", "-c", code],
            cwd=Path(__file__).resolve().parents[3],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("simulation-free", completed.stdout)

    def test_controller_only_bridge_does_not_import_ros(self) -> None:
        for name in tuple(sys.modules):
            if name == "rclpy" or name.startswith(("rclpy.", "sensor_msgs", "std_msgs")):
                sys.modules.pop(name, None)
        bridge = _load_controller_only_bridge(TELEOP_ROOT)
        self.assertTrue(callable(bridge.compute_joint_target))
        self.assertNotIn("rclpy", sys.modules)
        self.assertNotIn("sensor_msgs", sys.modules)

    def test_python313_collection_imports_without_rclpy(self) -> None:
        code = """
import sys
import projects.fr3_block_push.scripts.collect_dataset
assert 'rclpy' not in sys.modules
assert 'sensor_msgs' not in sys.modules
print('collection import is ROS-free')
"""
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=Path(__file__).resolve().parents[3],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("ROS-free", completed.stdout)


if __name__ == "__main__":
    unittest.main()
