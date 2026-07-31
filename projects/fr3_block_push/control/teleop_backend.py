"""Thin adapter around the current OMY_FRANKA_TELEOP controller functions."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path
import sys
import threading
import time
from types import ModuleType

import mujoco
import numpy as np


TELEOP_ENVIRONMENT_VARIABLE = "OMY_FRANKA_TELEOP_ROOT"
BRIDGE_RELATIVE_PATH = Path("launch/FR3_omy_bridge.py")
OMY_RELATIVE_PATH = Path("robotis_mujoco_menagerie/robotis_omy/scene.xml")


def resolve_teleop_root(
    cli_path: str | Path | None = None,
    *,
    study_root: Path | None = None,
) -> Path:
    """Resolve the external dependency in CLI, env, sibling order.

    An explicit source (CLI flag or environment variable) is authoritative:
    if it does not point at a valid dependency, this raises immediately
    instead of silently falling back to a sibling checkout.
    """

    def _validated(source: str, candidate: Path) -> Path | None:
        resolved = candidate.resolve()
        bridge = resolved / BRIDGE_RELATIVE_PATH
        return resolved if bridge.is_file() else None

    if cli_path is not None:
        candidate = Path(cli_path).expanduser()
        resolved = _validated("CLI --teleop-root", candidate)
        if resolved is not None:
            return resolved
        raise FileNotFoundError(
            "CLI --teleop-root does not point at a valid OMY_FRANKA_TELEOP "
            f"checkout: {candidate.resolve()} (missing {BRIDGE_RELATIVE_PATH})"
        )

    environment_path = os.environ.get(TELEOP_ENVIRONMENT_VARIABLE)
    if environment_path:
        candidate = Path(environment_path).expanduser()
        resolved = _validated(TELEOP_ENVIRONMENT_VARIABLE, candidate)
        if resolved is not None:
            return resolved
        raise FileNotFoundError(
            f"{TELEOP_ENVIRONMENT_VARIABLE} does not point at a valid "
            f"OMY_FRANKA_TELEOP checkout: {candidate.resolve()} "
            f"(missing {BRIDGE_RELATIVE_PATH})"
        )

    if study_root is None:
        study_root = Path(__file__).resolve().parents[3]
    attempted: list[str] = []
    for candidate in (
        study_root.parent / "OMY_FRANKA_TELEOP",
        study_root.parent / "omy_franka_teleop",
    ):
        resolved = _validated("sibling", candidate)
        if resolved is not None:
            return resolved
        attempted.append(f"sibling: {candidate.resolve()}")
    details = "\n  ".join(attempted) if attempted else "(no candidates)"
    raise FileNotFoundError(
        "Could not locate OMY_FRANKA_TELEOP. Set --teleop-root or "
        f"{TELEOP_ENVIRONMENT_VARIABLE}. Tried:\n  {details}"
    )


def _load_bridge(teleop_root: Path) -> ModuleType:
    bridge_path = teleop_root / BRIDGE_RELATIVE_PATH
    module_name = "_fr3_block_push_external_bridge"
    specification = importlib.util.spec_from_file_location(module_name, bridge_path)
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load teleop bridge: {bridge_path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    try:
        specification.loader.exec_module(module)
    except ModuleNotFoundError as error:
        if error.name and (
            error.name.startswith("rclpy")
            or error.name.startswith("sensor_msgs")
            or error.name.startswith("std_msgs")
        ):
            raise RuntimeError(
                "ROS 2 Python packages required by OMY_FRANKA_TELEOP are not "
                "available. Source the matching ROS 2 environment and use its "
                "Python interpreter."
            ) from error
        raise
    return module


@dataclass(frozen=True)
class TeleopState:
    mode: str
    active: bool
    ros_fresh: bool
    clutch_count: int
    command_position: np.ndarray
    target_position: np.ndarray
    actual_position: np.ndarray


class TeleopBackend:
    """Use external mapping/conditioning/DLS while accepting final model/data."""

    def __init__(
        self,
        teleop_root: Path,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> None:
        self.teleop_root = teleop_root
        self.model = model
        self.data = data
        self.bridge = _load_bridge(teleop_root)
        self.dt = float(model.opt.timestep)

        omy_path = teleop_root / OMY_RELATIVE_PATH
        if not omy_path.is_file():
            raise FileNotFoundError(f"OMY FK model not found: {omy_path}")
        self.omy_model = mujoco.MjModel.from_xml_path(str(omy_path))
        self.omy_data = mujoco.MjData(self.omy_model)
        mujoco.mj_resetDataKeyframe(
            self.omy_model,
            self.omy_data,
            self.bridge.keyframe_id(self.omy_model, "home"),
        )
        mujoco.mj_forward(self.omy_model, self.omy_data)

        self.omy_base_body_id = self.bridge.body_id(
            self.omy_model, self.bridge.OMY_BASE_BODY_NAME
        )
        self.omy_ee_site_id = self.bridge.site_id(
            self.omy_model, self.bridge.OMY_EE_SITE_NAME
        )
        self.omy_qpos_addresses = [
            int(self.omy_model.jnt_qposadr[self.omy_model.joint(name).id])
            for name in self.bridge.OMY_MUJOCO_JOINTS
        ]

        self.fr3_ee_site_id = self.bridge.site_id(
            model, self.bridge.FR3_EE_SITE_NAME
        )
        joint_ids = np.array(
            [model.joint(f"fr3_joint{i}").id for i in range(1, 8)],
            dtype=int,
        )
        self.fr3_qpos_indices = model.jnt_qposadr[joint_ids].astype(int)
        self.fr3_dof_indices = model.jnt_dofadr[joint_ids].astype(int)
        self.fr3_joint_lower = model.jnt_range[joint_ids, 0].copy()
        self.fr3_joint_upper = model.jnt_range[joint_ids, 1].copy()
        self.fr3_actuator_indices = np.array(
            [model.actuator(f"fr3_joint{i}").id for i in range(1, 8)],
            dtype=int,
        )
        self.fr3_gripper_actuator_id = model.actuator("fr3_gripper").id
        self.jacp = np.zeros((3, model.nv))
        self.jacr = np.zeros((3, model.nv))

        self.state_lock = threading.Lock()
        self.joint_positions = self.omy_data.qpos[
            self.omy_qpos_addresses
        ].copy()
        self.ros_node = None
        self.ros_thread: threading.Thread | None = None
        self.gripper_command = 0.0
        self.reset_controller_state()

    def initialize(self) -> None:
        rclpy = self.bridge.rclpy
        if not rclpy.ok():
            rclpy.init()
        self.ros_node = self.bridge.OmyPose(
            self.joint_positions,
            self.state_lock,
        )
        self.ros_thread = threading.Thread(
            target=rclpy.spin,
            args=(self.ros_node,),
            daemon=True,
        )
        self.ros_thread.start()

    def reset_controller_state(self) -> None:
        position, rotation = self.bridge.read_site_pose(
            self.data, self.fr3_ee_site_id
        )
        self.fr3_command_position = position.copy()
        self.fr3_command_rotation = rotation.copy()
        self.fr3_target_position = position.copy()
        self.fr3_target_rotation = rotation.copy()
        self.hold_q_target = self.data.qpos[self.fr3_qpos_indices].copy()
        self.nullspace_posture_reference = self.hold_q_target.copy()
        self.target_linear_velocity = np.zeros(3)
        self.target_angular_velocity = np.zeros(3)
        self.teleop_active = False
        self.clutch_count = 0
        self.omy_anchor_position = None
        self.omy_anchor_rotation = None
        self.fr3_anchor_position = position.copy()
        self.fr3_anchor_rotation = rotation.copy()
        self.ros_fresh = False

    def update_omy_state(self) -> tuple[np.ndarray, float, float, bool]:
        if self.ros_node is None:
            raise RuntimeError("initialize() must be called before teleoperation")
        with self.state_lock:
            target = self.joint_positions.copy()
            trigger = float(self.ros_node.trigger_position)
            gripper = float(self.ros_node.gripper_command)
            last_message = float(self.ros_node.last_message_time)
            has_state = bool(self.ros_node.has_joint_state)
        fresh = (
            has_state
            and time.perf_counter() - last_message <= self.bridge.ROS_TIMEOUT_S
        )
        return target, trigger, gripper, fresh

    def update_command(self, dt: float) -> None:
        omy_target, trigger, self.gripper_command, fresh = self.update_omy_state()
        self.ros_fresh = fresh
        if not fresh:
            self.teleop_active = False
            self.target_linear_velocity.fill(0.0)
            self.target_angular_velocity.fill(0.0)
            self.fr3_command_position = self.fr3_target_position.copy()
            self.fr3_command_rotation = self.fr3_target_rotation.copy()
            return

        for address, position in zip(self.omy_qpos_addresses, omy_target):
            self.omy_data.qpos[address] = position
        mujoco.mj_forward(self.omy_model, self.omy_data)
        omy_position, omy_rotation = self.bridge.read_site_pose(
            self.omy_data, self.omy_ee_site_id
        )

        if (
            not self.teleop_active
            and trigger <= self.bridge.TRIGGER_ON_THRESHOLD
        ):
            self.clutch_count += 1
            self.omy_anchor_position = omy_position.copy()
            self.omy_anchor_rotation = omy_rotation.copy()
            self.fr3_anchor_position = self.fr3_command_position.copy()
            self.fr3_anchor_rotation = self.fr3_command_rotation.copy()
            self.teleop_active = True
            self.target_linear_velocity.fill(0.0)
            self.target_angular_velocity.fill(0.0)
        elif (
            self.teleop_active
            and trigger >= self.bridge.TRIGGER_OFF_THRESHOLD
        ):
            self.teleop_active = False

        if self.teleop_active:
            desired_position, desired_rotation = self.bridge.make_desired_target(
                self.omy_anchor_position,
                self.omy_anchor_rotation,
                omy_position,
                omy_rotation,
                self.fr3_anchor_position,
                self.fr3_anchor_rotation,
            )
            mode = self.bridge.TELEOP_MODE
            if mode == "position_only":
                self.fr3_command_position = desired_position
                self.fr3_command_rotation = self.fr3_anchor_rotation.copy()
            elif mode == "orientation_only":
                self.fr3_command_position = self.fr3_anchor_position.copy()
                self.fr3_command_rotation = desired_rotation
            else:
                self.fr3_command_position = desired_position
                self.fr3_command_rotation = desired_rotation

        conditioned = self.bridge.condition_target(
            self.fr3_target_position,
            self.fr3_target_rotation,
            self.fr3_command_position,
            self.fr3_command_rotation,
            self.target_linear_velocity,
            self.target_angular_velocity,
            dt,
        )
        self.fr3_target_position = conditioned[0]
        self.fr3_target_rotation = conditioned[1]
        self.target_linear_velocity = conditioned[5]
        self.target_angular_velocity = conditioned[6]

    def apply_controller(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        dt: float,
    ) -> None:
        position_gap = np.linalg.norm(
            self.fr3_command_position - self.fr3_target_position
        )
        rotation_gap = np.linalg.norm(
            self.bridge.matrix_to_rotvec(
                self.fr3_command_rotation @ self.fr3_target_rotation.T
            )
        )
        target_follow_active = (
            self.teleop_active
            or position_gap >= 1e-6
            or rotation_gap >= 1e-4
        )
        if target_follow_active and self.ros_fresh:
            target, _ = self.bridge.compute_joint_target(
                model,
                data,
                self.fr3_ee_site_id,
                self.fr3_dof_indices,
                self.fr3_qpos_indices,
                self.fr3_joint_lower,
                self.fr3_joint_upper,
                self.fr3_target_position,
                self.fr3_target_rotation,
                self.hold_q_target,
                self.jacp,
                self.jacr,
                dt,
                teleop_mode=self.bridge.TELEOP_MODE,
                enable_nullspace_posture=self.bridge.ENABLE_NULLSPACE_POSTURE,
                q_posture_reference=self.nullspace_posture_reference,
            )
            self.hold_q_target = target.copy()
        data.ctrl[self.fr3_actuator_indices] = self.hold_q_target
        data.ctrl[self.fr3_gripper_actuator_id] = self.gripper_command

    def get_state(self) -> TeleopState:
        actual, _ = self.bridge.read_site_pose(self.data, self.fr3_ee_site_id)
        return TeleopState(
            mode=self.bridge.TELEOP_MODE,
            active=self.teleop_active,
            ros_fresh=self.ros_fresh,
            clutch_count=self.clutch_count,
            command_position=self.fr3_command_position.copy(),
            target_position=self.fr3_target_position.copy(),
            actual_position=actual,
        )

    def shutdown(self) -> None:
        if self.ros_node is not None:
            self.ros_node.destroy_node()
            self.ros_node = None
        if self.bridge.rclpy.ok():
            self.bridge.rclpy.shutdown()
