"""Deterministic task and FR3 state reset."""

from __future__ import annotations

from collections.abc import Mapping
import logging

import mujoco
import numpy as np

from .config import BlockPushConfig


LOGGER = logging.getLogger(__name__)


class ResetManager:
    def __init__(
        self,
        model: mujoco.MjModel,
        robot_home: Mapping[str, float],
        config: BlockPushConfig,
    ) -> None:
        config.validate()
        self.model = model
        self.config = config
        self.robot_home = dict(robot_home)
        self.reset_count = 0

        joint = model.joint(config.block_joint)
        if model.jnt_type[joint.id] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(f"{config.block_joint} must be a free joint")
        self.block_qpos_address = int(model.jnt_qposadr[joint.id])
        self.block_dof_address = int(model.jnt_dofadr[joint.id])
        self.block_initial_position = model.body(config.block_body).pos.copy()

        self.robot_qpos_addresses: dict[str, int] = {}
        for name in self.robot_home:
            joint_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                name,
            )
            if joint_id >= 0:
                self.robot_qpos_addresses[name] = int(
                    model.jnt_qposadr[joint_id]
                )
        for name in (f"fr3_joint{i}" for i in range(1, 8)):
            if name not in self.robot_qpos_addresses:
                raise ValueError(f"FR3 home keyframe is missing {name}")

        self.robot_actuator_ids = {
            name: model.actuator(name).id
            for name in (f"fr3_joint{i}" for i in range(1, 8))
        }
        self.gripper_actuator_id = model.actuator("fr3_gripper").id

    def reset(
        self,
        data: mujoco.MjData,
        *,
        seed: int | None = None,
    ) -> np.ndarray:
        before = data.qpos[
            self.block_qpos_address : self.block_qpos_address + 3
        ].copy()
        data.qvel[:] = 0.0
        data.act[:] = 0.0
        data.time = 0.0

        for name, address in self.robot_qpos_addresses.items():
            home = self.robot_home[name]
            data.qpos[address] = home
            if name in self.robot_actuator_ids:
                data.ctrl[self.robot_actuator_ids[name]] = home
        data.ctrl[self.gripper_actuator_id] = 0.0

        position = self.block_initial_position.copy()
        if self.config.reset_randomization_enabled:
            rng = np.random.default_rng(
                self.config.reset_seed if seed is None else seed
            )
            ranges = np.asarray(self.config.reset_xy_range)
            position[:2] += rng.uniform(-ranges, ranges)

        qpos = self.block_qpos_address
        data.qpos[qpos : qpos + 3] = position
        data.qpos[qpos + 3 : qpos + 7] = (1.0, 0.0, 0.0, 0.0)
        data.qvel[self.block_dof_address : self.block_dof_address + 6] = 0.0
        mujoco.mj_forward(self.model, data)

        self.reset_count += 1
        LOGGER.info(
            "reset %d: block %s -> %s",
            self.reset_count,
            before,
            position,
        )
        return position.copy()
