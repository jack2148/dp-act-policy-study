from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import unittest

import mujoco
import numpy as np

from projects.fr3_block_push.assets.scene_builder import build_scene
from projects.fr3_block_push.control.teleop_backend import resolve_teleop_root
from projects.fr3_block_push.environment import (
    BlockPushConfig,
    ResetManager,
    SuccessEvaluator,
)


STUDY_ROOT = Path(__file__).resolve().parents[3]


class ResolveTeleopRootTests(unittest.TestCase):
    def test_invalid_cli_path_raises_instead_of_falling_back(self) -> None:
        with self.assertRaises(FileNotFoundError):
            resolve_teleop_root("/does/not/exist", study_root=STUDY_ROOT)

    def test_invalid_env_path_raises_instead_of_falling_back(self) -> None:
        original = os.environ.get("OMY_FRANKA_TELEOP_ROOT")
        os.environ["OMY_FRANKA_TELEOP_ROOT"] = "/does/not/exist"
        try:
            with self.assertRaises(FileNotFoundError):
                resolve_teleop_root(study_root=STUDY_ROOT)
        finally:
            if original is None:
                del os.environ["OMY_FRANKA_TELEOP_ROOT"]
            else:
                os.environ["OMY_FRANKA_TELEOP_ROOT"] = original


class ConfigTests(unittest.TestCase):
    def test_default_config_is_valid(self) -> None:
        BlockPushConfig().validate()

    def test_invalid_workspace_is_rejected(self) -> None:
        config = replace(BlockPushConfig(), workspace_x_limits=(1.0, 0.0))
        with self.assertRaises(ValueError):
            config.validate()

    def test_success_thresholds_are_positive(self) -> None:
        config = replace(BlockPushConfig(), success_hold_seconds=0.0)
        with self.assertRaises(ValueError):
            config.validate()


class ModelFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = BlockPushConfig()
        teleop_root = resolve_teleop_root(study_root=STUDY_ROOT)
        cls.scene = build_scene(teleop_root)
        cls.model = cls.scene.model

    def make_data(self) -> tuple[mujoco.MjData, ResetManager]:
        data = mujoco.MjData(self.model)
        manager = ResetManager(self.model, self.scene.robot_home, self.config)
        manager.reset(data)
        return data, manager


class ModelTests(ModelFixture):
    def test_required_model_elements(self) -> None:
        model = self.model
        for name in (
            "push_table_geom",
            "push_block_geom",
            "push_goal_geom",
            "push_tool_geom",
        ):
            self.assertGreaterEqual(model.geom(name).id, 0)
        self.assertGreaterEqual(model.camera("policy_camera_top").id, 0)

    def test_block_joint_is_free(self) -> None:
        joint = self.model.joint(self.config.block_joint)
        self.assertEqual(
            self.model.jnt_type[joint.id],
            mujoco.mjtJoint.mjJNT_FREE,
        )

    def test_block_starts_on_table_and_goal_is_visual_only(self) -> None:
        table = self.model.body(self.config.table_body)
        table_geom = self.model.geom(self.config.table_geom)
        block = self.model.body(self.config.block_body)
        block_geom = self.model.geom(self.config.block_geom)
        table_top = table.pos[2] + table_geom.pos[2] + table_geom.size[2]
        block_bottom = block.pos[2] + block_geom.pos[2] - block_geom.size[2]
        self.assertAlmostEqual(table_top, block_bottom)
        goal = self.model.geom(self.config.goal_geom)
        self.assertEqual(goal.contype, 0)
        self.assertEqual(goal.conaffinity, 0)

    def test_pusher_can_contact_block(self) -> None:
        data, manager = self.make_data()
        tip = data.site_xpos[self.model.site(self.config.tool_tip_site).id]
        data.qpos[
            manager.block_qpos_address : manager.block_qpos_address + 2
        ] = tip[:2]
        mujoco.mj_forward(self.model, data)
        pairs = [
            {
                self.model.geom(data.contact[index].geom1).name,
                self.model.geom(data.contact[index].geom2).name,
            }
            for index in range(data.ncon)
        ]
        self.assertIn(
            {self.config.tool_geom, self.config.block_geom},
            pairs,
        )


class ResetTests(ModelFixture):
    def test_reset_restores_pose_and_zeros_velocity(self) -> None:
        data, manager = self.make_data()
        qpos = manager.block_qpos_address
        dof = manager.block_dof_address
        data.qpos[qpos : qpos + 3] += (0.1, 0.05, 0.2)
        data.qvel[dof : dof + 6] = 3.0
        expected = manager.reset(data)
        np.testing.assert_allclose(data.qpos[qpos : qpos + 3], expected)
        np.testing.assert_allclose(data.qvel[dof : dof + 6], 0.0)

    def test_reset_restores_all_fr3_home_joints(self) -> None:
        data, manager = self.make_data()
        for name, address in manager.robot_qpos_addresses.items():
            data.qpos[address] += 0.01
        manager.reset(data)
        for name, address in manager.robot_qpos_addresses.items():
            self.assertAlmostEqual(data.qpos[address], manager.robot_home[name])

    def test_repeated_default_reset_is_deterministic(self) -> None:
        data, manager = self.make_data()
        first = manager.reset(data)
        data.qpos[manager.block_qpos_address] += 0.2
        second = manager.reset(data)
        np.testing.assert_array_equal(first, second)

    def test_randomized_reset_is_seed_reproducible(self) -> None:
        config = replace(
            self.config,
            reset_randomization_enabled=True,
            reset_xy_range=(0.02, 0.02),
        )
        data = mujoco.MjData(self.model)
        manager = ResetManager(self.model, self.scene.robot_home, config)
        first = manager.reset(data, seed=123)
        second = manager.reset(data, seed=123)
        third = manager.reset(data, seed=456)
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, third))


class EvaluatorTests(ModelFixture):
    def set_block(
        self,
        data: mujoco.MjData,
        manager: ResetManager,
        xy: tuple[float, float],
        velocity_xy: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        data.qpos[
            manager.block_qpos_address : manager.block_qpos_address + 2
        ] = xy
        data.qvel[
            manager.block_dof_address : manager.block_dof_address + 2
        ] = velocity_xy
        mujoco.mj_forward(self.model, data)

    def test_outside_goal_is_not_success(self) -> None:
        data, _ = self.make_data()
        status = SuccessEvaluator(self.model, self.config).evaluate(data, 1.0)
        self.assertFalse(status.success)

    def test_inside_but_moving_is_not_success(self) -> None:
        data, manager = self.make_data()
        goal = self.model.body(self.config.goal_body).pos[:2]
        self.set_block(data, manager, tuple(goal), (0.02, 0.0))
        status = SuccessEvaluator(self.model, self.config).evaluate(data, 1.0)
        self.assertFalse(status.success)
        self.assertEqual(status.hold_time, 0.0)

    def test_inside_stopped_requires_hold_duration(self) -> None:
        data, manager = self.make_data()
        goal = self.model.body(self.config.goal_body).pos[:2]
        self.set_block(data, manager, tuple(goal))
        evaluator = SuccessEvaluator(self.model, self.config)
        first = evaluator.evaluate(data, self.config.success_hold_seconds / 2)
        second = evaluator.evaluate(data, self.config.success_hold_seconds / 2)
        self.assertFalse(first.success)
        self.assertTrue(second.success)
        self.assertEqual(second.reason, "success")

        self.set_block(data, manager, (0.2, 0.0))
        latched = evaluator.evaluate(data, 0.01)
        self.assertTrue(latched.success)
        self.assertEqual(latched.reason, "success")

    def test_outside_workspace_is_failure(self) -> None:
        data, manager = self.make_data()
        self.set_block(data, manager, (1.1, 0.0))
        status = SuccessEvaluator(self.model, self.config).evaluate(data, 0.01)
        self.assertTrue(status.failed)
        self.assertEqual(status.reason, "outside_workspace")


if __name__ == "__main__":
    unittest.main()
