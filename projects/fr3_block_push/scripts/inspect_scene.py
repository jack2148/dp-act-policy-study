#!/usr/bin/env python3
"""Compile and validate the combined FR3 block-push scene without ROS."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import mujoco
import numpy as np


STUDY_ROOT = Path(__file__).resolve().parents[3]
if str(STUDY_ROOT) not in sys.path:
    sys.path.insert(0, str(STUDY_ROOT))

from projects.fr3_block_push.assets.scene_builder import build_scene
from projects.fr3_block_push.control.teleop_backend import resolve_teleop_root
from projects.fr3_block_push.environment import BlockPushConfig, ResetManager


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"PASS: {message}")


def _name(model: mujoco.MjModel, geom_id: int) -> str:
    return (
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        or f"geom#{geom_id}"
    )


def _check_duplicate_names(model: mujoco.MjModel) -> None:
    categories = (
        (mujoco.mjtObj.mjOBJ_BODY, model.nbody),
        (mujoco.mjtObj.mjOBJ_JOINT, model.njnt),
        (mujoco.mjtObj.mjOBJ_GEOM, model.ngeom),
        (mujoco.mjtObj.mjOBJ_SITE, model.nsite),
        (mujoco.mjtObj.mjOBJ_CAMERA, model.ncam),
        (mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu),
    )
    for object_type, count in categories:
        names = [
            mujoco.mj_id2name(model, object_type, index)
            for index in range(count)
        ]
        named = [name for name in names if name is not None]
        _check(len(named) == len(set(named)), f"no duplicate {object_type.name} names")


def inspect(teleop_root: Path) -> None:
    config = BlockPushConfig()
    scene = build_scene(teleop_root)
    model = scene.model
    data = mujoco.MjData(model)
    reset = ResetManager(model, scene.robot_home, config)
    reset.reset(data)
    print(f"PASS: compiled final model from {scene.source_xml}")

    fr3_joints = [model.joint(f"fr3_joint{i}").id for i in range(1, 8)]
    fr3_actuators = [model.actuator(f"fr3_joint{i}").id for i in range(1, 8)]
    _check(len(set(fr3_joints)) == 7, "seven unique FR3 arm joints")
    _check(len(set(fr3_actuators)) == 7, "seven unique FR3 arm actuators")

    for object_type, name in (
        (mujoco.mjtObj.mjOBJ_BODY, config.table_body),
        (mujoco.mjtObj.mjOBJ_BODY, config.block_body),
        (mujoco.mjtObj.mjOBJ_BODY, config.goal_body),
        (mujoco.mjtObj.mjOBJ_JOINT, config.block_joint),
        (mujoco.mjtObj.mjOBJ_GEOM, config.tool_geom),
        (mujoco.mjtObj.mjOBJ_SITE, config.tool_tip_site),
        (mujoco.mjtObj.mjOBJ_CAMERA, config.camera),
        (mujoco.mjtObj.mjOBJ_CAMERA, config.viewer_camera),
    ):
        _check(mujoco.mj_name2id(model, object_type, name) >= 0, f"required name {name}")

    joint = model.joint(config.block_joint)
    _check(
        model.jnt_type[joint.id] == mujoco.mjtJoint.mjJNT_FREE,
        "block joint is free",
    )
    qpos_address = int(model.jnt_qposadr[joint.id])
    dof_address = int(model.jnt_dofadr[joint.id])
    print(
        f"PASS: block addresses qpos={qpos_address}:{qpos_address + 7}, "
        f"qvel={dof_address}:{dof_address + 6}"
    )

    table = model.body(config.table_body)
    table_geom = model.geom(config.table_geom)
    block = model.body(config.block_body)
    block_geom = model.geom(config.block_geom)
    table_top = float(table.pos[2] + table_geom.pos[2] + table_geom.size[2])
    block_bottom = float(block.pos[2] + block_geom.pos[2] - block_geom.size[2])
    _check(
        abs(table_top - block_bottom) <= 1e-9,
        f"block bottom aligns with table top ({table_top:.3f} m)",
    )

    goal_geom = model.geom(config.goal_geom)
    _check(
        goal_geom.contype == 0 and goal_geom.conaffinity == 0,
        "goal geometry is non-colliding",
    )
    goal_position = data.xpos[model.body(config.goal_body).id]
    tip_position = data.site_xpos[model.site(config.tool_tip_site).id]
    print(f"PASS: goal position {goal_position}")
    print(f"PASS: pusher tip position {tip_position}")
    print(
        "PASS: policy camera ids "
        f"top={model.camera(config.camera).id}, "
        f"operator={model.camera(config.viewer_camera).id}"
    )
    _check(abs(model.opt.timestep - 0.001) <= 1e-12, "model timestep is 0.001 s")
    _check_duplicate_names(model)

    contacts = [
        (
            _name(model, data.contact[index].geom1),
            _name(model, data.contact[index].geom2),
            float(data.contact[index].dist),
        )
        for index in range(data.ncon)
    ]
    penetrating = [contact for contact in contacts if contact[2] < -1e-6]
    _check(not penetrating, f"no initial penetration ({len(contacts)} contacts)")
    pusher_table = {
        config.tool_geom,
        config.table_geom,
    }
    _check(
        not any({first, second} == pusher_table for first, second, _ in contacts),
        "pusher does not initially contact table",
    )

    initial_block = data.xpos[model.body(config.block_body).id].copy()
    for _ in range(250):
        mujoco.mj_step(model, data)
    settled_block = data.xpos[model.body(config.block_body).id].copy()
    displacement = float(np.linalg.norm(settled_block - initial_block))
    _check(
        displacement < 0.002,
        f"block remains stable after gravity settling ({displacement:.6f} m)",
    )

    reset.reset(data)
    joint = model.joint(config.block_joint)
    block_qpos = int(model.jnt_qposadr[joint.id])
    tip = data.site_xpos[model.site(config.tool_tip_site).id]
    data.qpos[block_qpos : block_qpos + 2] = tip[:2]
    mujoco.mj_forward(model, data)
    contact_pairs = [
        {
            _name(model, data.contact[index].geom1),
            _name(model, data.contact[index].geom2),
        }
        for index in range(data.ncon)
    ]
    _check(
        {config.tool_geom, config.block_geom} in contact_pairs,
        "pusher geometry collides with block geometry",
    )
    print("SCENE INSPECTION: PASS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teleop-root", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        root = resolve_teleop_root(args.teleop_root, study_root=STUDY_ROOT)
        inspect(root)
    except Exception as error:
        print(f"SCENE INSPECTION: FAIL: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
