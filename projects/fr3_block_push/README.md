# FR3 block-pushing environment

## Purpose and ownership

This milestone adds a MuJoCo-only planar block-pushing task driven by the
existing OMY-to-FR3 teleoperation controller.

`OMY_FRANKA_TELEOP` remains the source of truth for OMY ROS joint input, OMY
MuJoCo FK, Cartesian mapping, cumulative clutch behavior, target conditioning,
velocity-DLS IK, null-space posture, FR3 commands, and FR3 assets. This project
owns the table, block, fixed goal, camera, primitive push tool, reset, task
evaluation, viewer, and final simulation loop. The dependency is one-way:
`dp-act-policy-study` imports the teleop implementation; the teleop repository
does not know about this task.

The runner owns one final FR3/task `MjModel`, one `MjData`, and one `mj_step`
loop. A separate OMY model is used only for leader FK, as in the existing
teleop implementation; it is not stepped as a second simulator.

## Dependency setup

The dependency root is resolved once, in this priority order:

1. `--teleop-root`
2. `OMY_FRANKA_TELEOP_ROOT`
3. sibling `../OMY_FRANKA_TELEOP` (lowercase sibling is also accepted)

The expected dependency contains `launch/FR3_omy_bridge.py`,
`mujoco_menagerie/franka_fr3/fr3.xml`, and the OMY model.

```bash
export OMY_FRANKA_TELEOP_ROOT=../OMY_FRANKA_TELEOP
```

## Inspect and run

Scene inspection does not require ROS or OMY hardware:

```bash
python projects/fr3_block_push/scripts/inspect_scene.py \
  --teleop-root ../OMY_FRANKA_TELEOP
```

Run teleoperation from an environment with ROS 2 sourced and the same Python
ABI as its `rclpy` installation:

```bash
python projects/fr3_block_push/scripts/run_teleop.py \
  --teleop-root ../OMY_FRANKA_TELEOP
```

Keyboard controls:

- `R`: deterministic reset
- `SPACE`: print task status
- `Q` or `ESC`: quit

The runner prints block position/speed, goal error, success hold time, task
state, teleop mode, ROS freshness, and reset count at 5 Hz.

Tests use the standard library and do not require ROS:

```bash
python -m unittest discover -s projects/fr3_block_push/tests -v
```

## Scene composition

At runtime, `scene_builder.py` parses the external FR3 XML, resolves its mesh
directory, merges the local task elements, and injects the local primitive
tool under the existing `fr3_hand` frame. It then compiles the resulting
in-memory XML once. No FR3 XML or controller implementation is copied, no
absolute local path is committed, and no generated artifact is saved.

This runtime-combined-XML approach is used because an absolute XML include
does not preserve the included FR3 file's relative mesh directory in the
installed MuJoCo version. The builder leaves both source repositories
unchanged.

## Task definition

The fixed table top is at `z=0.20 m`. A `0.30 kg` free-joint block with
`0.035 m` half-size starts at `(0.62, 0.00, 0.235)`. The non-colliding visual
goal is centered at `(0.80, 0.00, 0.202)` with `0.075 m` planar half-size.
The top policy camera is fixed above the workspace. A capsule pusher attached
to the FR3 hand ends just above the table in the home configuration.

Reset restores the seven FR3 joints and actuator targets from the external
FR3 `home` keyframe, closes the unused gripper command, restores the block
free-joint pose, zeros all velocities and actuator state, resets task timers,
and calls `mj_forward`. Reset is deterministic by default; seeded small XY
randomization can be enabled in `BlockPushConfig`.

Success requires the whole block (including a margin) to remain inside the
goal while planar speed is at most `0.01 m/s` for `0.5 s`. Leaving the
configured table workspace or reaching the `120 s` episode timeout is failure.
Evaluation reads state only and never changes the control loop.

## References, limitations, and next milestone

The supplied `Push_MuJoCo` files were used only to study general table,
free-block, visual-goal, camera, pusher, reset, and substep ideas. Their README
identifies <https://github.com/JericLew/Push_MuJoCo> and MuJoCo Menagerie
sources but provides no explicit project license. The originals therefore
remain local-only under `references/Push_MuJoCo/` and are ignored by Git.
Production code neither imports nor includes them and contains no Panda
dependency.

This is simulation-only and does not claim real-robot safety. It does not
include dataset recording, LeRobot integration, ACT, Diffusion Policy,
multi-camera data, multi-goal tasks, or a gripper task. The next milestone can
add a recorder after manual OMY contact behavior and camera framing have been
validated.
