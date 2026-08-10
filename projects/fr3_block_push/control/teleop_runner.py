"""Own the single final MuJoCo environment and existing teleop controller."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
import time
from typing import Callable

import mujoco
import mujoco.viewer
import numpy as np

from ..assets.scene_builder import build_scene
from ..environment import BlockPushConfig, ResetManager, SuccessEvaluator, TaskStatus
from .teleop_backend import TeleopBackend


@dataclass
class KeyRequests:
    reset: bool = False
    status: bool = False
    camera: bool = False
    quit: bool = False

    def callback(self, keycode: int) -> None:
        if keycode in (ord("R"), ord("r")):
            self.reset = True
        elif keycode == ord(" "):
            self.status = True
        elif keycode in (ord("V"), ord("v")):
            self.camera = True
        elif keycode in (ord("Q"), ord("q"), 256):
            self.quit = True


@dataclass(frozen=True)
class StepResult:
    task: TaskStatus
    reset_occurred: bool = False
    reset_source: str | None = None
    sample: "ControlSample | None" = None


@dataclass(frozen=True)
class ControlSample:
    """Pre-integration observation paired with the applied position target."""

    image: np.ndarray
    state: np.ndarray
    action: np.ndarray
    ee_pose: np.ndarray
    goal_distance: np.ndarray
    goal_delta: np.ndarray
    action_ee_pose: np.ndarray
    simulation_time: float


def _simulation_time_rewound(
    previous_time: float,
    current_time: float,
    timestep: float,
) -> bool:
    """Detect the MuJoCo viewer's built-in Reset action."""
    return current_time < previous_time - 0.5 * timestep


class TeleopRunner:
    """Public environment/control surface used by teleop and collection.

    This class owns exactly one task ``MjModel`` and one ``MjData``.  It does
    not import LeRobot or know anything about dataset episode buffers.
    """

    def __init__(
        self,
        teleop_root: str | Path,
        *,
        viewer_enabled: bool = True,
        key_callback: Callable[[int], None] | None = None,
        image_size: tuple[int, int] = (96, 96),
        input_backend: str = "ros",
        omy_endpoint: str = "tcp://127.0.0.1:5557",
        omy_stale_timeout: float = 0.2,
        success_margin: float | None = None,
    ) -> None:
        self.config = BlockPushConfig()
        if success_margin is not None:
            self.config = replace(self.config, success_margin=success_margin)
        self.scene = build_scene(Path(teleop_root))
        self.model = self.scene.model
        self.data = mujoco.MjData(self.model)
        self.reset_manager = ResetManager(
            self.model, self.scene.robot_home, self.config
        )
        self.evaluator = SuccessEvaluator(self.model, self.config)
        self.reset_manager.reset(self.data)

        self.backend = TeleopBackend(
            Path(teleop_root),
            self.model,
            self.data,
            input_backend=input_backend,
            teleop_mode=self.config.teleop_mode,
            omy_endpoint=omy_endpoint,
            omy_stale_timeout=omy_stale_timeout,
        )
        try:
            self.backend.initialize()
            self.backend.reset_controller_state()
            self._validate_position_action_semantics()
        except Exception:
            self.backend.shutdown()
            raise

        self.image_size = image_size
        self._renderer: mujoco.Renderer | None = None
        self._camera_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.config.camera
        )
        self._viewer_camera_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.config.viewer_camera
        )
        if self._camera_id < 0:
            self.backend.shutdown()
            raise ValueError(f"camera not found: {self.config.camera}")
        if self._viewer_camera_id < 0:
            self.backend.shutdown()
            raise ValueError(f"camera not found: {self.config.viewer_camera}")

        self._viewer_context = None
        self.viewer = None
        self._fixed_view_enabled = False
        try:
            if viewer_enabled:
                self._viewer_context = mujoco.viewer.launch_passive(
                    self.model, self.data, key_callback=key_callback
                )
                self.viewer = self._viewer_context.__enter__()
                with self.viewer.lock():
                    self.viewer.opt.flags[
                        mujoco.mjtVisFlag.mjVIS_CONTACTPOINT
                    ] = False
                    self.viewer.opt.flags[
                        mujoco.mjtVisFlag.mjVIS_CONTACTFORCE
                    ] = False
        except Exception:
            self.backend.shutdown()
            raise

        self.last_sim_time = float(self.data.time)
        self.last_status = self.evaluator.evaluate(self.data, 0.0)
        self._last_sync = time.perf_counter()

    def _validate_position_action_semantics(self) -> None:
        """Reject recording unless all seven controls are joint position refs."""
        for offset, actuator_id in enumerate(self.backend.fr3_actuator_indices, 1):
            expected_joint = self.model.joint(f"fr3_joint{offset}").id
            transmission_joint = int(self.model.actuator_trnid[actuator_id, 0])
            gain = float(self.model.actuator_gainprm[actuator_id, 0])
            bias_position = float(self.model.actuator_biasprm[actuator_id, 1])
            gear = self.model.actuator_gear[actuator_id]
            is_position_servo = (
                self.model.actuator_trntype[actuator_id]
                == mujoco.mjtTrn.mjTRN_JOINT
                and transmission_joint == expected_joint
                and self.model.actuator_gaintype[actuator_id]
                == mujoco.mjtGain.mjGAIN_FIXED
                and self.model.actuator_biastype[actuator_id]
                == mujoco.mjtBias.mjBIAS_AFFINE
                and gain > 0.0
                and np.isclose(bias_position, -gain)
                and np.isclose(gear[0], 1.0)
                and np.allclose(gear[1:], 0.0)
            )
            if not is_position_servo:
                name = self.model.actuator(actuator_id).name
                raise RuntimeError(
                    f"refusing to record {name}: actuator is not a direct "
                    "joint-position servo"
                )

    @property
    def simulation_time(self) -> float:
        return float(self.data.time)

    @property
    def timestep(self) -> float:
        return float(self.model.opt.timestep)

    @property
    def input_fresh(self) -> bool:
        return bool(self.backend.input_fresh)

    def get_fr3_joint_positions(self) -> np.ndarray:
        """Current seven FR3 arm joint positions in radians."""
        return np.asarray(
            self.data.qpos[self.backend.fr3_qpos_indices], dtype=np.float32
        ).copy()

    def get_block_goal_xy(self) -> tuple[np.ndarray, np.ndarray]:
        """Return current block and goal XY centers for evaluation logging."""
        block_id = self.model.body(self.config.block_body).id
        goal_id = self.model.body(self.config.goal_body).id
        return (
            np.asarray(self.data.xpos[block_id, :2], dtype=np.float32).copy(),
            np.asarray(self.data.xpos[goal_id, :2], dtype=np.float32).copy(),
        )

    def get_record_action(self) -> np.ndarray:
        """Seven position references actually applied to FR3 actuators, rad."""
        action = self.data.ctrl[self.backend.fr3_actuator_indices]
        if not np.allclose(action, self.backend.hold_q_target):
            raise RuntimeError("applied FR3 controls differ from controller target")
        return np.asarray(action, dtype=np.float32).copy()

    def _get_ee_pose(self) -> np.ndarray:
        site_id = self.backend.fr3_ee_site_id
        position = np.asarray(self.data.site_xpos[site_id], dtype=np.float64)
        rotation = np.asarray(self.data.site_xmat[site_id], dtype=np.float64)
        quaternion = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quaternion, rotation)
        return np.asarray(
            [*quaternion, *position], dtype=np.float32
        )

    def get_ee_pose(self) -> np.ndarray:
        """Current EE pose as [qw, qx, qy, qz, x, y, z]."""
        return self._get_ee_pose().copy()

    def get_action_ee_pose(self, action: np.ndarray) -> np.ndarray:
        """Kinematic EE pose corresponding to a seven-joint target."""
        values = np.asarray(action, dtype=np.float64).reshape(7)
        saved_qpos = self.data.qpos.copy()
        saved_qvel = self.data.qvel.copy()
        try:
            self.data.qpos[self.backend.fr3_qpos_indices] = values
            self.data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.data)
            return self._get_ee_pose().copy()
        finally:
            self.data.qpos[:] = saved_qpos
            self.data.qvel[:] = saved_qvel
            mujoco.mj_forward(self.model, self.data)

    def get_goal_distance_xy(self) -> np.ndarray:
        """Current Euclidean XY distance between block and goal centers."""
        block_xy, goal_xy = self.get_block_goal_xy()
        return np.asarray([np.linalg.norm(block_xy - goal_xy)], dtype=np.float32)

    def get_goal_delta_xy(self) -> np.ndarray:
        """Signed goal displacement as [goal_x-block_x, goal_y-block_y]."""
        block_xy, goal_xy = self.get_block_goal_xy()
        return np.asarray(goal_xy - block_xy, dtype=np.float32)

    def apply_policy_action(self, action: np.ndarray) -> None:
        """Apply a seven-joint position action produced by a policy."""
        values = np.asarray(action, dtype=np.float32).reshape(-1)
        if values.shape != (len(self.backend.fr3_actuator_indices),):
            raise ValueError(
                f"policy action must have shape (7,), got {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("policy action contains non-finite values")
        self.data.ctrl[self.backend.fr3_actuator_indices] = values

    def step_policy(self, *, sim_steps: int = 1) -> StepResult:
        """Advance MuJoCo using the action already applied by a policy."""
        if sim_steps <= 0:
            raise ValueError("sim_steps must be positive")
        lock = self.viewer.lock() if self.viewer is not None else nullcontext()
        with lock:
            for _ in range(sim_steps):
                mujoco.mj_step(self.model, self.data)
            self.last_status = self.evaluator.evaluate(
                self.data, self.timestep * sim_steps
            )
            self.last_sim_time = float(self.data.time)
        return StepResult(self.last_status)

    def render_policy_camera(self) -> np.ndarray:
        """Render policy_camera_top offscreen as RGB uint8 HWC, without flip."""
        lock = self.viewer.lock() if self.viewer is not None else nullcontext()
        with lock:
            return self._render_policy_camera_unlocked()

    def _render_policy_camera_unlocked(self) -> np.ndarray:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model, height=self.image_size[0], width=self.image_size[1]
            )
        self._renderer.update_scene(self.data, camera=self._camera_id)
        image = self._renderer.render().copy()
        if image.shape != (*self.image_size, 3) or image.dtype != np.uint8:
            raise RuntimeError(
                f"unexpected camera output: shape={image.shape}, dtype={image.dtype}"
            )
        return image

    def toggle_viewer_camera(self) -> str:
        """Switch the operator window between free and fixed overhead views."""
        if self.viewer is None:
            return "disabled"
        with self.viewer.lock():
            if self._fixed_view_enabled:
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                self._fixed_view_enabled = False
                view = "free"
            else:
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                self.viewer.cam.fixedcamid = self._viewer_camera_id
                self._fixed_view_enabled = True
                view = self.config.viewer_camera
        self.sync_viewer(force=True)
        return view

    def reset(self) -> None:
        lock = self.viewer.lock() if self.viewer is not None else nullcontext()
        with lock:
            self._reset_unlocked()

    def _reset_unlocked(self) -> None:
        self.reset_manager.reset(self.data)
        self.evaluator.reset()
        self.backend.reset_controller_state()
        self.last_sim_time = float(self.data.time)
        self.last_status = self.evaluator.evaluate(self.data, 0.0)

    def step(self, *, capture_sample: bool = False) -> StepResult:
        """Apply the existing controller once and advance one MuJoCo step."""
        reset_occurred = False
        reset_source = None
        sample = None
        lock = self.viewer.lock() if self.viewer is not None else nullcontext()
        with lock:
            if _simulation_time_rewound(
                self.last_sim_time, float(self.data.time), self.timestep
            ):
                self._reset_unlocked()
                reset_occurred = True
                reset_source = "MuJoCo UI"

            self.backend.update_command(self.timestep)
            self.backend.apply_controller(self.model, self.data, self.timestep)
            if capture_sample and self.backend.input_fresh:
                sample = ControlSample(
                    image=self._render_policy_camera_unlocked(),
                    state=self.get_fr3_joint_positions(),
                    action=self.get_record_action(),
                    ee_pose=self.get_ee_pose(),
                    goal_distance=self.get_goal_distance_xy(),
                    goal_delta=self.get_goal_delta_xy(),
                    action_ee_pose=self.get_action_ee_pose(
                        self.get_record_action()
                    ),
                    simulation_time=float(self.data.time),
                )
            mujoco.mj_step(self.model, self.data)
            self.last_status = self.evaluator.evaluate(self.data, self.timestep)
            self.last_sim_time = float(self.data.time)
        return StepResult(
            self.last_status, reset_occurred, reset_source, sample
        )

    def is_running(self) -> bool:
        return self.viewer is None or self.viewer.is_running()

    def sync_viewer(self, *, force: bool = False) -> None:
        if self.viewer is None:
            return
        now = time.perf_counter()
        if force or now - self._last_sync >= 1.0 / 30.0:
            self.viewer.sync()
            self._last_sync = now

    def status_line(self) -> str:
        block = self.data.xpos[self.model.body(self.config.block_body).id]
        teleop = self.backend.get_state()
        task = self.last_status
        state = task.reason or "running"
        return (
            f"task={state} block=({block[0]:.3f}, {block[1]:.3f}, "
            f"{block[2]:.3f}) speed={task.block_speed_xy:.4f}m/s "
            f"goal_error={task.block_goal_error_xy:.3f}m "
            f"hold={task.hold_time:.2f}s teleop={teleop.mode}/"
            f"{'active' if teleop.active else 'held'} "
            f"input={self.backend.input_backend}/"
            f"{'fresh' if self.backend.input_fresh else 'stale'} "
            f"resets={self.reset_manager.reset_count}"
        )

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._viewer_context is not None:
            self._viewer_context.__exit__(None, None, None)
            self._viewer_context = None
            self.viewer = None
        self.backend.shutdown()

    def __enter__(self) -> "TeleopRunner":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def run(
    teleop_root: str | Path,
    *,
    viewer_enabled: bool = True,
    max_seconds: float | None = None,
) -> None:
    """Backward-compatible plain teleoperation entry point."""
    keys = KeyRequests()
    with TeleopRunner(
        teleop_root,
        viewer_enabled=viewer_enabled,
        key_callback=keys.callback,
    ) as runner:
        print(f"Final MuJoCo model: {runner.scene.source_xml}")
        print(f"MuJoCo timestep: {runner.timestep:.6f} s")
        print(
            "Controls: UI Reset/R -> FR3 home | V camera | "
            "SPACE status | Q/ESC quit"
        )
        print("Waiting for /leader/joint_states...")
        start = time.perf_counter()
        next_tick = start
        last_print = start
        while runner.is_running() and not keys.quit:
            now = time.perf_counter()
            if max_seconds is not None and now - start >= max_seconds:
                break
            if keys.reset:
                runner.reset()
                keys.reset = False
                print("Reset from keyboard: FR3 restored to home pose", flush=True)

            if keys.camera:
                view = runner.toggle_viewer_camera()
                keys.camera = False
                print(f"Viewer camera: {view}", flush=True)

            result = runner.step()
            if result.reset_occurred:
                print(
                    f"Reset from {result.reset_source}: FR3 restored to home pose",
                    flush=True,
                )
            now = time.perf_counter()
            if keys.status or now - last_print >= 0.2:
                print(runner.status_line(), flush=True)
                keys.status = False
                last_print = now
            runner.sync_viewer()

            next_tick += runner.timestep
            delay = next_tick - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.perf_counter()
