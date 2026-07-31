"""Own the one final MuJoCo model, data instance, viewer, and step loop."""

from __future__ import annotations

from dataclasses import dataclass
import time

import mujoco
import mujoco.viewer

from ..assets.scene_builder import build_scene
from ..environment import (
    BlockPushConfig,
    ResetManager,
    SuccessEvaluator,
    TaskStatus,
)
from .teleop_backend import TeleopBackend


@dataclass
class KeyRequests:
    reset: bool = False
    status: bool = False
    quit: bool = False

    def callback(self, keycode: int) -> None:
        if keycode in (ord("R"), ord("r")):
            self.reset = True
        elif keycode == ord(" "):
            self.status = True
        elif keycode in (ord("Q"), ord("q"), 256):
            self.quit = True


def _status_line(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    config: BlockPushConfig,
    task: TaskStatus,
    backend: TeleopBackend,
    reset_count: int,
) -> str:
    block = data.xpos[model.body(config.block_body).id]
    teleop = backend.get_state()
    state = task.reason or "running"
    return (
        f"task={state} block=({block[0]:.3f}, {block[1]:.3f}, "
        f"{block[2]:.3f}) speed={task.block_speed_xy:.4f}m/s "
        f"goal_error={task.block_goal_error_xy:.3f}m "
        f"hold={task.hold_time:.2f}s teleop={teleop.mode}/"
        f"{'active' if teleop.active else 'held'} "
        f"ros={'fresh' if teleop.ros_fresh else 'stale'} resets={reset_count}"
    )


def run(
    teleop_root,
    *,
    viewer_enabled: bool = True,
    max_seconds: float | None = None,
) -> None:
    config = BlockPushConfig()
    scene = build_scene(teleop_root)
    model = scene.model
    data = mujoco.MjData(model)
    reset_manager = ResetManager(model, scene.robot_home, config)
    evaluator = SuccessEvaluator(model, config)
    reset_manager.reset(data)

    backend = TeleopBackend(teleop_root, model, data)
    backend.initialize()
    backend.reset_controller_state()
    keys = KeyRequests()
    viewer_context = (
        mujoco.viewer.launch_passive(model, data, key_callback=keys.callback)
        if viewer_enabled
        else None
    )

    start = time.perf_counter()
    next_tick = start
    last_sync = start
    last_print = start
    last_status = evaluator.evaluate(data, 0.0)
    try:
        if viewer_context is None:
            viewer = None
            keep_running = lambda: not keys.quit
        else:
            viewer = viewer_context.__enter__()
            keep_running = lambda: viewer.is_running() and not keys.quit

        print(f"Final MuJoCo model: {scene.source_xml}")
        print(f"MuJoCo timestep: {model.opt.timestep:.6f} s")
        print("Controls: R reset | SPACE status | Q/ESC quit")
        print("Waiting for /leader/joint_states...")

        while keep_running():
            now = time.perf_counter()
            if max_seconds is not None and now - start >= max_seconds:
                break
            if keys.reset:
                reset_manager.reset(data)
                evaluator.reset()
                backend.reset_controller_state()
                keys.reset = False

            backend.update_command(model.opt.timestep)
            backend.apply_controller(model, data, model.opt.timestep)
            mujoco.mj_step(model, data)
            last_status = evaluator.evaluate(data, model.opt.timestep)

            now = time.perf_counter()
            if keys.status or now - last_print >= 0.2:
                print(
                    _status_line(
                        model,
                        data,
                        config,
                        last_status,
                        backend,
                        reset_manager.reset_count,
                    ),
                    flush=True,
                )
                keys.status = False
                last_print = now
            if viewer is not None and now - last_sync >= 1.0 / 30.0:
                viewer.sync()
                last_sync = now

            next_tick += model.opt.timestep
            delay = next_tick - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.perf_counter()
    finally:
        if viewer_context is not None:
            viewer_context.__exit__(None, None, None)
        backend.shutdown()
