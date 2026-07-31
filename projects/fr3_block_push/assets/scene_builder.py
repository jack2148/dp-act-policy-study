"""Compose the external FR3 asset and the local pushing task."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco


ASSET_DIR = Path(__file__).resolve().parent
TASK_XML = ASSET_DIR / "block_push_task.xml"
TOOL_XML = ASSET_DIR / "push_tool.xml"
FR3_RELATIVE_XML = Path("mujoco_menagerie/franka_fr3/fr3.xml")
FR3_SCENE_RELATIVE_XML = Path("mujoco_menagerie/franka_fr3/scene.xml")


@dataclass(frozen=True)
class ComposedScene:
    model: mujoco.MjModel
    robot_home: dict[str, float]
    source_xml: Path


def _required(root: ET.Element, tag: str) -> ET.Element:
    element = root.find(tag)
    if element is None:
        element = ET.SubElement(root, tag)
    return element


def _merge_section(
    destination_root: ET.Element,
    source_root: ET.Element,
    tag: str,
) -> None:
    source = source_root.find(tag)
    if source is None:
        return
    destination = _required(destination_root, tag)
    for child in source:
        destination.append(copy.deepcopy(child))


def _joint_width(joint: ET.Element) -> int:
    return {"free": 7, "ball": 4}.get(joint.get("type", "hinge"), 1)


def _robot_home_from_xml(root: ET.Element) -> dict[str, float]:
    worldbody = root.find("worldbody")
    key = root.find("./keyframe/key[@name='home']")
    if worldbody is None or key is None or not key.get("qpos"):
        raise ValueError("FR3 XML must provide worldbody and the 'home' keyframe")

    values = [float(value) for value in key.get("qpos", "").split()]
    result: dict[str, float] = {}
    offset = 0
    for joint in worldbody.iter("joint"):
        width = _joint_width(joint)
        name = joint.get("name")
        if name and width == 1:
            result[name] = values[offset]
        offset += width
    if offset != len(values):
        raise ValueError(
            f"FR3 home keyframe has {len(values)} values for {offset} qpos entries"
        )
    return result


def _remove_keyframes(root: ET.Element) -> None:
    keyframe = root.find("keyframe")
    if keyframe is not None:
        root.remove(keyframe)


def compose_scene_xml(teleop_root: Path) -> tuple[str, dict[str, float], Path]:
    """Return deterministic combined XML without modifying either source repo."""
    teleop_root = teleop_root.resolve()
    robot_xml = teleop_root / FR3_RELATIVE_XML
    if not robot_xml.is_file():
        raise FileNotFoundError(
            f"FR3 asset not found: {robot_xml}. "
            "Expected the OMY_FRANKA_TELEOP Menagerie checkout."
        )

    robot_root = ET.parse(robot_xml).getroot()
    robot_home = _robot_home_from_xml(robot_root)
    _remove_keyframes(robot_root)

    compiler = _required(robot_root, "compiler")
    compiler.set("meshdir", str((robot_xml.parent / "assets").resolve()))

    task_root = ET.parse(TASK_XML).getroot()
    for tag in ("statistic", "visual"):
        existing = robot_root.find(tag)
        replacement = task_root.find(tag)
        if replacement is not None:
            if existing is not None:
                robot_root.remove(existing)
            robot_root.append(copy.deepcopy(replacement))
    for tag in ("asset", "worldbody", "contact"):
        _merge_section(robot_root, task_root, tag)

    tool_root = ET.parse(TOOL_XML).getroot()
    tool_body = tool_root.find("./worldbody/body[@name='push_tool']")
    fr3_hand = robot_root.find(".//body[@name='fr3_hand']")
    if tool_body is None or fr3_hand is None:
        raise ValueError("push tool body or FR3 hand attachment body is missing")
    fr3_hand.append(copy.deepcopy(tool_body))
    _merge_section(robot_root, tool_root, "contact")

    return ET.tostring(robot_root, encoding="unicode"), robot_home, robot_xml


def build_scene(teleop_root: Path, timestep: float = 0.001) -> ComposedScene:
    xml, robot_home, source_xml = compose_scene_xml(teleop_root)
    model = mujoco.MjModel.from_xml_string(xml)
    model.opt.timestep = timestep
    return ComposedScene(model=model, robot_home=robot_home, source_xml=source_xml)
