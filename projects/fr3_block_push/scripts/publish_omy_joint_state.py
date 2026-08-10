#!/usr/bin/env python3
"""Publish latest /leader/joint_states over localhost ZeroMQ (Python 3.10)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time


STUDY_ROOT = Path(__file__).resolve().parents[3]
if str(STUDY_ROOT) not in sys.path:
    sys.path.insert(0, str(STUDY_ROOT))

from projects.fr3_block_push.ipc.protocol import (
    OMY_JOINT_NAMES,
    OMY_ROS_JOINT_NAMES,
    OMY_TRIGGER_JOINT,
    ProtocolError,
    encode_message,
    extract_joint_position,
    reorder_joint_positions,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default="/leader/joint_states")
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5557")
    args = parser.parse_args()

    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    import zmq

    context = zmq.Context.instance()
    socket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.SNDHWM, 1)
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(args.endpoint)

    class PublisherNode(Node):
        def __init__(self) -> None:
            super().__init__("omy_joint_state_ipc_publisher")
            self.sequence = 0
            self.rejected_messages = 0
            self.subscription = self.create_subscription(
                JointState, args.topic, self.callback, 10
            )

        def callback(self, message: JointState) -> None:
            try:
                ordered = reorder_joint_positions(
                    message.name,
                    message.position,
                    allow_extra_joints=True,
                    required_joint_names=OMY_ROS_JOINT_NAMES,
                )
                trigger_position = extract_joint_position(
                    message.name,
                    message.position,
                    OMY_TRIGGER_JOINT,
                )
                stamp = message.header.stamp
                source_timestamp_ns = int(stamp.sec) * 1_000_000_000 + int(
                    stamp.nanosec
                )
                payload = encode_message(
                    sequence=self.sequence,
                    source_timestamp_ns=source_timestamp_ns,
                    sender_monotonic_ns=time.monotonic_ns(),
                    joint_names=OMY_JOINT_NAMES,
                    position=ordered,
                    trigger_position=trigger_position,
                )
            except ProtocolError as error:
                self.rejected_messages += 1
                if self.rejected_messages <= 3 or self.rejected_messages % 100 == 0:
                    self.get_logger().warning(
                        f"Rejected JointState #{self.rejected_messages}: {error}"
                    )
                return
            try:
                socket.send(payload, flags=zmq.NOBLOCK)
            except zmq.Again:
                self.rejected_messages += 1
                return
            self.sequence += 1

    rclpy.init()
    node = PublisherNode()
    print(f"Publishing {args.topic} latest state on {args.endpoint}", flush=True)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        socket.close(linger=0)
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
