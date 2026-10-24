"""Relay appended JSONL chess events to dedicated ROS 2 String topics."""

from __future__ import annotations

import json
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class MoveEventBridge(Node):
    def __init__(self) -> None:
        super().__init__("chessmate_move_event_bridge")
        self.declare_parameter(
            "event_log", "/home/ubuntu/data/chessmate_events/moves.jsonl"
        )
        self.declare_parameter("move_topic", "/chessmate/perception/move")
        self.declare_parameter("engine_topic", "/chessmate/engine/move")
        self.declare_parameter(
            "verified_topic", "/chessmate/perception/robot_move_verified"
        )
        self.declare_parameter("start_at_end", True)
        self.declare_parameter("poll_period_seconds", 0.1)

        self._event_log = Path(
            self.get_parameter("event_log").get_parameter_value().string_value
        )
        move_topic = self.get_parameter("move_topic").get_parameter_value().string_value
        engine_topic = (
            self.get_parameter("engine_topic").get_parameter_value().string_value
        )
        verified_topic = (
            self.get_parameter("verified_topic").get_parameter_value().string_value
        )
        start_at_end = (
            self.get_parameter("start_at_end").get_parameter_value().bool_value
        )
        poll_period = (
            self.get_parameter("poll_period_seconds")
            .get_parameter_value()
            .double_value
        )
        self._publishers = {
            "move_detected": self.create_publisher(String, move_topic, 10),
            "engine_move_requested": self.create_publisher(String, engine_topic, 10),
            "robot_move_verified": self.create_publisher(String, verified_topic, 10),
        }
        self._offset = (
            self._event_log.stat().st_size
            if start_at_end and self._event_log.exists()
            else 0
        )
        self._partial_line = ""
        self.create_timer(max(0.02, poll_period), self._poll)
        self.get_logger().info(
            f"Relaying {self._event_log} from byte {self._offset}: "
            f"human={move_topic}, engine={engine_topic}, verified={verified_topic}"
        )

    def _poll(self) -> None:
        if not self._event_log.exists():
            return
        size = self._event_log.stat().st_size
        if size < self._offset:
            self.get_logger().warning("Event log was truncated; restarting at byte 0")
            self._offset = 0
            self._partial_line = ""
        if size == self._offset:
            return

        with self._event_log.open() as stream:
            stream.seek(self._offset)
            chunk = stream.read()
            self._offset = stream.tell()

        lines = (self._partial_line + chunk).splitlines(keepends=True)
        self._partial_line = ""
        for line in lines:
            if not line.endswith("\n"):
                self._partial_line = line
                continue
            serialized = line.strip()
            if not serialized:
                continue
            try:
                event = json.loads(serialized)
            except json.JSONDecodeError as error:
                self.get_logger().error(f"Skipping invalid JSON event: {error}")
                continue
            event_type = event.get("event")
            publisher = self._publishers.get(event_type)
            if publisher is None:
                continue
            message = String()
            message.data = json.dumps(event, sort_keys=True)
            publisher.publish(message)
            self.get_logger().info(
                f"Published {event_type} {event.get('uci', '<missing uci>')}"
            )


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = MoveEventBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
