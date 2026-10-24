"""Reference Team 2 subscriber for Stockfish robot-move commands."""

from __future__ import annotations

import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class MoveEventListener(Node):
    def __init__(self) -> None:
        super().__init__("chessmate_move_event_listener")
        self.declare_parameter("engine_topic", "/chessmate/engine/move")
        topic = self.get_parameter("engine_topic").get_parameter_value().string_value
        self.create_subscription(String, topic, self._on_move, 10)
        self.get_logger().info(f"Listening for robot move commands on {topic}")

    def _on_move(self, message: String) -> None:
        try:
            event = json.loads(message.data)
        except json.JSONDecodeError as error:
            self.get_logger().error(f"Invalid move JSON: {error}")
            return
        if event.get("event") != "engine_move_requested" or "uci" not in event:
            self.get_logger().warning(
                "Ignoring message without an engine_move_requested UCI"
            )
            return
        self.get_logger().info(
            f"ROBOT MOVE uci={event['uci']} requested_elo="
            f"{event.get('requested_elo')} applied_uci_elo="
            f"{event.get('applied_uci_elo')} "
            f"fen_after={event.get('fen_after')}"
        )


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = MoveEventListener()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
