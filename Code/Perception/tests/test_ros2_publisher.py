import sys
from types import ModuleType

from chess_perception.ros2_publisher import Ros2JsonPublisher


class _String:
    def __init__(self) -> None:
        self.data = ""


class _Publisher:
    def __init__(self) -> None:
        self.messages: list[_String] = []

    def publish(self, message: _String) -> None:
        self.messages.append(message)


class _Node:
    def __init__(self) -> None:
        self.publisher = _Publisher()
        self.destroyed = False

    def create_publisher(
        self, message_type: type[_String], topic: str, depth: int
    ) -> _Publisher:
        assert message_type is _String
        assert topic == "/chessmate/perception/move"
        assert depth == 10
        return self.publisher

    def destroy_node(self) -> None:
        self.destroyed = True


def test_ros2_publisher_uses_string_message_and_cleans_up(monkeypatch) -> None:
    node = _Node()
    state = {"ok": False, "shutdown": False, "spin_count": 0}
    rclpy = ModuleType("rclpy")

    def ok() -> bool:
        return state["ok"]

    def init(args: list[str]) -> None:
        assert args == []
        state["ok"] = True

    def shutdown() -> None:
        state["ok"] = False
        state["shutdown"] = True

    def spin_once(spin_node: _Node, timeout_sec: float) -> None:
        assert spin_node is node
        assert timeout_sec == 0.0
        state["spin_count"] += 1

    rclpy.ok = ok
    rclpy.init = init
    rclpy.shutdown = shutdown
    rclpy.spin_once = spin_once
    rclpy.create_node = lambda name: node

    std_msgs = ModuleType("std_msgs")
    std_msgs_msg = ModuleType("std_msgs.msg")
    std_msgs_msg.String = _String
    std_msgs.msg = std_msgs_msg
    monkeypatch.setitem(sys.modules, "rclpy", rclpy)
    monkeypatch.setitem(sys.modules, "std_msgs", std_msgs)
    monkeypatch.setitem(sys.modules, "std_msgs.msg", std_msgs_msg)

    publisher = Ros2JsonPublisher("/chessmate/perception/move")
    publisher.publish('{"uci":"e2e4"}')
    publisher.close()

    assert [message.data for message in node.publisher.messages] == [
        '{"uci":"e2e4"}'
    ]
    assert state["spin_count"] == 1
    assert node.destroyed
    assert state["shutdown"]
