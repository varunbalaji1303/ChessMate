"""Optional ROS 2 publisher for accepted perception move events."""

from __future__ import annotations


class Ros2UnavailableError(RuntimeError):
    """Raised when ROS 2 Python packages are unavailable."""


class Ros2JsonPublisher:
    """Publish serialized JSON events with ``std_msgs/msg/String``.

    ROS imports intentionally happen during construction so the perception
    package remains usable on machines that do not have ROS 2 installed.
    """

    def __init__(
        self,
        topic: str,
        node_name: str = "chessmate_perception",
        qos_depth: int = 10,
    ) -> None:
        if not topic.strip():
            raise ValueError("ROS topic must not be empty")
        if qos_depth < 1:
            raise ValueError("ROS QoS depth must be at least 1")

        try:
            import rclpy
            from std_msgs.msg import String
        except ImportError as error:
            raise Ros2UnavailableError(
                "ROS 2 publishing requires rclpy and std_msgs. Source the ROS "
                "2 environment before running, or omit --ros-topic."
            ) from error

        self._rclpy = rclpy
        self._string_type = String
        self._owns_context = not rclpy.ok()
        if self._owns_context:
            # Do not let rclpy parse chess-perception command-line arguments.
            rclpy.init(args=[])
        self._node = rclpy.create_node(node_name)
        self._publisher = self._node.create_publisher(String, topic, qos_depth)
        self.topic = topic
        self.node_name = node_name
        self._closed = False

    def publish(self, serialized_event: str) -> None:
        if self._closed:
            raise RuntimeError("ROS 2 publisher is closed")
        message = self._string_type()
        message.data = serialized_event
        self._publisher.publish(message)
        # Give the middleware an opportunity to handle discovery and outbound
        # work without blocking the camera loop.
        self._rclpy.spin_once(self._node, timeout_sec=0.0)

    def close(self) -> None:
        if self._closed:
            return
        self._node.destroy_node()
        if self._owns_context and self._rclpy.ok():
            self._rclpy.shutdown()
        self._closed = True

    def __enter__(self) -> "Ros2JsonPublisher":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
