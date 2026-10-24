from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    event_log = LaunchConfiguration("event_log")
    state_file = LaunchConfiguration("state_file")
    move_topic = LaunchConfiguration("move_topic")
    engine_topic = LaunchConfiguration("engine_topic")
    verified_topic = LaunchConfiguration("verified_topic")
    service_name = LaunchConfiguration("service_name")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "event_log",
                default_value="/home/ubuntu/data/chessmate_events/moves.jsonl",
            ),
            DeclareLaunchArgument(
                "state_file",
                default_value="/home/ubuntu/data/chessmate_events/observation.json",
            ),
            DeclareLaunchArgument(
                "move_topic", default_value="/chessmate/perception/move"
            ),
            DeclareLaunchArgument(
                "engine_topic", default_value="/chessmate/engine/move"
            ),
            DeclareLaunchArgument(
                "verified_topic",
                default_value="/chessmate/perception/robot_move_verified",
            ),
            DeclareLaunchArgument(
                "service_name",
                default_value="/chessmate/perception/verify_board",
            ),
            Node(
                package="chessmate_perception_bridge",
                executable="move_event_bridge",
                name="chessmate_move_event_bridge",
                output="screen",
                parameters=[
                    {
                        "event_log": event_log,
                        "move_topic": move_topic,
                        "engine_topic": engine_topic,
                        "verified_topic": verified_topic,
                        "start_at_end": True,
                    }
                ],
            ),
            Node(
                package="chessmate_perception_bridge",
                executable="verification_service",
                name="chessmate_verification_service",
                output="screen",
                parameters=[
                    {
                        "state_file": state_file,
                        "service_name": service_name,
                    }
                ],
            ),
        ]
    )
