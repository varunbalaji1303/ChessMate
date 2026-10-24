"""Verify an expected FEN against Team 1's latest stable observation."""

from __future__ import annotations

import json
from pathlib import Path
import time

from chessmate_interfaces.srv import VerifyBoard
import rclpy
from rclpy.node import Node

from .fen import occupied_squares_from_fen


class VerificationService(Node):
    def __init__(self) -> None:
        super().__init__("chessmate_verification_service")
        self.declare_parameter(
            "state_file", "/home/ubuntu/data/chessmate_events/observation.json"
        )
        self.declare_parameter(
            "service_name", "/chessmate/perception/verify_board"
        )
        self.declare_parameter("default_maximum_age_seconds", 2.0)
        self._state_file = Path(
            self.get_parameter("state_file").get_parameter_value().string_value
        )
        service_name = (
            self.get_parameter("service_name").get_parameter_value().string_value
        )
        self._default_max_age = (
            self.get_parameter("default_maximum_age_seconds")
            .get_parameter_value()
            .double_value
        )
        self.create_service(VerifyBoard, service_name, self._verify)
        self.get_logger().info(
            f"Serving {service_name} from stable state {self._state_file}"
        )

    def _failure(
        self,
        response: VerifyBoard.Response,
        message: str,
        age: float = -1.0,
    ) -> VerifyBoard.Response:
        response.success = False
        response.mismatched_squares = []
        response.observation_age_seconds = float(age)
        response.message = message
        return response

    def _verify(
        self,
        request: VerifyBoard.Request,
        response: VerifyBoard.Response,
    ) -> VerifyBoard.Response:
        try:
            payload = json.loads(self._state_file.read_text())
        except FileNotFoundError:
            return self._failure(response, f"state file not found: {self._state_file}")
        except (OSError, json.JSONDecodeError) as error:
            return self._failure(response, f"could not read state file: {error}")

        if not payload.get("stable"):
            return self._failure(response, "latest observation is not stable")
        try:
            timestamp = float(payload["timestamp"])
            observed = set(payload["observation"]["occupied_squares"])
            expected = occupied_squares_from_fen(request.expected_fen)
        except (KeyError, TypeError, ValueError) as error:
            return self._failure(response, f"invalid state or expected FEN: {error}")

        age = max(0.0, time.time() - timestamp)
        maximum_age = (
            float(request.maximum_age_seconds)
            if request.maximum_age_seconds > 0.0
            else self._default_max_age
        )
        if age > maximum_age:
            return self._failure(
                response,
                f"observation is stale ({age:.3f}s > {maximum_age:.3f}s)",
                age,
            )

        mismatches = sorted(expected.symmetric_difference(observed))
        response.success = len(mismatches) <= request.maximum_mismatches
        response.mismatched_squares = mismatches
        response.observation_age_seconds = float(age)
        response.message = (
            "board matches expected FEN"
            if response.success
            else f"board differs on {len(mismatches)} square(s)"
        )
        return response


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = VerificationService()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
