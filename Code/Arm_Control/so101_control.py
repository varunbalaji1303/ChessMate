#!/usr/bin/env python3
"""Cycle an SO-101 follower arm through the CS 6341 Part 2 poses.

The arrival time for each pose is measured from the call to ``send_action``
until every joint is within the configured tolerance of its target.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path
from typing import Any


JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

POSITIONS: tuple[tuple[str, dict[str, float]], ...] = (
    (
        "Position 0",
        {
            "shoulder_pan.pos": 0.0,
            "shoulder_lift.pos": -30.0,
            "elbow_flex.pos": 30.0,
            "wrist_flex.pos": 70.0,
            "wrist_roll.pos": 0.0,
            "gripper.pos": 0.0,
        },
    ),
    (
        "Position 1",
        {
            "shoulder_pan.pos": 45.0,
            "shoulder_lift.pos": -20.0,
            "elbow_flex.pos": 30.0,
            "wrist_flex.pos": 60.0,
            "wrist_roll.pos": -30.0,
            "gripper.pos": 50.0,
        },
    ),
    (
        "Position 2",
        {
            "shoulder_pan.pos": -45.0,
            "shoulder_lift.pos": -10.0,
            "elbow_flex.pos": 0.0,
            "wrist_flex.pos": -30.0,
            "wrist_roll.pos": 30.0,
            "gripper.pos": 30.0,
        },
    ),
)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run the three CS 6341 Part 2 SO-101 joint configurations."
    )
    parser.add_argument(
        "--port",
        help="Robot serial port. If omitted, a single connected USB modem port is auto-detected.",
    )
    parser.add_argument("--robot-id", default="team08_arm", help="LeRobot calibration ID.")
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=script_dir,
        help="Directory containing <robot-id>.json (default: script directory).",
    )
    parser.add_argument(
        "--mode",
        choices=("timed", "keyboard"),
        default="timed",
        help="Advance after a fixed interval or on Enter (default: timed).",
    )
    parser.add_argument("--cycles", type=int, default=1, help="Number of complete 3-pose cycles.")
    parser.add_argument(
        "--interval",
        type=float,
        default=3.0,
        help="Minimum seconds between commands in timed mode (default: 3.0).",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1.0,
        help="Absolute per-joint arrival tolerance (default: 1.0).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Maximum seconds allowed to reach a pose (default: 10.0).",
    )
    parser.add_argument(
        "--poll-rate",
        type=float,
        default=20.0,
        help="Joint-state checks per second (default: 20).",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the clear-workspace confirmation prompt.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned commands without connecting to hardware.",
    )
    args = parser.parse_args()

    if args.cycles < 1:
        parser.error("--cycles must be at least 1")
    if args.interval <= 0 or args.tolerance <= 0 or args.timeout <= 0 or args.poll_rate <= 0:
        parser.error("--interval, --tolerance, --timeout, and --poll-rate must be positive")
    return args


def detect_port(explicit_port: str | None) -> str:
    if explicit_port:
        port = Path(explicit_port)
        if not port.exists():
            raise FileNotFoundError(f"Serial port does not exist: {port}")
        return str(port)

    patterns = (
        "/dev/tty.usbmodem*",
        "/dev/ttyACM*",
        "/dev/ttyUSB*",
        "/dev/cu.usbmodem*",
    )
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise RuntimeError(
                "Multiple candidate robot ports found: " + ", ".join(matches) + ". Pass --port explicitly."
            )
    raise RuntimeError("No SO-101 serial port found. Connect and power the arm, or pass --port.")


def validate_calibration(calibration_dir: Path, robot_id: str) -> Path:
    path = calibration_dir.expanduser().resolve() / f"{robot_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Calibration file not found: {path}")
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    missing = [joint for joint in JOINTS if joint not in data]
    if missing:
        raise ValueError(f"Calibration file is missing joints: {', '.join(missing)}")
    return path


def import_robot_api() -> tuple[type[Any], type[Any]]:
    try:
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    except ImportError:
        # Compatibility with older LeRobot releases used by the assignment handout.
        from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
    return SO101Follower, SO101FollowerConfig


def joint_errors(observation: dict[str, Any], target: dict[str, float]) -> dict[str, float]:
    missing = sorted(set(target) - set(observation))
    if missing:
        raise KeyError(f"Observation is missing joint values: {', '.join(missing)}")
    return {key: abs(float(observation[key]) - desired) for key, desired in target.items()}


def print_position_table(
    label: str,
    target: dict[str, float],
    command: dict[str, float],
    observation: dict[str, Any],
    elapsed: float,
    tolerance: float,
) -> None:
    errors = joint_errors(observation, target)
    reached = all(error <= tolerance for error in errors.values())
    print("\n" + "=" * 80)
    print(f"{label}: {'TARGET REACHED' if reached else 'TIMEOUT'} in {elapsed:.3f} s")
    print(f"{'Joint':<18} {'Target':>10} {'Command':>10} {'Final':>10} {'Error':>10} {'OK':>6}")
    print("-" * 80)
    for key, desired in target.items():
        actual = float(observation[key])
        error = errors[key]
        print(
            f"{key.removesuffix('.pos'):<18} {desired:>10.2f} {command[key]:>10.2f} {actual:>10.2f} "
            f"{error:>10.2f} {('yes' if error <= tolerance else 'NO'):>6}"
        )
    print("=" * 80, flush=True)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def wait_for_target(
    robot: Any,
    target: dict[str, float],
    initial_command: dict[str, float],
    started_at: float,
    tolerance: float,
    timeout: float,
    poll_rate: float,
) -> tuple[bool, float, dict[str, Any], dict[str, float]]:
    poll_period = 1.0 / poll_rate
    observation: dict[str, Any] = {}
    command = dict(initial_command)
    next_correction_at = started_at + 1.0
    while True:
        observation = robot.get_observation()
        now = time.perf_counter()
        errors = joint_errors(observation, target)
        if all(error <= tolerance for error in errors.values()):
            return True, now - started_at, observation, command
        if now - started_at >= timeout:
            return False, now - started_at, observation, command

        # Loaded joints can settle just outside the required +/-1 degree band.
        # After the initial motion has had time to settle, apply small bounded
        # command corrections while continuing to evaluate the original target.
        if now >= next_correction_at:
            corrected_command = dict(command)
            for key, desired in target.items():
                if key == "gripper.pos" or errors[key] <= tolerance:
                    continue
                feedback_step = clamp(desired - float(observation[key]), -2.0, 2.0)
                total_adjustment = clamp(command[key] + feedback_step - desired, -5.0, 5.0)
                corrected_command[key] = desired + total_adjustment
            if corrected_command != command:
                command = robot.send_action(corrected_command)
            next_correction_at = now + 0.5
        time.sleep(poll_period)


def print_plan(args: argparse.Namespace, port: str, calibration_path: Path) -> None:
    print("CS 6341 Part 2 - SO-101 Control")
    print(f"Port:        {port}")
    print(f"Robot ID:    {args.robot_id}")
    print(f"Calibration: {calibration_path}")
    print(f"Mode:        {args.mode}")
    print(f"Cycles:      {args.cycles}")
    print(f"Tolerance:   +/-{args.tolerance:.2f}")
    print(f"Timeout:     {args.timeout:.2f} s")
    if args.mode == "timed":
        print(f"Interval:    {args.interval:.2f} s")
    for label, target in POSITIONS:
        values = ", ".join(f"{key}={value:.1f}" for key, value in target.items())
        print(f"{label}: {values}")


def main() -> int:
    args = parse_args()
    try:
        calibration_path = validate_calibration(args.calibration_dir, args.robot_id)
        port = detect_port(args.port)
    except (FileNotFoundError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print_plan(args, port, calibration_path)
    if args.dry_run:
        print("\nDry run complete; no hardware command was sent.")
        return 0

    if not args.yes:
        input(
            "\nClear the arm's workspace, verify the correct power supply, "
            "and press Enter to connect (Ctrl-C to cancel): "
        )

    try:
        SO101Follower, SO101FollowerConfig = import_robot_api()
    except ImportError as exc:
        print(
            "ERROR: LeRobot with the Feetech dependency is not installed in this Python environment.\n"
            "Run this script from the LeRobot checkout using the command in PART2_README.md.\n"
            f"Details: {exc}",
            file=sys.stderr,
        )
        return 2

    config = SO101FollowerConfig(
        port=port,
        id=args.robot_id,
        calibration_dir=args.calibration_dir.expanduser().resolve(),
        use_degrees=True,
    )
    robot = SO101Follower(config)
    connected = False
    try:
        print("\nConnecting to the SO-101...", flush=True)
        robot.connect()
        connected = True
        print("Connected. Starting pose cycle.", flush=True)

        for cycle_index in range(args.cycles):
            print(f"\n######## Cycle {cycle_index + 1}/{args.cycles} ########", flush=True)
            for label, target in POSITIONS:
                if args.mode == "keyboard":
                    input(f"Press Enter to command {label}: ")
                print(f"\nCommanding {label}...", flush=True)
                started_at = time.perf_counter()
                initial_command = robot.send_action(target)
                reached, elapsed, observation, final_command = wait_for_target(
                    robot,
                    target,
                    initial_command,
                    started_at,
                    args.tolerance,
                    args.timeout,
                    args.poll_rate,
                )
                print_position_table(
                    label,
                    target,
                    final_command,
                    observation,
                    elapsed,
                    args.tolerance,
                )
                if not reached:
                    raise TimeoutError(
                        f"{label} did not reach every target within +/-{args.tolerance:.2f} "
                        f"in {args.timeout:.2f} s; stopping for safety."
                    )
                if args.mode == "timed":
                    remaining = args.interval - (time.perf_counter() - started_at)
                    if remaining > 0:
                        time.sleep(remaining)

        print("\nAll requested cycles completed successfully.", flush=True)
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted by user; stopping.", file=sys.stderr, flush=True)
        return 130
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        if connected or robot.is_connected:
            print("Disconnecting and disabling torque...", flush=True)
            try:
                robot.disconnect()
            except Exception as exc:
                print(f"WARNING: disconnect reported an error: {exc}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
