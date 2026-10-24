"""Interactive live-camera workflows for perception calibration and testing."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Protocol

import chess
import cv2

from .board import BoardNotFoundError, BoardRectifier
from .config import BoardConfig
from .grid_calibration import GridCalibration
from .move_inference import (
    MoveCandidate,
    infer_legal_move,
    observation_matches_board,
    verify_move_result,
)
from .opponent import EngineMove
from .occupancy import (
    DEFAULT_OCCUPANCY_ROI,
    EmptyBoardModel,
    OccupancyObservation,
    draw_occupancy_overlay,
    square_change_probabilities,
)
from .stability import FrameStabilityDetector


class EventPublisher(Protocol):
    def publish(self, serialized_event: str) -> None: ...


class Opponent(Protocol):
    def choose_move(self, board: chess.Board) -> EngineMove: ...


def _emit_event(
    event: dict[str, object],
    event_log: Path | None,
    event_publisher: EventPublisher | None,
) -> None:
    serialized = json.dumps(event, sort_keys=True)
    print(serialized, flush=True)
    if event_log is not None:
        event_log.parent.mkdir(parents=True, exist_ok=True)
        with event_log.open("a") as stream:
            stream.write(serialized + "\n")
    if event_publisher is not None:
        event_publisher.publish(serialized)


def write_observation_state(
    path: Path,
    logical_fen: str,
    observation: OccupancyObservation,
    motion_score: float,
    timestamp: float | None = None,
) -> None:
    """Atomically write the latest stable board observation for integration."""

    payload = {
        "schema_version": 1,
        "timestamp": time.time() if timestamp is None else timestamp,
        "stable": True,
        "motion_score": motion_score,
        "logical_fen": logical_fen,
        "observation": observation.as_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w") as stream:
        json.dump(payload, stream, sort_keys=True)
        stream.write("\n")
    temporary_path.replace(path)


def open_camera(device: int) -> cv2.VideoCapture:
    if sys.platform == "darwin":
        return cv2.VideoCapture(device, cv2.CAP_AVFOUNDATION)
    return cv2.VideoCapture(device)


def _status_line(image, text: str, color=(0, 0, 255)) -> None:
    cv2.rectangle(image, (0, 0), (image.shape[1], 42), (255, 255, 255), -1)
    cv2.putText(
        image,
        text,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        color,
        2,
        cv2.LINE_AA,
    )


def capture_empty_board_model(
    config: BoardConfig,
    grid_calibration: GridCalibration,
    device: int,
    output: Path,
    frame_count: int = 30,
    roi_normalized: tuple[
        float, float, float, float
    ] = DEFAULT_OCCUPANCY_ROI,
) -> int:
    """Interactively collect stable empty-board frames and save a model."""

    if frame_count < 5:
        print("At least 5 frames are required", file=sys.stderr)
        return 2
    capture = open_camera(device)
    if not capture.isOpened():
        print(f"Could not open camera device {device}", file=sys.stderr)
        return 2

    rectifier = BoardRectifier(config, grid_calibration)
    stability = FrameStabilityDetector()
    frames = []
    collecting = False
    print(
        "Remove every piece from the board. Press SPACE to begin collecting "
        "stable frames; r resets; q quits."
    )
    try:
        while True:
            ok, camera_frame = capture.read()
            if not ok:
                print("Camera frame capture failed", file=sys.stderr)
                return 4
            try:
                result = rectifier.rectify(camera_frame)
                stability_result = stability.update(result.rectified)
                preview = result.debug_grid.copy()
                if collecting and stability_result.stable:
                    frames.append(result.rectified.copy())
                status = (
                    f"Collecting empty board: {len(frames)}/{frame_count}"
                    if collecting
                    else "SPACE: collect empty-board reference"
                )
                status += f" | motion {stability_result.motion_score:.4f}"
                _status_line(
                    preview,
                    status,
                    (0, 150, 0) if stability_result.stable else (0, 0, 255),
                )
            except BoardNotFoundError as error:
                preview = camera_frame.copy()
                _status_line(preview, str(error))
                stability.reset()

            cv2.imshow("Capture empty board", preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return 0
            if key == ord("r"):
                frames.clear()
                collecting = False
                stability.reset()
            if key == ord(" "):
                frames.clear()
                collecting = True
                stability.reset()
            if len(frames) >= frame_count:
                model = EmptyBoardModel.from_frames(
                    frames, config, roi_normalized=roi_normalized
                )
                model.save(output)
                print(f"Saved empty-board model to {output.resolve()}")
                return 0
    finally:
        capture.release()
        cv2.destroyAllWindows()


def _emit_move_event(
    candidate: MoveCandidate,
    board: chess.Board,
    event_log: Path | None,
    event_publisher: EventPublisher | None = None,
    inference_mode: str | None = None,
    stabilization_seconds: float | None = None,
) -> None:
    event = {
        "event": "move_detected",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **candidate.as_dict(board),
    }
    if inference_mode is not None:
        event["inference_mode"] = inference_mode
    if stabilization_seconds is not None:
        event["stabilization_seconds"] = round(stabilization_seconds, 6)
    _emit_event(event, event_log, event_publisher)


def _emit_engine_move_requested(
    decision: EngineMove,
    board: chess.Board,
    event_log: Path | None,
    event_publisher: EventPublisher | None,
) -> None:
    next_board = board.copy(stack=False)
    next_board.push(decision.move)
    event: dict[str, object] = {
        "event": "engine_move_requested",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "actor": "robot",
        "color": "black",
        "uci": decision.move.uci(),
        "san": decision.san,
        "fen_before": board.fen(),
        "fen_after": next_board.fen(),
        "engine": decision.engine_name,
        "requested_elo": decision.strength.requested_elo,
        "applied_uci_elo": decision.strength.applied_uci_elo,
        "strength_mode": decision.strength.mode,
        "exact_elo": decision.strength.exact,
        "think_time_seconds": decision.think_time_seconds,
    }
    _emit_event(event, event_log, event_publisher)


def _emit_robot_move_verified(
    decision: EngineMove,
    board: chess.Board,
    event_log: Path | None,
    event_publisher: EventPublisher | None,
) -> None:
    next_board = board.copy(stack=False)
    next_board.push(decision.move)
    _emit_event(
        {
            "event": "robot_move_verified",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "actor": "robot",
            "color": "black",
            "uci": decision.move.uci(),
            "san": decision.san,
            "fen_before": board.fen(),
            "fen_after": next_board.fen(),
        },
        event_log,
        event_publisher,
    )


def _accept_human_move(
    candidate: MoveCandidate,
    board: chess.Board,
    event_log: Path | None,
    event_publisher: EventPublisher | None,
    inference_mode: str | None,
    stabilization_seconds: float | None,
    opponent: Opponent | None,
) -> EngineMove | None:
    _emit_move_event(
        candidate,
        board,
        event_log,
        event_publisher,
        inference_mode,
        stabilization_seconds,
    )
    board.push(candidate.move)
    if opponent is None or board.is_game_over():
        return None
    decision = opponent.choose_move(board)
    _emit_engine_move_requested(decision, board, event_log, event_publisher)
    return decision


def _validate_evaluation_plan(
    board: chess.Board,
    plan: list[dict[str, object]],
) -> str | None:
    validation_board = board.copy(stack=False)
    for index, trial in enumerate(plan, start=1):
        uci = trial.get("uci")
        if not isinstance(uci, str):
            return f"evaluation trial {index} is missing string field 'uci'"
        expected_fen = trial.get("fen_before")
        if expected_fen is not None and expected_fen != validation_board.fen():
            return f"evaluation trial {index} fen_before does not match the plan"
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            return f"evaluation trial {index} has invalid UCI: {uci}"
        if move not in validation_board.legal_moves:
            return f"evaluation trial {index} is illegal: {uci}"
        validation_board.push(move)
    return None


def _emit_evaluation_trial(
    trial: dict[str, object],
    candidate: MoveCandidate | None,
    board: chess.Board,
    output: Path,
    inference_mode: str | None,
    stabilization_seconds: float | None,
) -> None:
    expected_uci = str(trial["uci"])
    observed_uci = candidate.move.uci() if candidate is not None else None
    outcome = (
        "rejected"
        if candidate is None
        else "correct"
        if observed_uci == expected_uci
        else "wrong_candidate"
    )
    event: dict[str, object] = {
        "event": "evaluation_trial",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "trial_id": trial.get("trial_id"),
        "category": trial.get("category", "unclassified"),
        "condition": trial.get("condition", "centered"),
        "fen_before": board.fen(),
        "expected_uci": expected_uci,
        "observed_uci": observed_uci,
        "outcome": outcome,
    }
    if candidate is not None:
        event.update(
            {
                "confidence": round(candidate.confidence, 6),
                "loss": round(candidate.loss, 6),
                "occupancy_loss": round(candidate.occupancy_loss, 6),
                "change_loss": round(candidate.change_loss, 6),
                "runner_up_gap": round(candidate.runner_up_gap, 6),
                "occupancy_mismatches": list(candidate.occupancy_mismatches),
                "inference_mode": inference_mode,
            }
        )
    if stabilization_seconds is not None:
        event["stabilization_seconds"] = round(stabilization_seconds, 6)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")
    print(json.dumps(event, sort_keys=True), flush=True)


def track_moves(
    config: BoardConfig,
    grid_calibration: GridCalibration,
    empty_model: EmptyBoardModel,
    device: int,
    fen: str,
    auto_accept: bool = False,
    minimum_confidence: float = 0.72,
    event_log: Path | None = None,
    event_publisher: EventPublisher | None = None,
    state_file: Path | None = None,
    state_interval_seconds: float = 0.2,
    evaluation_plan: list[dict[str, object]] | None = None,
    evaluation_log: Path | None = None,
    opponent: Opponent | None = None,
) -> int:
    """Track a known chess game and emit legal UCI move events."""

    try:
        board = chess.Board() if fen == "start" else chess.Board(fen)
    except ValueError as error:
        print(f"Invalid FEN: {error}", file=sys.stderr)
        return 2
    if opponent is not None and board.turn != chess.WHITE:
        print(
            "Stockfish mode requires a position with White (the human) to move",
            file=sys.stderr,
        )
        return 2
    if opponent is not None and evaluation_plan is not None:
        print("Stockfish mode cannot be combined with evaluation mode", file=sys.stderr)
        return 2
    if evaluation_plan is not None:
        plan_error = _validate_evaluation_plan(board, evaluation_plan)
        if plan_error is not None:
            print(f"Invalid evaluation plan: {plan_error}", file=sys.stderr)
            return 2
        if evaluation_log is None:
            print("An evaluation log is required with an evaluation plan", file=sys.stderr)
            return 2

    capture = open_camera(device)
    if not capture.isOpened():
        print(f"Could not open camera device {device}", file=sys.stderr)
        return 2

    rectifier = BoardRectifier(config, grid_calibration)
    stability = FrameStabilityDetector()
    previous_rectified = None
    current_rectified = None
    pending: MoveCandidate | None = None
    repeated_candidate_frames = 0
    previous_candidate_uci: str | None = None
    last_state_write = 0.0
    activity_started_at: float | None = None
    pending_inference_mode: str | None = None
    pending_stabilization_seconds: float | None = None
    evaluation_index = 0
    current_stable = False
    pending_robot_move: EngineMove | None = None

    print(
        "Move tracker started. Press a to accept the displayed candidate, "
        "r to re-arm from the current expected position, or q to quit."
    )
    if evaluation_plan is not None:
        print(
            "Evaluation mode: perform the planned move, wait for a stable "
            "result, then press e to record and advance."
        )
    if opponent is not None:
        print(
            "Human is White. After each accepted White move, Stockfish selects "
            "Black's reply; tracking resumes only after the camera verifies it."
        )
    try:
        while True:
            ok, camera_frame = capture.read()
            if not ok:
                print("Camera frame capture failed", file=sys.stderr)
                return 4

            try:
                result = rectifier.rectify(camera_frame)
                current_rectified = result.rectified
                stability_result = stability.update(current_rectified)
                current_stable = stability_result.stable
                observation = empty_model.observe(current_rectified, config)
                preview = draw_occupancy_overlay(
                    current_rectified, observation, config
                )
                probabilities = observation.probability_by_square()

                now = time.monotonic()
                if (
                    state_file is not None
                    and stability_result.stable
                    and now - last_state_write >= state_interval_seconds
                ):
                    write_observation_state(
                        state_file,
                        board.fen(),
                        observation,
                        stability_result.motion_score,
                    )
                    last_state_write = now

                if pending_robot_move is not None:
                    if not stability_result.stable:
                        status = (
                            f"WAIT: robot motion for {pending_robot_move.move.uci()} "
                            f"({stability_result.motion_score:.4f})"
                        )
                    else:
                        verification = verify_move_result(
                            board,
                            pending_robot_move.move,
                            probabilities,
                        )
                        if verification.success:
                            _emit_robot_move_verified(
                                pending_robot_move,
                                board,
                                event_log,
                                event_publisher,
                            )
                            verified_uci = pending_robot_move.move.uci()
                            board.push(pending_robot_move.move)
                            pending_robot_move = None
                            previous_rectified = current_rectified.copy()
                            pending = None
                            repeated_candidate_frames = 0
                            previous_candidate_uci = None
                            activity_started_at = None
                            pending_inference_mode = None
                            pending_stabilization_seconds = None
                            status = f"ROBOT MOVE VERIFIED: {verified_uci}; HUMAN READY"
                        else:
                            mismatches = ",".join(verification.mismatched_squares[:5])
                            extra = max(0, len(verification.mismatched_squares) - 5)
                            if extra:
                                mismatches += f",+{extra}"
                            status = (
                                f"ROBOT: execute {pending_robot_move.move.uci()} "
                                f"| mismatches {mismatches or 'none'}"
                            )
                elif not stability_result.stable:
                    if previous_rectified is not None and activity_started_at is None:
                        activity_started_at = now
                    status = f"WAIT: motion {stability_result.motion_score:.4f}"
                    pending = None
                    repeated_candidate_frames = 0
                elif previous_rectified is None:
                    if observation_matches_board(board, probabilities):
                        previous_rectified = current_rectified.copy()
                        status = "ARMED: board matches expected FEN"
                    else:
                        status = "SYNC ERROR: observed board does not match expected FEN"
                elif observation_matches_board(board, probabilities):
                    # Slowly refresh the pre-move image to absorb small lighting drift.
                    previous_rectified = current_rectified.copy()
                    pending = None
                    repeated_candidate_frames = 0
                    previous_candidate_uci = None
                    activity_started_at = None
                    pending_inference_mode = None
                    pending_stabilization_seconds = None
                    status = f"READY: {board.fullmove_number} {'white' if board.turn else 'black'}"
                else:
                    if activity_started_at is None:
                        activity_started_at = now
                    change_probabilities = square_change_probabilities(
                        previous_rectified,
                        current_rectified,
                        config,
                        roi_normalized=empty_model.roi_normalized,
                        crop_size_px=empty_model.crop_size_px,
                    )
                    pending = infer_legal_move(
                        board,
                        probabilities,
                        change_probabilities,
                    )
                    inference_mode = "occupancy + change"
                    if pending is None:
                        # Illumination or small camera vibration can make the
                        # before/after change map noisy even when the current
                        # occupancy clearly identifies an ordinary move.
                        pending = infer_legal_move(board, probabilities)
                        inference_mode = "occupancy fallback"
                    if pending is None:
                        status = "NO UNIQUE LEGAL MOVE: wait or correct the board"
                        if evaluation_plan is not None:
                            status += " | e: record rejection"
                        repeated_candidate_frames = 0
                        previous_candidate_uci = None
                    else:
                        uci = pending.move.uci()
                        if uci == previous_candidate_uci:
                            repeated_candidate_frames += 1
                        else:
                            repeated_candidate_frames = 1
                            previous_candidate_uci = uci
                            pending_stabilization_seconds = (
                                now - activity_started_at
                            )
                        pending_inference_mode = inference_mode
                        control = (
                            "e: record trial"
                            if evaluation_plan is not None
                            else "a: accept"
                        )
                        status = (
                            f"CANDIDATE {uci} ({pending.san}) "
                            f"confidence={pending.confidence:.2f} "
                            f"[{inference_mode}] | {control}"
                        )
                        if (
                            auto_accept
                            and pending.confidence >= minimum_confidence
                            and repeated_candidate_frames >= 5
                        ):
                            pending_robot_move = _accept_human_move(
                                pending,
                                board,
                                event_log,
                                event_publisher,
                                pending_inference_mode,
                                pending_stabilization_seconds,
                                opponent,
                            )
                            previous_rectified = current_rectified.copy()
                            pending = None
                            repeated_candidate_frames = 0
                            previous_candidate_uci = None
                            activity_started_at = None
                            pending_inference_mode = None
                            pending_stabilization_seconds = None
                            status = (
                                "AUTO-ACCEPTED; waiting for robot"
                                if pending_robot_move is not None
                                else "AUTO-ACCEPTED; waiting for next move"
                            )

                _status_line(
                    preview,
                    status,
                    (0, 150, 0) if "READY" in status or "ARMED" in status else (0, 0, 255),
                )
            except BoardNotFoundError as error:
                current_stable = False
                preview = camera_frame.copy()
                _status_line(preview, str(error))
                stability.reset()

            cv2.imshow("ChessMate move tracker", preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return 0
            if key == ord("r"):
                previous_rectified = None
                pending = None
                activity_started_at = None
                pending_inference_mode = None
                pending_stabilization_seconds = None
                stability.reset()
            if key == ord("e") and evaluation_plan is not None:
                if not current_stable or current_rectified is None:
                    print("Evaluation trial not recorded: wait for a stable frame")
                elif evaluation_index >= len(evaluation_plan):
                    print("Evaluation plan is already complete")
                else:
                    trial = evaluation_plan[evaluation_index]
                    if evaluation_log is None:
                        raise AssertionError("evaluation log unexpectedly missing")
                    _emit_evaluation_trial(
                        trial,
                        pending,
                        board,
                        evaluation_log,
                        pending_inference_mode,
                        pending_stabilization_seconds,
                    )
                    expected_move = chess.Move.from_uci(str(trial["uci"]))
                    board.push(expected_move)
                    evaluation_index += 1
                    previous_rectified = current_rectified.copy()
                    pending = None
                    repeated_candidate_frames = 0
                    previous_candidate_uci = None
                    activity_started_at = None
                    pending_inference_mode = None
                    pending_stabilization_seconds = None
                    print(
                        f"Recorded evaluation trial {evaluation_index}/"
                        f"{len(evaluation_plan)}"
                    )
                    if evaluation_index >= len(evaluation_plan):
                        print(f"Evaluation complete: {evaluation_log.resolve()}")
                        return 0
            if (
                key == ord("a")
                and evaluation_plan is None
                and pending is not None
                and current_rectified is not None
            ):
                pending_robot_move = _accept_human_move(
                    pending,
                    board,
                    event_log,
                    event_publisher,
                    pending_inference_mode,
                    pending_stabilization_seconds,
                    opponent,
                )
                previous_rectified = current_rectified.copy()
                pending = None
                repeated_candidate_frames = 0
                previous_candidate_uci = None
                activity_started_at = None
                pending_inference_mode = None
                pending_stabilization_seconds = None
    finally:
        capture.release()
        cv2.destroyAllWindows()
