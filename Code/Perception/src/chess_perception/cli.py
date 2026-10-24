"""Command-line interface for camera and image-based board localization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

from .board import BoardNotFoundError, BoardRectifier, RectificationResult
from .config import BoardConfig, CORNER_NAMES, load_board_config
from .evaluation import evaluate_event_log, load_json_lines, write_evaluation_report
from .grid_calibration import (
    GRID_CORNER_NAMES,
    GridCalibration,
    load_grid_calibration,
    save_grid_calibration,
)
from .live import capture_empty_board_model, track_moves
from .opponent import StockfishOpponent, StockfishUnavailableError
from .occupancy import EmptyBoardModel, draw_occupancy_overlay


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chess-perception")
    subparsers = parser.add_subparsers(dest="command", required=True)

    image_parser = subparsers.add_parser("image", help="process one board image")
    image_parser.add_argument("image", type=Path)
    image_parser.add_argument("--config", type=Path, default=Path("config/board.toml"))
    image_parser.add_argument("--output-dir", type=Path, default=Path("artifacts/debug"))
    image_parser.add_argument(
        "--grid-calibration",
        type=Path,
        default=Path("config/grid_calibration.json"),
    )

    camera_parser = subparsers.add_parser("camera", help="process a USB camera")
    camera_parser.add_argument("--device", type=int, default=0)
    camera_parser.add_argument("--config", type=Path, default=Path("config/board.toml"))
    camera_parser.add_argument("--output-dir", type=Path, default=Path("artifacts/debug"))
    camera_parser.add_argument(
        "--grid-calibration",
        type=Path,
        default=Path("config/grid_calibration.json"),
    )
    camera_parser.add_argument(
        "--empty-model",
        type=Path,
        default=Path("config/empty_board.npz"),
        help="optional empty-board model used to overlay occupancy",
    )

    calibrate_parser = subparsers.add_parser(
        "calibrate-grid", help="click the four corners of the playable 8x8 grid"
    )
    calibrate_parser.add_argument("--device", type=int, default=0)
    calibrate_parser.add_argument(
        "--config", type=Path, default=Path("config/board.toml")
    )
    calibrate_parser.add_argument(
        "--output",
        type=Path,
        default=Path("config/grid_calibration.json"),
    )

    empty_parser = subparsers.add_parser(
        "capture-empty", help="capture a multi-frame empty-board reference"
    )
    empty_parser.add_argument("--device", type=int, default=0)
    empty_parser.add_argument(
        "--config", type=Path, default=Path("config/board.toml")
    )
    empty_parser.add_argument(
        "--grid-calibration",
        type=Path,
        default=Path("config/grid_calibration.json"),
    )
    empty_parser.add_argument(
        "--output", type=Path, default=Path("config/empty_board.npz")
    )
    empty_parser.add_argument("--frames", type=int, default=30)
    empty_parser.add_argument(
        "--roi",
        nargs=4,
        type=float,
        metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
        default=(0.05, 0.05, 0.95, 0.95),
        help=(
            "normalized occupancy region inside each square; default: "
            "0.05 0.05 0.95 0.95 (nearly the full square)"
        ),
    )

    track_parser = subparsers.add_parser(
        "track-moves",
        help="infer legal moves from stable before/after board observations",
    )
    track_parser.add_argument("--device", type=int, default=0)
    track_parser.add_argument(
        "--config", type=Path, default=Path("config/board.toml")
    )
    track_parser.add_argument(
        "--grid-calibration",
        type=Path,
        default=Path("config/grid_calibration.json"),
    )
    track_parser.add_argument(
        "--empty-model",
        type=Path,
        default=Path("config/empty_board.npz"),
    )
    track_parser.add_argument(
        "--fen",
        default="start",
        help="expected initial FEN or 'start' for the normal starting position",
    )
    track_parser.add_argument("--auto-accept", action="store_true")
    track_parser.add_argument("--minimum-confidence", type=float, default=0.72)
    track_parser.add_argument("--event-log", type=Path)
    track_parser.add_argument(
        "--stockfish",
        action="store_true",
        help=(
            "make the human White and use Stockfish for every Black reply; "
            "the reply is emitted before camera verification"
        ),
    )
    track_parser.add_argument(
        "--stockfish-path",
        default="stockfish",
        help="Stockfish executable or command name (default: stockfish)",
    )
    track_parser.add_argument(
        "--stockfish-elo",
        type=int,
        default=1200,
        help="requested Stockfish Elo (default: 1200)",
    )
    track_parser.add_argument(
        "--stockfish-time",
        type=float,
        default=0.2,
        help="seconds allowed for each Stockfish reply (default: 0.2)",
    )
    track_parser.add_argument(
        "--state-file",
        type=Path,
        help=(
            "atomically write the latest stable occupancy observation as JSON; "
            "used by the ROS verification bridge"
        ),
    )
    track_parser.add_argument(
        "--state-interval",
        type=float,
        default=0.2,
        help="minimum seconds between state-file updates (default: 0.2)",
    )
    track_parser.add_argument(
        "--ros-topic",
        help=(
            "publish tracker JSON events in std_msgs/msg/String on this ROS 2 "
            "topic"
        ),
    )
    track_parser.add_argument(
        "--ros-node-name",
        default="chessmate_perception",
        help="ROS 2 node name (default: chessmate_perception)",
    )
    track_parser.add_argument(
        "--ros-qos-depth",
        type=int,
        default=10,
        help="ROS 2 publisher queue depth (default: 10)",
    )
    track_parser.add_argument(
        "--evaluation-plan",
        type=Path,
        help="ordered JSONL trial plan; enables the e-to-record evaluation mode",
    )
    track_parser.add_argument(
        "--evaluation-log",
        type=Path,
        default=Path("artifacts/evaluation/trials.jsonl"),
        help="evaluation trial results (default: artifacts/evaluation/trials.jsonl)",
    )

    list_parser = subparsers.add_parser(
        "list-cameras", help="probe camera device indices available to OpenCV"
    )
    list_parser.add_argument(
        "--max-devices",
        type=int,
        default=8,
        help="probe indices from 0 up to this value (default: 8)",
    )

    marker_parser = subparsers.add_parser(
        "generate-markers", help="create printable marker PNG files"
    )
    marker_parser.add_argument("--config", type=Path, default=Path("config/board.toml"))
    marker_parser.add_argument("--output-dir", type=Path, default=Path("artifacts/markers"))
    marker_parser.add_argument("--size-px", type=int, default=800)

    evaluate_parser = subparsers.add_parser(
        "evaluate-events",
        help="compare an accepted-move JSONL log with a labeled JSONL plan",
    )
    evaluate_parser.add_argument("--event-log", type=Path, required=True)
    evaluate_parser.add_argument("--expected", type=Path, required=True)
    evaluate_parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/evaluation/report.json"),
    )
    evaluate_parser.add_argument(
        "--confidence-threshold", type=float, default=0.72
    )
    return parser


def _save_result(result: RectificationResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "detected_markers.png": result.debug_original,
        "rectified_board.png": result.rectified,
        "rectified_grid.png": result.debug_grid,
    }
    for name, image in outputs.items():
        path = output_dir / name
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"failed to write {path}")
    print(f"Saved debug output to {output_dir.resolve()}")


def _run_image(
    config: BoardConfig,
    image_path: Path,
    output_dir: Path,
    grid_calibration: GridCalibration | None,
) -> int:
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Could not read image: {image_path}", file=sys.stderr)
        return 2
    try:
        result = BoardRectifier(config, grid_calibration).rectify(image)
    except BoardNotFoundError as error:
        print(f"Board detection failed: {error}", file=sys.stderr)
        return 3
    _save_result(result, output_dir)
    return 0


def _run_camera(
    config: BoardConfig,
    device: int,
    output_dir: Path,
    grid_calibration: GridCalibration | None,
    empty_model: EmptyBoardModel | None,
) -> int:
    capture = _open_camera(device)
    if not capture.isOpened():
        print(f"Could not open camera device {device}", file=sys.stderr)
        return 2

    rectifier = BoardRectifier(config, grid_calibration)
    latest_result: RectificationResult | None = None
    latest_observation = None
    latest_occupancy_preview: np.ndarray | None = None
    print("Press s to save debug images; press q to quit.")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("Camera frame capture failed", file=sys.stderr)
                return 4
            try:
                latest_result = rectifier.rectify(frame)
                if empty_model is None:
                    preview = latest_result.debug_grid
                    latest_observation = None
                    latest_occupancy_preview = None
                else:
                    latest_observation = empty_model.observe(
                        latest_result.rectified, config
                    )
                    latest_occupancy_preview = draw_occupancy_overlay(
                        latest_result.rectified, latest_observation, config
                    )
                    preview = latest_occupancy_preview
            except BoardNotFoundError as error:
                preview = frame.copy()
                cv2.putText(
                    preview,
                    str(error),
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("ChessMate board", preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return 0
            if key == ord("s") and latest_result is not None:
                _save_result(latest_result, output_dir)
                if (
                    latest_observation is not None
                    and latest_occupancy_preview is not None
                ):
                    occupancy_image = output_dir / "occupancy_overlay.png"
                    if not cv2.imwrite(
                        str(occupancy_image), latest_occupancy_preview
                    ):
                        raise RuntimeError(f"failed to write {occupancy_image}")
                    observation_path = output_dir / "board_observation.json"
                    observation_path.write_text(
                        json.dumps(latest_observation.as_dict(), indent=2) + "\n"
                    )
    finally:
        capture.release()
        cv2.destroyAllWindows()


def _calibrate_grid(config: BoardConfig, device: int, output: Path) -> int:
    capture = _open_camera(device)
    if not capture.isOpened():
        print(f"Could not open camera device {device}", file=sys.stderr)
        return 2

    rectifier = BoardRectifier(config)
    frozen: np.ndarray | None = None
    print("Place an empty board in view. Press SPACE to freeze a frame; q quits.")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("Camera frame capture failed", file=sys.stderr)
                return 4

            latest_rectified: np.ndarray | None = None
            try:
                result = rectifier.rectify(frame)
                latest_rectified = result.rectified
                preview = result.rectified
                status = "SPACE: freeze this marker-rectified frame"
            except BoardNotFoundError as error:
                preview = frame.copy()
                status = str(error)

            display = preview.copy()
            cv2.putText(
                display,
                status,
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Grid calibration", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return 0
            if key == ord(" ") and latest_rectified is not None:
                frozen = latest_rectified.copy()
                break
    finally:
        capture.release()

    if frozen is None:
        cv2.destroyAllWindows()
        return 1

    selected: list[tuple[int, int]] = []

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(selected) < 4:
            selected.append((x, y))

    cv2.setMouseCallback("Grid calibration", on_mouse)
    print(
        "Click playable-grid corners in order: top-left, top-right, "
        "bottom-right, bottom-left. Press r to reset, s to save, q to cancel."
    )
    try:
        while True:
            display = frozen.copy()
            for index, point in enumerate(selected):
                cv2.circle(display, point, 7, (0, 0, 255), -1)
                cv2.putText(
                    display,
                    GRID_CORNER_NAMES[index],
                    (point[0] + 8, point[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            if len(selected) > 1:
                cv2.polylines(
                    display,
                    [np.asarray(selected, dtype=np.int32).reshape((-1, 1, 2))],
                    len(selected) == 4,
                    (0, 255, 0),
                    2,
                )
            cv2.imshow("Grid calibration", display)
            key = cv2.waitKey(20) & 0xFF
            if key == ord("q"):
                return 0
            if key == ord("r"):
                selected.clear()
            if key == ord("s"):
                if len(selected) != 4:
                    print("Select all four grid corners before saving.")
                    continue
                calibration = GridCalibration.from_pixel_corners(
                    np.asarray(selected, dtype=np.float32),
                    config.output_size_px,
                )
                save_grid_calibration(calibration, output)
                print(f"Saved playable-grid calibration to {output.resolve()}")
                return 0
    finally:
        cv2.destroyAllWindows()


def _open_camera(device: int) -> cv2.VideoCapture:
    """Open a camera using the native macOS backend when applicable."""

    if sys.platform == "darwin":
        return cv2.VideoCapture(device, cv2.CAP_AVFOUNDATION)
    return cv2.VideoCapture(device)


def _list_cameras(max_devices: int) -> int:
    if max_devices < 1:
        print("--max-devices must be at least 1", file=sys.stderr)
        return 2

    print("Probing camera indices (a camera light may turn on briefly)...")
    found = 0
    for device in range(max_devices):
        capture = _open_camera(device)
        try:
            if not capture.isOpened():
                continue
            ok, frame = capture.read()
            if not ok or frame is None:
                print(f"  device {device}: opened, but no frame was received")
                continue
            height, width = frame.shape[:2]
            fps = capture.get(cv2.CAP_PROP_FPS)
            fps_text = f", {fps:.1f} fps" if fps > 0 else ""
            print(f"  device {device}: {width}x{height}{fps_text}")
            found += 1
        finally:
            capture.release()

    if found == 0:
        print(
            "No cameras produced frames. Check the USB connection and allow "
            "camera access for Terminal in System Settings > Privacy & Security > Camera.",
            file=sys.stderr,
        )
        return 1
    return 0


def _generate_markers(config: BoardConfig, output_dir: Path, size_px: int) -> int:
    if size_px < 100:
        print("Marker size must be at least 100 px", file=sys.stderr)
        return 2
    dictionary_id = getattr(cv2.aruco, config.aruco_dictionary, None)
    if dictionary_id is None:
        print(f"Unknown dictionary: {config.aruco_dictionary}", file=sys.stderr)
        return 2
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    for corner_name in CORNER_NAMES:
        marker_id = config.markers[corner_name].marker_id
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, size_px)
        path = output_dir / f"marker_{marker_id}_{corner_name}.png"
        if not cv2.imwrite(str(path), marker):
            raise RuntimeError(f"failed to write {path}")
    print(f"Saved marker images to {output_dir.resolve()}")
    return 0


def _optional_grid_calibration(path: Path) -> GridCalibration | None:
    if not path.exists():
        print(
            f"No playable-grid calibration at {path}; using marker corners. "
            "Run 'chess-perception calibrate-grid' for accurate square crops."
        )
        return None
    try:
        calibration = load_grid_calibration(path)
    except (OSError, ValueError) as error:
        raise ValueError(f"invalid grid calibration {path}: {error}") from error
    print(f"Loaded playable-grid calibration from {path}")
    return calibration


def _required_grid_calibration(path: Path) -> GridCalibration:
    if not path.exists():
        raise ValueError(
            f"playable-grid calibration not found at {path}; run "
            "'chess-perception calibrate-grid' first"
        )
    calibration = _optional_grid_calibration(path)
    if calibration is None:
        raise AssertionError("required calibration unexpectedly missing")
    return calibration


def main() -> int:
    args = _parser().parse_args()
    if args.command == "list-cameras":
        return _list_cameras(args.max_devices)
    if args.command == "evaluate-events":
        try:
            report = evaluate_event_log(
                args.event_log,
                args.expected,
                args.confidence_threshold,
            )
            write_evaluation_report(report, args.output)
        except (OSError, ValueError) as error:
            print(f"Evaluation error: {error}", file=sys.stderr)
            return 2
        print(json.dumps(report, indent=2, sort_keys=True))
        print(f"Saved evaluation report to {args.output.resolve()}")
        return 0

    try:
        config = load_board_config(args.config)
    except (OSError, ValueError) as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2

    if args.command == "image":
        try:
            grid_calibration = _optional_grid_calibration(args.grid_calibration)
        except ValueError as error:
            print(str(error), file=sys.stderr)
            return 2
        return _run_image(
            config, args.image, args.output_dir, grid_calibration
        )
    if args.command == "camera":
        try:
            grid_calibration = _optional_grid_calibration(args.grid_calibration)
            empty_model = (
                EmptyBoardModel.load(args.empty_model)
                if args.empty_model.exists()
                else None
            )
        except ValueError as error:
            print(str(error), file=sys.stderr)
            return 2
        if empty_model is not None:
            if grid_calibration is None:
                print(
                    "An empty-board model requires the playable-grid "
                    "calibration used to create it.",
                    file=sys.stderr,
                )
                return 2
            print(f"Loaded empty-board model from {args.empty_model}")
        else:
            print(
                f"No empty-board model at {args.empty_model}; showing grid only."
            )
        return _run_camera(
            config,
            args.device,
            args.output_dir,
            grid_calibration,
            empty_model,
        )
    if args.command == "calibrate-grid":
        return _calibrate_grid(config, args.device, args.output)
    if args.command == "capture-empty":
        try:
            grid_calibration = _required_grid_calibration(args.grid_calibration)
        except ValueError as error:
            print(str(error), file=sys.stderr)
            return 2
        return capture_empty_board_model(
            config,
            grid_calibration,
            args.device,
            args.output,
            frame_count=args.frames,
            roi_normalized=tuple(args.roi),
        )
    if args.command == "track-moves":
        try:
            grid_calibration = _required_grid_calibration(args.grid_calibration)
            empty_model = EmptyBoardModel.load(args.empty_model)
        except (OSError, ValueError) as error:
            print(f"Tracker configuration error: {error}", file=sys.stderr)
            return 2
        evaluation_plan = None
        if args.evaluation_plan is not None:
            try:
                evaluation_plan = load_json_lines(args.evaluation_plan)
            except (OSError, ValueError) as error:
                print(f"Evaluation plan error: {error}", file=sys.stderr)
                return 2
        event_publisher = None
        opponent = None
        if args.ros_topic is not None:
            from .ros2_publisher import (
                Ros2JsonPublisher,
                Ros2UnavailableError,
            )

            try:
                event_publisher = Ros2JsonPublisher(
                    topic=args.ros_topic,
                    node_name=args.ros_node_name,
                    qos_depth=args.ros_qos_depth,
                )
            except (Ros2UnavailableError, ValueError) as error:
                print(f"ROS 2 configuration error: {error}", file=sys.stderr)
                return 2
            print(
                f"Publishing tracker events on ROS 2 topic "
                f"{event_publisher.topic}"
            )
        if args.stockfish:
            try:
                opponent = StockfishOpponent(
                    executable=args.stockfish_path,
                    requested_elo=args.stockfish_elo,
                    think_time_seconds=args.stockfish_time,
                ).start()
            except (StockfishUnavailableError, ValueError) as error:
                if event_publisher is not None:
                    event_publisher.close()
                print(f"Stockfish configuration error: {error}", file=sys.stderr)
                return 2
            strength = opponent.strength
            if strength is None:
                raise AssertionError("Stockfish strength unexpectedly missing")
            qualifier = "exact" if strength.exact else "closest supported"
            print(
                f"Stockfish opponent ready: requested Elo {strength.requested_elo}, "
                f"applied UCI Elo {strength.applied_uci_elo} ({qualifier})"
            )
        try:
            try:
                return track_moves(
                    config=config,
                    grid_calibration=grid_calibration,
                    empty_model=empty_model,
                    device=args.device,
                    fen=args.fen,
                    auto_accept=args.auto_accept,
                    minimum_confidence=args.minimum_confidence,
                    event_log=args.event_log,
                    event_publisher=event_publisher,
                    state_file=args.state_file,
                    state_interval_seconds=args.state_interval,
                    evaluation_plan=evaluation_plan,
                    evaluation_log=args.evaluation_log,
                    opponent=opponent,
                )
            except StockfishUnavailableError as error:
                print(f"Stockfish runtime error: {error}", file=sys.stderr)
                return 5
        finally:
            if opponent is not None:
                opponent.close()
            if event_publisher is not None:
                event_publisher.close()
    if args.command == "generate-markers":
        return _generate_markers(config, args.output_dir, args.size_px)
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
