# ChessMate Team 1: Perception

This package implements the camera side of ChessMate. The first milestone is
deliberately small and testable:

1. detect four ArUco markers around the chessboard;
2. rectify the board into a stable top-down image;
3. overlay a labeled 8x8 grid; and
4. save debug images for calibration and later occupancy detection.

The core vision code is independent of ROS. Move tracking can optionally
publish accepted events through ROS 2 while remaining usable on computers
without ROS installed.

## Marker layout

Use markers 0-3, aligned with the chessboard and placed outside its corners:

```text
marker 0 (top-left)          marker 1 (top-right)
          +------------------+
          |    chessboard    |
          +------------------+
marker 3 (bottom-left)       marker 2 (bottom-right)
```

The configured marker corner touching the board is used as the corresponding
board corner. The default corner indices are based on OpenCV's canonical
ArUco order: top-left=0, top-right=1, bottom-right=2, bottom-left=3.

## Setup

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

`opencv-contrib-python` is required because the ArUco module is part of the
OpenCV contrib distribution.

## Generate markers

```bash
chess-perception generate-markers --output-dir artifacts/markers
```

Print the markers without scaling them non-uniformly. Measure and record their
physical side length before later pose-calibration work.

## Process one image

```bash
chess-perception image path/to/board.jpg \
  --config config/board.toml \
  --output-dir artifacts/debug
```

Outputs:

- `detected_markers.png`: marker detections on the original image;
- `rectified_board.png`: normalized top-down board image;
- `rectified_grid.png`: board image with labeled square boundaries.

## Use a live USB camera

First list the camera indices visible to OpenCV:

```bash
chess-perception list-cameras
```

Camera indices depend on connection order. On the current setup the robot
camera is device `0`; rerun `list-cameras` after reconnecting devices:

```bash
chess-perception camera --device 0 --config config/board.toml
```

Press `s` to save the current debug images and `q` to quit.

On macOS, allow camera access for the terminal application running the command
under **System Settings > Privacy & Security > Camera**. If the external camera
does not appear, reconnect it directly rather than through an unpowered hub,
then rerun `list-cameras`.

## Calibrate the playable 8x8 grid

ArUco markers establish a stable outer coordinate system, but a chessboard may
have a labeled border between those markers and the playable squares. Calibrate
that inset before building occupancy detection:

```bash
chess-perception calibrate-grid --device 0
```

1. Wait until all four markers are detected.
2. Press Space to freeze a clear, empty-board frame.
3. Click the playable-grid corners in this exact order: top-left (`a8`),
   top-right (`h8`), bottom-right (`h1`), bottom-left (`a1`).
4. Press `s` to save, `r` to reset incorrect clicks, or `q` to cancel.

The calibration is saved to `config/grid_calibration.json`. Subsequent `camera`
and `image` commands load it automatically and warp only the playable 8x8 area
before drawing the square grid.

## Tests

```bash
python -m pytest -q
```

## Continue through the full perception pipeline

The complete calibration, occupancy, move-tracking, robot-verification, and
evaluation sequence is documented in
[`docs/PERCEPTION_RUNBOOK.md`](docs/PERCEPTION_RUNBOOK.md).
The file/JSON contract for the controls team is documented in
[`docs/TEAM2_INTERFACE.md`](docs/TEAM2_INTERFACE.md).
The physical evaluation and ROS handoff procedure is in
[`docs/FINAL_VALIDATION.md`](docs/FINAL_VALIDATION.md).

Main commands:

```bash
chess-perception calibrate-grid --device 0
chess-perception capture-empty --device 0 --frames 30
chess-perception camera --device 0
chess-perception track-moves --device 0 --fen start --stockfish \
  --event-log artifacts/events/moves.jsonl
```

With `--stockfish`, the human is always White. An accepted human move emits a
`move_detected` event followed by an `engine_move_requested` Black reply. The
tracker will not accept another human move until the camera emits
`robot_move_verified` for the physical robot result.

The requested target is Elo 1200. Stockfish 19 exposes a calibrated UCI range
starting at 1320, so events explicitly record both `requested_elo: 1200` and
`applied_uci_elo: 1320` rather than claiming an unsupported exact rating.

To publish accepted moves as JSON using `std_msgs/msg/String`, first source a
ROS 2 installation that provides `rclpy` and `std_msgs`, then run:

```bash
chess-perception track-moves --device 0 --fen start \
  --event-log artifacts/events/moves.jsonl \
  --ros-topic /chessmate/perception/move
```

## Acceptance gate for milestone 1

- All four markers are detected consistently.
- The rectified board always has `a8` at the top-left and `h1` at the
  bottom-right.
- Grid lines visually align with all 64 squares.
- Rectification remains stable across at least 100 captured frames.

Implemented follow-on milestones include the empty-board appearance model,
per-square occupancy probabilities, temporal stability gating, legal-move
inference, capture/castling handling, JSON move events, and commanded-move
verification. Do not start piece classification until the geometric and
occupancy pipeline is reliable on the final camera mount.
