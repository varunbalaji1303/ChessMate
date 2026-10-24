# ChessMate perception runbook

This is the order to follow after the permanent camera mount is installed.
Do not skip calibration steps: every later model assumes that each square is
mapped to the same pixels in every frame.

## 1. Lock the physical setup

- Fix the camera, chessboard, markers, and robot base in place.
- Point the camera toward the center of the board.
- Keep all four ArUco markers fully visible.
- Use stable, diffuse lighting and avoid strong shadows.
- Disable autofocus, auto-framing, and digital zoom when the camera permits it.

Changing the camera, board, marker, resolution, focus, or zoom invalidates the
grid and empty-board calibration.

## 2. Activate the project

```bash
cd /Users/pesy/School/PhD/CS6341-Robotics/Project/Team1
source .venv/bin/activate
chess-perception list-cameras
```

The current robot camera is device 0.

## 3. Verify marker rectification

```bash
chess-perception camera --device 0
```

Required result:

- all four marker IDs are detected;
- `a8` is at the top-left and `h1` at the bottom-right;
- the image does not jump between frames.

## 4. Calibrate the playable grid

```bash
chess-perception calibrate-grid --device 0
```

Press Space to freeze a clear empty-board frame. Click these four points in
order:

1. top-left corner of the playable `a8` square;
2. top-right corner of the playable `h8` square;
3. bottom-right corner of the playable `h1` square;
4. bottom-left corner of the playable `a1` square.

Press `s` to save or `r` to reset. The output is
`config/grid_calibration.json`.

Reopen the camera view and confirm that every green line follows a playable
square boundary. Coordinate labels printed around the physical board must not
appear inside the rectified 8x8 image.

## 5. Capture the empty-board appearance model

Remove all pieces, keep the lighting in its normal operating state, and run:

```bash
chess-perception capture-empty --device 0 --frames 30
```

Press Space once the board is clear and still. The pipeline collects only
stable frames and writes `config/empty_board.npz`.

This model stores a median reference crop and a noise-derived threshold for
each of the 64 squares. With the camera mounted overhead, occupancy is measured
from nearly the complete square: `left=0.05, top=0.05, right=0.95,
bottom=0.95`. The 5% edge guard avoids grid-line noise caused by small
rectification errors. The cyan rectangle in the live overlay shows the region
being measured. Recreate the model after changing lighting, focus, exposure,
camera pose, board pose, grid calibration, or occupancy-region settings.

The region can be changed while recapturing the empty board. To include every
pixel in each square, use:

```bash
chess-perception capture-empty --device 0 --frames 30 \
  --roi 0 0 1 1
```

The default 5% guard is recommended because it is less sensitive to grid-line
jitter while still seeing essentially the entire piece.

The occupancy score combines color, brightness, and silhouette-edge changes.
The brightness and edge terms are important for detecting white pieces on
light squares and black pieces on dark squares.

## 6. Validate occupancy

```bash
chess-perception camera --device 0
```

When an empty model exists, the live camera automatically overlays occupancy:

- green: empty;
- red: occupied;
- number: occupied probability.

Place one piece at a time on center, edge, and corner squares. Test both piece
colors and every piece type. Tune the model only from recorded failures; do not
choose a threshold from a single favorable position.

Acceptance target: at least 99% correct square classifications over a labeled
set of representative board images.

## 7. Detect moves from a known chess position

Set up the normal initial chess position and run:

```bash
chess-perception track-moves --device 0 --fen start \
  --stockfish --stockfish-elo 1200 \
  --event-log artifacts/events/moves.jsonl
```

The tracker waits for the image to become stable, compares observed occupancy
with the logical `python-chess` state, and scores only legal moves. It also uses
per-square visual change, which allows it to distinguish captures whose target
square remains occupied. Frame-wide auto-exposure changes are removed from the
motion score to prevent stability-message flicker.

Inference first combines occupancy with the before/after change map. If that
map is noisy, an ordinary move can use the occupancy-only fallback shown in
the live status. During board synchronization, at most two probabilities in
the uncertainty range `0.42` through `0.58` are treated as unknown instead of
hard contradictions.

Controls:

- `a`: accept the displayed move candidate and advance logical state;
- `r`: re-arm against the current logical state;
- `q`: quit.

After manual validation, automatic acceptance can be tested with:

```bash
chess-perception track-moves --device 0 --fen start --auto-accept \
  --minimum-confidence 0.80 \
  --event-log artifacts/events/moves.jsonl
```

Each accepted move is printed as one JSON object. Team 2 can consume that
event format without depending on OpenCV.

In Stockfish mode, accepting the White human move immediately prints a second
`engine_move_requested` JSON event containing Black's `uci`, `san`, and the
pre/post FENs. Team 2 should execute that UCI move. The tracker then blocks new
human input until vision confirms the expected board and emits
`robot_move_verified`. Stockfish 19 cannot set native UCI Elo below 1320, so a
1200 request is transparently reported as requested 1200 / applied 1320.

To publish the same accepted event through ROS 2 as `std_msgs/msg/String`:

```bash
chess-perception track-moves --device 0 --fen start \
  --event-log artifacts/events/moves.jsonl \
  --ros-topic /chessmate/perception/move
```

This option requires a sourced ROS 2 environment with `rclpy` and `std_msgs`.
The message `data` field is the same JSON string printed to the terminal and
written to the event log.

## 8. Verify commanded robot moves

The integration layer can call `verify_move_result()` from
`chess_perception.move_inference` after the arm retreats. It compares the live
occupancy probabilities with the expected `python-chess` position and returns
the mismatched squares. Team 2 should commit its logical move only when this
verification succeeds.

## 9. Evaluation

Record and label at least:

- 20 normal moves distributed across the board;
- 10 captures;
- both king-side and queen-side castling when safe;
- pieces placed slightly off center;
- several representative lighting conditions;
- failure cases with a hand or robot arm still in view.

Report square-occupancy accuracy, move accuracy, rejection rate, false accepts,
confidence, and stabilization time. A wrong confident move is more dangerous
than a rejected move, so ambiguous observations must remain unaccepted.

Use the included labeled evaluation mode rather than estimating these values
manually:

```bash
chess-perception track-moves --device 0 --fen start \
  --evaluation-plan config/evaluation_plan.jsonl \
  --evaluation-log artifacts/evaluation/run-overhead-01.jsonl
```

After each planned move is stable, press `e`. Then create the metrics report:

```bash
chess-perception evaluate-events \
  --event-log artifacts/evaluation/run-overhead-01.jsonl \
  --expected config/evaluation_plan.jsonl \
  --output artifacts/evaluation/run-overhead-01-report.json
```

See `docs/FINAL_VALIDATION.md` for the move sequence, off-center trials, ROS
container commands, and acceptance gate.

## Scope boundary

This pipeline tracks a game from a known starting position. It does not attempt
to reconstruct an arbitrary chess position from one photograph. A 13-class
square classifier (empty plus twelve colored piece types) is only needed for
that separate capability. During a synchronized game, `python-chess` already
provides piece identity and the appropriate grasp profile.
