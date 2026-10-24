# Team 1 to Team 2 interface

Team 1 owns the synchronized `python-chess` state, detects the White human
move, asks Stockfish for Black's response, and verifies the resulting physical
board. Team 2 owns grasp selection and arm motion for the requested Black move.

## Move event

`chess-perception track-moves` prints one JSON object per accepted move, can
append the same objects to a JSON Lines file, and can publish the identical
serialized object through ROS 2.

Example:

```json
{
  "event": "move_detected",
  "uci": "e2e4",
  "san": "e4",
  "confidence": 0.96,
  "loss": 0.02,
  "runner_up_gap": 0.18,
  "expected_changed_squares": ["e2", "e4"],
  "occupancy_mismatches": [],
  "fen_before": "...",
  "fen_after": "..."
}
```

Team 2 must validate `uci` against its own current `python-chess` board before
accepting it. `confidence` is a perception score, not proof of legality or
physical success.

## Stockfish robot command

Run the tracker with `--stockfish`. The human is always White and the first
accepted event is therefore a White `move_detected`. The next event is the
Black command Team 2 must execute:

```json
{
  "event": "engine_move_requested",
  "actor": "robot",
  "color": "black",
  "uci": "e7e5",
  "san": "e5",
  "engine": "Stockfish 19",
  "requested_elo": 1200,
  "applied_uci_elo": 1320,
  "strength_mode": "nearest_supported_uci_elo",
  "exact_elo": false,
  "fen_before": "...",
  "fen_after": "..."
}
```

Stockfish 19 does not expose native UCI settings below 1320, so the requested
1200 target is clamped visibly to its closest supported value. Team 2 should
command only `engine_move_requested` events and should reject a command whose
`fen_before` does not match its synchronized state.

After the arm completes and retreats, Team 1 emits `robot_move_verified` with
the same `uci` and commits the Black move. Until that event, no new human move
is accepted.

## Board observation

Pressing `s` in the `camera` command after an empty model is loaded writes:

- `artifacts/debug/occupancy_overlay.png`;
- `artifacts/debug/board_observation.json`.

The JSON contains occupied squares plus per-square occupancy probability,
appearance-change score, and threshold. This is the initial file-based schema
for integration testing; it can later become a ROS message without changing
the core vision code.

## Robot-move verification

After the arm has retreated and the image is stable, Team 2 can call:

```python
from chess_perception.move_inference import verify_move_result

result = verify_move_result(board_before, commanded_move, occupancy_probabilities)
if result.success:
    board_before.push(commanded_move)
else:
    print(result.mismatched_squares)
```

The logical board must be committed only after verification succeeds. A
capture should be checked after the captured piece has been removed and the
moving piece has been placed on the destination.

## ROS integration

The tracker can publish accepted moves on a configurable ROS 2 topic using
`std_msgs/msg/String`. The message's `data` field contains exactly the JSON
object documented above.

Source the ROS 2 environment, start Team 1 with:

```bash
chess-perception track-moves --device 0 --fen start \
  --stockfish --stockfish-elo 1200 \
  --event-log /Users/pesy/data/chessmate_events/moves.jsonl \
  --state-file /Users/pesy/data/chessmate_events/observation.json
```

Inspect messages from another ROS 2 terminal with:

```bash
ros2 topic echo /chessmate/engine/move std_msgs/msg/String
```

A minimal Team 2 callback is:

```python
import json
from std_msgs.msg import String

def on_engine_move(message: String) -> None:
    event = json.loads(message.data)
    if event["event"] != "engine_move_requested":
        return
    uci = event["uci"]
    # Validate fen_before and uci, then command the robot arm.
```

When running directly in a sourced ROS environment, `--ros-topic` can publish
all three event types on one JSON event-bus topic. In the current Mac + Docker
setup, use the shared JSONL file instead; the bridge routes each type to its
dedicated topic.

Board-observation publishing and a commanded-move verification service remain
available through the existing Jazzy container bridge. The Mac tracker writes:

```bash
--event-log /Users/pesy/data/chessmate_events/moves.jsonl \
--state-file /Users/pesy/data/chessmate_events/observation.json
```

Inside `ros2-jazzy`, start both endpoints with:

```bash
source /opt/ros/jazzy/setup.bash
source /home/ubuntu/data/chessmate_ros2_ws/install/setup.bash
ros2 launch chessmate_perception_bridge perception_bridge.launch.py
```

The `/chessmate/perception/verify_board` service accepts an expected
post-command FEN, maximum observation age, and allowed mismatch count. It
returns success, observation age, and the exact mismatched squares. The OpenCV,
occupancy, and legal-move modules remain unit-testable outside ROS.

Bridge topic routing:

- `/chessmate/perception/move`: accepted White `move_detected` events;
- `/chessmate/engine/move`: Black `engine_move_requested` robot commands;
- `/chessmate/perception/robot_move_verified`: confirmed robot results.
