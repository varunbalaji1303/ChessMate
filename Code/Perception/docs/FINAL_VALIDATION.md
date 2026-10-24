# ChessMate Team 1 final validation

This checklist closes the remaining perception work on the final overhead
camera mount. Do not enable automatic acceptance until the evaluation report
meets the acceptance gate at the end of this document.

## 1. Freeze the physical geometry

Secure the camera and board so neither can move during a game. With the empty
board under normal lighting, recreate both geometry-dependent artifacts:

```bash
cd /Users/pesy/School/PhD/CS6341-Robotics/Project/Team1
source .venv/bin/activate
chess-perception calibrate-grid --device 0
chess-perception capture-empty --device 0 --frames 30
chess-perception camera --device 0
```

In the final camera check, all square boundaries must align and white/black
pieces must be detected on both light and dark squares. Recalibrate after any
camera, board, focus, exposure, or lighting change.

## 2. Start the existing ROS 2 Jazzy environment

The existing Docker image is reused; do not install another ROS distribution.
It mounts `/Users/pesy/data` at `/home/ubuntu/data` and contains the built
workspace `/home/ubuntu/data/chessmate_ros2_ws`.

Start the ROS relay and verification service:

```bash
docker start ros2-jazzy
docker exec -it ros2-jazzy bash
source /opt/ros/jazzy/setup.bash
source /home/ubuntu/data/chessmate_ros2_ws/install/setup.bash
ros2 launch chessmate_perception_bridge perception_bridge.launch.py
```

On macOS, start perception with shared event and observation files:

```bash
chess-perception track-moves --device 0 --fen start \
  --stockfish --stockfish-elo 1200 \
  --event-log /Users/pesy/data/chessmate_events/moves.jsonl \
  --state-file /Users/pesy/data/chessmate_events/observation.json
```

The human is White and moves first. The relay publishes human detections on
`/chessmate/perception/move`, Stockfish Black commands for Team 2 on
`/chessmate/engine/move`, and confirmed physical completions on
`/chessmate/perception/robot_move_verified`, all as `std_msgs/msg/String`.

Stockfish 19 has a native minimum `UCI_Elo` of 1320. The command keeps the
project's requested target at 1200, applies the closest supported value 1320,
and records both values plus `exact_elo: false` in every engine request.

## 3. Confirm ROS topic delivery

In another shell inside the container:

```bash
source /opt/ros/jazzy/setup.bash
source /home/ubuntu/data/chessmate_ros2_ws/install/setup.bash
ros2 run chessmate_perception_bridge move_event_listener
```

Accept one White move in the Mac tracker. The listener must print Stockfish's
Black UCI command and post-move FEN. After Team 2 performs it, the verified
topic must publish the same UCI. The reusable automated test is:

```bash
docker exec ros2-jazzy bash \
  /home/ubuntu/data/chessmate_ros2_ws/scripts/smoke_test.sh
```

## 4. Verify a robot result through ROS

Team 2 computes the expected post-move FEN and calls:

```bash
ros2 service call /chessmate/perception/verify_board \
  chessmate_interfaces/srv/VerifyBoard \
  "{expected_fen: 'EXPECTED_FEN', maximum_age_seconds: 2.0, maximum_mismatches: 0}"
```

The service rejects missing, unstable, or stale observations and returns every
mismatched square. Team 2 commits its logical move only when `success=True`.

## 5. Run the labeled 20-move evaluation

Reset to the standard initial position and use a new result filename:

```bash
chess-perception track-moves --device 0 --fen start \
  --evaluation-plan config/evaluation_plan.jsonl \
  --evaluation-log artifacts/evaluation/run-overhead-01.jsonl \
  --state-file /Users/pesy/data/chessmate_events/observation.json
```

For every planned physical move, remove your hands, wait for a stable result,
and press `e`. Evaluation mode records a correct candidate, wrong candidate,
or rejection and then advances using the known expected move. Do not press
`a` in this mode.

Perform these moves in order:

```text
 1. e2e4     2. d7d5     3. e4d5 capture     4. d8d5 capture
 5. b1c3     6. d5d8     7. g1f3             8. g8f6
 9. f1e2    10. e7e6    11. e1g1 castle     12. f8e7
13. d2d4    14. e8g8 castle
15. c1f4    16. b8c6
17. c3b5    18. f6d5    19. f4g3            20. a7a6
```

For moves 17-20, place the destination piece slightly off center while keeping
its base entirely inside the correct square. During castling, move both king
and rook before pressing `e`.

Generate the report:

```bash
chess-perception evaluate-events \
  --event-log artifacts/evaluation/run-overhead-01.jsonl \
  --expected config/evaluation_plan.jsonl \
  --output artifacts/evaluation/run-overhead-01-report.json
```

Repeat under the normal lighting conditions expected in the demonstration.

## 6. Acceptance gate

Before enabling `--auto-accept`, require:

- zero wrong or extra accepted moves;
- all 20 scripted trials recorded;
- at least 95% total move accuracy;
- both captures and both castlings correct;
- no wrong candidate at or above the chosen auto-accept threshold;
- mean stabilization time recorded and acceptable for the demonstration;
- ROS topic and verification service tests passing.

Start with `--minimum-confidence 0.80`. Lower it only after multiple labeled
runs show zero false accepts at the proposed threshold. A rejected move is
safer than a confidently wrong robot command.
