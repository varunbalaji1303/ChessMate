#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/jazzy/setup.bash
source /home/ubuntu/data/chessmate_ros2_ws/install/setup.bash
set -u

event_log="$(mktemp /home/ubuntu/data/chessmate_events/ros_smoke.XXXXXX.jsonl)"
bridge_log="$(mktemp /tmp/chessmate_bridge.XXXXXX.log)"
listener_log="$(mktemp /tmp/chessmate_listener.XXXXXX.log)"
service_log="$(mktemp /tmp/chessmate_service.XXXXXX.log)"
state_file="$(mktemp /home/ubuntu/data/chessmate_events/ros_state.XXXXXX.json)"
jq --argjson now "$(date +%s)" '.timestamp = $now' \
  /home/ubuntu/data/chessmate_events/observation.json >"$state_file"

process_ids=()
cleanup() {
  for process_id in "${process_ids[@]}"; do
    kill "$process_id" 2>/dev/null || true
  done
}
trap cleanup EXIT

bridge_bin="/home/ubuntu/data/chessmate_ros2_ws/install/chessmate_perception_bridge/lib/chessmate_perception_bridge"

"$bridge_bin/move_event_listener" \
  >"$listener_log" 2>&1 &
process_ids+=("$!")

"$bridge_bin/move_event_bridge" --ros-args \
  -p event_log:="$event_log" \
  -p start_at_end:=false \
  >"$bridge_log" 2>&1 &
process_ids+=("$!")

sleep 1
printf '%s\n' '{"event":"engine_move_requested","uci":"e7e5","requested_elo":1200,"applied_uci_elo":1320,"fen_before":"before","fen_after":"after"}' >>"$event_log"
sleep 2

if ! grep -q 'ROBOT MOVE uci=e7e5' "$listener_log"; then
  echo "Engine move topic smoke test failed"
  cat "$bridge_log"
  cat "$listener_log"
  exit 1
fi
echo "Engine move topic smoke test passed"

"$bridge_bin/verification_service" --ros-args \
  -p state_file:="$state_file" \
  >"$service_log" 2>&1 &
process_ids+=("$!")
sleep 1

service_output="$(ros2 service call \
  /chessmate/perception/verify_board \
  chessmate_interfaces/srv/VerifyBoard \
  "{expected_fen: 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1', maximum_age_seconds: 30.0, maximum_mismatches: 0}")"
echo "$service_output"
if ! grep -q 'success=True' <<<"$service_output"; then
  echo "Verification service smoke test failed"
  cat "$service_log"
  exit 1
fi
echo "Verification service smoke test passed"

mismatch_output="$(ros2 service call \
  /chessmate/perception/verify_board \
  chessmate_interfaces/srv/VerifyBoard \
  "{expected_fen: '8/8/8/8/8/8/8/8 w - - 0 1', maximum_age_seconds: 30.0, maximum_mismatches: 0}")"
echo "$mismatch_output"
if ! grep -q 'success=False' <<<"$mismatch_output"; then
  echo "Verification mismatch smoke test failed"
  cat "$service_log"
  exit 1
fi
echo "Verification mismatch smoke test passed"
