# CS 6341 Part 2 - SO-101 Arm Control

`so101_control.py` implements all code requirements from Part 2 of the mid-term
assignment. It commands the three required poses, detects arrival when every
joint is within `+/-1.0` of its target, measures elapsed time from the command,
and prints a video-friendly final joint-position table. Small, bounded feedback
corrections compensate for static error on loaded degree joints without changing
the assignment target or tolerance.

The defaults match this arm:

- Port: auto-detected (`/dev/tty.usbmodem5A680130241` when prepared)
- Robot ID: `team08_arm`
- Calibration: `team08_arm.json` in this directory
- Mode: timed, with a minimum of 3 seconds between commands
- Cycles: one complete pass through Position 0, Position 1, and Position 2

## Prepare the environment once

From this directory:

```bash
cd lerobot
/Users/pesy/.local/bin/uv sync --locked --extra feetech
```

## Validate without moving the arm

```bash
./run_part2.sh --dry-run
```

## Record the required video

1. Power the follower arm and connect its USB cable.
2. Clear the arm's full workspace and start recording both the arm and terminal.
3. Run:

```bash
./run_part2.sh
```

4. At the safety prompt, verify the workspace is clear and press Enter.
5. Keep the arm and the terminal tables visible through all three positions.

If LeRobot reports a calibration mismatch, press Enter to load the existing
`team08_arm` calibration. Its gripper zero was updated to the measured reachable
endpoint; the previous calibration is preserved as `team08_arm.before_part2_fix.json`.
The script always disconnects and disables torque on
normal completion, timeout, error, or Ctrl-C.

Useful alternatives:

```bash
# Advance manually for easier filming
./run_part2.sh --mode keyboard

# Repeat the three-position sequence twice
./run_part2.sh --cycles 2

# Select the port explicitly
./run_part2.sh \
  --port /dev/tty.usbmodem5A680130241
```

Do not use `--yes` until the arm's workspace has been checked. If a pose fails
to reach tolerance within 10 seconds, the script prints the final readings,
stops the sequence, disconnects, and disables torque.
