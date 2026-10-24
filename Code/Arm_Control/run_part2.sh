#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
python_path="$script_dir/lerobot/.venv/bin/python"

if [ ! -x "$python_path" ]; then
    echo "LeRobot environment not found at: $python_path" >&2
    echo "Prepare it with the first command in PART2_README.md." >&2
    exit 2
fi

exec "$python_path" "$script_dir/so101_control.py" "$@"
