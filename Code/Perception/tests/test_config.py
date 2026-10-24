from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chess_perception.config import load_board_config  # noqa: E402


class BoardConfigTests(unittest.TestCase):
    def test_default_config(self) -> None:
        config = load_board_config(PROJECT_ROOT / "config" / "board.toml")
        self.assertEqual(config.output_size_px, 800)
        self.assertEqual(config.files, "abcdefgh")
        self.assertEqual(config.ranks, "87654321")
        self.assertEqual(config.markers["top_left"].marker_id, 0)
        self.assertEqual(config.markers["bottom_left"].inner_corner_index, 1)

    def test_rejects_duplicate_marker_ids(self) -> None:
        invalid = """
[board]
output_size_px = 800
files = "abcdefgh"
ranks = "87654321"
[aruco]
dictionary = "DICT_4X4_50"
[aruco.top_left]
id = 0
inner_corner_index = 2
[aruco.top_right]
id = 0
inner_corner_index = 3
[aruco.bottom_right]
id = 2
inner_corner_index = 0
[aruco.bottom_left]
id = 3
inner_corner_index = 1
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.toml"
            path.write_text(invalid)
            with self.assertRaisesRegex(ValueError, "must be unique"):
                load_board_config(path)


if __name__ == "__main__":
    unittest.main()
