from pathlib import Path

import cv2
import numpy as np

from chess_perception.board import BoardRectifier
from chess_perception.config import load_board_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _synthetic_board() -> np.ndarray:
    canvas = np.full((1000, 1000, 3), 255, dtype=np.uint8)
    board_start = 200
    square_size = 75

    for row in range(8):
        for column in range(8):
            value = 225 if (row + column) % 2 == 0 else 70
            y0 = board_start + row * square_size
            x0 = board_start + column * square_size
            canvas[y0 : y0 + square_size, x0 : x0 + square_size] = value

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_size = 100
    placements = {
        0: (100, 100),
        1: (800, 100),
        2: (800, 800),
        3: (100, 800),
    }
    for marker_id, (x, y) in placements.items():
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_size)
        canvas[y : y + marker_size, x : x + marker_size] = cv2.cvtColor(
            marker, cv2.COLOR_GRAY2BGR
        )
    return canvas


def test_rectifies_synthetic_board() -> None:
    config = load_board_config(PROJECT_ROOT / "config" / "board.toml")
    result = BoardRectifier(config).rectify(_synthetic_board())

    assert result.rectified.shape == (800, 800, 3)
    assert result.debug_grid.shape == result.rectified.shape
    assert result.homography.shape == (3, 3)
    assert result.board_corners_px.shape == (4, 2)

    expected = np.array(
        [[199, 199], [800, 199], [800, 800], [199, 800]], dtype=np.float32
    )
    assert np.allclose(result.board_corners_px, expected, atol=2.0)
