from pathlib import Path

import cv2
import numpy as np

from chess_perception.config import load_board_config
from chess_perception.occupancy import EmptyBoardModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _empty_board() -> np.ndarray:
    image = np.zeros((800, 800, 3), dtype=np.uint8)
    for row in range(8):
        for column in range(8):
            value = 220 if (row + column) % 2 == 0 else 65
            image[
                row * 100 : (row + 1) * 100,
                column * 100 : (column + 1) * 100,
            ] = value
    return image


def test_detects_piece_on_one_square() -> None:
    config = load_board_config(PROJECT_ROOT / "config" / "board.toml")
    empty = _empty_board()
    frames = [empty.copy() for _ in range(8)]
    model = EmptyBoardModel.from_frames(frames, config)

    assert model.roi_normalized == (0.05, 0.05, 0.95, 0.95)

    current = empty.copy()
    cv2.circle(current, (450, 450), 28, (20, 20, 220), -1)
    observation = model.observe(current, config)

    assert observation.occupied_squares() == {"e4"}
    assert observation.probability_by_square()["e4"] > 0.9


def test_detects_white_piece_on_light_square() -> None:
    config = load_board_config(PROJECT_ROOT / "config" / "board.toml")
    empty = _empty_board()
    model = EmptyBoardModel.from_frames([empty.copy() for _ in range(8)], config)

    current = empty.copy()
    # A low-contrast synthetic pawn on the light e4 square. Its fill is close
    # to the board color; the outline represents the piece's normal shading.
    cv2.circle(current, (450, 425), 11, (235, 235, 235), -1)
    cv2.circle(current, (450, 425), 11, (205, 205, 205), 2)
    cv2.ellipse(current, (450, 460), (22, 25), 0, 0, 360, (235, 235, 235), -1)
    cv2.ellipse(current, (450, 460), (22, 25), 0, 0, 360, (205, 205, 205), 2)

    observation = model.observe(current, config)

    assert observation.occupied_squares() == {"e4"}
    assert observation.probability_by_square()["e4"] > 0.6


def test_model_round_trip(tmp_path: Path) -> None:
    config = load_board_config(PROJECT_ROOT / "config" / "board.toml")
    empty = _empty_board()
    model = EmptyBoardModel.from_frames([empty.copy() for _ in range(5)], config)
    path = tmp_path / "empty_board.npz"
    model.save(path)
    restored = EmptyBoardModel.load(path)

    assert restored.labels == model.labels
    assert np.array_equal(restored.reference_crops, model.reference_crops)
    assert np.allclose(restored.thresholds, model.thresholds)
    assert np.allclose(restored.roi_normalized, model.roi_normalized)
