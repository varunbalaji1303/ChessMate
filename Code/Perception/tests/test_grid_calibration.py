from pathlib import Path

import numpy as np
import pytest

from chess_perception.grid_calibration import (
    GridCalibration,
    load_grid_calibration,
    save_grid_calibration,
)


def test_grid_calibration_round_trip(tmp_path: Path) -> None:
    corners_px = np.array(
        [[100, 80], [720, 90], [740, 730], [90, 710]], dtype=np.float32
    )
    calibration = GridCalibration.from_pixel_corners(corners_px, 800)
    path = tmp_path / "grid.json"
    save_grid_calibration(calibration, path)
    restored = load_grid_calibration(path)

    assert np.allclose(restored.pixel_corners(800), corners_px, atol=0.01)


def test_grid_calibration_rejects_degenerate_polygon() -> None:
    corners = np.array(
        [[0.1, 0.1], [0.2, 0.1], [0.2, 0.11], [0.1, 0.11]],
        dtype=np.float32,
    )
    with pytest.raises(ValueError, match="too small or degenerate"):
        GridCalibration(corners)
