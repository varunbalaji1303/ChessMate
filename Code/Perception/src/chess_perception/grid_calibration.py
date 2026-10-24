"""Persistent calibration for the playable 8x8 area inside the board image."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


GRID_CORNER_NAMES = ("top_left", "top_right", "bottom_right", "bottom_left")


@dataclass(frozen=True)
class GridCalibration:
    """Playable-grid corners normalized within the marker-rectified image."""

    corners_normalized: np.ndarray

    def __post_init__(self) -> None:
        corners = np.asarray(self.corners_normalized, dtype=np.float32)
        if corners.shape != (4, 2):
            raise ValueError("grid calibration must contain four 2D corners")
        if not np.isfinite(corners).all():
            raise ValueError("grid calibration contains non-finite coordinates")
        if np.any(corners < 0.0) or np.any(corners > 1.0):
            raise ValueError("normalized grid corners must be between 0 and 1")
        polygon_area = 0.5 * abs(
            np.dot(corners[:, 0], np.roll(corners[:, 1], 1))
            - np.dot(corners[:, 1], np.roll(corners[:, 0], 1))
        )
        if polygon_area < 0.25:
            raise ValueError(
                "grid calibration polygon is too small or degenerate; select "
                "the four corners of the entire playable 8x8 area"
            )
        object.__setattr__(self, "corners_normalized", corners)

    @classmethod
    def from_pixel_corners(
        cls, corners_px: np.ndarray, image_size_px: int
    ) -> "GridCalibration":
        if image_size_px < 2:
            raise ValueError("image_size_px must be at least 2")
        corners = np.asarray(corners_px, dtype=np.float32)
        return cls(corners / float(image_size_px - 1))

    def pixel_corners(self, image_size_px: int) -> np.ndarray:
        if image_size_px < 2:
            raise ValueError("image_size_px must be at least 2")
        return self.corners_normalized * float(image_size_px - 1)


def save_grid_calibration(calibration: GridCalibration, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "corner_order": list(GRID_CORNER_NAMES),
        "corners_normalized": calibration.corners_normalized.tolist(),
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n")


def load_grid_calibration(path: str | Path) -> GridCalibration:
    input_path = Path(path)
    payload = json.loads(input_path.read_text())
    if tuple(payload.get("corner_order", ())) != GRID_CORNER_NAMES:
        raise ValueError(
            "grid calibration corner order must be top_left, top_right, "
            "bottom_right, bottom_left"
        )
    try:
        corners = np.asarray(payload["corners_normalized"], dtype=np.float32)
    except KeyError as error:
        raise ValueError("missing corners_normalized in grid calibration") from error
    return GridCalibration(corners)
