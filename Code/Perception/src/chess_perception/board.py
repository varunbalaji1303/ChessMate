"""ArUco-based chessboard localization and top-down rectification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .config import BoardConfig, CORNER_NAMES
from .grid_calibration import GridCalibration


class BoardNotFoundError(RuntimeError):
    """Raised when the required board markers are not visible."""


@dataclass(frozen=True)
class RectificationResult:
    rectified: np.ndarray
    debug_original: np.ndarray
    debug_grid: np.ndarray
    homography: np.ndarray
    board_corners_px: np.ndarray


def _aruco_dictionary(name: str) -> Any:
    try:
        dictionary_id = getattr(cv2.aruco, name)
    except AttributeError as error:
        raise ValueError(f"unsupported ArUco dictionary: {name}") from error
    return cv2.aruco.getPredefinedDictionary(dictionary_id)


class BoardRectifier:
    """Detect the four configured markers and normalize the board image."""

    def __init__(
        self,
        config: BoardConfig,
        grid_calibration: GridCalibration | None = None,
    ) -> None:
        self.config = config
        self.grid_calibration = grid_calibration
        dictionary = _aruco_dictionary(config.aruco_dictionary)
        parameters = cv2.aruco.DetectorParameters()
        self._detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    def rectify(self, image: np.ndarray) -> RectificationResult:
        if image is None or image.size == 0:
            raise ValueError("input image is empty")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _rejected = self._detector.detectMarkers(gray)
        if ids is None:
            raise BoardNotFoundError("no ArUco markers detected")

        detections = {
            int(marker_id): marker_corners.reshape(4, 2)
            for marker_id, marker_corners in zip(ids.flatten(), corners)
        }
        expected_ids = {
            self.config.markers[name].marker_id for name in CORNER_NAMES
        }
        missing = sorted(expected_ids - set(detections))
        if missing:
            raise BoardNotFoundError(f"missing required marker IDs: {missing}")

        source_points = np.array(
            [
                detections[self.config.markers[name].marker_id][
                    self.config.markers[name].inner_corner_index
                ]
                for name in CORNER_NAMES
            ],
            dtype=np.float32,
        )

        size = self.config.output_size_px
        destination_points = np.array(
            [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]],
            dtype=np.float32,
        )
        marker_homography = cv2.getPerspectiveTransform(
            source_points, destination_points
        )
        marker_rectified = cv2.warpPerspective(
            image, marker_homography, (size, size)
        )

        if self.grid_calibration is None:
            homography = marker_homography
            rectified = marker_rectified
        else:
            grid_source_points = self.grid_calibration.pixel_corners(size).astype(
                np.float32
            )
            grid_homography = cv2.getPerspectiveTransform(
                grid_source_points, destination_points
            )
            rectified = cv2.warpPerspective(
                marker_rectified, grid_homography, (size, size)
            )
            homography = grid_homography @ marker_homography

        debug_original = image.copy()
        cv2.aruco.drawDetectedMarkers(debug_original, corners, ids)
        polygon = np.round(source_points).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(debug_original, [polygon], True, (0, 255, 0), 3)

        debug_grid = draw_labeled_grid(rectified, self.config)
        return RectificationResult(
            rectified=rectified,
            debug_original=debug_original,
            debug_grid=debug_grid,
            homography=homography,
            board_corners_px=source_points,
        )


def draw_labeled_grid(image: np.ndarray, config: BoardConfig) -> np.ndarray:
    """Overlay square boundaries and algebraic labels on a rectified board."""

    debug = image.copy()
    height, width = debug.shape[:2]
    square_w = width / 8.0
    square_h = height / 8.0

    for index in range(9):
        x = round(index * square_w)
        y = round(index * square_h)
        cv2.line(debug, (x, 0), (x, height - 1), (0, 255, 0), 1)
        cv2.line(debug, (0, y), (width - 1, y), (0, 255, 0), 1)

    font_scale = max(0.35, min(width, height) / 1600.0)
    for row, rank in enumerate(config.ranks):
        for column, file_name in enumerate(config.files):
            label = f"{file_name}{rank}"
            origin = (
                round(column * square_w + 5),
                round(row * square_h + 18),
            )
            cv2.putText(
                debug,
                label,
                origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
    return debug
