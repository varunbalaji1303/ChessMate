"""Utilities for addressing and extracting the 64 rectified board squares."""

from __future__ import annotations

from collections.abc import Iterator

import cv2
import numpy as np

from .config import BoardConfig


def iter_squares(config: BoardConfig) -> Iterator[tuple[int, int, str]]:
    """Yield row, column, and algebraic label in image scan order."""

    for row, rank in enumerate(config.ranks):
        for column, file_name in enumerate(config.files):
            yield row, column, f"{file_name}{rank}"


def square_labels(config: BoardConfig) -> tuple[str, ...]:
    return tuple(label for _row, _column, label in iter_squares(config))


def extract_square_crops(
    rectified: np.ndarray,
    config: BoardConfig,
    margin_ratio: float = 0.12,
    output_size_px: int = 64,
    roi_normalized: tuple[float, float, float, float] | None = None,
) -> dict[str, np.ndarray]:
    """Extract equally sized square interiors, excluding noisy grid borders."""

    if rectified is None or rectified.size == 0:
        raise ValueError("rectified image is empty")
    if not 0.0 <= margin_ratio < 0.45:
        raise ValueError("margin_ratio must be between 0 and 0.45")
    if output_size_px < 16:
        raise ValueError("output_size_px must be at least 16")
    if roi_normalized is not None:
        left, top, right, bottom = roi_normalized
        if not (
            0.0 <= left < right <= 1.0
            and 0.0 <= top < bottom <= 1.0
            and right - left >= 0.2
            and bottom - top >= 0.2
        ):
            raise ValueError(
                "roi_normalized must be left, top, right, bottom within 0-1 "
                "and span at least 20% of the square"
            )

    height, width = rectified.shape[:2]
    crops: dict[str, np.ndarray] = {}
    for row, column, label in iter_squares(config):
        x0 = round(column * width / 8)
        x1 = round((column + 1) * width / 8)
        y0 = round(row * height / 8)
        y1 = round((row + 1) * height / 8)
        if roi_normalized is None:
            margin_x = round((x1 - x0) * margin_ratio)
            margin_y = round((y1 - y0) * margin_ratio)
            crop_x0, crop_x1 = x0 + margin_x, x1 - margin_x
            crop_y0, crop_y1 = y0 + margin_y, y1 - margin_y
        else:
            left, top, right, bottom = roi_normalized
            crop_x0 = round(x0 + (x1 - x0) * left)
            crop_x1 = round(x0 + (x1 - x0) * right)
            crop_y0 = round(y0 + (y1 - y0) * top)
            crop_y1 = round(y0 + (y1 - y0) * bottom)
        interior = rectified[crop_y0:crop_y1, crop_x0:crop_x1]
        if interior.size == 0:
            raise ValueError(f"empty crop generated for square {label}")
        crops[label] = cv2.resize(
            interior,
            (output_size_px, output_size_px),
            interpolation=cv2.INTER_AREA,
        )
    return crops


def square_center_px(
    label: str, image_shape: tuple[int, ...], config: BoardConfig
) -> tuple[int, int]:
    for row, column, candidate in iter_squares(config):
        if candidate == label:
            height, width = image_shape[:2]
            return (
                round((column + 0.5) * width / 8),
                round((row + 0.5) * height / 8),
            )
    raise ValueError(f"unknown square label: {label}")
