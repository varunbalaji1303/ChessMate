"""Empty-board modeling and per-square occupancy detection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import BoardConfig
from .squares import extract_square_crops, iter_squares, square_labels


# Use nearly the complete square for an overhead camera.  The small edge guard
# prevents tiny rectification errors from turning grid lines into false change.
DEFAULT_OCCUPANCY_ROI = (0.05, 0.05, 0.95, 0.95)
EMPTY_MODEL_FORMAT_VERSION = 4


def _edge_magnitude(image: np.ndarray) -> np.ndarray:
    """Return a normalized grayscale edge-magnitude image."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    # 4 * sqrt(2) is the maximum magnitude of a 3x3 Sobel response for a
    # normalized step edge.
    return np.sqrt(gradient_x**2 + gradient_y**2) / (4.0 * np.sqrt(2.0))


def crop_difference_score(reference: np.ndarray, current: np.ndarray) -> float:
    """Return a 0-1 appearance-change score for two same-sized square crops."""

    if reference.shape != current.shape:
        raise ValueError("reference and current crops must have the same shape")
    reference_blur = cv2.GaussianBlur(reference, (5, 5), 0)
    current_blur = cv2.GaussianBlur(current, (5, 5), 0)
    reference_lab = cv2.cvtColor(reference_blur, cv2.COLOR_BGR2LAB).astype(np.float32)
    current_lab = cv2.cvtColor(current_blur, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_delta = np.abs(reference_lab - current_lab) / 255.0

    # General color/appearance change. This remains strong for black pieces
    # and for pieces whose color differs clearly from the square.
    color_delta = np.mean(lab_delta, axis=2)
    color_score = 0.7 * np.percentile(color_delta, 85) + 0.3 * np.mean(
        color_delta
    )

    # Do not average the L channel with the two nearly unchanged chroma
    # channels. A white piece on a light square is often visible mainly from
    # its highlights and shadows.
    luminance_delta = lab_delta[:, :, 0]
    luminance_score = 0.9 * np.percentile(
        luminance_delta, 90
    ) + 0.1 * np.mean(luminance_delta)

    # A similarly colored piece still introduces a silhouette, rings, and
    # contours. Comparing gradients captures those structural changes.
    edge_delta = np.abs(
        _edge_magnitude(reference_blur) - _edge_magnitude(current_blur)
    )
    edge_score = 0.75 * np.percentile(edge_delta, 95) + 0.25 * np.mean(
        edge_delta
    )

    return float(max(color_score, luminance_score, edge_score))


@dataclass(frozen=True)
class OccupancyObservation:
    labels: tuple[str, ...]
    scores: np.ndarray
    thresholds: np.ndarray
    probabilities: np.ndarray
    occupied: np.ndarray
    roi_normalized: tuple[float, float, float, float]

    def probability_by_square(self) -> dict[str, float]:
        return {
            label: float(probability)
            for label, probability in zip(self.labels, self.probabilities)
        }

    def score_by_square(self) -> dict[str, float]:
        return {label: float(score) for label, score in zip(self.labels, self.scores)}

    def occupied_squares(self) -> set[str]:
        return {
            label for label, occupied in zip(self.labels, self.occupied) if occupied
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "occupied_squares": sorted(self.occupied_squares()),
            "occupancy_roi": list(self.roi_normalized),
            "squares": {
                label: {
                    "occupied": bool(occupied),
                    "probability": round(float(probability), 6),
                    "change_score": round(float(score), 6),
                    "threshold": round(float(threshold), 6),
                }
                for label, occupied, probability, score, threshold in zip(
                    self.labels,
                    self.occupied,
                    self.probabilities,
                    self.scores,
                    self.thresholds,
                )
            },
        }


@dataclass(frozen=True)
class EmptyBoardModel:
    labels: tuple[str, ...]
    reference_crops: np.ndarray
    thresholds: np.ndarray
    roi_normalized: tuple[float, float, float, float] = DEFAULT_OCCUPANCY_ROI
    crop_size_px: int = 64

    def __post_init__(self) -> None:
        if len(self.labels) != 64:
            raise ValueError("empty-board model must contain 64 square labels")
        if self.reference_crops.shape != (
            64,
            self.crop_size_px,
            self.crop_size_px,
            3,
        ):
            raise ValueError("unexpected empty-board reference crop shape")
        if self.thresholds.shape != (64,):
            raise ValueError("empty-board model must contain 64 thresholds")
        left, top, right, bottom = self.roi_normalized
        if not (0.0 <= left < right <= 1.0 and 0.0 <= top < bottom <= 1.0):
            raise ValueError("invalid occupancy ROI")

    @classmethod
    def from_frames(
        cls,
        rectified_frames: list[np.ndarray],
        config: BoardConfig,
        roi_normalized: tuple[
            float, float, float, float
        ] = DEFAULT_OCCUPANCY_ROI,
        crop_size_px: int = 64,
        minimum_threshold: float = 0.040,
    ) -> "EmptyBoardModel":
        if len(rectified_frames) < 5:
            raise ValueError("at least 5 empty-board frames are required")

        labels = square_labels(config)
        frame_crops = []
        for frame in rectified_frames:
            crops = extract_square_crops(
                frame,
                config,
                output_size_px=crop_size_px,
                roi_normalized=roi_normalized,
            )
            frame_crops.append(np.stack([crops[label] for label in labels]))
        samples = np.stack(frame_crops)
        reference = np.median(samples, axis=0).astype(np.uint8)

        noise_scores = np.zeros((len(samples), 64), dtype=np.float32)
        for frame_index, sample in enumerate(samples):
            for square_index in range(64):
                noise_scores[frame_index, square_index] = crop_difference_score(
                    reference[square_index], sample[square_index]
                )
        thresholds = np.maximum(
            minimum_threshold,
            np.percentile(noise_scores, 99, axis=0) + 0.025,
        ).astype(np.float32)
        return cls(
            labels=labels,
            reference_crops=reference,
            thresholds=thresholds,
            roi_normalized=roi_normalized,
            crop_size_px=crop_size_px,
        )

    def observe(self, rectified: np.ndarray, config: BoardConfig) -> OccupancyObservation:
        if square_labels(config) != self.labels:
            raise ValueError("board labels do not match the empty-board model")
        crops = extract_square_crops(
            rectified,
            config,
            output_size_px=self.crop_size_px,
            roi_normalized=self.roi_normalized,
        )
        scores = np.asarray(
            [
                crop_difference_score(self.reference_crops[index], crops[label])
                for index, label in enumerate(self.labels)
            ],
            dtype=np.float32,
        )
        scale = np.maximum(self.thresholds * 0.22, 0.012)
        logits = np.clip((scores - self.thresholds) / scale, -20.0, 20.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        occupied = scores >= self.thresholds
        return OccupancyObservation(
            labels=self.labels,
            scores=scores,
            thresholds=self.thresholds.copy(),
            probabilities=probabilities.astype(np.float32),
            occupied=occupied,
            roi_normalized=self.roi_normalized,
        )

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            labels=np.asarray(self.labels),
            reference_crops=self.reference_crops,
            thresholds=self.thresholds,
            roi_normalized=np.asarray(self.roi_normalized, dtype=np.float32),
            crop_size_px=np.asarray(self.crop_size_px),
            format_version=np.asarray(EMPTY_MODEL_FORMAT_VERSION),
        )

    @classmethod
    def load(cls, path: str | Path) -> "EmptyBoardModel":
        with np.load(Path(path), allow_pickle=False) as data:
            if (
                "format_version" not in data
                or int(data["format_version"].item())
                != EMPTY_MODEL_FORMAT_VERSION
            ):
                raise ValueError(
                    "empty-board model uses an older occupancy region; "
                    "run 'chess-perception capture-empty' again"
                )
            return cls(
                labels=tuple(str(value) for value in data["labels"].tolist()),
                reference_crops=data["reference_crops"].astype(np.uint8),
                thresholds=data["thresholds"].astype(np.float32),
                roi_normalized=tuple(
                    float(value) for value in data["roi_normalized"].tolist()
                ),
                crop_size_px=int(data["crop_size_px"].item()),
            )


def draw_occupancy_overlay(
    rectified: np.ndarray,
    observation: OccupancyObservation,
    config: BoardConfig,
) -> np.ndarray:
    """Draw occupancy decisions and probabilities over a rectified board."""

    debug = rectified.copy()
    height, width = debug.shape[:2]
    for row, column, label in iter_squares(config):
        index = observation.labels.index(label)
        occupied = bool(observation.occupied[index])
        probability = float(observation.probabilities[index])
        color = (0, 0, 255) if occupied else (0, 180, 0)
        x0 = round(column * width / 8)
        x1 = round((column + 1) * width / 8)
        y0 = round(row * height / 8)
        y1 = round((row + 1) * height / 8)
        cv2.rectangle(debug, (x0, y0), (x1 - 1, y1 - 1), color, 2)
        roi_left, roi_top, roi_right, roi_bottom = observation.roi_normalized
        roi_x0 = round(x0 + (x1 - x0) * roi_left)
        roi_x1 = round(x0 + (x1 - x0) * roi_right)
        roi_y0 = round(y0 + (y1 - y0) * roi_top)
        roi_y1 = round(y0 + (y1 - y0) * roi_bottom)
        cv2.rectangle(
            debug,
            (roi_x0, roi_y0),
            (roi_x1, roi_y1),
            (255, 170, 0),
            1,
        )
        cv2.putText(
            debug,
            f"{label} {probability:.2f}",
            (x0 + 4, y0 + 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color,
            1,
            cv2.LINE_AA,
        )
    return debug


def square_change_probabilities(
    previous_rectified: np.ndarray,
    current_rectified: np.ndarray,
    config: BoardConfig,
    roi_normalized: tuple[
        float, float, float, float
    ] = DEFAULT_OCCUPANCY_ROI,
    crop_size_px: int = 64,
    change_threshold: float = 0.055,
) -> dict[str, float]:
    previous = extract_square_crops(
        previous_rectified,
        config,
        output_size_px=crop_size_px,
        roi_normalized=roi_normalized,
    )
    current = extract_square_crops(
        current_rectified,
        config,
        output_size_px=crop_size_px,
        roi_normalized=roi_normalized,
    )
    probabilities: dict[str, float] = {}
    for label in square_labels(config):
        score = crop_difference_score(previous[label], current[label])
        logit = np.clip((score - change_threshold) / 0.015, -20.0, 20.0)
        probabilities[label] = float(1.0 / (1.0 + np.exp(-logit)))
    return probabilities
