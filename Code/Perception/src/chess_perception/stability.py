"""Temporal stability gating used to avoid reading a moving hand or arm."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class StabilityResult:
    stable: bool
    stable_frames: int
    motion_score: float


class FrameStabilityDetector:
    def __init__(
        self,
        motion_threshold: float = 0.012,
        required_stable_frames: int = 8,
        comparison_size_px: int = 160,
    ) -> None:
        self.motion_threshold = motion_threshold
        self.required_stable_frames = required_stable_frames
        self.comparison_size_px = comparison_size_px
        self._previous: np.ndarray | None = None
        self._stable_frames = 0

    def reset(self) -> None:
        self._previous = None
        self._stable_frames = 0

    def update(self, image: np.ndarray) -> StabilityResult:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        normalized = cv2.resize(
            gray,
            (self.comparison_size_px, self.comparison_size_px),
            interpolation=cv2.INTER_AREA,
        )
        normalized = cv2.GaussianBlur(normalized, (5, 5), 0)
        if self._previous is None:
            self._previous = normalized
            return StabilityResult(False, 0, 1.0)

        # Remove the median frame-wide brightness shift before measuring
        # motion. Webcam auto-exposure can otherwise make a completely still
        # board alternate between stable and unstable.
        delta = normalized.astype(np.float32) - self._previous.astype(np.float32)
        delta -= np.median(delta)
        motion_score = float(np.mean(np.abs(delta)) / 255.0)
        self._previous = normalized
        if motion_score <= self.motion_threshold:
            self._stable_frames += 1
        else:
            self._stable_frames = 0
        return StabilityResult(
            stable=self._stable_frames >= self.required_stable_frames,
            stable_frames=self._stable_frames,
            motion_score=motion_score,
        )
