import numpy as np

from chess_perception.stability import FrameStabilityDetector


def test_stability_requires_consecutive_quiet_frames() -> None:
    detector = FrameStabilityDetector(required_stable_frames=3)
    still = np.full((200, 200, 3), 100, dtype=np.uint8)

    assert not detector.update(still).stable
    assert not detector.update(still).stable
    assert not detector.update(still).stable
    assert detector.update(still).stable

    changed = still.copy()
    changed[:, :100] = 240
    result = detector.update(changed)
    assert not result.stable
    assert result.stable_frames == 0


def test_stability_ignores_global_exposure_flicker() -> None:
    detector = FrameStabilityDetector(required_stable_frames=2)
    board = np.full((200, 200, 3), 100, dtype=np.uint8)

    assert not detector.update(board).stable
    assert not detector.update(board + 8).stable
    result = detector.update(board)

    assert result.stable
    assert result.motion_score == 0.0
