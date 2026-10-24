import json

import numpy as np

from chess_perception.live import write_observation_state
from chess_perception.occupancy import OccupancyObservation


def test_writes_atomic_stable_observation_state(tmp_path) -> None:
    labels = tuple(f"square-{index}" for index in range(64))
    observation = OccupancyObservation(
        labels=labels,
        scores=np.zeros(64, dtype=np.float32),
        thresholds=np.full(64, 0.04, dtype=np.float32),
        probabilities=np.concatenate(
            (np.ones(2, dtype=np.float32), np.zeros(62, dtype=np.float32))
        ),
        occupied=np.concatenate(
            (np.ones(2, dtype=bool), np.zeros(62, dtype=bool))
        ),
        roi_normalized=(0.05, 0.05, 0.95, 0.95),
    )
    path = tmp_path / "observation.json"

    write_observation_state(
        path,
        logical_fen="8/8/8/8/8/8/8/8 w - - 0 1",
        observation=observation,
        motion_score=0.003,
        timestamp=123.5,
    )

    payload = json.loads(path.read_text())
    assert payload["schema_version"] == 1
    assert payload["timestamp"] == 123.5
    assert payload["stable"] is True
    assert payload["observation"]["occupied_squares"] == ["square-0", "square-1"]
    assert not (tmp_path / ".observation.json.tmp").exists()
