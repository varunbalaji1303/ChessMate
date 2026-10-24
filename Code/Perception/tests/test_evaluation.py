import json

import pytest

from chess_perception.evaluation import evaluate_event_log


def _write_jsonl(path, records) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def test_evaluates_correct_wrong_missing_and_extra_events(tmp_path) -> None:
    expected_path = tmp_path / "expected.jsonl"
    event_path = tmp_path / "events.jsonl"
    _write_jsonl(
        expected_path,
        [
            {"trial_id": "n1", "category": "normal", "uci": "e2e4"},
            {"trial_id": "c1", "category": "capture", "uci": "e4d5"},
            {"trial_id": "n2", "category": "normal", "uci": "g1f3"},
        ],
    )
    _write_jsonl(
        event_path,
        [
            {
                "event": "move_detected",
                "uci": "e2e4",
                "confidence": 0.9,
                "stabilization_seconds": 0.5,
                "occupancy_mismatches": [],
            },
            {
                "event": "move_detected",
                "uci": "e4e5",
                "confidence": 0.8,
                "occupancy_mismatches": ["d7"],
            },
        ],
    )

    report = evaluate_event_log(event_path, expected_path)

    assert report["correct"] == 1
    assert report["wrong_accepts"] == 1
    assert report["missing_or_rejected"] == 1
    assert report["move_accuracy"] == pytest.approx(1 / 3, abs=1e-6)
    assert report["accepted_accuracy"] == 0.5
    assert report["events_with_occupancy_mismatches"] == 1
    assert report["mean_stabilization_seconds"] == 0.5


def test_reports_extra_event_as_false_accept(tmp_path) -> None:
    expected_path = tmp_path / "expected.jsonl"
    event_path = tmp_path / "events.jsonl"
    _write_jsonl(expected_path, [{"uci": "e2e4"}])
    _write_jsonl(
        event_path,
        [
            {"event": "move_detected", "uci": "e2e4", "confidence": 0.9},
            {"event": "move_detected", "uci": "d7d5", "confidence": 0.9},
        ],
    )

    report = evaluate_event_log(event_path, expected_path)

    assert report["extra_events"] == 1
    assert report["false_accepts_total"] == 1


def test_evaluation_trial_rejection_does_not_shift_later_moves(tmp_path) -> None:
    expected_path = tmp_path / "expected.jsonl"
    event_path = tmp_path / "events.jsonl"
    _write_jsonl(
        expected_path,
        [{"uci": "e2e4"}, {"uci": "d7d5"}, {"uci": "e4d5"}],
    )
    _write_jsonl(
        event_path,
        [
            {
                "event": "evaluation_trial",
                "expected_uci": "e2e4",
                "observed_uci": "e2e4",
                "outcome": "correct",
                "confidence": 0.9,
            },
            {
                "event": "evaluation_trial",
                "expected_uci": "d7d5",
                "observed_uci": None,
                "outcome": "rejected",
            },
            {
                "event": "evaluation_trial",
                "expected_uci": "e4d5",
                "observed_uci": "e4d5",
                "outcome": "correct",
                "confidence": 0.8,
            },
        ],
    )

    report = evaluate_event_log(event_path, expected_path)

    assert report["correct"] == 2
    assert report["missing_or_rejected"] == 1
    assert report["wrong_accepts"] == 0
