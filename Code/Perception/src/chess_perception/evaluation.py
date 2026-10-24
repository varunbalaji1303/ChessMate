"""Offline metrics for labeled ChessMate move-tracking sessions."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any


def load_json_lines(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON in {path} line {line_number}: {error}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(f"{path} line {line_number} must be an object")
            records.append(record)
    return records


def _mean_or_none(values: list[float]) -> float | None:
    return round(mean(values), 6) if values else None


def evaluate_event_log(
    event_log: Path,
    expected_path: Path,
    confidence_threshold: float = 0.72,
) -> dict[str, Any]:
    """Compare accepted move events with an ordered labeled trial plan."""

    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence threshold must be between 0 and 1")
    events = [
        record
        for record in load_json_lines(event_log)
        if record.get("event") == "move_detected"
    ]
    all_records = load_json_lines(event_log)
    trial_records = [
        record for record in all_records if record.get("event") == "evaluation_trial"
    ]
    if trial_records:
        events = trial_records
    expected = load_json_lines(expected_path)
    for index, trial in enumerate(expected, start=1):
        if not isinstance(trial.get("uci"), str):
            raise ValueError(f"expected trial {index} is missing string field 'uci'")

    correct = 0
    wrong = 0
    missing = 0
    category_results: dict[str, dict[str, int]] = {}
    details: list[dict[str, Any]] = []
    confidences: list[float] = []
    correct_confidences: list[float] = []
    stabilization_times: list[float] = []
    auto_eligible_correct = 0
    auto_eligible_wrong = 0

    for index, trial in enumerate(expected):
        category = str(trial.get("category", "unclassified"))
        category_summary = category_results.setdefault(
            category, {"expected": 0, "correct": 0, "wrong": 0, "missing": 0}
        )
        category_summary["expected"] += 1
        if index >= len(events):
            missing += 1
            category_summary["missing"] += 1
            details.append(
                {
                    "trial_id": trial.get("trial_id", index + 1),
                    "category": category,
                    "expected_uci": trial["uci"],
                    "observed_uci": None,
                    "outcome": "missing_or_rejected",
                }
            )
            continue

        event = events[index]
        observed_uci = event.get("uci")
        if event.get("event") == "evaluation_trial":
            observed_uci = event.get("observed_uci")
            if event.get("outcome") == "rejected":
                missing += 1
                category_summary["missing"] += 1
                details.append(
                    {
                        "trial_id": trial.get("trial_id", index + 1),
                        "category": category,
                        "expected_uci": trial["uci"],
                        "observed_uci": None,
                        "outcome": "rejected",
                    }
                )
                continue
        confidence = float(event.get("confidence", 0.0))
        confidences.append(confidence)
        if "stabilization_seconds" in event:
            stabilization_times.append(float(event["stabilization_seconds"]))
        is_correct = observed_uci == trial["uci"]
        if is_correct:
            correct += 1
            category_summary["correct"] += 1
            correct_confidences.append(confidence)
            if confidence >= confidence_threshold:
                auto_eligible_correct += 1
            outcome = "correct"
        else:
            wrong += 1
            category_summary["wrong"] += 1
            if confidence >= confidence_threshold:
                auto_eligible_wrong += 1
            outcome = "wrong_accept"
        details.append(
            {
                "trial_id": trial.get("trial_id", index + 1),
                "category": category,
                "condition": trial.get("condition", "centered"),
                "expected_uci": trial["uci"],
                "observed_uci": observed_uci,
                "confidence": confidence,
                "outcome": outcome,
                "occupancy_mismatches": event.get("occupancy_mismatches", []),
            }
        )

    extra_events = max(0, len(events) - len(expected))
    wrong_or_extra = wrong + extra_events
    mismatch_events = sum(bool(event.get("occupancy_mismatches")) for event in events)
    expected_count = len(expected)
    observed_aligned = min(len(events), expected_count)
    report: dict[str, Any] = {
        "event_log": str(event_log),
        "expected_plan": str(expected_path),
        "confidence_threshold": confidence_threshold,
        "total_expected": expected_count,
        "accepted_events": len(events),
        "correct": correct,
        "wrong_accepts": wrong,
        "missing_or_rejected": missing,
        "extra_events": extra_events,
        "false_accepts_total": wrong_or_extra,
        "move_accuracy": round(correct / expected_count, 6)
        if expected_count
        else None,
        "accepted_accuracy": round(correct / observed_aligned, 6)
        if observed_aligned
        else None,
        "mean_confidence": _mean_or_none(confidences),
        "mean_correct_confidence": _mean_or_none(correct_confidences),
        "mean_stabilization_seconds": _mean_or_none(stabilization_times),
        "events_with_occupancy_mismatches": mismatch_events,
        "auto_accept_eligible_correct": auto_eligible_correct,
        "auto_accept_eligible_wrong": auto_eligible_wrong,
        "by_category": category_results,
        "details": details,
    }
    return report


def write_evaluation_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
