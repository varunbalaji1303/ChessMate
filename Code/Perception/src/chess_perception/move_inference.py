"""Infer a human chess move from vision while enforcing chess legality."""

from __future__ import annotations

from dataclasses import dataclass
import math

import chess


@dataclass(frozen=True)
class MoveCandidate:
    move: chess.Move
    san: str
    confidence: float
    loss: float
    occupancy_loss: float
    change_loss: float
    runner_up_gap: float
    expected_changed_squares: tuple[str, ...]
    occupancy_mismatches: tuple[str, ...]

    def as_dict(self, board: chess.Board) -> dict[str, object]:
        next_board = board.copy(stack=False)
        next_board.push(self.move)
        return {
            "uci": self.move.uci(),
            "san": self.san,
            "confidence": round(self.confidence, 6),
            "loss": round(self.loss, 6),
            "occupancy_loss": round(self.occupancy_loss, 6),
            "change_loss": round(self.change_loss, 6),
            "runner_up_gap": round(self.runner_up_gap, 6),
            "expected_changed_squares": list(self.expected_changed_squares),
            "occupancy_mismatches": list(self.occupancy_mismatches),
            "fen_before": board.fen(),
            "fen_after": next_board.fen(),
        }


@dataclass(frozen=True)
class MoveVerification:
    success: bool
    move_uci: str
    expected_fen: str
    mismatched_squares: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "success": self.success,
            "move": self.move_uci,
            "expected_fen": self.expected_fen,
            "mismatched_squares": list(self.mismatched_squares),
        }


def occupancy_for_board(board: chess.Board) -> dict[str, bool]:
    return {
        chess.square_name(square): board.piece_at(square) is not None
        for square in chess.SQUARES
    }


def changed_squares_for_move(board: chess.Board, move: chess.Move) -> set[str]:
    """Return squares whose visible contents should change for a legal move."""

    changed = {
        chess.square_name(move.from_square),
        chess.square_name(move.to_square),
    }
    if board.is_castling(move):
        if chess.square_file(move.to_square) > chess.square_file(move.from_square):
            rook_from = chess.H1 if board.turn == chess.WHITE else chess.H8
            rook_to = chess.F1 if board.turn == chess.WHITE else chess.F8
        else:
            rook_from = chess.A1 if board.turn == chess.WHITE else chess.A8
            rook_to = chess.D1 if board.turn == chess.WHITE else chess.D8
        changed.update((chess.square_name(rook_from), chess.square_name(rook_to)))
    if board.is_en_passant(move):
        captured_square = move.to_square + (-8 if board.turn == chess.WHITE else 8)
        changed.add(chess.square_name(captured_square))
    return changed


def _bernoulli_loss(probability: float, expected: bool) -> float:
    probability = min(max(float(probability), 1e-5), 1.0 - 1e-5)
    return -math.log(probability if expected else 1.0 - probability)


def infer_legal_move(
    board: chess.Board,
    occupancy_probabilities: dict[str, float],
    change_probabilities: dict[str, float] | None = None,
    maximum_loss: float = 0.42,
    minimum_runner_up_gap: float = 0.008,
) -> MoveCandidate | None:
    """Score all legal moves against observed occupancy and appearance change."""

    expected_labels = {chess.square_name(square) for square in chess.SQUARES}
    missing = expected_labels - set(occupancy_probabilities)
    if missing:
        raise ValueError(f"missing occupancy probabilities: {sorted(missing)}")

    scored: list[
        tuple[
            float,
            float,
            float,
            chess.Move,
            str,
            tuple[str, ...],
            tuple[str, ...],
        ]
    ] = []
    current_occupancy = occupancy_for_board(board)
    for move in board.legal_moves:
        san = board.san(move)
        candidate_board = board.copy(stack=False)
        candidate_board.push(move)
        expected_occupancy = occupancy_for_board(candidate_board)

        changed = changed_squares_for_move(board, move)
        # Evidence on the source/destination squares is much more informative
        # than a persistent, unrelated false positive elsewhere on the board.
        # Clip only unchanged-square outliers so a stale square cannot suppress
        # the confidence of every otherwise well-supported move.
        occupancy_weighted_loss = 0.0
        occupancy_weight_total = 0.0
        for label in expected_labels:
            square_loss = _bernoulli_loss(
                occupancy_probabilities[label], expected_occupancy[label]
            )
            weight = 6.0 if label in changed else 1.0
            if label not in changed:
                square_loss = min(square_loss, 2.0)
            occupancy_weighted_loss += weight * square_loss
            occupancy_weight_total += weight
        occupancy_loss = occupancy_weighted_loss / occupancy_weight_total

        if change_probabilities is None:
            change_loss = 0.0
            occupancy_weight = 1.0
        else:
            missing_change = expected_labels - set(change_probabilities)
            if missing_change:
                raise ValueError(
                    f"missing change probabilities: {sorted(missing_change)}"
                )
            change_weighted_loss = 0.0
            change_weight_total = 0.0
            for label in expected_labels:
                square_loss = _bernoulli_loss(
                    change_probabilities[label], label in changed
                )
                weight = 6.0 if label in changed else 1.0
                if label not in changed:
                    square_loss = min(square_loss, 2.0)
                change_weighted_loss += weight * square_loss
                change_weight_total += weight
            change_loss = change_weighted_loss / change_weight_total
            occupancy_weight = 0.68

        loss = occupancy_weight * occupancy_loss + (1.0 - occupancy_weight) * change_loss
        mismatches = tuple(
            sorted(
                label
                for label in expected_labels
                if (occupancy_probabilities[label] >= 0.5)
                != expected_occupancy[label]
            )
        )

        # Prefer a move whose expected occupancy actually differs from the
        # current logical board; this prevents a capture destination from being
        # treated as the only useful signal.
        if not any(
            current_occupancy[label] != expected_occupancy[label]
            for label in expected_labels
        ):
            loss += 0.05
        scored.append(
            (
                loss,
                occupancy_loss,
                change_loss,
                move,
                san,
                tuple(sorted(changed)),
                mismatches,
            )
        )

    if not scored:
        return None
    scored.sort(key=lambda item: item[0])
    (
        best_loss,
        best_occupancy_loss,
        best_change_loss,
        best_move,
        best_san,
        changed,
        mismatches,
    ) = scored[0]
    runner_up_loss = scored[1][0] if len(scored) > 1 else best_loss + 1.0
    gap = runner_up_loss - best_loss
    if best_loss > maximum_loss or gap < minimum_runner_up_gap:
        return None

    fit = max(0.0, min(1.0, 1.0 - best_loss / maximum_loss))
    separation = max(0.0, min(1.0, gap / 0.12))
    confidence = 0.7 * fit + 0.3 * separation
    return MoveCandidate(
        move=best_move,
        san=best_san,
        confidence=confidence,
        loss=best_loss,
        occupancy_loss=best_occupancy_loss,
        change_loss=best_change_loss,
        runner_up_gap=gap,
        expected_changed_squares=changed,
        occupancy_mismatches=mismatches,
    )


def observation_matches_board(
    board: chess.Board,
    occupancy_probabilities: dict[str, float],
    maximum_mismatches: int = 0,
    uncertainty_margin: float = 0.08,
    maximum_uncertain: int = 2,
) -> bool:
    """Return whether confident observations agree with the logical board.

    Probabilities close to 0.5 are treated as unknown rather than as hard
    contradictions. A small cap prevents a generally poor observation from
    being accepted merely because many squares are uncertain.
    """

    expected = occupancy_for_board(board)
    lower = 0.5 - uncertainty_margin
    upper = 0.5 + uncertainty_margin
    mismatches = 0
    uncertain = 0
    for label, occupied in expected.items():
        probability = occupancy_probabilities[label]
        if lower <= probability <= upper:
            uncertain += 1
        elif (probability > upper) != occupied:
            mismatches += 1
    return mismatches <= maximum_mismatches and uncertain <= maximum_uncertain


def verify_move_result(
    board_before: chess.Board,
    move: chess.Move,
    occupancy_probabilities: dict[str, float],
    maximum_mismatches: int = 0,
) -> MoveVerification:
    """Verify that an observed board matches the expected result of a move."""

    if move not in board_before.legal_moves:
        raise ValueError(f"move is not legal in the supplied position: {move.uci()}")
    expected_board = board_before.copy(stack=False)
    expected_board.push(move)
    expected = occupancy_for_board(expected_board)
    missing = set(expected) - set(occupancy_probabilities)
    if missing:
        raise ValueError(f"missing occupancy probabilities: {sorted(missing)}")
    mismatches = tuple(
        sorted(
            label
            for label, occupied in expected.items()
            if (occupancy_probabilities[label] >= 0.5) != occupied
        )
    )
    return MoveVerification(
        success=len(mismatches) <= maximum_mismatches,
        move_uci=move.uci(),
        expected_fen=expected_board.fen(),
        mismatched_squares=mismatches,
    )
