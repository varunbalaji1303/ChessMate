import chess

from chess_perception.move_inference import (
    changed_squares_for_move,
    infer_legal_move,
    observation_matches_board,
    occupancy_for_board,
    verify_move_result,
)


def _probabilities_for_board(board: chess.Board) -> dict[str, float]:
    return {
        label: 0.999 if occupied else 0.001
        for label, occupied in occupancy_for_board(board).items()
    }


def _change_probabilities(changed: set[str]) -> dict[str, float]:
    return {
        chess.square_name(square): (
            0.999 if chess.square_name(square) in changed else 0.001
        )
        for square in chess.SQUARES
    }


def test_infers_normal_move_from_starting_position() -> None:
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    after = board.copy(stack=False)
    after.push(move)

    candidate = infer_legal_move(
        board,
        _probabilities_for_board(after),
        _change_probabilities({"e2", "e4"}),
    )

    assert candidate is not None
    assert candidate.move == move


def test_infers_second_move_when_source_probability_is_borderline() -> None:
    board = chess.Board()
    board.push_uci("e2e3")
    move = chess.Move.from_uci("d7d5")
    after = board.copy(stack=False)
    after.push(move)
    probabilities = {
        label: 0.92 if occupied else 0.08
        for label, occupied in occupancy_for_board(after).items()
    }
    probabilities["d7"] = 0.52
    probabilities["d5"] = 0.999

    candidate = infer_legal_move(board, probabilities)

    assert candidate is not None
    assert candidate.move == move


def test_board_match_tolerates_only_a_few_uncertain_squares() -> None:
    board = chess.Board()
    probabilities = _probabilities_for_board(board)
    probabilities["a3"] = 0.52

    assert observation_matches_board(board, probabilities)

    probabilities["b3"] = 0.48
    probabilities["c3"] = 0.51
    assert not observation_matches_board(board, probabilities)


def test_infers_capture_when_destination_stays_occupied() -> None:
    board = chess.Board("8/8/8/3p4/4P3/8/4K3/7k w - - 0 1")
    move = chess.Move.from_uci("e4d5")
    assert move in board.legal_moves
    after = board.copy(stack=False)
    after.push(move)

    candidate = infer_legal_move(
        board,
        _probabilities_for_board(after),
        _change_probabilities({"e4", "d5"}),
    )

    assert candidate is not None
    assert candidate.move == move


def test_capture_confidence_tolerates_unrelated_stale_square() -> None:
    board = chess.Board()
    for uci in ("e2e4", "d7d5"):
        board.push_uci(uci)
    move = chess.Move.from_uci("e4d5")
    after = board.copy(stack=False)
    after.push(move)
    probabilities = _probabilities_for_board(after)
    # A stale d7 false positive is unrelated to the e4xd5 capture and should
    # be reported as a mismatch without destroying candidate confidence.
    probabilities["d7"] = 0.999

    candidate = infer_legal_move(
        board,
        probabilities,
        _change_probabilities({"e4", "d5"}),
    )

    assert candidate is not None
    assert candidate.move == move
    assert candidate.confidence > 0.7
    assert candidate.occupancy_mismatches == ("d7",)


def test_castling_marks_king_and_rook_squares_changed() -> None:
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    move = chess.Move.from_uci("e1g1")
    assert changed_squares_for_move(board, move) == {"e1", "f1", "g1", "h1"}


def test_verifies_commanded_robot_move() -> None:
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    after = board.copy(stack=False)
    after.push(move)

    result = verify_move_result(board, move, _probabilities_for_board(after))

    assert result.success
    assert result.mismatched_squares == ()


def test_reports_square_that_does_not_match_commanded_move() -> None:
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    probabilities = _probabilities_for_board(board)

    result = verify_move_result(board, move, probabilities)

    assert not result.success
    assert result.mismatched_squares == ("e2", "e4")
