import json

import chess
import chess.engine

from chess_perception.live import _accept_human_move
from chess_perception.move_inference import MoveCandidate
from chess_perception.opponent import EngineMove, EngineStrength, strength_for_options


def _option(name: str, option_type: str, default, minimum=None, maximum=None):
    return chess.engine.Option(name, option_type, default, minimum, maximum, [])


def test_requested_1200_uses_stockfish_minimum_without_claiming_exact() -> None:
    strength = strength_for_options(
        1200,
        {
            "UCI_LimitStrength": _option(
                "UCI_LimitStrength", "check", False
            ),
            "UCI_Elo": _option("UCI_Elo", "spin", 1320, 1320, 3190),
        },
    )

    assert strength.requested_elo == 1200
    assert strength.applied_uci_elo == 1320
    assert strength.mode == "nearest_supported_uci_elo"
    assert not strength.exact


def test_native_supported_elo_is_applied_exactly() -> None:
    strength = strength_for_options(
        1500,
        {
            "UCI_LimitStrength": _option(
                "UCI_LimitStrength", "check", False
            ),
            "UCI_Elo": _option("UCI_Elo", "spin", 1320, 1320, 3190),
        },
    )

    assert strength.applied_uci_elo == 1500
    assert strength.mode == "native_uci_elo"
    assert strength.exact


class _BlackOpponent:
    def choose_move(self, board: chess.Board) -> EngineMove:
        move = chess.Move.from_uci("e7e5")
        assert board.turn == chess.BLACK
        assert move in board.legal_moves
        return EngineMove(
            move=move,
            san=board.san(move),
            engine_name="Stockfish test",
            strength=EngineStrength(1200, 1320, "nearest_supported_uci_elo"),
            think_time_seconds=0.2,
        )


def test_accepting_white_move_logs_human_then_requests_black_reply(tmp_path) -> None:
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    candidate = MoveCandidate(
        move=move,
        san=board.san(move),
        confidence=0.9,
        loss=0.1,
        occupancy_loss=0.1,
        change_loss=0.1,
        runner_up_gap=0.2,
        expected_changed_squares=("e2", "e4"),
        occupancy_mismatches=(),
    )
    event_log = tmp_path / "moves.jsonl"

    reply = _accept_human_move(
        candidate,
        board,
        event_log,
        None,
        "test",
        0.5,
        _BlackOpponent(),
    )

    assert reply is not None
    assert reply.move.uci() == "e7e5"
    assert board.fen().startswith("rnbqkbnr/pppppppp/8/8/4P3")
    assert board.turn == chess.BLACK
    events = [json.loads(line) for line in event_log.read_text().splitlines()]
    assert [event["event"] for event in events] == [
        "move_detected",
        "engine_move_requested",
    ]
    assert events[1]["uci"] == "e7e5"
    assert events[1]["requested_elo"] == 1200
    assert events[1]["applied_uci_elo"] == 1320
    assert events[1]["color"] == "black"
