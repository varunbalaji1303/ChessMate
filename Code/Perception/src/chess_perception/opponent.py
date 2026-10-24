"""Stockfish-backed opponent move selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

import chess
import chess.engine


class StockfishUnavailableError(RuntimeError):
    """Raised when Stockfish cannot be started or configured."""


@dataclass(frozen=True)
class EngineStrength:
    requested_elo: int
    applied_uci_elo: int
    mode: str

    @property
    def exact(self) -> bool:
        return self.requested_elo == self.applied_uci_elo


@dataclass(frozen=True)
class EngineMove:
    move: chess.Move
    san: str
    engine_name: str
    strength: EngineStrength
    think_time_seconds: float


def resolve_stockfish_path(command: str | Path) -> str:
    """Resolve a Stockfish executable supplied as a path or command name."""

    candidate = str(command)
    if "/" in candidate:
        path = Path(candidate).expanduser()
        if path.is_file():
            return str(path.resolve())
        raise StockfishUnavailableError(f"Stockfish executable not found: {path}")
    resolved = shutil.which(candidate)
    if resolved is None:
        raise StockfishUnavailableError(
            f"Stockfish executable '{candidate}' was not found on PATH"
        )
    return resolved


def strength_for_options(
    requested_elo: int,
    options: dict[str, chess.engine.Option],
) -> EngineStrength:
    """Select the closest honest UCI strength supported by an engine."""

    if requested_elo <= 0:
        raise ValueError("Stockfish Elo must be positive")
    elo_option = options.get("UCI_Elo")
    limit_option = options.get("UCI_LimitStrength")
    if elo_option is None or limit_option is None:
        raise StockfishUnavailableError(
            "Stockfish does not expose UCI_LimitStrength and UCI_Elo"
        )
    minimum = int(elo_option.min) if elo_option.min is not None else requested_elo
    maximum = int(elo_option.max) if elo_option.max is not None else requested_elo
    applied = min(max(requested_elo, minimum), maximum)
    mode = "native_uci_elo" if applied == requested_elo else "nearest_supported_uci_elo"
    return EngineStrength(requested_elo, applied, mode)


class StockfishOpponent:
    """Choose Black replies with a persistent Stockfish UCI process."""

    def __init__(
        self,
        executable: str | Path = "stockfish",
        requested_elo: int = 1200,
        think_time_seconds: float = 0.2,
    ) -> None:
        if think_time_seconds <= 0:
            raise ValueError("Stockfish think time must be positive")
        self.executable = executable
        self.requested_elo = requested_elo
        self.think_time_seconds = think_time_seconds
        self._engine: chess.engine.SimpleEngine | None = None
        self.strength: EngineStrength | None = None
        self.engine_name = "Stockfish"

    def start(self) -> "StockfishOpponent":
        if self._engine is not None:
            return self
        executable = resolve_stockfish_path(self.executable)
        engine: chess.engine.SimpleEngine | None = None
        try:
            engine = chess.engine.SimpleEngine.popen_uci(executable)
            strength = strength_for_options(self.requested_elo, engine.options)
            engine.configure(
                {
                    "UCI_LimitStrength": True,
                    "UCI_Elo": strength.applied_uci_elo,
                }
            )
        except (OSError, chess.engine.EngineError, ValueError) as error:
            if engine is not None:
                try:
                    engine.quit()
                except Exception:
                    pass
            raise StockfishUnavailableError(
                f"Could not start/configure Stockfish: {error}"
            ) from error
        self._engine = engine
        self.strength = strength
        self.engine_name = str(engine.id.get("name", "Stockfish"))
        return self

    def choose_move(self, board: chess.Board) -> EngineMove:
        if self._engine is None or self.strength is None:
            raise StockfishUnavailableError("Stockfish has not been started")
        if board.turn != chess.BLACK:
            raise ValueError("Stockfish opponent can only move when Black is to move")
        if board.is_game_over():
            raise ValueError("Stockfish cannot move because the game is over")
        try:
            result = self._engine.play(
                board,
                chess.engine.Limit(time=self.think_time_seconds),
            )
        except chess.engine.EngineError as error:
            raise StockfishUnavailableError(f"Stockfish move failed: {error}") from error
        if result.move is None or result.move not in board.legal_moves:
            raise StockfishUnavailableError("Stockfish did not return a legal move")
        return EngineMove(
            move=result.move,
            san=board.san(result.move),
            engine_name=self.engine_name,
            strength=self.strength,
            think_time_seconds=self.think_time_seconds,
        )

    def close(self) -> None:
        if self._engine is not None:
            try:
                self._engine.quit()
            finally:
                self._engine = None

    def __enter__(self) -> "StockfishOpponent":
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
