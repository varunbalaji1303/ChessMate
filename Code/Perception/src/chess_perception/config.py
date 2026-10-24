"""Configuration parsing for the chessboard perception pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


CORNER_NAMES = ("top_left", "top_right", "bottom_right", "bottom_left")


@dataclass(frozen=True)
class MarkerSpec:
    marker_id: int
    inner_corner_index: int

    def __post_init__(self) -> None:
        if self.marker_id < 0:
            raise ValueError("marker_id must be non-negative")
        if self.inner_corner_index not in range(4):
            raise ValueError("inner_corner_index must be between 0 and 3")


@dataclass(frozen=True)
class BoardConfig:
    output_size_px: int
    files: str
    ranks: str
    aruco_dictionary: str
    markers: dict[str, MarkerSpec]

    def __post_init__(self) -> None:
        if self.output_size_px < 80:
            raise ValueError("output_size_px must be at least 80")
        if len(self.files) != 8 or len(set(self.files)) != 8:
            raise ValueError("files must contain exactly 8 unique labels")
        if len(self.ranks) != 8 or len(set(self.ranks)) != 8:
            raise ValueError("ranks must contain exactly 8 unique labels")
        missing = set(CORNER_NAMES) - set(self.markers)
        if missing:
            raise ValueError(f"missing marker configuration: {sorted(missing)}")
        marker_ids = [self.markers[name].marker_id for name in CORNER_NAMES]
        if len(set(marker_ids)) != 4:
            raise ValueError("the four board-corner marker IDs must be unique")


def load_board_config(path: str | Path) -> BoardConfig:
    """Load and validate a board configuration from TOML."""

    config_path = Path(path)
    with config_path.open("rb") as stream:
        raw = tomllib.load(stream)

    try:
        board = raw["board"]
        aruco = raw["aruco"]
        markers = {
            name: MarkerSpec(
                marker_id=int(aruco[name]["id"]),
                inner_corner_index=int(aruco[name]["inner_corner_index"]),
            )
            for name in CORNER_NAMES
        }
        return BoardConfig(
            output_size_px=int(board["output_size_px"]),
            files=str(board["files"]),
            ranks=str(board["ranks"]),
            aruco_dictionary=str(aruco["dictionary"]),
            markers=markers,
        )
    except KeyError as error:
        raise ValueError(f"missing required config key: {error}") from error
