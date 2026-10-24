"""Small FEN helpers that avoid a runtime python-chess dependency."""

from __future__ import annotations


def occupied_squares_from_fen(fen: str) -> set[str]:
    """Return occupied algebraic squares from a complete or board-only FEN."""

    placement = fen.strip().split()[0]
    ranks = placement.split("/")
    if len(ranks) != 8:
        raise ValueError("FEN placement must contain eight ranks")

    occupied: set[str] = set()
    for row, rank_text in enumerate(ranks):
        file_index = 0
        rank_number = 8 - row
        for character in rank_text:
            if character.isdigit():
                count = int(character)
                if count < 1 or count > 8:
                    raise ValueError("invalid empty-square count in FEN")
                file_index += count
            elif character in "prnbqkPRNBQK":
                if file_index >= 8:
                    raise ValueError("too many squares in FEN rank")
                occupied.add(f"{chr(ord('a') + file_index)}{rank_number}")
                file_index += 1
            else:
                raise ValueError(f"invalid FEN piece character: {character}")
        if file_index != 8:
            raise ValueError("each FEN rank must describe eight squares")
    return occupied
