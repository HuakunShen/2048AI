"""Default pattern library for the universal value function (plan §4.2, App. B).

All shapes are relative-coordinate :class:`~src.ntuple.pattern.Pattern` objects,
so the compiler generates their orientations and translations on any board. A
pattern whose footprint is larger than the board simply contributes zero
placements there (e.g. ``line_6`` on 4×4), which is exactly how one model serves
3×3 through 8×8.

* ``CORE`` — shapes that fit a 4×4 board, carrying 4×4 performance.
* ``EXTENDED`` — longer shapes that add capacity on larger boards.
"""
from src.ntuple.pattern import Pattern

SQUARE_2X2 = Pattern("square_2x2", ((0, 0), (0, 1), (1, 0), (1, 1)))
LINE_4 = Pattern("line_4", ((0, 0), (0, 1), (0, 2), (0, 3)))
L_4 = Pattern("l_4", ((0, 0), (1, 0), (2, 0), (2, 1)))
RECT_2X3 = Pattern("rect_2x3", ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)))
CORNER_6 = Pattern("corner_6", ((0, 0), (0, 1), (0, 2), (1, 0), (2, 0), (1, 1)))
LINE_5 = Pattern("line_5", ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4)))
LINE_6 = Pattern("line_6", ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5)))
SNAKE_6 = Pattern("snake_6", ((0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3)))

# Extra 6-tuple shapes that fit 4×4, for the specialist's capacity (distinct
# lookup tables ≈ more expressive V, following strong published 8×6-tuple nets).
ROW_PLUS_6 = Pattern("row_plus_6", ((0, 0), (0, 1), (0, 2), (0, 3), (1, 1), (1, 2)))
ROW_EDGE_6 = Pattern("row_edge_6", ((0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)))

CORE = [SQUARE_2X2, LINE_4, L_4, RECT_2X3, CORNER_6]
EXTENDED = [LINE_5, LINE_6, SNAKE_6]
DEFAULT_PATTERNS = CORE + EXTENDED

# 4×4 specialist: small tables (square/line-4) + six distinct 6-tuple tables for
# maximum capacity on a fixed board. Pair with a wide alphabet (18 -> 131072) and
# deep expectimax to chase the highest possible tile.
SPECIALIST_4X4 = [SQUARE_2X2, LINE_4, RECT_2X3, CORNER_6, SNAKE_6,
                  ROW_PLUS_6, ROW_EDGE_6]


def with_alphabet(patterns, alphabet):
    """Clone patterns with a wider tile alphabet (e.g. 18 -> tiles up to 131072).

    Used by the 4×4 specialist so the value function can *distinguish* the high
    endgame tiles (32768/65536/131072) instead of clipping them all to 32768.
    """
    from dataclasses import replace
    return [replace(p, alphabet=alphabet) for p in patterns]
