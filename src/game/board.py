"""Generic H×W 2048 board engine (Milestone M1 of the variable-grid plan).

The existing :class:`~src.game.model.staticboardImpl.NumpyStaticBoard` only works
on **square** boards: its ``move`` iterates ``range(len(matrix))`` (= number of
rows) even for column moves, so a 3×7 board silently loses columns. This module
is a clean, dynamic replacement that supports any ``3 ≤ H, W ≤ 8`` (and beyond),
keeping the exact 2048 slide/merge semantics.

Design decisions
----------------
* **Exponent representation.** A board is an ``int8`` array of *exponents*:
  ``0`` = empty, ``k`` = tile ``2**k``. This is what the n-tuple value function
  indexes on directly and what the vectorized GPU engine already uses, so no
  conversion happens on the hot path. :func:`from_values` / :func:`to_values`
  bridge to the legacy raw-value boards for parity tests.
* **Afterstate / spawn separation.** :func:`move` returns
  ``(afterstate, reward, changed)`` — the deterministic board *after* merges,
  *before* the random spawn — exactly the triple afterstate-TD needs. The random
  tile is added separately by :func:`spawn`, which takes an explicit
  ``np.random.Generator`` so seeded games are reproducible.
* **Merge reward in raw units.** ``reward`` is the sum of the raw values of tiles
  created this move (merging two ``2**k`` gives ``2**(k+1)`` and adds
  ``2**(k+1)``), matching the legacy engine's score increment.
* **numba hot paths.** :func:`move` and :func:`is_done` are ``@njit``; the
  per-line collapse is inlined into ``_move`` to avoid per-call allocation.

Directions are the module ints :data:`UP`/:data:`DOWN`/:data:`LEFT`/:data:`RIGHT`
(0..3). :data:`DIR_FROM_STR` maps the legacy string constants for interop.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from numba import njit

from src.game import utils as _u

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
DIRECTIONS = (UP, DOWN, LEFT, RIGHT)
DIR_FROM_STR = {_u.UP: UP, _u.DOWN: DOWN, _u.LEFT: LEFT, _u.RIGHT: RIGHT}
DIR_TO_STR = {v: k for k, v in DIR_FROM_STR.items()}

MAX_EXPONENT = 15               # tiles clipped to 2**15 = 32768 in the LUT alphabet

# Powers of two as int64 so merge rewards never overflow an int8 shift in numba.
_POW2 = (np.int64(2) ** np.arange(20)).astype(np.int64)


@njit(cache=True)
def _move(board, direction):
    """Slide+merge ``board`` (int8 exponents) in ``direction``.

    Returns ``(afterstate, reward, changed)``. ``afterstate`` is a fresh array;
    ``board`` is never mutated. Merge reward is in raw tile units.
    """
    H, W = board.shape
    out = np.zeros((H, W), dtype=board.dtype)
    reward = np.int64(0)
    changed = False
    horizontal = (direction == LEFT or direction == RIGHT)
    toward_zero = (direction == LEFT or direction == UP)
    n = W if horizontal else H          # length of each line
    nlines = H if horizontal else W     # number of lines
    seq = np.empty(n, dtype=np.int64)   # non-zero tiles in travel order

    for li in range(nlines):
        # Gather non-zero exponents nearest-wall-first.
        m = 0
        if toward_zero:
            for j in range(n):
                v = board[li, j] if horizontal else board[j, li]
                if v != 0:
                    seq[m] = v
                    m += 1
        else:
            for j in range(n - 1, -1, -1):
                v = board[li, j] if horizontal else board[j, li]
                if v != 0:
                    seq[m] = v
                    m += 1
        # Merge equal neighbours once, writing straight into ``out``.
        k = 0
        i = 0
        while i < m:
            if i + 1 < m and seq[i] == seq[i + 1]:
                mval = seq[i] + 1
                reward += _POW2[mval]
                pos = k if toward_zero else n - 1 - k
                if horizontal:
                    out[li, pos] = mval
                else:
                    out[pos, li] = mval
                k += 1
                i += 2
            else:
                pos = k if toward_zero else n - 1 - k
                if horizontal:
                    out[li, pos] = seq[i]
                else:
                    out[pos, li] = seq[i]
                k += 1
                i += 1
        # Did this line change?
        for j in range(n):
            a = out[li, j] if horizontal else out[j, li]
            b = board[li, j] if horizontal else board[j, li]
            if a != b:
                changed = True
    return out, reward, changed


@njit(cache=True)
def _is_done(board):
    """True iff no empty cell and no adjacent equal pair (no legal move)."""
    H, W = board.shape
    for r in range(H):
        for c in range(W):
            if board[r, c] == 0:
                return False
            if c + 1 < W and board[r, c] == board[r, c + 1]:
                return False
            if r + 1 < H and board[r, c] == board[r + 1, c]:
                return False
    return True


# --------------------------------------------------------------------------- #
# Public API (thin Python wrappers around the njit kernels).
# --------------------------------------------------------------------------- #
def empty(H: int, W: int) -> np.ndarray:
    """An all-zero ``H×W`` exponent board."""
    return np.zeros((H, W), dtype=np.int8)


def move(board: np.ndarray, direction: int) -> Tuple[np.ndarray, int, bool]:
    """``(afterstate, reward, changed)`` for ``direction`` (does not spawn)."""
    b = np.ascontiguousarray(board, dtype=np.int8)
    after, reward, changed = _move(b, direction)
    return after, int(reward), bool(changed)


def all_afterstates(board: np.ndarray):
    """All four ``(afterstate, reward, changed)`` triples, indexed by direction."""
    b = np.ascontiguousarray(board, dtype=np.int8)
    return [move(b, d) for d in DIRECTIONS]


def is_done(board: np.ndarray) -> bool:
    return bool(_is_done(np.ascontiguousarray(board, dtype=np.int8)))


def spawn(board: np.ndarray, rng: np.random.Generator,
          inplace: bool = False) -> Tuple[np.ndarray, bool]:
    """Place one random tile (exp 1 = "2" w.p. 0.9, exp 2 = "4" w.p. 0.1).

    Uniform over empty cells using ``rng`` (reproducible). Returns
    ``(board, spawned)``; ``spawned`` is False only on a full board.
    """
    empties = np.argwhere(board == 0)
    if len(empties) == 0:
        return board, False
    if not inplace:
        board = board.copy()
    r, c = empties[rng.integers(len(empties))]
    board[r, c] = 1 if rng.random() < 0.9 else 2
    return board, True


def new_game(H: int, W: int, rng: np.random.Generator) -> np.ndarray:
    """Fresh board with two random spawns (standard 2048 start)."""
    b = empty(H, W)
    spawn(b, rng, inplace=True)
    spawn(b, rng, inplace=True)
    return b


def max_tile(board: np.ndarray) -> int:
    """Largest raw tile value on the board (0 for an empty board)."""
    m = int(board.max())
    return (1 << m) if m > 0 else 0


def board_key(board: np.ndarray) -> bytes:
    """Compact, shape-aware hash key for transposition tables.

    Prefixed with ``H`` and ``W`` so different shapes (and H×W vs W×H) never
    collide, followed by the raw exponent bytes.
    """
    b = np.ascontiguousarray(board, dtype=np.int8)
    H, W = b.shape
    return bytes((H, W)) + b.tobytes()


# --------------------------------------------------------------------------- #
# Bridges to the legacy raw-value engine (for parity tests & GUI interop).
# --------------------------------------------------------------------------- #
def from_values(matrix: np.ndarray) -> np.ndarray:
    """Raw tile values (0, 2, 4, 8, …) → exponent board (0, 1, 2, 3, …)."""
    m = np.asarray(matrix)
    out = np.zeros(m.shape, dtype=np.int8)
    nz = m > 0
    out[nz] = np.round(np.log2(m[nz])).astype(np.int8)
    return out


def to_values(board: np.ndarray) -> np.ndarray:
    """Exponent board → raw tile values (int64), inverse of :func:`from_values`."""
    b = np.asarray(board)
    out = np.zeros(b.shape, dtype=np.int64)
    nz = b > 0
    out[nz] = (np.int64(1) << b[nz].astype(np.int64))
    return out
