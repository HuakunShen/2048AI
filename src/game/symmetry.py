"""Dihedral symmetry helpers for patterns and boards (Milestone M2).

Two related uses:

* **Pattern orientations** — :func:`orient_cells` applies the 8 plane symmetries
  (the dihedral group D4) to a pattern's relative coordinates and normalises each
  result to the non-negative quadrant. This is how one n-tuple table is *shared*
  across a board's symmetries: reading a pattern in all of its orientations (and
  all translations) into a single table makes the value function invariant to
  board symmetries by construction.

* **Board automorphisms** — :func:`board_automorphisms` returns the symmetries
  that map an ``H×W`` board *to itself* as flat-index permutations. For a square
  board that is all 8 of D4; for a non-square board only the 4 that preserve the
  shape (identity, 180° rotation, horizontal flip, vertical flip). The remaining
  four (90°/270° rotation, the two diagonal reflections) map ``H×W → W×H`` and
  are handled as cross-shape sharing elsewhere (plan §4.4).
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

# The 8 elements of D4 acting on a coordinate ``(r, c)`` in the plane. Four
# rotations, and each rotation composed with a horizontal flip ``c -> -c``.
_D4 = (
    lambda r, c: (r, c),        # identity
    lambda r, c: (c, -r),       # rot90
    lambda r, c: (-r, -c),      # rot180
    lambda r, c: (-c, r),       # rot270
    lambda r, c: (r, -c),       # flip
    lambda r, c: (c, r),        # flip · rot90  (main diagonal)
    lambda r, c: (-r, c),       # flip · rot180 (vertical flip)
    lambda r, c: (-c, -r),      # flip · rot270 (anti-diagonal)
)


def orient_cells(cells: List[Tuple[int, int]]) -> List[np.ndarray]:
    """All distinct oriented footprints of ``cells`` under D4.

    Each result is an ``[L, 2]`` int array of ``(row, col)`` coordinates
    normalised so the minimum row and column are 0, with the **same index order**
    as the input (so ``result[j]`` corresponds to input cell ``j``). Orientations
    that produce an identical ordered coordinate sequence are de-duplicated, so a
    symmetric pattern yields fewer than 8 orientations.
    """
    seen = set()
    out = []
    for g in _D4:
        pts = np.array([g(r, c) for r, c in cells], dtype=np.int64)
        pts -= pts.min(axis=0)                      # normalise to quadrant
        key = tuple(map(tuple, pts.tolist()))
        if key not in seen:
            seen.add(key)
            out.append(pts)
    return out


def _flat_perms(H: int, W: int):
    """The 8 D4 transforms of the ``H×W`` index grid, as flattened arrays.

    Generated as the four rotations of the grid and of its mirror (the approach
    proven correct in ``ntuple.py``), which are guaranteed distinct for a square
    grid. Returns ``(same, cross)`` where ``same`` keep the ``H×W`` shape (the
    board automorphisms) and ``cross`` map to ``W×H`` (odd rotations of a
    rectangle), used for cross-shape sharing.
    """
    grid = np.arange(H * W).reshape(H, W)
    same, cross = [], []
    for flip in (False, True):
        base = np.fliplr(grid) if flip else grid
        for k in range(4):
            v = np.rot90(base, k)
            (same if v.shape == (H, W) else cross).append(v.flatten())
    return same, cross


def board_automorphisms(H: int, W: int) -> np.ndarray:
    """Flat-index permutations mapping an ``H×W`` board to itself.

    ``perm[p]`` is the source cell whose value lands at flat position ``p`` after
    the transform. 8 permutations for a square board, 4 for a rectangle.
    """
    same, _ = _flat_perms(H, W)
    # De-duplicate (a square grid's identity/flip variants are all distinct, but
    # guard against accidental repeats for tiny boards).
    uniq, out = set(), []
    for p in same:
        key = tuple(p.tolist())
        if key not in uniq:
            uniq.add(key)
            out.append(p)
    return np.array(out, dtype=np.int64)
