"""Vectorized 2048 engine: step thousands of boards per call (CPU numpy / GPU torch).

Rationale
---------
The per-board :class:`NumpyStaticBoard` engine is perfect for classical agents and
1-ply n-tuple play, but GPU reinforcement learning needs to advance *many* games
at once so the network sees large batches. A single 2048 move decomposes into
four independent **row** operations, and a row of 4 cells (each an exponent in
0..15) has only ``16**4 = 65,536`` possible states. So we precompute, once, the
full transition table for the canonical "collapse toward index 0" operation and
then every move becomes a table lookup applied to a whole batch of boards.

Exactness
---------
The lookup tables are built by *calling the real* ``NumpyStaticBoard.collapse_array``
on all 65,536 rows, so the vectorized engine reproduces the existing engine's
merge, tie-break, and scoring semantics bit-for-bit. ``tests/test_vectorboard.py``
asserts this on 10^5 random boards x 4 directions.

Board representation
--------------------
Boards are **exponent** grids (int8), shape ``[N, 4, 4]``, where 0 means empty and
``k`` means tile ``2**k``. Convert with :func:`tiles_to_exp` / :func:`exp_to_tiles`.

Direction mapping (matches NumpyStaticBoard):
  LEFT  = collapse rows toward col 0        (reverse=True on rows)
  RIGHT = collapse rows toward col 3        (reverse=False)
  UP    = collapse cols toward row 0        (reverse=True on cols)
  DOWN  = collapse cols toward row 3        (reverse=False)
"""
from __future__ import annotations

import numpy as np

from src.game.model.staticboardImpl import NumpyStaticBoard
from src.game.utils import UP, DOWN, LEFT, RIGHT

GRID = 4
N_ROW_STATES = 16 ** GRID          # 65,536
POW4 = (16 ** np.arange(GRID)).astype(np.int64)   # [1, 16, 256, 4096]
MAX_EXPONENT = 15

# Direction order used by the vectorized batch move (indices into the stacked
# output of move_batch). Kept explicit so callers can map action ids <-> names.
DIRECTIONS = (UP, DOWN, LEFT, RIGHT)


def _build_row_luts():
    """Build (result_index, reward, changed) tables for collapse-toward-index-0.

    Row ``index`` encodes 4 exponents base-16 (col 0 is the least significant
    digit). The canonical op is ``collapse_array(row, reverse=True)`` — the LEFT
    primitive; every other direction is a flip/transpose of it.
    """
    result = np.zeros(N_ROW_STATES, dtype=np.int64)
    reward = np.zeros(N_ROW_STATES, dtype=np.int64)
    changed = np.zeros(N_ROW_STATES, dtype=bool)

    for index in range(N_ROW_STATES):
        exps = [(index >> (4 * c)) & 0xF for c in range(GRID)]
        tiles = np.array([0 if e == 0 else (1 << e) for e in exps], dtype=np.int64)
        out, score, chg = NumpyStaticBoard.collapse_array(tiles.copy(), reverse=True)
        out_exps = [0 if v == 0 else int(round(np.log2(v))) for v in out]
        out_index = sum(e << (4 * c) for c, e in enumerate(out_exps))
        result[index] = out_index
        reward[index] = score
        changed[index] = chg
    return result, reward, changed


# Built once at import (≈65k numba-backed calls, well under a second).
_ROW_RESULT, _ROW_REWARD, _ROW_CHANGED = _build_row_luts()


# --------------------------------------------------------------------------- #
# Encoding helpers.
# --------------------------------------------------------------------------- #
def tiles_to_exp(boards: np.ndarray) -> np.ndarray:
    """Tile-value boards ``[..., 4, 4]`` -> exponent boards (int8)."""
    boards = np.asarray(boards)
    exp = np.zeros_like(boards, dtype=np.int8)
    nz = boards > 0
    exp[nz] = np.log2(boards[nz]).round().astype(np.int8)
    return exp


def exp_to_tiles(exp: np.ndarray) -> np.ndarray:
    """Exponent boards -> tile-value boards (int64)."""
    exp = np.asarray(exp)
    out = np.zeros(exp.shape, dtype=np.int64)
    nz = exp > 0
    out[nz] = (1 << exp[nz].astype(np.int64))
    return out


def _rows_to_index(rows_exp: np.ndarray) -> np.ndarray:
    """``[..., 4]`` exponent rows -> ``[...]`` base-16 indices."""
    return (rows_exp.astype(np.int64) * POW4).sum(axis=-1)


def _index_to_rows(index: np.ndarray) -> np.ndarray:
    """``[...]`` base-16 indices -> ``[..., 4]`` exponent rows (int8)."""
    shifts = (4 * np.arange(GRID)).astype(np.int64)
    return ((index[..., None] >> shifts) & 0xF).astype(np.int8)


# --------------------------------------------------------------------------- #
# Batched move — numpy.
# --------------------------------------------------------------------------- #
def _left_collapse(boards_exp: np.ndarray):
    """Collapse rows toward col 0 for a batch ``[N,4,4]``. -> (new, reward, changed)."""
    idx = _rows_to_index(boards_exp)                 # [N,4]
    new_idx = _ROW_RESULT[idx]                       # [N,4]
    reward = _ROW_REWARD[idx].sum(axis=1)            # [N]
    changed = _ROW_CHANGED[idx].any(axis=1)          # [N]
    new_boards = _index_to_rows(new_idx)             # [N,4,4]
    return new_boards, reward, changed


def move_batch(boards_exp: np.ndarray, direction: str):
    """Apply ``direction`` to a batch of exponent boards ``[N,4,4]``.

    Returns ``(afterstate_exp [N,4,4], reward [N], changed [N])``. No tile is
    spawned — this is the afterstate, matching ``NumpyStaticBoard.move(inplace=False)``.
    """
    b = boards_exp
    if direction == LEFT:
        out, r, c = _left_collapse(b)
    elif direction == RIGHT:
        out, r, c = _left_collapse(b[:, :, ::-1])
        out = out[:, :, ::-1]
    elif direction == UP:
        out, r, c = _left_collapse(b.transpose(0, 2, 1))
        out = out.transpose(0, 2, 1)
    elif direction == DOWN:
        out, r, c = _left_collapse(b.transpose(0, 2, 1)[:, :, ::-1])
        out = out[:, :, ::-1].transpose(0, 2, 1)
    else:
        raise ValueError(f"invalid direction: {direction}")
    return np.ascontiguousarray(out), r, c


def all_afterstates(boards_exp: np.ndarray):
    """Afterstates for all 4 directions at once.

    Returns ``(after [4,N,4,4], reward [4,N], changed [4,N])`` in DIRECTIONS order.
    """
    outs, rewards, changes = [], [], []
    for d in DIRECTIONS:
        o, r, c = move_batch(boards_exp, d)
        outs.append(o); rewards.append(r); changes.append(c)
    return np.stack(outs), np.stack(rewards), np.stack(changes)


# --------------------------------------------------------------------------- #
# Batched spawn + done-check.
# --------------------------------------------------------------------------- #
def spawn_batch(boards_exp: np.ndarray, rng: np.random.Generator, mask=None):
    """Spawn one tile (2 w.p. .9, 4 w.p. .1) on a random empty cell per board.

    ``mask`` (bool ``[N]``) restricts spawning to selected boards (e.g. only ones
    that just changed). Operates in place and returns the array.
    """
    n = boards_exp.shape[0]
    flat = boards_exp.reshape(n, 16)
    idx = np.arange(n) if mask is None else np.nonzero(mask)[0]
    for i in idx:
        empties = np.nonzero(flat[i] == 0)[0]
        if len(empties) == 0:
            continue
        cell = empties[rng.integers(len(empties))]
        flat[i, cell] = 2 if rng.random() >= 0.1 else 1   # exponent 1==>2, 2==>4
    return boards_exp


def done_batch(boards_exp: np.ndarray) -> np.ndarray:
    """Vectorized game-over check for a batch ``[N,4,4]`` -> bool ``[N]``.

    A board is done iff it has no empty cell AND no two orthogonally-adjacent
    equal cells (no horizontal or vertical merge available).
    """
    b = boards_exp
    has_empty = (b == 0).any(axis=(1, 2))
    horiz_merge = (b[:, :, :-1] == b[:, :, 1:]).any(axis=(1, 2))
    vert_merge = (b[:, :-1, :] == b[:, 1:, :]).any(axis=(1, 2))
    return ~(has_empty | horiz_merge | vert_merge)


# --------------------------------------------------------------------------- #
# GPU engine (torch) — same row LUTs, all ops batched on-device.
# --------------------------------------------------------------------------- #
class TorchVectorEngine:
    """Batched 2048 engine on a torch device (default CUDA).

    Boards are int64 exponent tensors ``[N,4,4]`` on ``device``. All methods keep
    data resident on the device — no host<->GPU syncs on the hot path — so tens of
    thousands of games can be stepped per call for GPU self-play RL.
    """

    def __init__(self, device: str = "cuda"):
        import torch
        self.torch = torch
        self.device = device
        self.result = torch.from_numpy(_ROW_RESULT).to(device)          # [65536]
        self.reward = torch.from_numpy(_ROW_REWARD).to(device)
        self.changed = torch.from_numpy(_ROW_CHANGED).to(device)        # bool
        self.pow4 = torch.tensor([1, 16, 256, 4096], device=device, dtype=torch.int64)
        self.shifts = (torch.arange(GRID, device=device) * 4)

    def new_boards(self, n: int):
        """Fresh boards ``[n,4,4]`` each seeded with two spawned tiles."""
        b = self.torch.zeros((n, GRID, GRID), dtype=self.torch.int64, device=self.device)
        self.spawn(b)
        self.spawn(b)
        return b

    def _left(self, b):
        idx = (b * self.pow4).sum(-1)                                   # [N,4]
        new = (self.result[idx].unsqueeze(-1) >> self.shifts) & 0xF     # [N,4,4]
        reward = self.reward[idx].sum(-1)                               # [N]
        changed = self.changed[idx].any(-1)                            # [N]
        return new, reward, changed

    def move(self, b, direction: str):
        """Apply ``direction`` to ``[N,4,4]`` -> ``(after, reward, changed)``."""
        if direction == LEFT:
            out, r, c = self._left(b)
        elif direction == RIGHT:
            out, r, c = self._left(b.flip(-1)); out = out.flip(-1)
        elif direction == UP:
            out, r, c = self._left(b.transpose(1, 2)); out = out.transpose(1, 2)
        elif direction == DOWN:
            out, r, c = self._left(b.transpose(1, 2).flip(-1))
            out = out.flip(-1).transpose(1, 2)
        else:
            raise ValueError(f"invalid direction: {direction}")
        return out.contiguous(), r, c

    def all_afterstates(self, b):
        """``(after [4,N,4,4], reward [4,N], changed [4,N])`` in DIRECTIONS order."""
        outs, rs, cs = [], [], []
        for d in DIRECTIONS:
            o, r, c = self.move(b, d)
            outs.append(o); rs.append(r); cs.append(c)
        t = self.torch
        return t.stack(outs), t.stack(rs), t.stack(cs)

    def spawn(self, b, mask=None):
        """Spawn one tile per board (2 w.p. .9, 4 w.p. .1) on a random empty cell.

        In place. ``mask`` (bool ``[N]``) restricts which boards spawn.
        """
        t = self.torch
        n = b.shape[0]
        flat = b.view(n, 16)
        empty = flat == 0
        # Uniform pick among empties: max of uniforms restricted to empty cells.
        r = t.rand((n, 16), device=self.device)
        r[~empty] = -1.0
        cell = r.argmax(1)                                             # [N]
        has_empty = empty.any(1)
        tile = t.where(t.rand(n, device=self.device) < 0.1,
                       t.tensor(2, device=self.device),
                       t.tensor(1, device=self.device))               # exp 2==4, 1==2
        active = has_empty if mask is None else (has_empty & mask)
        rows = t.arange(n, device=self.device)[active]
        flat[rows, cell[active]] = tile[active]
        return b

    def done(self, b):
        """Game-over check ``[N,4,4]`` -> bool ``[N]``."""
        has_empty = (b == 0).any(dim=(1, 2))
        horiz = (b[:, :, :-1] == b[:, :, 1:]).any(dim=(1, 2))
        vert = (b[:, :-1, :] == b[:, 1:, :]).any(dim=(1, 2))
        return ~(has_empty | horiz | vert)
