"""N-tuple network + afterstate TD-learning agent for 2048.

Why this instead of the DQN? In 2048 the *action*-value differences are tiny
relative to the *state* value, so a Q(s,a) network tends to collapse to
Q(s,a) ~= V(s) and its greedy policy degenerates to near-random. The standard,
high-performance approach (Szubert & Jaskowski, 2014) instead learns a value
function over **afterstates** (the deterministic board right after a move's
merges, before the random tile spawns) and selects actions by

    a* = argmax_a  [ reward(s, a) + V(afterstate(s, a)) ]

The explicit reward term sharply separates actions, and V is approximated by an
**n-tuple network**: a set of lookup tables indexed by small groups of cells.
Board symmetries (the dihedral group of 8) share weights, which massively
improves sample efficiency. This fits the existing engine directly:
``NumpyStaticBoard.move(board, dir, inplace=False)`` already returns exactly the
(afterstate, merge_reward, changed) triple we need.
"""
from __future__ import annotations

import numpy as np

from src.game.model.staticboardImpl import NumpyStaticBoard
from src.game.utils import ARROW_KEYS

# Row-major cell indices of the 4x4 board:
#   0  1  2  3
#   4  5  6  7
#   8  9 10 11
#  12 13 14 15
# Four overlapping 6-cell tuples (two "row bands" + two 3x2 rectangles). With
# all 8 board symmetries these cover the board densely and reach the 2048 tile.
TUPLES = (
    (0, 1, 2, 3, 4, 5),
    (4, 5, 6, 7, 8, 9),
    (0, 1, 2, 4, 5, 6),
    (4, 5, 6, 8, 9, 10),
)
TUPLE_LEN = 6
MAX_EXPONENT = 15            # tile 2^15; indices stay < 16**6
INDEX_BASE = 16


def _symmetry_perms() -> np.ndarray:
    """The 8 dihedral symmetries as permutations of the 16 flat cell indices.

    ``perm[p]`` is the original cell that lands at canonical position ``p`` after
    the transform, so reading a canonical tuple ``T`` from a symmetric board is
    just ``board_flat[perm[T]]``.
    """
    grid = np.arange(16).reshape(4, 4)
    perms = []
    for flip in (False, True):
        base = np.fliplr(grid) if flip else grid
        for k in range(4):
            perms.append(np.rot90(base, k).flatten())
    return np.array(perms, dtype=np.int64)  # [8, 16]


class NTupleNetwork:
    """Set of per-tuple lookup tables with symmetric weight sharing."""

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.tuples = TUPLES
        self.n_tuples = len(TUPLES)
        self.n_syms = 8
        self.n_instances = self.n_tuples * self.n_syms

        perms = _symmetry_perms()
        # CELLS[i] = the 6 source cells read for instance i (tuple i//8, sym i%8).
        cells = []
        for t in self.tuples:
            t_arr = np.array(t, dtype=np.int64)
            for perm in perms:
                cells.append(perm[t_arr])
        self.CELLS = np.array(cells, dtype=np.int64)          # [n_instances, 6]
        self.POW = (INDEX_BASE ** np.arange(TUPLE_LEN)).astype(np.int64)  # [6]

        table_size = INDEX_BASE ** TUPLE_LEN                  # 16**6 = 16,777,216
        self.LUT = [np.zeros(table_size, dtype=np.float32) for _ in range(self.n_tuples)]

    # -- indexing -------------------------------------------------------
    @staticmethod
    def _exponents(board: np.ndarray) -> np.ndarray:
        b = np.asarray(board).reshape(-1)
        e = np.zeros(16, dtype=np.int64)
        nz = b > 0
        e[nz] = np.log2(b[nz]).round().astype(np.int64)
        np.clip(e, 0, MAX_EXPONENT, out=e)
        return e

    def _indices(self, board: np.ndarray) -> np.ndarray:
        e = self._exponents(board)
        return (e[self.CELLS] * self.POW).sum(axis=1)         # [n_instances]

    def _value_from_indices(self, idx: np.ndarray) -> float:
        total = 0.0
        for t in range(self.n_tuples):
            s = t * self.n_syms
            total += float(self.LUT[t][idx[s:s + self.n_syms]].sum())
        return total

    # -- public API -----------------------------------------------------
    def value(self, board: np.ndarray) -> float:
        return self._value_from_indices(self._indices(board))

    def update(self, board: np.ndarray, target: float) -> float:
        """TD update V(board) toward target; returns the pre-update value."""
        idx = self._indices(board)
        v = self._value_from_indices(idx)
        # Split the step across all instances so the summed value moves by
        # alpha * (target - v) regardless of how many tables/symmetries there are.
        delta = self.alpha * (target - v) / self.n_instances
        for t in range(self.n_tuples):
            s = t * self.n_syms
            np.add.at(self.LUT[t], idx[s:s + self.n_syms], delta)
        return v

    def save(self, path: str):
        np.savez_compressed(path, alpha=self.alpha,
                            **{f"lut{t}": self.LUT[t] for t in range(self.n_tuples)})

    def load(self, path: str):
        data = np.load(path)
        self.alpha = float(data["alpha"])
        for t in range(self.n_tuples):
            self.LUT[t] = data[f"lut{t}"].astype(np.float32)


def _init_board() -> np.ndarray:
    board = NumpyStaticBoard.get_empty_matrix(4, 4)          # int64 zeros
    NumpyStaticBoard.set_random_cell(board, inplace=True)
    NumpyStaticBoard.set_random_cell(board, inplace=True)
    return board


def _best_action(net: NTupleNetwork, board: np.ndarray):
    """Return (afterstate, reward) maximizing reward + V(afterstate), or None."""
    best = None
    best_val = -np.inf
    for direction in ARROW_KEYS:
        after, reward, changed = NumpyStaticBoard.move(board, direction, inplace=False)
        if not changed:
            continue
        val = reward + net.value(after)
        if val > best_val:
            best_val = val
            best = (after, reward)
    return best


def play_game(net: NTupleNetwork, learn: bool = True):
    """Play one game (greedy on reward + V). If ``learn``, apply afterstate TD.

    Returns (score, max_tile, moves).
    """
    board = _init_board()
    score = 0
    max_tile = int(board.max())
    moves = 0

    cur = _best_action(net, board)
    while cur is not None:
        after, reward = cur
        score += int(reward)
        max_tile = max(max_tile, int(after.max()))
        moves += 1

        s_next = after.copy()
        NumpyStaticBoard.set_random_cell(s_next, inplace=True)

        if NumpyStaticBoard.compute_is_done(s_next):
            if learn:
                net.update(after, 0.0)      # no future reward past a terminal
            break

        nxt = _best_action(net, s_next)
        if nxt is None:                     # defensive; shouldn't happen if not done
            if learn:
                net.update(after, 0.0)
            break

        next_after, next_reward = nxt
        if learn:
            target = next_reward + net.value(next_after)
            net.update(after, target)
        cur = nxt

    return score, max_tile, moves


# ------------------------------------------------------------------------- #
# Optional Player wrapper so the trained agent can drive the pygame GUI /
# the shared Player.run() loop (e.g. to watch it play).
# ------------------------------------------------------------------------- #
def make_player(net: NTupleNetwork, game, quiet: bool = False, ui: bool = True):
    from src.agent.agent import Player

    class NTuplePlayer(Player):
        def get_move(self):
            matrix = self._game.get_matrix()
            matrix = matrix if isinstance(matrix, np.ndarray) else matrix.detach().cpu().numpy()
            best = None
            best_val = -np.inf
            best_dir = ARROW_KEYS[0]
            for direction in ARROW_KEYS:
                after, reward, changed = NumpyStaticBoard.move(matrix, direction, inplace=False)
                if not changed:
                    continue
                val = reward + net.value(after)
                if val > best_val:
                    best_val = val
                    best_dir = direction
            return best_dir

    return NTuplePlayer(game=game, quiet=quiet, ui=ui)
