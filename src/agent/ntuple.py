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
from numba import njit

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

# Track B (B2): a larger 8-pattern set. A superset of the 4 above plus four more
# diverse local shapes (an axis/column reach, a diagonal staircase, the central
# block, and an L). With 8-fold symmetry each, this covers more board features →
# a more expressive V. Memory: 16**6 * 4 B = 64 MB per table -> 512 MB total.
TUPLES_8 = (
    (0, 1, 2, 3, 4, 5),      # horizontal band
    (4, 5, 6, 7, 8, 9),      # horizontal band (shifted)
    (0, 1, 2, 4, 5, 6),      # 3x2 rectangle
    (4, 5, 6, 8, 9, 10),     # 3x2 rectangle (shifted)
    (0, 1, 4, 5, 8, 12),     # left column + corner reach
    (0, 1, 5, 6, 10, 11),    # diagonal staircase
    (1, 2, 5, 6, 9, 10),     # central 3x2
    (0, 4, 5, 8, 9, 10),     # L-shape, lower-left
)

TUPLE_SETS = {"4": TUPLES, "8": TUPLES_8}

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


@njit(cache=True)
def _decode_exp(board_flat):
    """int64 board [16] (power-of-two tiles or 0) -> exponents [16], clipped 0..15."""
    e = np.empty(16, dtype=np.int64)
    for i in range(16):
        v = board_flat[i]
        if v <= 0:
            e[i] = 0
        else:
            ex = 0
            while v > 1:            # exact log2 for powers of two
                v >>= 1
                ex += 1
            e[i] = ex if ex <= 15 else 15
    return e


@njit(cache=True)
def _instance_indices(e, cells, pow_):
    """Per-instance table indices [n_instances] from exponents + cell map."""
    n_instances = cells.shape[0]
    tuple_len = cells.shape[1]
    idxs = np.empty(n_instances, dtype=np.int64)
    for inst in range(n_instances):
        idx = 0
        for c in range(tuple_len):
            idx += e[cells[inst, c]] * pow_[c]
        idxs[inst] = idx
    return idxs


@njit(cache=True)
def _value_njit(board_flat, cells, pow_, lut, n_syms):
    """Fast V(board): exponent-decode + n-tuple index + table-sum in pure loops.

    Instance ``i`` belongs to tuple ``i // n_syms``. Matches
    ``_value_from_indices(_indices(board))`` exactly but with no per-call numpy
    allocation — the expectimax / training hot path.
    """
    e = _decode_exp(board_flat)
    idxs = _instance_indices(e, cells, pow_)
    total = 0.0
    for inst in range(idxs.shape[0]):
        total += lut[inst // n_syms, idxs[inst]]
    return total


@njit(cache=True)
def _update_njit(board_flat, cells, pow_, lut, n_syms, alpha, target):
    """Standard TD update: move summed V toward target by alpha*(target-v)."""
    e = _decode_exp(board_flat)
    idxs = _instance_indices(e, cells, pow_)
    n_instances = idxs.shape[0]
    v = 0.0
    for inst in range(n_instances):
        v += lut[inst // n_syms, idxs[inst]]
    delta = alpha * (target - v) / n_instances
    for inst in range(n_instances):
        lut[inst // n_syms, idxs[inst]] += delta
    return v


@njit(cache=True)
def _update_tc_njit(board_flat, cells, pow_, lut, acc_e, acc_a, n_syms, alpha, target):
    """Temporal-coherence TD update: per-weight step scaled by |E|/A.

    Each weight keeps accumulators E (net signed error) and A (total absolute
    error). Coherent updates (|E|/A -> 1) take a full step; oscillating ones
    (|E|/A -> 0) are damped. This converges faster and more stably than a fixed
    step (Beal & Smith; Szubert & Jaskowski 2014).
    """
    e = _decode_exp(board_flat)
    idxs = _instance_indices(e, cells, pow_)
    n_instances = idxs.shape[0]
    v = 0.0
    for inst in range(n_instances):
        v += lut[inst // n_syms, idxs[inst]]
    error = target - v
    for inst in range(n_instances):
        t = inst // n_syms
        idx = idxs[inst]
        a = acc_a[t, idx]
        lr = (abs(acc_e[t, idx]) / a) if a > 0.0 else 1.0
        lut[t, idx] += alpha * lr * error / n_instances
        acc_e[t, idx] += error
        acc_a[t, idx] += abs(error)
    return v


class NTupleNetwork:
    """Set of per-tuple lookup tables with symmetric weight sharing."""

    def __init__(self, alpha: float = 0.1, tuples=TUPLES, tc: bool = False):
        self.alpha = alpha
        self.n_syms = 8
        self.tc = tc
        self.POW = (INDEX_BASE ** np.arange(TUPLE_LEN)).astype(np.int64)  # [6]
        self._configure(tuples)
        # Temporal-coherence accumulators (allocated only when tc=True).
        if tc:
            self.E = np.zeros_like(self.LUT)
            self.A = np.zeros_like(self.LUT)
        else:
            self.E = self.A = None

    def _configure(self, tuples):
        """(Re)build the cell map and LUT for a given tuple set. Reused by load()
        so a checkpoint restores its own pattern set regardless of construction."""
        self.tuples = tuple(tuple(int(c) for c in t) for t in tuples)
        self.n_tuples = len(self.tuples)
        self.n_instances = self.n_tuples * self.n_syms
        perms = _symmetry_perms()
        # CELLS[i] = the source cells read for instance i (tuple i//8, sym i%8).
        cells = []
        for t in self.tuples:
            t_arr = np.array(t, dtype=np.int64)
            for perm in perms:
                cells.append(perm[t_arr])
        self.CELLS = np.array(cells, dtype=np.int64)          # [n_instances, 6]
        table_size = INDEX_BASE ** TUPLE_LEN                  # 16**6 = 16,777,216
        # One contiguous [n_tuples, table_size] array; self.LUT[t] is a row view,
        # so save()/load() keep working, and it feeds the njit kernels directly.
        self.LUT = np.zeros((self.n_tuples, table_size), dtype=np.float32)

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
        board_flat = np.ascontiguousarray(board, dtype=np.int64).reshape(-1)
        return float(_value_njit(board_flat, self.CELLS, self.POW, self.LUT, self.n_syms))

    def update(self, board: np.ndarray, target: float) -> float:
        """TD update V(board) toward target; returns the pre-update value."""
        board_flat = np.ascontiguousarray(board, dtype=np.int64).reshape(-1)
        if self.tc:
            return float(_update_tc_njit(board_flat, self.CELLS, self.POW, self.LUT,
                                         self.E, self.A, self.n_syms, self.alpha, target))
        return float(_update_njit(board_flat, self.CELLS, self.POW, self.LUT,
                                  self.n_syms, self.alpha, target))

    def save(self, path: str):
        # Persist the tuple set so load() can restore any pattern configuration.
        np.savez_compressed(
            path, alpha=self.alpha,
            tuples=np.array([list(t) for t in self.tuples], dtype=np.int64),
            **{f"lut{t}": self.LUT[t] for t in range(self.n_tuples)})

    def load(self, path: str):
        data = np.load(path)
        self.alpha = float(data["alpha"])
        if "tuples" in data:                       # self-describing checkpoint
            self._configure([tuple(row) for row in data["tuples"]])
        for t in range(self.n_tuples):             # older files: 4-tuple default
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
