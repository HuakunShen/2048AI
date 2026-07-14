"""M1 acceptance tests: the generic H×W engine (``src/game/board.py``).

Covers the plan's §9.1 checklist:
  * tile-mass conservation and correct merge reward on 3×3–8×8 random boards,
  * afterstate/spawn separation (illegal move never spawns),
  * transpose equivalence (5×6 move maps to the transposed 6×5 move),
  * bit-exact parity with the legacy square engine on 4×4.

Correctness is cross-checked against an *independent* pure-Python reference
collapse, so a shared bug in the njit kernel cannot hide.
"""
import numpy as np
import pytest

from src.game import board as B
from src.game.model.staticboardImpl import NumpyStaticBoard
from src.game.utils import UP, DOWN, LEFT, RIGHT

SHAPES = [(3, 3), (3, 4), (4, 3), (4, 4), (4, 5), (5, 4),
          (5, 5), (5, 6), (6, 5), (6, 6), (3, 7), (8, 8)]
DIRS = [B.UP, B.DOWN, B.LEFT, B.RIGHT]


def _random_boards(n, H, W, rng, max_exp=13, fill=0.6):
    b = np.zeros((n, H, W), dtype=np.int8)
    draw = rng.random((n, H, W))
    cells = draw < fill
    b[cells] = rng.integers(1, max_exp + 1, size=int(cells.sum()), dtype=np.int8)
    return b


# --- independent reference implementation (deliberately not the njit one) --- #
def _ref_collapse(line, toward_zero):
    vals = [int(v) for v in line if v != 0]
    if not toward_zero:
        vals = vals[::-1]
    out, reward, i = [], 0, 0
    while i < len(vals):
        if i + 1 < len(vals) and vals[i] == vals[i + 1]:
            out.append(vals[i] + 1)
            reward += 1 << (vals[i] + 1)
            i += 2
        else:
            out.append(vals[i])
            i += 1
    res = [0] * len(line)
    for j, v in enumerate(out):
        res[j if toward_zero else len(line) - 1 - j] = v
    return res, reward


def _ref_move(board, direction):
    H, W = board.shape
    out = np.zeros_like(board)
    reward = 0
    if direction in (B.LEFT, B.RIGHT):
        toward = direction == B.LEFT
        for r in range(H):
            res, rw = _ref_collapse(board[r, :], toward)
            out[r, :] = res
            reward += rw
    else:
        toward = direction == B.UP
        for c in range(W):
            res, rw = _ref_collapse(board[:, c], toward)
            out[:, c] = res
            reward += rw
    return out, reward, not np.array_equal(out, board)


class TestMoveCorrectness:
    @pytest.mark.parametrize("H,W", SHAPES)
    def test_matches_reference_and_conserves_mass(self, H, W):
        rng = np.random.default_rng(H * 100 + W)
        boards = _random_boards(2000, H, W, rng)
        for d in DIRS:
            for b in boards:
                after, reward, changed = B.move(b, d)
                ref_after, ref_reward, ref_changed = _ref_move(b, d)
                assert np.array_equal(after, ref_after), (H, W, d)
                assert reward == ref_reward, (H, W, d)
                assert changed == ref_changed, (H, W, d)
                # raw tile mass is invariant under a move (no spawn).
                assert B.to_values(after).sum() == B.to_values(b).sum()

    def test_illegal_move_does_not_change_board(self):
        # A board where LEFT is illegal (already packed & unmergeable per row).
        b = np.array([[1, 2, 3, 4], [4, 3, 2, 1],
                      [1, 2, 3, 4], [4, 3, 2, 1]], dtype=np.int8)
        after, reward, changed = B.move(b, B.LEFT)
        assert not changed
        assert reward == 0
        assert np.array_equal(after, b)


class TestSpawnSeparation:
    def test_spawn_adds_one_tile_reproducibly(self):
        b = B.empty(4, 4)
        r1 = B.spawn(b, np.random.default_rng(0))[0]
        r2 = B.spawn(b, np.random.default_rng(0))[0]
        assert np.array_equal(r1, r2)             # same seed → same spawn
        assert (r1 > 0).sum() == 1
        assert set(np.unique(r1[r1 > 0]).tolist()) <= {1, 2}
        assert (b == 0).all()                     # non-inplace left b untouched

    def test_move_never_spawns(self):
        rng = np.random.default_rng(1)
        for b in _random_boards(500, 4, 4, rng):
            for d in DIRS:
                after, _, changed = B.move(b, d)
                if changed:
                    # afterstate has exactly the same number of tiles or fewer
                    # (merges reduce count); a spawn would *add* one.
                    assert (after > 0).sum() <= (b > 0).sum()

    def test_full_board_spawn_fails(self):
        b = np.ones((3, 3), dtype=np.int8)
        out, ok = B.spawn(b, np.random.default_rng(2))
        assert not ok


class TestTransposeEquivalence:
    @pytest.mark.parametrize("H,W", [(5, 6), (3, 7), (4, 5)])
    def test_transpose_maps_directions(self, H, W):
        # Transpose swaps LEFT<->UP and RIGHT<->DOWN.
        pairs = [(B.LEFT, B.UP), (B.RIGHT, B.DOWN),
                 (B.UP, B.LEFT), (B.DOWN, B.RIGHT)]
        rng = np.random.default_rng(H + W)
        for b in _random_boards(500, H, W, rng):
            bt = np.ascontiguousarray(b.T)
            for d, dt in pairs:
                after, reward, changed = B.move(b, d)
                after_t, reward_t, changed_t = B.move(bt, dt)
                assert np.array_equal(after.T, after_t)
                assert reward == reward_t
                assert changed == changed_t


class TestLegacyParity:
    @pytest.mark.parametrize("N", [3, 4, 5])
    def test_bit_exact_vs_numpy_static_board(self, N):
        """On square boards the new engine must match the legacy engine exactly."""
        rng = np.random.default_rng(7 * N)
        boards = _random_boards(3000, N, N, rng, max_exp=13, fill=0.55)
        str_dir = {B.UP: UP, B.DOWN: DOWN, B.LEFT: LEFT, B.RIGHT: RIGHT}
        for b in boards:
            raw = B.to_values(b)
            for d in DIRS:
                after, reward, changed = B.move(b, d)
                ref_after, ref_reward, ref_changed = NumpyStaticBoard.move(
                    raw, str_dir[d], inplace=False)
                assert np.array_equal(B.to_values(after), ref_after), (N, d)
                assert reward == ref_reward, (N, d)
                assert bool(changed) == bool(ref_changed), (N, d)

    def test_is_done_matches_legacy(self):
        rng = np.random.default_rng(99)
        boards = _random_boards(3000, 4, 4, rng, max_exp=5, fill=0.95)
        for b in boards:
            assert B.is_done(b) == NumpyStaticBoard.compute_is_done(B.to_values(b))
