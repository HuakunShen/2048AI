"""Unit tests for the n-tuple network, afterstate assumptions, and expectimax.

These protect the load-bearing invariants the RL players rely on:
  * symmetric weight sharing makes ``V`` invariant to the 8 board symmetries,
  * ``move(..., inplace=False)`` really returns the pre-spawn afterstate,
  * a TD ``update`` moves ``value`` toward its target and converges,
  * expectimax at depth 1 reproduces the greedy n-tuple policy exactly.
"""
import numpy as np
import pytest

from src.game.model.staticboardImpl import NumpyStaticBoard
from src.game.utils import ARROW_KEYS
from src.agent.ntuple import NTupleNetwork, _best_action
from src.agent.expectimax import ExpectimaxNTuple


def _random_board(rng, max_exp=11):
    """A plausible board: mostly empty, a few power-of-two tiles."""
    board = np.zeros(16, dtype=np.int64)
    k = rng.integers(3, 10)
    cells = rng.choice(16, size=k, replace=False)
    exps = rng.integers(1, max_exp, size=k)
    board[cells] = 2 ** exps
    return board.reshape(4, 4)


class TestSymmetry:
    def test_value_invariant_under_dihedral_group(self):
        rng = np.random.default_rng(0)
        net = NTupleNetwork()
        # Put nonzero weights in the tables so the test is meaningful.
        for lut in net.LUT:
            lut[:] = rng.standard_normal(lut.shape).astype(np.float32) * 0.01
        for _ in range(20):
            board = _random_board(rng)
            base = net.value(board)
            for k in range(4):
                assert net.value(np.rot90(board, k)) == pytest.approx(base, abs=1e-3)
            assert net.value(np.fliplr(board)) == pytest.approx(base, abs=1e-3)
            assert net.value(np.flipud(board)) == pytest.approx(base, abs=1e-3)


class TestAfterstate:
    def test_move_inplace_false_returns_afterstate_without_spawn(self):
        rng = np.random.default_rng(1)
        for _ in range(50):
            board = _random_board(rng)
            for d in ARROW_KEYS:
                before_nonzero = int((board != 0).sum())
                after, reward, changed = NumpyStaticBoard.move(board, d, inplace=False)
                if not changed:
                    continue
                # A merge only ever reduces the tile count; a spawn would add one.
                # The afterstate must therefore have <= the original nonzero count.
                assert int((after != 0).sum()) <= before_nonzero
                assert reward >= 0
                # Original board is untouched (inplace=False).
                assert not np.array_equal(after, board) or reward == 0


class TestTDUpdate:
    def test_update_moves_value_toward_target(self):
        net = NTupleNetwork(alpha=0.1)
        board = np.array([[2, 4, 8, 16],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0]], dtype=np.int64)
        target = 100.0
        prev = net.value(board)
        net.update(board, target)
        assert abs(net.value(board) - target) < abs(prev - target)

    def test_repeated_updates_converge_to_target(self):
        net = NTupleNetwork(alpha=0.1)
        board = np.array([[2, 4, 8, 16],
                          [32, 64, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0]], dtype=np.int64)
        target = 250.0
        for _ in range(200):
            net.update(board, target)
        assert net.value(board) == pytest.approx(target, abs=1.0)


class TestExpectimaxCorrectness:
    def test_depth1_equals_greedy(self):
        """Expectimax at depth 1 must pick the same move as greedy reward+V."""
        rng = np.random.default_rng(2)
        net = NTupleNetwork()
        for lut in net.LUT:
            lut[:] = rng.standard_normal(lut.shape).astype(np.float32) * 0.01

        searcher = ExpectimaxNTuple(net, depth=1)
        for _ in range(100):
            board = _random_board(rng)
            best = _best_action(net, board)
            if best is None:               # no valid move; greedy also declines
                assert searcher.get_move(board) is None
                continue
            # Recompute greedy's chosen direction to compare against expectimax.
            greedy_dir, greedy_val = None, -np.inf
            for d in ARROW_KEYS:
                after, reward, changed = NumpyStaticBoard.move(board, d, inplace=False)
                if not changed:
                    continue
                v = reward + net.value(after)
                if v > greedy_val:
                    greedy_val, greedy_dir = v, d
            assert searcher.get_move(board) == greedy_dir

    def test_deeper_search_runs_and_returns_valid_move(self):
        rng = np.random.default_rng(3)
        net = NTupleNetwork()
        for lut in net.LUT:
            lut[:] = rng.standard_normal(lut.shape).astype(np.float32) * 0.01
        searcher = ExpectimaxNTuple(net, depth=3, max_chance_cells=6)
        for _ in range(10):
            board = _random_board(rng)
            direction = searcher.get_move(board)
            if direction is None:
                continue
            _, _, changed = NumpyStaticBoard.move(board, direction, inplace=False)
            assert changed                 # never returns an invalid move
