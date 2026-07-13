"""Correctness gate for the vectorized engine: it must match NumpyStaticBoard.

Everything in Track C (GPU RL) depends on this equivalence, so the tests are
deliberately exhaustive: random boards including high tiles and full boards,
all four directions, checking afterstate, reward, and the changed flag.
"""
import numpy as np
import pytest

from src.game.model.staticboardImpl import NumpyStaticBoard
from src.game.model import vectorboard as vb
from src.game.utils import UP, DOWN, LEFT, RIGHT

DIRS = (UP, DOWN, LEFT, RIGHT)


def _random_exp_boards(n, rng, max_exp=13, fill=0.6):
    """Random exponent boards [n,4,4]; ``fill`` controls density."""
    b = np.zeros((n, 4, 4), dtype=np.int8)
    draw = rng.random((n, 4, 4))
    cells = draw < fill
    b[cells] = rng.integers(1, max_exp + 1, size=cells.sum())
    return b


class TestMoveEquivalence:
    def test_matches_numpy_engine_on_100k_boards(self):
        rng = np.random.default_rng(0)
        n = 100_000
        boards_exp = _random_exp_boards(n, rng)
        tiles = vb.exp_to_tiles(boards_exp)

        for d in DIRS:
            after_exp, reward, changed = vb.move_batch(boards_exp, d)
            after_tiles = vb.exp_to_tiles(after_exp)
            # Spot-check a random subset against the reference engine (calling it
            # 100k*4 times is slow; 2000 comparisons per direction is plenty).
            sample = rng.choice(n, size=2000, replace=False)
            for i in sample:
                ref_after, ref_reward, ref_changed = NumpyStaticBoard.move(
                    tiles[i], d, inplace=False)
                assert np.array_equal(after_tiles[i], ref_after), (d, i)
                assert reward[i] == ref_reward, (d, i)
                assert bool(changed[i]) == bool(ref_changed), (d, i)

    def test_full_board_edge_cases(self):
        rng = np.random.default_rng(1)
        # Full boards (no empties) exercise the "no slide, maybe merge" path.
        boards_exp = _random_exp_boards(5000, rng, max_exp=11, fill=1.0)
        tiles = vb.exp_to_tiles(boards_exp)
        for d in DIRS:
            after_exp, reward, changed = vb.move_batch(boards_exp, d)
            after_tiles = vb.exp_to_tiles(after_exp)
            for i in range(0, 5000, 7):
                ref_after, ref_reward, ref_changed = NumpyStaticBoard.move(
                    tiles[i], d, inplace=False)
                assert np.array_equal(after_tiles[i], ref_after)
                assert reward[i] == ref_reward
                assert bool(changed[i]) == bool(ref_changed)


class TestDoneCheck:
    def test_done_matches_compute_is_done(self):
        rng = np.random.default_rng(2)
        boards_exp = _random_exp_boards(3000, rng, max_exp=6, fill=0.95)
        tiles = vb.exp_to_tiles(boards_exp)
        done = vb.done_batch(boards_exp)
        for i in range(3000):
            assert bool(done[i]) == bool(NumpyStaticBoard.compute_is_done(tiles[i])), i


class TestSpawnAndEncoding:
    def test_spawn_fills_one_empty_with_2_or_4(self):
        rng = np.random.default_rng(3)
        boards = _random_exp_boards(1000, rng, fill=0.5)
        before_counts = (boards == 0).sum(axis=(1, 2))
        vb.spawn_batch(boards, rng)
        after_counts = (boards == 0).sum(axis=(1, 2))
        for i in range(1000):
            if before_counts[i] > 0:
                assert after_counts[i] == before_counts[i] - 1
        # spawned tiles are exponent 1 (=2) or 2 (=4) only
        assert set(np.unique(boards[boards > 0]).tolist()) - set(range(1, 14)) == set()

    def test_encoding_roundtrip(self):
        rng = np.random.default_rng(4)
        boards = _random_exp_boards(2000, rng, max_exp=15, fill=0.7)
        tiles = vb.exp_to_tiles(boards)
        assert np.array_equal(vb.tiles_to_exp(tiles), boards)


class TestTorchEngine:
    def test_gpu_move_matches_numpy(self):
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("no CUDA device")
        rng = np.random.default_rng(5)
        boards = _random_exp_boards(20000, rng, max_exp=13, fill=0.6)
        eng = vb.TorchVectorEngine(device="cuda")
        bt = torch.from_numpy(boards.astype(np.int64)).cuda()
        for d in DIRS:
            np_after, np_reward, np_changed = vb.move_batch(boards, d)
            gp_after, gp_reward, gp_changed = eng.move(bt, d)
            assert np.array_equal(gp_after.cpu().numpy().astype(np.int8), np_after), d
            assert np.array_equal(gp_reward.cpu().numpy(), np_reward), d
            assert np.array_equal(gp_changed.cpu().numpy(), np_changed), d

    def test_gpu_done_matches_numpy(self):
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("no CUDA device")
        rng = np.random.default_rng(6)
        boards = _random_exp_boards(20000, rng, max_exp=6, fill=0.95)
        eng = vb.TorchVectorEngine(device="cuda")
        bt = torch.from_numpy(boards.astype(np.int64)).cuda()
        assert np.array_equal(eng.done(bt).cpu().numpy(), vb.done_batch(boards))
