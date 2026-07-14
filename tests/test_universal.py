"""M3 acceptance tests: the universal value function.

The load-bearing properties from the plan:
  * board-symmetry invariance ``V(B) == V(gB)`` (§3.1, §9.2),
  * TD/TC convergence toward a target,
  * cross-size scale stability — one update moves ``V`` by a size-independent
    amount thanks to placement-mean aggregation (§2.3, §5.1, §9.2),
  * one model evaluates every shape,
  * save/load round-trip and schema-mismatch fail-fast (§9.2, §5.5).
"""
import numpy as np
import pytest

from src.game import board as B
from src.game.symmetry import board_automorphisms
from src.ntuple.universal_value import UniversalNTuple
from src.ntuple import library as lib


def _random_board(H, W, rng, max_exp=11, fill=0.6):
    b = np.zeros((H, W), dtype=np.int8)
    cells = rng.random((H, W)) < fill
    b[cells] = rng.integers(1, max_exp + 1, size=int(cells.sum()), dtype=np.int8)
    return b


class TestSymmetryInvariance:
    @pytest.mark.parametrize("H,W", [(4, 4), (5, 5), (5, 6), (3, 7)])
    def test_value_invariant_under_board_symmetry(self, H, W):
        net = UniversalNTuple(patterns=lib.CORE, tc=False)
        rng = np.random.default_rng(H * 10 + W)
        net.LUT[:] = rng.standard_normal(net.total).astype(np.float32)  # non-trivial V
        perms = board_automorphisms(H, W)
        for _ in range(20):
            b = _random_board(H, W, rng)
            base = net.value(b)
            for perm in perms:
                gb = b.reshape(-1)[perm].reshape(H, W)
                assert net.value(gb) == pytest.approx(base, rel=1e-4, abs=1e-3)


class TestLearning:
    def test_td_update_moves_toward_target(self):
        net = UniversalNTuple(patterns=lib.CORE, alpha=0.5, tc=True)
        rng = np.random.default_rng(0)
        b = _random_board(4, 4, rng)
        target = 5000.0
        for _ in range(200):
            net.update(b, target)
        assert net.value(b) == pytest.approx(target, rel=0.05)

    def test_afterstate_decision_runs_any_shape(self):
        net = UniversalNTuple(patterns=lib.CORE, tc=True)
        rng = np.random.default_rng(1)
        for H, W in [(3, 3), (4, 4), (5, 6), (8, 8)]:
            b = B.new_game(H, W, rng)
            best, best_v = None, -np.inf
            for d in B.DIRECTIONS:
                after, reward, changed = B.move(b, d)
                if changed:
                    v = reward + net.value(after)
                    if v > best_v:
                        best_v, best = v, d
            assert best is not None                    # some legal move exists


class TestScaleStability:
    def test_single_update_size_independent(self):
        """A first update from V=0 moves V by ~α·(#patterns present), not by area."""
        net4 = UniversalNTuple(patterns=lib.CORE, alpha=0.5, tc=True)
        net8 = UniversalNTuple(patterns=lib.CORE, alpha=0.5, tc=True)
        rng = np.random.default_rng(3)
        b4 = _random_board(4, 4, rng)
        b8 = _random_board(8, 8, rng)
        net4.update(b4, 1000.0)
        net8.update(b8, 1000.0)
        dv4, dv8 = net4.value(b4), net8.value(b8)
        # Both move a similar (small) fraction toward 1000 despite 8×8 having far
        # more placements — the placement mean removes the size dependence.
        assert dv4 > 0 and dv8 > 0
        assert 0.25 < dv4 / dv8 < 4.0


class TestPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        net = UniversalNTuple(patterns=lib.CORE, tc=True)
        rng = np.random.default_rng(4)
        for _ in range(50):
            net.update(_random_board(5, 5, rng), 2000.0)
        p = str(tmp_path / "m.npz")
        net.save(p)
        b = _random_board(5, 5, np.random.default_rng(9))
        expect = net.value(b)
        net2 = UniversalNTuple(patterns=lib.CORE, tc=True)
        net2.load(p)
        assert net2.value(b) == pytest.approx(expect, rel=1e-5)

    def test_schema_mismatch_fails(self, tmp_path):
        net = UniversalNTuple(patterns=lib.CORE)
        p = str(tmp_path / "m.npz")
        net.save(p)
        other = UniversalNTuple(patterns=lib.DEFAULT_PATTERNS)   # different schema
        with pytest.raises(ValueError, match="schema mismatch"):
            other.load(p)
