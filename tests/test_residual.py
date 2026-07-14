"""M4 acceptance tests: the conditional residual head.

The residual must (a) preserve board-symmetry invariance (roles are
symmetry-invariant, so V stays invariant), (b) actually be exercised and learn,
and (c) round-trip through save/load. It is allocated only for small-table
patterns (plan §3.3).
"""
import numpy as np
import pytest

from src.game.symmetry import board_automorphisms
from src.ntuple.universal_value import UniversalNTuple, RESIDUAL_MAX_LEN
from src.ntuple import library as lib


def _random_board(H, W, rng, max_exp=11, fill=0.6):
    b = np.zeros((H, W), dtype=np.int8)
    cells = rng.random((H, W)) < fill
    b[cells] = rng.integers(1, max_exp + 1, size=int(cells.sum()), dtype=np.int8)
    return b


class TestResidualAllocation:
    def test_only_small_patterns_get_residual(self):
        net = UniversalNTuple(patterns=lib.DEFAULT_PATTERNS, residual=True)
        for k, p in enumerate(net.patterns):
            has_res = net.res_offsets[k] >= 0
            assert has_res == (p.length <= RESIDUAL_MAX_LEN)
        assert net.res_total > 0

    def test_requires_tc(self):
        with pytest.raises(ValueError, match="requires tc"):
            UniversalNTuple(patterns=lib.CORE, residual=True, tc=False)


class TestResidualSymmetry:
    @pytest.mark.parametrize("H,W", [(4, 4), (5, 6)])
    def test_symmetry_preserved_with_residual(self, H, W):
        net = UniversalNTuple(patterns=lib.CORE, residual=True)
        rng = np.random.default_rng(H + W)
        net.LUT[:] = rng.standard_normal(net.total).astype(np.float32)
        net.R[:] = rng.standard_normal(net.res_total).astype(np.float32)
        perms = board_automorphisms(H, W)
        for _ in range(15):
            b = _random_board(H, W, rng)
            base = net.value(b)
            for perm in perms:
                gb = b.reshape(-1)[perm].reshape(H, W)
                assert net.value(gb) == pytest.approx(base, rel=1e-4, abs=1e-3)


class TestResidualLearning:
    def test_residual_tables_move_and_help_fit(self):
        net = UniversalNTuple(patterns=lib.CORE, residual=True, rho=0.25)
        rng = np.random.default_rng(0)
        b = _random_board(4, 4, rng)
        for _ in range(150):
            net.update(b, 4000.0)
        assert np.abs(net.R).sum() > 0                # residual actually learned
        assert net.value(b) == pytest.approx(4000.0, rel=0.05)

    def test_save_load_roundtrip_with_residual(self, tmp_path):
        net = UniversalNTuple(patterns=lib.CORE, residual=True)
        rng = np.random.default_rng(1)
        for _ in range(40):
            net.update(_random_board(4, 4, rng), 3000.0)
        p = str(tmp_path / "m.npz")
        net.save(p)
        b = _random_board(4, 4, np.random.default_rng(5))
        expect = net.value(b)
        net2 = UniversalNTuple(patterns=lib.CORE, residual=True)
        net2.load(p)
        assert net2.value(b) == pytest.approx(expect, rel=1e-5)
