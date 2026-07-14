"""M7 acceptance tests: multi-stage tables + weight promotion.

Verifies the two mechanics of §7.1:
  * **fallback read** — an entry never visited at a high stage reads the deepest
    visited lower stage (so a high stage inherits lower-stage knowledge),
  * **weight promotion** — the first update at a stage seeds the entry from the
    lower stage, then updates independently without disturbing the lower stage.
Plus learning, save/load (with stage config + visited markers), and parallel
training with stages.
"""
import numpy as np
import pytest

from src.game import board as B
from src.ntuple.universal_value import UniversalNTuple
from src.ntuple import library as lib


def _rand_board(H, W, rng, max_exp, fill=0.6):
    b = np.zeros((H, W), dtype=np.int8)
    cells = rng.random((H, W)) < fill
    b[cells] = rng.integers(1, max_exp + 1, size=int(cells.sum()), dtype=np.int8)
    return b


class TestConfig:
    def test_stage_count_and_shapes(self):
        net = UniversalNTuple(patterns=lib.CORE, stages=[13, 15])
        assert net.n_stages == 3
        assert net.LUT.shape == (3, net.total)
        assert net.A.shape == (3, net.total)

    def test_stage_from_max_tile(self):
        net = UniversalNTuple(patterns=lib.CORE, stages=[13, 15])
        assert net._stage(np.array([[10, 5], [3, 0]], dtype=np.int8)) == 0   # <8192
        assert net._stage(np.array([[13, 5], [3, 0]], dtype=np.int8)) == 1   # 8192
        assert net._stage(np.array([[16, 5], [3, 0]], dtype=np.int8)) == 2   # >=32768

    def test_requires_tc(self):
        with pytest.raises(ValueError, match="requires tc"):
            UniversalNTuple(patterns=lib.CORE, stages=[13], tc=False)


class TestFallback:
    def test_unvisited_high_stage_reads_stage_zero(self):
        # All entries unvisited (A==0) -> every stage falls back to stage 0.
        ms = UniversalNTuple(patterns=lib.CORE, stages=[13, 15])
        ss = UniversalNTuple(patterns=lib.CORE)
        R = np.random.default_rng(0).standard_normal(ms.total).astype(np.float32)
        ms.LUT[0, :] = R
        ss.LUT[:] = R
        rng = np.random.default_rng(1)
        b_hi = _rand_board(4, 4, rng, max_exp=11)
        b_hi[0, 0] = 15                          # force stage 2
        assert ms._stage(b_hi) == 2
        assert ms.value(b_hi) == pytest.approx(ss.value(b_hi), rel=1e-5, abs=1e-4)


class TestPromotion:
    def test_first_update_seeds_from_lower_stage_and_leaves_it_intact(self):
        net = UniversalNTuple(patterns=lib.CORE, stages=[13], alpha=0.5)
        net.LUT[0, :] = 5.0
        net.A[0, :] = 1.0                        # mark stage 0 visited
        b = _rand_board(4, 4, np.random.default_rng(2), max_exp=11)
        b[0, 0] = 14                             # stage 1
        assert net._stage(b) == 1
        before = net.value(b)                    # fallback -> stage 0 (all 5.0)
        assert before == pytest.approx(5.0 * sum(
            1 for cp in net.compiler.get(4, 4).compiled if cp.n_instances))
        stage0_before = net.LUT[0].copy()
        for _ in range(50):
            net.update(b, 1000.0)
        assert net.value(b) > before             # stage 1 moved toward target
        assert np.array_equal(net.LUT[0], stage0_before)   # stage 0 untouched


class TestLearning:
    def test_staged_net_learns(self):
        net = UniversalNTuple(patterns=lib.CORE, stages=[11], alpha=0.5)
        from src.training.selfplay import play_game
        rng = np.random.default_rng(0)
        base = np.mean([play_game(net, 4, 4, np.random.default_rng(7), learn=False)[1]
                        for _ in range(30)])
        for _ in range(400):
            play_game(net, 4, 4, rng, learn=True)
        after = np.mean([play_game(net, 4, 4, np.random.default_rng(7), learn=False)[1]
                         for _ in range(30)])
        assert after > base


class TestPersistence:
    def test_save_load_preserves_staged_value(self, tmp_path):
        net = UniversalNTuple(patterns=lib.CORE, stages=[13])
        rng = np.random.default_rng(3)
        for _ in range(60):
            b = _rand_board(4, 4, rng, max_exp=11)
            b[0, 0] = 14                         # some stage-1 states
            net.update(b, 800.0)
        p = str(tmp_path / "m.npz")
        net.save(p)
        probe = _rand_board(4, 4, np.random.default_rng(9), max_exp=11)
        probe[1, 1] = 14
        expect = net.value(probe)
        net2 = UniversalNTuple(patterns=lib.CORE, stages=[13])
        net2.load(p)
        assert net2.value(probe) == pytest.approx(expect, rel=1e-5)

    def test_stage_mismatch_fails(self, tmp_path):
        net = UniversalNTuple(patterns=lib.CORE, stages=[13])
        p = str(tmp_path / "m.npz")
        net.save(p)
        with pytest.raises(ValueError, match="stage config mismatch"):
            UniversalNTuple(patterns=lib.CORE, stages=[13, 15]).load(p)


class TestParallelWithStages:
    def test_parallel_training_with_stages(self):
        from src.training.parallel import train_parallel, build_net
        cfg = {"patterns": "core", "alphabet": 16, "alpha": 0.5, "tc": True,
               "residual": False, "rho": 0.25, "alpha_residual": 0.1,
               "stages": [11]}
        net = train_parallel(cfg, shapes=[(4, 4)], weights=[1.0], workers=4,
                             total_games=600, eval_every=10_000, report_every=20)
        assert net.LUT.shape == (2, net.total)
        assert np.abs(net.LUT).sum() > 0
