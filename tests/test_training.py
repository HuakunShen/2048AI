"""Training-infrastructure tests: transition quota + generic self-play loop."""
import numpy as np

from src.training.curriculum import TransitionQuota
from src.training.selfplay import play_game
from src.ntuple.universal_value import UniversalNTuple
from src.ntuple import library as lib


class TestTransitionQuota:
    def test_balances_by_transitions_not_episodes(self):
        # 8x8 games are "longer": simulate by recording more transitions for it.
        q = TransitionQuota([(4, 4), (8, 8)], [0.5, 0.5])
        move_cost = {0: 100, 1: 400}       # 8x8 game yields 4x the moves
        for _ in range(400):
            i = q.next_index()
            q.record(i, move_cost[i])
        shares = q.realised_shares()
        # Despite very different game lengths, transition shares track the target.
        assert abs(shares[0] - 0.5) < 0.1 and abs(shares[1] - 0.5) < 0.1

    def test_respects_uneven_targets(self):
        q = TransitionQuota([(4, 4), (5, 5), (6, 6)], [0.6, 0.3, 0.1])
        for _ in range(3000):
            q.record(q.next_index(), 10)
        shares = q.realised_shares()
        assert np.allclose(shares, [0.6, 0.3, 0.1], atol=0.05)


class TestSelfPlay:
    def test_learns_on_two_shapes_with_one_model(self):
        net = UniversalNTuple(patterns=lib.CORE, alpha=0.5, tc=True)
        rng = np.random.default_rng(0)
        # Baseline greedy max-tile on each shape before training.
        def avg_max(H, W, n, learn):
            g = np.random.default_rng(123)
            return np.mean([play_game(net, H, W, g, learn=learn)[1] for _ in range(n)])

        before44 = avg_max(4, 4, 30, learn=False)
        before55 = avg_max(5, 5, 30, learn=False)
        for _ in range(400):
            play_game(net, 4, 4, rng, learn=True)
            play_game(net, 5, 5, rng, learn=True)
        after44 = avg_max(4, 4, 30, learn=False)
        after55 = avg_max(5, 5, 30, learn=False)
        # One shared model improves on both shapes.
        assert after44 > before44
        assert after55 > before55
