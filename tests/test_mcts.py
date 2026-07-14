"""M8 acceptance tests: stochastic MCTS (research module).

MCTS need not beat expectimax to be *correct* — the plan gates its promotion on a
budget-matched win (§6.4). These tests check it is a valid, reproducible searcher:
returns legal moves on every shape, plays games to completion, respects the visit
budget, and improves a random-init net's play over the untrained baseline.
"""
import numpy as np
import pytest

from src.game import board as B
from src.ntuple.universal_value import UniversalNTuple
from src.ntuple import library as lib
from src.search.mcts import StochasticMCTS, play_game


def _net(seed=0):
    net = UniversalNTuple(patterns=lib.CORE, tc=False)
    net.LUT[:] = np.random.default_rng(seed).standard_normal(net.total).astype(np.float32)
    return net


class TestValidity:
    @pytest.mark.parametrize("H,W", [(3, 3), (4, 4), (5, 6), (8, 8)])
    def test_returns_legal_move(self, H, W):
        net = _net(H + W)
        mcts = StochasticMCTS(net, iterations=50, seed=1)
        b = B.new_game(H, W, np.random.default_rng(H + W))
        d = mcts.get_move(b)
        assert d in B.DIRECTIONS
        # the chosen move must actually be legal (change the board)
        assert B.move(b, d)[2]

    def test_none_when_stuck(self):
        net = _net(0)
        mcts = StochasticMCTS(net, iterations=10)
        # A full unmergeable board has no legal move.
        b = np.array([[1, 2, 3, 4], [4, 3, 2, 1],
                      [1, 2, 3, 4], [4, 3, 2, 1]], dtype=np.int8)
        assert mcts.get_move(b) is None


class TestReproducibility:
    def test_same_seed_same_move(self):
        net = _net(2)
        b = B.new_game(4, 4, np.random.default_rng(5))
        m1 = StochasticMCTS(net, iterations=100, seed=42).get_move(b)
        m2 = StochasticMCTS(net, iterations=100, seed=42).get_move(b)
        assert m1 == m2


class TestBudget:
    def test_visits_sum_to_iterations(self):
        net = _net(3)
        mcts = StochasticMCTS(net, iterations=200, seed=0)
        root_visits = {}

        # Wrap to capture the root after search: re-run get_move logic inline.
        from src.search.mcts import _Node
        root = _Node(B.all_afterstates(B.new_game(4, 4, np.random.default_rng(1))))
        for _ in range(200):
            mcts._simulate(root, 0)
        assert root.n.sum() == 200


class TestPlays:
    @pytest.mark.parametrize("H,W", [(4, 4), (5, 5)])
    def test_plays_full_game(self, H, W):
        net = _net(7)
        mcts = StochasticMCTS(net, iterations=40, seed=0)
        score, mt, moves = play_game(net, H, W, np.random.default_rng(0), mcts)
        assert moves > 0 and mt >= 4

    def test_beats_untrained_after_training(self):
        # A briefly-trained net + MCTS should outplay an untrained net + MCTS.
        from src.training.selfplay import play_game as td_game
        trained = UniversalNTuple(patterns=lib.CORE, alpha=0.5, tc=True)
        rng = np.random.default_rng(0)
        for _ in range(300):
            td_game(trained, 4, 4, rng, learn=True)
        untrained = UniversalNTuple(patterns=lib.CORE, tc=True)

        def avg_max(net, n):
            g = np.random.default_rng(123)
            mcts = StochasticMCTS(net, iterations=30, seed=1)
            return np.mean([play_game(net, 4, 4, g, mcts)[1] for _ in range(6)])

        assert avg_max(trained, 6) > avg_max(untrained, 6)
