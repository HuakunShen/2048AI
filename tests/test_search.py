"""M6 acceptance tests: generic node-budget expectimax.

The load-bearing anchor is that ``depth == 1`` reproduces the greedy policy
exactly, so search is a strict extension of the trained value function. Also
checks the node budget is honoured and the searcher runs on every shape.
"""
import numpy as np
import pytest

from src.game import board as B
from src.ntuple.universal_value import UniversalNTuple
from src.ntuple import library as lib
from src.search.expectimax import UniversalExpectimax, play_game
from src.training.selfplay import best_action


def _net(seed=0):
    net = UniversalNTuple(patterns=lib.CORE, tc=False)
    net.LUT[:] = np.random.default_rng(seed).standard_normal(net.total).astype(np.float32)
    return net


class TestGreedyAnchor:
    @pytest.mark.parametrize("H,W", [(4, 4), (5, 5), (5, 6)])
    def test_depth1_matches_greedy(self, H, W):
        net = _net(H + W)
        search = UniversalExpectimax(net, depth=1, adaptive=None)
        rng = np.random.default_rng(H * W)
        for _ in range(60):
            b = B.new_game(H, W, rng)
            for _ in range(15):
                greedy = best_action(net, b)
                sd = search.get_move(b)
                if greedy is None:
                    assert sd is None
                    break
                # Reconstruct greedy's direction to compare against the searcher.
                gdir, gval = None, -np.inf
                for d in B.DIRECTIONS:
                    after, reward, changed = B.move(b, d)
                    if changed and reward + net.value(after) > gval:
                        gval, gdir = reward + net.value(after), d
                assert sd == gdir
                after, _, _ = B.move(b, sd)
                b = after
                B.spawn(b, rng, inplace=True)
                if B.is_done(b):
                    break


class TestBudget:
    def test_node_budget_respected(self):
        net = _net(1)
        search = UniversalExpectimax(net, depth=6, adaptive=None,
                                     node_budget=500, max_chance_cells=8)
        b = B.new_game(6, 6, np.random.default_rng(2))
        search.get_move(b)
        # The counter may overshoot by one chance node's fan-out, never unbounded.
        assert search._nodes <= 500 + 8


class TestMultiShape:
    @pytest.mark.parametrize("H,W", [(3, 3), (4, 4), (5, 6), (8, 8)])
    def test_runs_and_returns_valid_move(self, H, W):
        net = _net(9)
        search = UniversalExpectimax(net, depth=2, max_chance_cells=4)
        b = B.new_game(H, W, np.random.default_rng(H + W))
        d = search.get_move(b)
        assert d in B.DIRECTIONS

    def test_full_game_terminates(self):
        net = _net(3)
        search = UniversalExpectimax(net, adaptive=[(0.3, 2)], else_depth=1,
                                     max_chance_cells=4)
        score, mt, moves = play_game(net, 4, 4, np.random.default_rng(0), search)
        assert moves > 0 and mt >= 2
