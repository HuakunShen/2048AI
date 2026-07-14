"""Hogwild parallel training: workers must share one table and learn into it."""
import numpy as np
import pytest

from src.training.parallel import train_parallel, build_net


CFG = {"patterns": "core", "alphabet": 16, "alpha": 0.5, "tc": True,
       "residual": False, "rho": 0.25, "alpha_residual": 0.1}


def test_shared_tables_are_writable_views():
    from src.training.parallel import _make_raws, _bind
    net = build_net(CFG)
    raws = _make_raws(net)
    _bind(net, raws)
    net.LUT[5] = 3.14                       # write via the numpy view
    shared = np.frombuffer(raws["LUT"], dtype=np.float32)
    assert shared[5] == pytest.approx(3.14)  # ...visible in the shared buffer


def test_parallel_training_learns():
    net = train_parallel(CFG, shapes=[(4, 4)], weights=[1.0], workers=4,
                         total_games=800, eval_every=10_000, report_every=20)
    # Workers wrote into the shared tables.
    assert np.abs(net.LUT).sum() > 0
    # And the learned value plays better than an untrained net.
    from src.training.selfplay import play_game
    rng = np.random.default_rng(0)
    trained = np.mean([play_game(net, 4, 4, rng, learn=False)[1] for _ in range(40)])
    untrained_net = build_net(CFG)
    g = np.random.default_rng(0)
    untrained = np.mean([play_game(untrained_net, 4, 4, g, learn=False)[1]
                         for _ in range(40)])
    assert trained > untrained
