"""Generic afterstate-TD self-play for any board shape (Milestone M3/M5).

The learning loop is identical to the 4×4 n-tuple agent's, but shape-parameterised
and running on the generic :mod:`src.game.board` engine and the
:class:`~src.ntuple.universal_value.UniversalNTuple`. Decision rule:

    a* = argmax_a [ reward(s, a) + V(afterstate(s, a)) ]

and the afterstate-TD(0) target is ``reward' + V(next_afterstate)`` (0 at a
terminal). An explicit ``np.random.Generator`` makes every game reproducible.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from src.game import board as B


def best_action(net, board: np.ndarray):
    """``(afterstate, reward)`` maximising ``reward + V(afterstate)``, or None."""
    best, best_val = None, -np.inf
    for d in B.DIRECTIONS:
        after, reward, changed = B.move(board, d)
        if not changed:
            continue
        val = reward + net.value(after)
        if val > best_val:
            best_val, best = val, (after, reward)
    return best


def play_game(net, H: int, W: int, rng: np.random.Generator,
              learn: bool = True) -> Tuple[int, int, int]:
    """Play one game on an ``H×W`` board. Returns ``(score, max_tile, moves)``."""
    board = B.new_game(H, W, rng)
    score = 0
    max_tile = B.max_tile(board)
    moves = 0

    cur = best_action(net, board)
    while cur is not None:
        after, reward = cur
        score += reward
        max_tile = max(max_tile, B.max_tile(after))
        moves += 1

        s_next = after.copy()
        B.spawn(s_next, rng, inplace=True)

        if B.is_done(s_next):
            if learn:
                net.update(after, 0.0)
            break

        nxt = best_action(net, s_next)
        if nxt is None:                     # defensive: no legal move though not done
            if learn:
                net.update(after, 0.0)
            break

        next_after, next_reward = nxt
        if learn:
            net.update(after, next_reward + net.value(next_after))
        cur = nxt

    return score, max_tile, moves
