"""Stochastic MCTS with an n-tuple value leaf (Milestone M8, plan §6.4).

2048 is a single-agent stochastic game, so this is **not** two-player PUCT: the
tree alternates our **decision** nodes (a max choice over the 4 moves) and the
environment's **chance** nodes (the random tile spawn). Each simulation:

  1. **Select** an action at a decision node by UCT, with Q min-max normalised
     inside the node so the exploration term stays meaningful against the large
     absolute value scale of a 2048 value function.
  2. **Chance**: sample a real spawn (2 w.p. 0.9, 4 w.p. 0.1) on the afterstate,
     using a private RNG so seeded games stay reproducible.
  3. **Evaluate**: a freshly expanded child is bootstrapped with
     ``reward + V(afterstate)`` (the learned value replaces a random rollout);
     an already-expanded child recurses. Rewards accumulate up the tree, so with
     more iterations the search replaces ``V`` with sampled reward + deeper ``V``.

Per plan §6.4 this is a **research module**: it only belongs on the default path
if it beats expectimax at equal budget on multiple seed sets. The move returned
is the most-visited root action.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.game import board as B

_EPS = 1e-9


class _Node:
    __slots__ = ("actions", "n", "w", "children", "expanded")

    def __init__(self, afterstates):
        # Legal actions as (direction, afterstate, reward).
        self.actions: List[Tuple[int, np.ndarray, int]] = [
            (d, after, reward) for d, (after, reward, changed) in enumerate(afterstates)
            if changed
        ]
        na = len(self.actions)
        self.n = np.zeros(na, dtype=np.float64)
        self.w = np.zeros(na, dtype=np.float64)
        self.children: Dict[Tuple[int, bytes], "_Node"] = {}


class StochasticMCTS:
    """MCTS over afterstates with an n-tuple ``V`` leaf, for any board shape."""

    def __init__(self, net, iterations: int = 200, c: float = 1.5,
                 max_depth: int = 40, seed: int = 0):
        self.net = net
        self.iterations = iterations
        self.c = c
        self.max_depth = max_depth
        self._rng = np.random.default_rng(seed)

    def _select(self, node: _Node) -> int:
        n, w = node.n, node.w
        total = n.sum()
        q = w / np.maximum(n, 1.0)
        # Min-max normalise Q inside the node so `c` is comparable to the
        # exploration term regardless of the value function's absolute scale.
        lo, hi = q.min(), q.max()
        qn = (q - lo) / (hi - lo) if hi > lo else np.zeros_like(q)
        ucb = qn + self.c * np.sqrt(math.log(total + 1.0) / (n + 1.0))
        # Unvisited actions get +inf priority so each is tried once first.
        ucb = np.where(n == 0, np.inf, ucb)
        return int(np.argmax(ucb))

    def _simulate(self, node: _Node, depth: int) -> float:
        if not node.actions:
            return 0.0
        ai = self._select(node)
        d, after, reward = node.actions[ai]

        s_next = after.copy()
        B.spawn(s_next, self._rng, inplace=True)

        if depth + 1 >= self.max_depth or B.is_done(s_next):
            value = reward + self.net.value(after)
        else:
            key = (ai, B.board_key(s_next))
            child = node.children.get(key)
            if child is None:
                node.children[key] = _Node(B.all_afterstates(s_next))
                value = reward + self.net.value(after)      # leaf bootstrap
            else:
                value = reward + self._simulate(child, depth + 1)

        node.n[ai] += 1.0
        node.w[ai] += value
        return value

    def get_move(self, board: np.ndarray) -> Optional[int]:
        """Best direction for ``board`` (most-visited root action), or None."""
        root = _Node(B.all_afterstates(board))
        if not root.actions:
            return None
        for _ in range(self.iterations):
            self._simulate(root, 0)
        best = int(np.argmax(root.n))
        return root.actions[best][0]


def play_game(net, H: int, W: int, rng: np.random.Generator,
              mcts: StochasticMCTS) -> Tuple[int, int, int]:
    """Play one full game driven by MCTS. Returns (score, max_tile, moves)."""
    board = B.new_game(H, W, rng)
    score, max_tile, moves = 0, B.max_tile(board), 0
    while True:
        d = mcts.get_move(board)
        if d is None:
            break
        after, reward, changed = B.move(board, d)
        if not changed:
            break
        score += reward
        max_tile = max(max_tile, B.max_tile(after))
        moves += 1
        board = after
        B.spawn(board, rng, inplace=True)
        if B.is_done(board):
            break
    return score, max_tile, moves
