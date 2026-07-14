"""Generic node-budget expectimax over afterstates (Milestone M6).

The universal-value counterpart to :mod:`src.agent.expectimax`: it searches on
the dynamic :mod:`src.game.board` engine and evaluates leaves with a
:class:`~src.ntuple.universal_value.UniversalNTuple`, so **one searcher serves
every board shape**.

Search structure from a spawned state ``s``:

    best_action_value(s, d):     # MAX node — our move
        max over valid a of  reward(s, a) + evaluate_afterstate(after(s, a), d-1)

    evaluate_afterstate(w, d):   # w is the board right after a merge
        d == 0  -> V(w)                          # learned leaf
        d  > 0  -> CHANCE node: expectation over empty cells and {2:.9, 4:.1}

``depth == 1`` reproduces the greedy policy exactly (the correctness anchor).

Dynamic boards need size-relative budgeting (plan §6.1): an 8×8 opening has a
huge chance-branching factor, so depth is chosen by the **empty-cell ratio**
(shallow when open, deep in the endgame) and a chance node with many empties is
**sampled** rather than fully expanded. A hard ``node_budget`` bounds tail
latency. The transposition table is keyed shape-aware via
:func:`~src.game.board.board_key`, and sampling uses a private RNG so seeded
games stay reproducible.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from src.game import board as B

# Spawned tile exponents: 2 (=exp 1) w.p. 0.9, 4 (=exp 2) w.p. 0.1.
SPAWN_EXP = ((1, 0.9), (2, 0.1))

# Default adaptive rule as (max_empty_ratio, depth), first match wins (plan §6.1).
DEFAULT_ADAPTIVE = [(0.15, 4), (0.30, 3), (0.55, 2)]
DEFAULT_ELSE_DEPTH = 1


class UniversalExpectimax:
    """Depth-limited expectimax with a universal n-tuple ``V`` as leaf."""

    def __init__(self, net, depth: int = 3,
                 adaptive: Optional[List[Tuple[float, int]]] = None,
                 else_depth: int = DEFAULT_ELSE_DEPTH,
                 max_chance_cells: int = 8, node_budget: int = 200_000,
                 seed: int = 0):
        self.net = net
        self.depth = depth
        self.adaptive = adaptive
        self.else_depth = else_depth
        self.max_chance_cells = max_chance_cells
        self.node_budget = node_budget
        self._rng = np.random.default_rng(seed)
        self._tt: dict = {}
        self._nodes = 0

    def _effective_depth(self, board: np.ndarray) -> int:
        if not self.adaptive:
            return self.depth
        ratio = float((board == 0).mean())
        for max_ratio, d in self.adaptive:
            if ratio <= max_ratio:
                return d
        return self.else_depth

    def _evaluate_afterstate(self, after: np.ndarray, depth: int) -> float:
        if depth <= 0 or self._nodes >= self.node_budget:
            return self.net.value(after)

        key = (B.board_key(after), depth)
        cached = self._tt.get(key)
        if cached is not None:
            return cached

        empty = np.argwhere(after == 0)
        n_empty = len(empty)
        if n_empty == 0:
            val = self.net.value(after)
            self._tt[key] = val
            return val

        if n_empty > self.max_chance_cells:
            pick = self._rng.choice(n_empty, size=self.max_chance_cells, replace=False)
            cells = empty[pick]
        else:
            cells = empty

        self._nodes += len(cells)
        total = 0.0
        for (r, c) in cells:
            for exp, p in SPAWN_EXP:
                after[r, c] = exp
                total += p * self._best_action_value(after, depth)
                after[r, c] = 0
        val = total / len(cells)
        self._tt[key] = val
        return val

    def _best_action_value(self, state: np.ndarray, depth: int) -> float:
        best = -np.inf
        for d in B.DIRECTIONS:
            after, reward, changed = B.move(state, d)
            if not changed:
                continue
            val = reward + self._evaluate_afterstate(after, depth - 1)
            if val > best:
                best = val
        return 0.0 if best == -np.inf else best

    def get_move(self, board: np.ndarray) -> Optional[int]:
        """Best direction for ``board`` (root MAX node), or None if no move."""
        self._tt.clear()
        self._nodes = 0
        depth = self._effective_depth(board)
        best_dir, best_val = None, -np.inf
        for d in B.DIRECTIONS:
            after, reward, changed = B.move(board, d)
            if not changed:
                continue
            val = reward + self._evaluate_afterstate(after, depth - 1)
            if val > best_val:
                best_val, best_dir = val, d
        return best_dir


def play_game(net, H: int, W: int, rng: np.random.Generator,
              searcher: UniversalExpectimax) -> Tuple[int, int, int]:
    """Play one full game driven by the searcher. Returns (score, max_tile, moves)."""
    board = B.new_game(H, W, rng)
    score, max_tile, moves = 0, B.max_tile(board), 0
    while True:
        d = searcher.get_move(board)
        if d is None:
            break
        after, reward, changed = B.move(board, d)
        score += reward
        max_tile = max(max_tile, B.max_tile(after))
        moves += 1
        board = after
        B.spawn(board, rng, inplace=True)
        if B.is_done(board):
            break
    return score, max_tile, moves
