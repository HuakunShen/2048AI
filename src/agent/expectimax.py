"""Expectimax search over afterstates with an n-tuple leaf evaluation.

The trained :class:`~src.agent.ntuple.NTupleNetwork` gives a strong value
function ``V(afterstate)`` but the greedy player in ``ntuple.py`` only uses it at
1 ply: ``a* = argmax_a [reward(s,a) + V(afterstate(s,a))]``. This module deepens
that lookahead with **expectimax**, which is the natural search for 2048 because
the environment alternates between *our* deterministic move (a **max** node) and
the game's random tile spawn (a **chance** node, over which we take an
*expectation*).

The tree is expressed entirely over **afterstates** so it reuses the exact same
engine primitive the rest of the codebase relies on:
``NumpyStaticBoard.move(board, dir, inplace=False)`` -> ``(afterstate, reward, changed)``.

Structure of one search from a state ``s`` (a board with a tile already spawned):

    best_action_value(s, d):        # MAX node — our move
        max over valid a of  reward(s,a) + evaluate_afterstate(afterstate(s,a), d-1)

    evaluate_afterstate(w, d):      # w is the board right after a merge
        d == 0            -> V(w)                          # learned leaf
        d  > 0            -> CHANCE node: average over every empty cell c and
                             tile in {2 (p=.9), 4 (p=.1)} of
                             best_action_value(w with c:=tile, d)

``depth == 1`` reduces evaluate_afterstate to ``V(afterstate)``, i.e. it
reproduces the greedy n-tuple player exactly (this is the correctness anchor,
asserted in the tests).

Cost control (all optional, on by default at the aggressive settings):
  * **Transposition table** — memoize ``evaluate_afterstate`` on
    ``(afterstate.tobytes(), depth)`` within a single root search.
  * **Adaptive depth** — search shallow when the board is open (many empties, low
    stakes) and deep in the endgame (few empties, where lookahead pays). This is
    what keeps depth-3 search tractable.
  * **Chance-node sampling** — when a chance node has more than
    ``max_chance_cells`` empty cells, evaluate a uniform random subset and average
    (an unbiased estimate). Uses a *private* RNG so it never perturbs the game's
    tile-spawn stream, keeping seeded benchmarks reproducible.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from src.game.model.staticboardImpl import NumpyStaticBoard
from src.game.utils import ARROW_KEYS
from src.agent.ntuple import NTupleNetwork

# Tile spawned on an empty cell: 2 with prob 0.9, 4 with prob 0.1 (matches
# NumpyStaticBoard.set_random_cell).
SPAWN_TILES = ((2, 0.9), (4, 0.1))


class ExpectimaxNTuple:
    """Depth-limited expectimax with an n-tuple ``V`` as the leaf evaluator.

    Parameters
    ----------
    net:
        A (trained) NTupleNetwork used to evaluate leaf afterstates.
    depth:
        Base search depth in *move plies*. ``depth=1`` == greedy.
    adaptive:
        If set, override ``depth`` per-move using ``adaptive`` as a list of
        ``(max_empty, depth)`` rules, first match wins. E.g.
        ``[(3, 5), (7, 3), (16, 2)]`` -> depth 5 when <=3 empties, depth 3 when
        4-7, depth 2 otherwise.
    max_chance_cells:
        Expand at most this many empty cells at a chance node; sample if more.
    seed:
        Seed for the private sampling RNG (kept separate from the game RNG).
    """

    def __init__(
        self,
        net: NTupleNetwork,
        depth: int = 3,
        adaptive: Optional[list] = None,
        max_chance_cells: int = 16,
        seed: int = 0,
    ):
        self.net = net
        self.depth = depth
        self.adaptive = adaptive
        self.max_chance_cells = max_chance_cells
        self._rng = np.random.default_rng(seed)
        self._tt: dict = {}

    # -- search ---------------------------------------------------------
    def _effective_depth(self, board: np.ndarray) -> int:
        if not self.adaptive:
            return self.depth
        n_empty = int((board == 0).sum())
        for max_empty, d in self.adaptive:
            if n_empty <= max_empty:
                return d
        return self.adaptive[-1][1]

    def _evaluate_afterstate(self, after: np.ndarray, depth: int) -> float:
        """Expected value of an afterstate: leaf V at depth 0, else a chance node."""
        if depth <= 0:
            return self.net.value(after)

        key = (after.tobytes(), depth)
        cached = self._tt.get(key)
        if cached is not None:
            return cached

        empty = np.argwhere(after == 0)
        n_empty = len(empty)
        if n_empty == 0:
            # No spawn possible -> the state is effectively terminal for the
            # opponent; fall back to the leaf value.
            val = self.net.value(after)
            self._tt[key] = val
            return val

        if n_empty > self.max_chance_cells:
            pick = self._rng.choice(n_empty, size=self.max_chance_cells, replace=False)
            cells = empty[pick]
        else:
            cells = empty

        total = 0.0
        for (r, c) in cells:
            for tile, p_tile in SPAWN_TILES:
                after[r, c] = tile
                total += p_tile * self._best_action_value(after, depth)
                after[r, c] = 0
        val = total / len(cells)          # uniform over the (sampled) cells
        self._tt[key] = val
        return val

    def _best_action_value(self, state: np.ndarray, depth: int) -> float:
        """MAX node: best over valid moves of reward + deeper afterstate value."""
        best = -np.inf
        for direction in ARROW_KEYS:
            after, reward, changed = NumpyStaticBoard.move(state, direction, inplace=False)
            if not changed:
                continue
            val = reward + self._evaluate_afterstate(after, depth - 1)
            if val > best:
                best = val
        if best == -np.inf:
            return 0.0                    # terminal: no future reward
        return best

    def get_move(self, board: np.ndarray) -> Optional[str]:
        """Return the best direction for ``board`` (root MAX node), or None."""
        self._tt.clear()
        depth = self._effective_depth(board)
        best_dir = None
        best_val = -np.inf
        for direction in ARROW_KEYS:
            after, reward, changed = NumpyStaticBoard.move(board, direction, inplace=False)
            if not changed:
                continue
            val = reward + self._evaluate_afterstate(after, depth - 1)
            if val > best_val:
                best_val = val
                best_dir = direction
        return best_dir


def make_player(net: NTupleNetwork, game, depth: int = 3, adaptive=None,
                quiet: bool = False, ui: bool = True):
    """Wrap an ExpectimaxNTuple searcher as a Player for GUI / Player.run()."""
    from src.agent.agent import Player

    searcher = ExpectimaxNTuple(net, depth=depth, adaptive=adaptive)

    class ExpectimaxPlayer(Player):
        def get_move(self):
            matrix = self._game.get_matrix()
            matrix = matrix if isinstance(matrix, np.ndarray) else matrix.detach().cpu().numpy()
            direction = searcher.get_move(np.asarray(matrix))
            return direction if direction is not None else ARROW_KEYS[0]

    return ExpectimaxPlayer(game=game, quiet=quiet, ui=ui)
