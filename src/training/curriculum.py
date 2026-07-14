"""Transition-quota shape sampling (plan §5.3).

Mixed-size training must be balanced by **transitions**, not episodes: an 8×8
game produces far more moves than a 4×4 game, so round-robin-by-episode lets big
boards silently dominate the updates (plan §2.4). :class:`TransitionQuota` tracks
moves per shape and always hands out the shape furthest *below* its target share,
driving the realised transition mix toward the configured proportions.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


class TransitionQuota:
    """Pick the most under-served shape by cumulative transition share."""

    def __init__(self, shapes: List[Tuple[int, int]], targets: List[float]):
        assert len(shapes) == len(targets)
        self.shapes = list(shapes)
        t = np.asarray(targets, dtype=np.float64)
        self.targets = t / t.sum()
        self.counts = np.zeros(len(shapes), dtype=np.float64)

    def next_index(self) -> int:
        total = self.counts.sum()
        if total <= 0:
            return int(np.argmax(self.targets))
        share = self.counts / total
        return int(np.argmax(self.targets - share))

    def next_shape(self) -> Tuple[int, int]:
        return self.shapes[self.next_index()]

    def record(self, index: int, transitions: int):
        self.counts[index] += transitions

    def realised_shares(self) -> np.ndarray:
        total = self.counts.sum()
        return self.counts / total if total > 0 else self.counts
