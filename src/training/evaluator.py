"""Offline evaluation over one or more board shapes (plan §5.5, §8.1).

Greedy games (no learning) on a private RNG so evaluation never perturbs the
training stream. Reports reach-rates for the standard tile thresholds plus score
and max-tile summaries per shape.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple

import numpy as np

from src.training.selfplay import play_game

THRESHOLDS = (512, 1024, 2048, 4096, 8192, 16384, 32768)


def evaluate_shape(net, H: int, W: int, games: int,
                   seed: int = 12345) -> Dict:
    rng = np.random.default_rng(seed)
    scores, maxes = [], []
    for _ in range(games):
        score, mt, _ = play_game(net, H, W, rng, learn=False)
        scores.append(score)
        maxes.append(mt)
    scores, maxes = np.array(scores), np.array(maxes)
    return {
        "shape": (H, W),
        "games": games,
        "mean_score": float(scores.mean()),
        "mean_max": float(maxes.mean()),
        "best_tile": int(maxes.max()),
        "reach": {t: float((maxes >= t).mean()) for t in THRESHOLDS},
        "dist": Counter((1 << int(np.log2(m)) if m > 0 else 0 for m in maxes)),
    }


def evaluate(net, shapes: List[Tuple[int, int]], games: int,
             seed: int = 12345) -> List[Dict]:
    return [evaluate_shape(net, H, W, games, seed + i)
            for i, (H, W) in enumerate(shapes)]


def format_stats(stats: Dict) -> str:
    r = stats["reach"]
    H, W = stats["shape"]
    hi = " ".join(f"{t}:{r[t]*100:.0f}%" for t in (1024, 2048, 4096, 8192)
                  if r[t] > 0) or "—"
    return (f"{H}x{W} | mean_score {stats['mean_score']:.0f} "
            f"mean_max {stats['mean_max']:.0f} best {stats['best_tile']} | {hi}")
