"""Evaluate a trained 2048 DQN checkpoint (headless, greedy play).

Reports score/tile statistics and the distribution of final max tiles over
many games, and compares against an untrained network as a baseline.

Examples
--------
    uv run evaluate_model.py models/dqn_2048_XXXX/best_model.pth
    uv run evaluate_model.py models/dqn_2048_XXXX/best_model.pth --games 200
"""
import sys
import argparse
from collections import Counter

import numpy as np
import torch

from src.game.controller.game import Game
from src.game.model.staticboardImpl import NumpyStaticBoard
from src.game.utils import ARROW_KEYS
from src.agent.agentImpl import DQNPlayer


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def play_games(agent: DQNPlayer, games: int):
    maxes, scores = [], []
    for _ in range(games):
        agent._game.restart()
        while not agent._game.get_is_done():
            idx, mask = agent._select_action(agent._matrix_np(), greedy=True)
            if not mask.any():
                break
            agent._game.move(ARROW_KEYS[idx])
            if agent._game.has_won():
                break
        maxes.append(int(agent._game.get_max_val()))
        scores.append(agent._game.get_score())
    return np.array(scores), np.array(maxes)


def report(name: str, scores: np.ndarray, maxes: np.ndarray):
    print(f"\n=== {name} ({len(scores)} games) ===")
    print(f"  score : mean {scores.mean():8.0f} | median {np.median(scores):8.0f} | max {scores.max():8.0f}")
    print(f"  tile  : mean {maxes.mean():8.0f} | median {np.median(maxes):8.0f} | max {maxes.max():8.0f}")
    dist = Counter(maxes.tolist())
    print("  max-tile distribution:")
    for tile in sorted(dist):
        n = dist[tile]
        bar = "#" * int(round(40 * n / len(maxes)))
        print(f"    {tile:6d}: {n:4d} ({100 * n / len(maxes):5.1f}%) {bar}")
    for thresh in (256, 512, 1024, 2048):
        rate = 100 * np.mean(maxes >= thresh)
        print(f"  reach >= {thresh:5d}: {rate:5.1f}%")


def main():
    parser = argparse.ArgumentParser("Evaluate a 2048 DQN checkpoint")
    parser.add_argument("checkpoint", help="path to a .pth checkpoint")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--no-baseline", action="store_true",
                        help="skip the untrained-network baseline comparison")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    game = Game(seed=None, static_board=NumpyStaticBoard)
    agent = DQNPlayer(game=game, quiet=True, ui=False, device=device)

    if not args.no_baseline:
        b_scores, b_maxes = play_games(agent, args.games)
        report("UNTRAINED baseline (masked-greedy)", b_scores, b_maxes)

    agent.load_model(args.checkpoint)
    print(f"\nLoaded checkpoint: {args.checkpoint} (trained steps: {agent.total_steps})")
    t_scores, t_maxes = play_games(agent, args.games)
    report(f"TRAINED model", t_scores, t_maxes)


if __name__ == "__main__":
    main()
