"""Train an n-tuple network on 2048 by afterstate TD-learning (headless).

This is the high-performance, sample-efficient RL approach for 2048: it learns
a value function over afterstates using overlapping n-tuple lookup tables with
8-fold symmetric weight sharing. It is CPU/tabular (no GPU needed) and typically
climbs past 512 -> 1024 within a few thousand games and reaches 2048.

Examples
--------
    uv run train_ntuple.py                      # default 20k games
    uv run train_ntuple.py --games 50000        # train longer
    uv run train_ntuple.py --smoke              # 300-game sanity run
"""
import time
import argparse
from pathlib import Path
from collections import deque, Counter

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from src.agent.ntuple import NTupleNetwork, play_game


def evaluate(net: NTupleNetwork, games: int):
    scores, maxes = [], []
    for _ in range(games):
        score, max_tile, _ = play_game(net, learn=False)
        scores.append(score)
        maxes.append(max_tile)
    scores, maxes = np.array(scores), np.array(maxes)
    return {
        "mean_score": float(scores.mean()),
        "max_score": int(scores.max()),
        "mean_max": float(maxes.mean()),
        "best_tile": int(maxes.max()),
        "reach1024": float(np.mean(maxes >= 1024)),
        "reach2048": float(np.mean(maxes >= 2048)),
        "dist": Counter(maxes.tolist()),
    }


def main():
    parser = argparse.ArgumentParser("2048 n-tuple TD trainer")
    parser.add_argument("--games", type=int, default=20000)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--eval-games", type=int, default=200)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.games = 300
        args.eval_every = 100
        args.eval_games = 50

    timestamp = int(time.time())
    name = f"ntuple_2048_{timestamp}"
    ckpt_dir = Path("models") / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(f"runs/{name}")
    print(f"Checkpoints -> {ckpt_dir}\nTensorBoard -> runs/{name}")

    net = NTupleNetwork(alpha=args.alpha)
    print(f"n-tuple net: {net.n_tuples} tuples x {net.n_syms} symmetries "
          f"= {net.n_instances} instances, {len(net.LUT)} LUTs of {net.LUT[0].size:,} each")

    recent_score = deque(maxlen=100)
    recent_max = deque(maxlen=100)
    best_eval = 0.0
    t0 = time.time()

    for game in range(1, args.games + 1):
        score, max_tile, moves = play_game(net, learn=True)
        recent_score.append(score)
        recent_max.append(max_tile)

        if game % 100 == 0:
            avg_s = np.mean(recent_score)
            avg_m = np.mean(recent_max)
            writer.add_scalar("Train/avg_score_100", avg_s, game)
            writer.add_scalar("Train/avg_max_100", avg_m, game)
            rate = game / (time.time() - t0)
            print(f"game {game:6d} | avg_score {avg_s:7.0f} | avg_max {avg_m:6.0f} "
                  f"| last_max {max_tile:5d} | {rate:5.1f} games/s")

        if game % args.eval_every == 0:
            stats = evaluate(net, args.eval_games)
            writer.add_scalar("Eval/mean_score", stats["mean_score"], game)
            writer.add_scalar("Eval/mean_max_tile", stats["mean_max"], game)
            writer.add_scalar("Eval/best_tile", stats["best_tile"], game)
            writer.add_scalar("Eval/reach1024_rate", stats["reach1024"], game)
            writer.add_scalar("Eval/reach2048_rate", stats["reach2048"], game)
            dist = " ".join(f"{t}:{n}" for t, n in sorted(stats["dist"].items()))
            print(f"  [eval @{game}] mean_score {stats['mean_score']:.0f} "
                  f"mean_max {stats['mean_max']:.0f} best {stats['best_tile']} "
                  f"| 1024 {stats['reach1024']*100:.0f}% 2048 {stats['reach2048']*100:.0f}% "
                  f"| dist {dist}")
            score_metric = stats["mean_score"]
            if score_metric >= best_eval:
                best_eval = score_metric
                net.save(str(ckpt_dir / "best_model.npz"))

    net.save(str(ckpt_dir / "final_model.npz"))
    elapsed = time.time() - t0
    print(f"\nDone: {args.games} games in {elapsed/60:.1f} min. Final evaluation...")
    stats = evaluate(net, max(500, args.eval_games))
    print(f"  mean_score {stats['mean_score']:.0f} | mean_max {stats['mean_max']:.0f} "
          f"| best {stats['best_tile']} | reach1024 {stats['reach1024']*100:.1f}% "
          f"| reach2048 {stats['reach2048']*100:.1f}%")
    with open(ckpt_dir / "test_results.txt", "w") as f:
        f.write(
            f"games: {args.games}\ntrain_minutes: {elapsed/60:.2f}\n"
            f"mean_score: {stats['mean_score']:.1f}\nmean_max_tile: {stats['mean_max']:.1f}\n"
            f"best_tile: {stats['best_tile']}\n"
            f"reach1024_rate: {stats['reach1024']:.3f}\nreach2048_rate: {stats['reach2048']:.3f}\n"
        )


if __name__ == "__main__":
    main()
