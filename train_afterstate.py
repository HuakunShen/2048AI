"""Train the neural afterstate value network (Track C1) on GPU self-play.

The NN counterpart to the tabular n-tuple: learns V(afterstate) and acts by
argmax_a [reward + V(afterstate)]. Runs entirely batched on the vectorized GPU
engine. This is the neural method designed to succeed where the DQN failed.

Examples
--------
    uv run train_afterstate.py --smoke
    uv run train_afterstate.py --games 2000000 --n-envs 4096
"""
import time
import argparse
from pathlib import Path
from collections import deque

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from src.agent.afterstate_net import AfterstateValueNet, train, evaluate
from src.game.model.vectorboard import TorchVectorEngine


def main():
    p = argparse.ArgumentParser("2048 neural afterstate value trainer")
    p.add_argument("--games", type=int, default=2_000_000)
    p.add_argument("--n-envs", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-final", type=float, default=1e-4)
    p.add_argument("--eval-every", type=int, default=100_000)
    p.add_argument("--eval-games", type=int, default=3000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.games, args.n_envs, args.eval_every = 40_000, 1024, 15_000

    name = f"afterstate_{int(time.time())}"
    ckpt_dir = Path("models") / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(f"runs/{name}")
    print(f"Checkpoints -> {ckpt_dir}\nTensorBoard -> runs/{name}\n"
          f"device={args.device} n_envs={args.n_envs} games={args.games:,}")

    net = AfterstateValueNet().to(args.device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"net params: {n_params:,}")

    recent = deque(maxlen=400)
    t0 = time.time()
    best = {"val": -1.0, "last_log": 0.0}

    def log_cb(games_done, loss, fin_max):
        recent.extend(fin_max.cpu().tolist())
        now = time.time()
        if now - best["last_log"] >= 5.0:          # throttle: at most every 5s
            best["last_log"] = now
            avg_max = float(np.mean(recent)) if recent else 0.0
            rate = games_done / (now - t0)
            writer.add_scalar("Train/loss", loss, games_done)
            writer.add_scalar("Train/avg_max_tile", avg_max, games_done)
            print(f"game {games_done:8d} | loss {loss:7.3f} | avg_max {avg_max:6.0f} "
                  f"| {rate:6.0f} games/s")

    def eval_cb(games_done, _stats):
        stats = evaluate(net, TorchVectorEngine(args.device), n_games=args.eval_games)
        r = stats["reach"]
        for t in (512, 1024, 2048, 4096, 8192):
            writer.add_scalar(f"Eval/reach{t}", r[t], games_done)
        writer.add_scalar("Eval/mean_score", stats["mean_score"], games_done)
        print(f"\n  [eval @{games_done}] reach 512:{r[512]*100:.0f}% 1024:{r[1024]*100:.0f}% "
              f"2048:{r[2048]*100:.0f}% 4096:{r[4096]*100:.0f}% 8192:{r[8192]*100:.0f}% "
              f"| mean_score {stats['mean_score']:.0f} best {stats['best_tile']}")
        torch.save(net.state_dict(), ckpt_dir / "final_model.pt")
        if r[2048] >= best["val"]:
            best["val"] = r[2048]
            torch.save(net.state_dict(), ckpt_dir / "best_model.pt")

    train(games_target=args.games, n_envs=args.n_envs, lr=args.lr,
          lr_final=args.lr_final, device=args.device, net=net,
          log_cb=log_cb, eval_cb=eval_cb, eval_every=args.eval_every)

    stats = evaluate(net, TorchVectorEngine(args.device), n_games=max(3000, args.eval_games))
    r = stats["reach"]
    print(f"\nDone in {(time.time()-t0)/60:.1f} min. Final: reach2048 {r[2048]*100:.1f}% "
          f"4096 {r[4096]*100:.1f}% 8192 {r[8192]*100:.1f}% | best {stats['best_tile']}")
    torch.save(net.state_dict(), ckpt_dir / "final_model.pt")
    with open(ckpt_dir / "test_results.txt", "w") as f:
        f.write(f"games: {args.games}\nn_envs: {args.n_envs}\n"
                f"reach2048: {r[2048]:.3f}\nreach4096: {r[4096]:.3f}\n"
                f"reach8192: {r[8192]:.3f}\nbest_tile: {stats['best_tile']}\n"
                f"mean_score: {stats['mean_score']:.1f}\n")


if __name__ == "__main__":
    main()
