"""Train the Universal N-Tuple value function across board shapes (M3/M5).

One model, many shapes. Shapes are sampled by a transition quota so long games on
large boards don't dominate the updates (plan §5.3). Afterstate TD(0) with
temporal-coherence adaptive step sizes, exactly as the 4×4 agent, but on the
generic engine.

Examples
--------
    uv run train_universal.py --smoke
    uv run train_universal.py --shapes 4x4 5x5 --games 40000
    uv run train_universal.py --shapes 4x4:0.5 5x5:0.3 6x6:0.2 --games 200000
"""
import os
import time
import argparse
from pathlib import Path
from collections import deque, defaultdict

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from src.ntuple.universal_value import UniversalNTuple
from src.ntuple import library as lib
from src.training.curriculum import TransitionQuota
from src.training.selfplay import play_game
from src.training.evaluator import evaluate, format_stats
from src.training.parallel import train_parallel


def parse_shapes(tokens):
    """['4x4:0.5', '5x5'] -> (shapes, weights). Missing weights default to 1."""
    shapes, weights = [], []
    for tok in tokens:
        if ":" in tok:
            hw, w = tok.split(":")
            weight = float(w)
        else:
            hw, weight = tok, 1.0
        H, W = map(int, hw.lower().split("x"))
        shapes.append((H, W))
        weights.append(weight)
    return shapes, weights


def main():
    p = argparse.ArgumentParser("Universal n-tuple trainer")
    p.add_argument("--shapes", nargs="+", default=["4x4", "5x5"],
                   help="e.g. 4x4 5x5  or  4x4:0.5 5x5:0.3 6x6:0.2")
    p.add_argument("--held-out", nargs="+", default=[],
                   help="shapes to evaluate but never train on (§1.2 generalization)")
    p.add_argument("--patterns", choices=["core", "default", "specialist"],
                   default="core")
    p.add_argument("--alphabet", type=int, default=16,
                   help="tile alphabet; 18 covers up to 131072 (4×4 specialist)")
    p.add_argument("--games", type=int, default=40000)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--no-tc", action="store_true")
    p.add_argument("--residual", action="store_true",
                   help="enable the M4 per-role residual head (small patterns)")
    p.add_argument("--rho", type=float, default=0.25)
    p.add_argument("--alpha-residual", type=float, default=0.1)
    p.add_argument("--stages", default="",
                   help="multi-stage exponent thresholds, e.g. 13,15 (M7 weight "
                        "promotion; splits tables by max-tile stage)")
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--eval-games", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=0,
                   help="parallel self-play workers (0=auto=cores-2; 1=deterministic)")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.games, args.eval_every, args.eval_games = 1500, 500, 60
    workers = args.workers or max(1, (os.cpu_count() or 2) - 2)

    shapes, weights = parse_shapes(args.shapes)
    patterns = {"core": lib.CORE, "default": lib.DEFAULT_PATTERNS,
                "specialist": lib.SPECIALIST_4X4}[args.patterns]
    if args.alphabet != 16:
        patterns = lib.with_alphabet(patterns, args.alphabet)
    stages = [int(s) for s in args.stages.split(",")] if args.stages else None
    quota = TransitionQuota(shapes, weights)
    net = UniversalNTuple(patterns=patterns, alpha=args.alpha, tc=not args.no_tc,
                          residual=args.residual, rho=args.rho,
                          alpha_residual=args.alpha_residual, stages=stages)

    name = f"universal_{'x'.join(f'{h}-{w}' for h, w in shapes)}_{int(time.time())}"
    ckpt_dir = Path("models") / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(f"runs/{name}")
    print(f"Checkpoints -> {ckpt_dir}\nTensorBoard -> runs/{name}")
    print(f"patterns={args.patterns} ({len(patterns)}) tables={net.total*4/1e6:.0f}MB "
          f"tc={not args.no_tc} residual={args.residual} workers={workers} "
          f"shapes={shapes} weights={quota.targets.round(2).tolist()}")

    best = {"metric": 0.0}

    def run_eval(step, model):
        """Periodic eval + best-checkpoint, shared by both training paths."""
        metric = 0.0
        for s in evaluate(model, shapes, args.eval_games):
            print("  [eval] " + format_stats(s))
            h, w = s["shape"]
            for t, v in s["reach"].items():
                writer.add_scalar(f"Eval/{h}x{w}_reach{t}", v, step)
            writer.add_scalar(f"Eval/{h}x{w}_mean_score", s["mean_score"], step)
            metric += s["mean_score"]
        if metric >= best["metric"]:
            best["metric"] = metric
            model.save(str(ckpt_dir / "best_model.npz"))

    t0 = time.time()
    if workers > 1:
        cfg = {"patterns": args.patterns, "alphabet": args.alphabet,
               "alpha": args.alpha, "tc": not args.no_tc, "residual": args.residual,
               "rho": args.rho, "alpha_residual": args.alpha_residual,
               "stages": stages}
        net = train_parallel(cfg, shapes, weights, workers, args.games,
                             args.eval_every, eval_cb=run_eval, seed=args.seed)
    else:
        rng = np.random.default_rng(args.seed)
        recent_max = defaultdict(lambda: deque(maxlen=100))
        for game in range(1, args.games + 1):
            idx = quota.next_index()
            H, W = shapes[idx]
            _, mt, moves = play_game(net, H, W, rng, learn=True)
            quota.record(idx, moves)
            recent_max[(H, W)].append(mt)
            if game % 200 == 0:
                rate = game / (time.time() - t0)
                parts = " ".join(f"{h}x{w}:{np.mean(recent_max[(h, w)]):.0f}"
                                 for h, w in shapes if recent_max[(h, w)])
                print(f"game {game:6d} | {rate:5.1f} g/s | avg_max {parts} "
                      f"| shares {quota.realised_shares().round(2).tolist()}")
            if game % args.eval_every == 0:
                run_eval(game, net)

    net.save(str(ckpt_dir / "final_model.npz"))
    print(f"\nDone: {args.games} games in {(time.time()-t0)/60:.1f} min. Final eval:")
    for s in evaluate(net, shapes, max(200, args.eval_games)):
        print("  [train] " + format_stats(s))
    if args.held_out:
        held, _ = parse_shapes(args.held_out)
        print("  held-out shapes (never trained — §1.2 generalization):")
        for s in evaluate(net, held, max(200, args.eval_games)):
            print("  [held-out] " + format_stats(s))


if __name__ == "__main__":
    main()
