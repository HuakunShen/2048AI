"""Evaluate a Universal N-Tuple checkpoint across shapes (greedy + expectimax).

Loads a checkpoint produced by ``train_universal.py`` and reports per-shape
reach-rates and scores, optionally under node-budget expectimax search. Games are
split across all CPU cores (``--procs``). Writes a machine-readable JSON so runs
can be diffed against a baseline (plan M0).

Examples
--------
    uv run evaluate_universal.py models/<run>/best_model.npz --shapes 4x4 5x5
    uv run evaluate_universal.py models/<run>/final_model.npz --patterns specialist \
        --alphabet 18 --shapes 4x4 --expectimax --adaptive 0.15:4 0.30:3 --games 500
"""
import json
import time
import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np

from src.ntuple.universal_value import UniversalNTuple
from src.ntuple import library as lib
from src.training.selfplay import play_game
from src.training.evaluator import format_stats, THRESHOLDS
from src.search.expectimax import UniversalExpectimax, play_game as expectimax_game

_PAT = {"core": lambda: lib.CORE, "default": lambda: lib.DEFAULT_PATTERNS,
        "specialist": lambda: lib.SPECIALIST_4X4}
_WORKER = {}


def parse_shape(tok):
    H, W = map(int, tok.lower().split("x"))
    return H, W


def parse_adaptive(tokens):
    return [(float(a.split(":")[0]), int(a.split(":")[1])) for a in tokens]


def _load_net(model, patterns_name, alphabet, residual, stages):
    patterns = _PAT[patterns_name]()
    if alphabet != 16:
        patterns = lib.with_alphabet(patterns, alphabet)
    net = UniversalNTuple(patterns=patterns, residual=residual, stages=stages)
    net.load(model)
    return net


def _init(model, patterns_name, alphabet, residual, stages):
    _WORKER["net"] = _load_net(model, patterns_name, alphabet, residual, stages)


def _chunk(args):
    """Play ``n`` games (greedy or expectimax) -> (scores, maxes)."""
    H, W, n, seed, adaptive, depth = args
    net = _WORKER["net"]
    rng = np.random.default_rng(seed)
    scores, maxes = [], []
    search = (UniversalExpectimax(net, depth=depth, adaptive=adaptive, seed=seed)
              if depth is not None else None)
    for _ in range(n):
        if search is not None:
            s, mt, _ = expectimax_game(net, H, W, rng, search)
        else:
            s, mt, _ = play_game(net, H, W, rng, learn=False)
        scores.append(s)
        maxes.append(mt)
    return scores, maxes


def parallel_eval(pool, H, W, games, adaptive, depth, seed=12345):
    """Split ``games`` across the pool; aggregate to a stats dict."""
    procs = pool._processes
    sizes = [games // procs + (1 if i < games % procs else 0) for i in range(procs)]
    tasks = [(H, W, n, seed + 1000 * i, adaptive, depth)
             for i, n in enumerate(sizes) if n]
    scores, maxes = [], []
    for sc, mx in pool.map(_chunk, tasks):
        scores.extend(sc)
        maxes.extend(mx)
    maxes = np.array(maxes)
    return {"shape": (H, W), "games": len(maxes), "mean_score": float(np.mean(scores)),
            "mean_max": float(maxes.mean()), "best_tile": int(maxes.max()),
            "reach": {t: float((maxes >= t).mean()) for t in THRESHOLDS}}


def main():
    p = argparse.ArgumentParser("Universal n-tuple evaluator")
    p.add_argument("model")
    p.add_argument("--patterns", choices=["core", "default", "specialist"],
                   default="core")
    p.add_argument("--alphabet", type=int, default=16)
    p.add_argument("--shapes", nargs="+", default=["4x4"])
    p.add_argument("--residual", action="store_true")
    p.add_argument("--stages", default="", help="e.g. 13,15 (must match training)")
    p.add_argument("--games", type=int, default=300)
    p.add_argument("--procs", type=int, default=0, help="0 = auto (cores-1)")
    p.add_argument("--expectimax", action="store_true")
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--adaptive", nargs="+", default=[],
                   help="ratio:depth rules, e.g. 0.15:4 0.30:3 0.55:2")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    import os
    procs = args.procs or max(1, (os.cpu_count() or 2) - 1)
    shapes = [parse_shape(s) for s in args.shapes]
    adaptive = parse_adaptive(args.adaptive) if args.adaptive else None
    report = {"model": args.model, "patterns": args.patterns, "games": args.games,
              "procs": procs, "greedy": [], "expectimax": []}
    print(f"model={args.model} patterns={args.patterns} alphabet={args.alphabet} "
          f"games={args.games} procs={procs}")

    t0 = time.time()
    ctx = mp.get_context("fork")
    stages = [int(s) for s in args.stages.split(",")] if args.stages else None
    with ctx.Pool(procs, initializer=_init,
                  initargs=(args.model, args.patterns, args.alphabet,
                            args.residual, stages)) as pool:
        for H, W in shapes:
            s = parallel_eval(pool, H, W, args.games, None, None)
            print("  [greedy]     " + format_stats(s))
            report["greedy"].append(s)
        if args.expectimax:
            for H, W in shapes:
                s = parallel_eval(pool, H, W, args.games, adaptive, args.depth)
                r = s["reach"]
                hi = " ".join(f"{t}:{r[t]*100:.0f}%" for t in
                              (2048, 4096, 8192, 16384, 32768) if r[t] > 0) or "—"
                print(f"  [expectimax] {H}x{W} | mean_score {s['mean_score']:.0f} "
                      f"best {s['best_tile']} | {hi}")
                report["expectimax"].append(s)
    print(f"done in {(time.time()-t0)/60:.1f} min")

    out = args.out or (Path(args.model).parent / "eval_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
