"""Unified benchmark harness for 2048 agents.

Every agent is evaluated under *identical* conditions — the same fixed list of
seeds, the same headless engine, the same reported metrics — so results across
methods (greedy n-tuple, expectimax, ...) are directly comparable. Each run
appends one row to ``docs/results.md`` so the table accumulates over time.

Examples
--------
    uv run benchmark.py ntuple --model models/<run>/best_model.npz --games 300 --procs 20
    uv run benchmark.py expectimax --model models/<run>/best_model.npz --depth 2 --games 300 --procs 20
    uv run benchmark.py expectimax --model ... --adaptive 3:5,7:3,16:2 --games 200

Metrics: reach-rate for each of 1024/2048/4096/8192/16384/32768, mean/median/max
score, mean moves per game, and moves/second (the cost of the method).
"""
from __future__ import annotations

import time
import argparse
import datetime
import multiprocessing as mp
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from src.game.model.staticboardImpl import NumpyStaticBoard
from src.agent.ntuple import NTupleNetwork
from src.agent.expectimax import ExpectimaxNTuple

REACH_TILES = (1024, 2048, 4096, 8192, 16384, 32768)
RESULTS_MD = Path("docs/results.md")

Policy = Callable[[np.ndarray], Optional[str]]


# --------------------------------------------------------------------------- #
# Headless seeded playout — shared by every agent.
# --------------------------------------------------------------------------- #
def play_headless(policy: Policy, seed: int, max_moves: int = 1_000_000):
    """Play one game to termination under ``policy``, seeded for reproducibility.

    ``policy(board) -> direction`` must return a board-changing direction, or
    None when no move is possible. Returns ``(score, max_tile, moves)``.
    """
    NumpyStaticBoard.set_random_seed(seed)
    board = NumpyStaticBoard.get_empty_matrix(4, 4)
    NumpyStaticBoard.set_random_cell(board, inplace=True)
    NumpyStaticBoard.set_random_cell(board, inplace=True)

    score = 0
    moves = 0
    while moves < max_moves:
        direction = policy(board)
        if direction is None:
            break
        after, reward, changed = NumpyStaticBoard.move(board, direction, inplace=False)
        if not changed:                 # policy must only pick valid moves
            break
        score += int(reward)
        board = after
        NumpyStaticBoard.set_random_cell(board, inplace=True)
        moves += 1
        if NumpyStaticBoard.compute_is_done(board):
            break
    return score, int(board.max()), moves


# --------------------------------------------------------------------------- #
# Policy construction (per worker, loads the model once).
# --------------------------------------------------------------------------- #
def build_policy(cfg: dict) -> Policy:
    net = NTupleNetwork()
    net.load(cfg["model"])
    kind = cfg["kind"]
    if kind == "ntuple":                # greedy 1-ply
        searcher = ExpectimaxNTuple(net, depth=1)
        return searcher.get_move
    if kind == "expectimax":
        searcher = ExpectimaxNTuple(
            net,
            depth=cfg["depth"],
            adaptive=cfg["adaptive"],
            max_chance_cells=cfg["max_chance_cells"],
        )
        return searcher.get_move
    raise ValueError(f"unknown agent kind: {kind}")


_WORKER_POLICY: Optional[Policy] = None


def _init_worker(cfg: dict):
    global _WORKER_POLICY
    _WORKER_POLICY = build_policy(cfg)


def _run_seed(seed: int):
    t0 = time.time()
    score, max_tile, moves = play_headless(_WORKER_POLICY, seed)
    return seed, score, max_tile, moves, time.time() - t0


# --------------------------------------------------------------------------- #
# Reporting.
# --------------------------------------------------------------------------- #
def summarize(rows, label: str, cfg_str: str, wall_s: float):
    maxes = np.array([r[2] for r in rows])
    scores = np.array([r[1] for r in rows])
    moves = np.array([r[3] for r in rows])
    n = len(rows)
    reach = {t: float(np.mean(maxes >= t)) for t in REACH_TILES}
    total_moves = int(moves.sum())
    stats = {
        "label": label,
        "config": cfg_str,
        "games": n,
        "reach": reach,
        "mean_score": float(scores.mean()),
        "median_score": float(np.median(scores)),
        "max_score": int(scores.max()),
        "mean_moves": float(moves.mean()),
        "best_tile": int(maxes.max()),
        "moves_per_s": total_moves / wall_s if wall_s > 0 else 0.0,
        "wall_s": wall_s,
    }
    return stats


def print_summary(s: dict):
    print(f"\n=== {s['label']}  ({s['config']}) ===")
    print(f"games: {s['games']}   wall: {s['wall_s']:.1f}s   "
          f"moves/s: {s['moves_per_s']:.0f}   mean_moves/game: {s['mean_moves']:.0f}")
    reach_str = "  ".join(f"{t}:{s['reach'][t]*100:5.1f}%" for t in REACH_TILES)
    print(f"reach   {reach_str}")
    print(f"score   mean {s['mean_score']:.0f}   median {s['median_score']:.0f}   "
          f"max {s['max_score']}   best_tile {s['best_tile']}")


def append_results_md(s: dict, model: str):
    RESULTS_MD.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "| date | agent | config | model | games | @1024 | @2048 | @4096 | @8192 "
        "| @16384 | mean score | median | max tile | moves/game | moves/s |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    if not RESULTS_MD.exists():
        RESULTS_MD.write_text("# Benchmark results\n\n"
                              "Rows appended by `benchmark.py`. Same seed list per run "
                              "(default seeds 1..N) so agents are directly comparable.\n\n"
                              + header)
    date = datetime.date.today().isoformat()
    r = s["reach"]
    row = (f"| {date} | {s['label']} | {s['config']} | {Path(model).parent.name} "
           f"| {s['games']} | {r[1024]*100:.1f}% | {r[2048]*100:.1f}% | {r[4096]*100:.1f}% "
           f"| {r[8192]*100:.1f}% | {r[16384]*100:.1f}% | {s['mean_score']:.0f} "
           f"| {s['median_score']:.0f} | {s['best_tile']} | {s['mean_moves']:.0f} "
           f"| {s['moves_per_s']:.0f} |\n")
    with RESULTS_MD.open("a") as f:
        f.write(row)
    print(f"\nappended row to {RESULTS_MD}")


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser("2048 agent benchmark")
    sub = parser.add_subparsers(dest="kind", required=True)

    p_nt = sub.add_parser("ntuple", help="greedy 1-ply n-tuple")
    p_ex = sub.add_parser("expectimax", help="expectimax over afterstates")
    p_ex.add_argument("--depth", type=int, default=2)
    p_ex.add_argument("--adaptive", type=str, default=None,
                      help="comma rules 'maxEmpty:depth', e.g. 3:5,7:3,16:2")
    p_ex.add_argument("--max-chance-cells", type=int, default=16)

    for p in (p_nt, p_ex):
        p.add_argument("--model", required=True)
        p.add_argument("--games", type=int, default=300)
        p.add_argument("--procs", type=int, default=max(1, mp.cpu_count() - 4))
        p.add_argument("--seed-start", type=int, default=1)
        p.add_argument("--no-log", action="store_true", help="don't append to results.md")
    args = parser.parse_args()

    seeds = list(range(args.seed_start, args.seed_start + args.games))

    adaptive = None
    cfg_str = "greedy(d=1)"
    if args.kind == "expectimax":
        if args.adaptive:
            adaptive = [tuple(int(x) for x in rule.split(":"))
                        for rule in args.adaptive.split(",")]
            cfg_str = f"adaptive[{args.adaptive}]"
        else:
            cfg_str = f"depth={args.depth}"
    cfg = {
        "kind": args.kind,
        "model": args.model,
        "depth": getattr(args, "depth", 1),
        "adaptive": adaptive,
        "max_chance_cells": getattr(args, "max_chance_cells", 16),
    }

    label = "n-tuple greedy" if args.kind == "ntuple" else "expectimax n-tuple"
    print(f"Benchmarking {label} [{cfg_str}] on {args.games} games "
          f"(seeds {seeds[0]}..{seeds[-1]}), {args.procs} procs\nmodel: {args.model}")

    t0 = time.time()
    if args.procs > 1:
        with mp.Pool(args.procs, initializer=_init_worker, initargs=(cfg,)) as pool:
            rows = []
            for i, res in enumerate(pool.imap_unordered(_run_seed, seeds), 1):
                rows.append(res)
                if i % 25 == 0 or i == len(seeds):
                    print(f"  {i}/{len(seeds)} games done", end="\r", flush=True)
    else:
        _init_worker(cfg)
        rows = [_run_seed(s) for s in seeds]
    wall = time.time() - t0

    s = summarize(rows, label, cfg_str, wall)
    print_summary(s)
    if not args.no_log:
        append_results_md(s, args.model)


if __name__ == "__main__":
    main()
