"""Freeze a model's evaluation metrics as a baseline (Milestone M0).

Records greedy reach-rates/scores for a trained model on a fixed seed set, plus
the git commit, model path and pattern/alphabet config, into ``docs/baseline.json``
so any later refactor can be compared against it with one command (plan §9 M0:
"任何重构可一键与基线比较").

Usage:
    uv run scripts/freeze_baseline.py                      # newest universal model
    uv run scripts/freeze_baseline.py models/<run>/final_model.npz \
        --patterns specialist --alphabet 18 --shapes 4x4 --games 500
"""
from __future__ import annotations

import os
import sys
import json
import glob
import argparse
import subprocess
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ntuple import library as lib
from src.ntuple.universal_value import UniversalNTuple
from src.training.evaluator import evaluate

_PAT = {"core": lib.CORE, "default": lib.DEFAULT_PATTERNS, "specialist": lib.SPECIALIST_4X4}


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", nargs="?", default=None)
    ap.add_argument("--patterns", choices=list(_PAT), default="specialist")
    ap.add_argument("--alphabet", type=int, default=18)
    ap.add_argument("--shapes", nargs="+", default=["4x4"])
    ap.add_argument("--games", type=int, default=500)
    ap.add_argument("--out", default="docs/baseline.json")
    args = ap.parse_args()

    model = args.model
    if model is None:
        cands = sorted(glob.glob("models/universal_*/final_model.npz")
                       + glob.glob("models/universal_*/best_model.npz"),
                       key=os.path.getmtime)
        if not cands:
            raise SystemExit("no universal model found under models/universal_*/")
        model = cands[-1]

    patterns = _PAT[args.patterns]
    if args.alphabet != 16:
        patterns = lib.with_alphabet(patterns, args.alphabet)
    net = UniversalNTuple(patterns=patterns, residual=False)
    net.load(model)

    shapes = [tuple(map(int, s.lower().split("x"))) for s in args.shapes]
    print(f"evaluating {model} on {shapes} ({args.games} games each, greedy)...")
    stats = evaluate(net, shapes, args.games)

    baseline = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "model": model,
        "patterns": args.patterns,
        "alphabet": args.alphabet,
        "schema_hash": net.hash,
        "games_per_shape": args.games,
        "results": [
            {"shape": list(s["shape"]), "mean_score": s["mean_score"],
             "mean_max": s["mean_max"], "best_tile": s["best_tile"],
             "reach": {str(t): v for t, v in s["reach"].items()}}
            for s in stats
        ],
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(baseline, f, indent=2)
    for s in stats:
        r = s["reach"]
        H, W = s["shape"]
        print(f"  {H}x{W}: mean_score {s['mean_score']:.0f} best {s['best_tile']} "
              f"| 2048 {r[2048]*100:.0f}% 4096 {r[4096]*100:.0f}% 8192 {r[8192]*100:.0f}%")
    print(f"wrote {args.out} (commit {baseline['git_commit']})")


if __name__ == "__main__":
    main()
