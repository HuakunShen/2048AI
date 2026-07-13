"""Dump a golden trace so the TS engine + value function can be checked bit-for-bit.

Plays a fixed-seed greedy n-tuple game, samples boards evenly across the whole
game (early open boards through the endgame), and records for each board:
``V(board)`` and, for all 4 directions, ``(afterstate, reward, changed)``. A few
hand-crafted edge cases (empty, full-no-merge, high tiles) are appended.

The TS test ``web/src/lib/engine/golden.test.ts`` must reproduce every field.

Run from the repo root:  ``uv run scripts/dump_golden.py``
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.agent.ntuple import NTupleNetwork
from src.game.model.staticboardImpl import NumpyStaticBoard
from src.game.utils import ARROW_KEYS


def find_default_model() -> str:
    candidates = sorted(glob.glob("models/ntuple_2048_*/final_model.npz"))
    if not candidates:
        candidates = sorted(glob.glob("models/ntuple_2048_*/best_model.npz"))
    if not candidates:
        raise SystemExit("no trained n-tuple model found under models/ntuple_2048_*/")
    return candidates[-1]


def board_record(net: NTupleNetwork, board: np.ndarray) -> dict:
    board = np.asarray(board, dtype=np.int64)
    rec = {
        "tiles": board.reshape(-1).tolist(),     # row-major tile values (0 = empty)
        "value": float(net.value(board)),
        "moves": {},
    }
    for d in ARROW_KEYS:
        after, reward, changed = NumpyStaticBoard.move(board, d, inplace=False)
        rec["moves"][d] = {
            "after": np.asarray(after, dtype=np.int64).reshape(-1).tolist(),
            "reward": int(reward),
            "changed": bool(changed),
        }
    return rec


def play_and_collect(net: NTupleNetwork, seed: int) -> list:
    """Greedy afterstate game; return every pre-move board encountered."""
    NumpyStaticBoard.set_random_seed(seed)
    board = NumpyStaticBoard.get_init_matrix().astype(np.int64)
    boards = []
    while True:
        best_dir, best_val = None, -1e18
        for d in ARROW_KEYS:
            after, reward, changed = NumpyStaticBoard.move(board, d, inplace=False)
            if not changed:
                continue
            val = reward + net.value(after)
            if val > best_val:
                best_val, best_dir = val, d
        if best_dir is None:
            break
        boards.append(board.copy())
        board, _, _ = NumpyStaticBoard.move(board, best_dir, inplace=True)
        NumpyStaticBoard.set_random_cell(board, inplace=True)
        if NumpyStaticBoard.compute_is_done(board):
            boards.append(board.copy())
            break
    return boards


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", nargs="?", default=None)
    ap.add_argument("--out", default="web/src/lib/engine/__fixtures__/golden.json")
    ap.add_argument("--seed", type=int, default=2048)
    ap.add_argument("--samples", type=int, default=200)
    args = ap.parse_args()

    net = NTupleNetwork()
    net.load(args.model or find_default_model())

    boards = play_and_collect(net, args.seed)
    if len(boards) > args.samples:
        idxs = np.linspace(0, len(boards) - 1, args.samples).astype(int)
        sampled = [boards[i] for i in idxs]
    else:
        sampled = boards

    edge_cases = [
        np.zeros((4, 4), dtype=np.int64),                                    # empty
        np.array([[2, 2, 2, 2], [4, 4, 4, 4], [2, 2, 4, 4], [8, 8, 8, 8]]),  # merges
        np.array([[2, 4, 8, 16], [32, 64, 128, 256],
                  [512, 1024, 2048, 4096], [2, 4, 8, 16]]),                  # high tiles
        np.array([[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]]),  # full, no merge
    ]
    records = [board_record(net, b) for b in sampled]
    records += [board_record(net, b) for b in edge_cases]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"seed": args.seed, "boards": records}, f)
    print(f"[golden] game length {len(boards)} moves; wrote {len(records)} boards -> {args.out}")


if __name__ == "__main__":
    main()
