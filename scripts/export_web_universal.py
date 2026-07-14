"""Export a trained Universal N-Tuple model to a compact sparse binary for the web.

The universal value function shares one dense table per pattern across all board
shapes, so the browser only needs the (small) per-pattern tables plus the pattern
schema — it compiles placements for whatever H×W the user picks. Only the
**base** tables are shipped (the residual head is training-only), so the browser
computes exactly the base value this script also records in the golden file.

Outputs (default web/static/model/):
    lut{k}_{p}.bin    uint32[nnz] indices ++ float32[nnz] values   (pattern k, part p)
    manifest.json     { alphabet, patterns:[{id,cells}], tableSizes, parts, shapes }
    golden.json       [{ shape:[H,W], tiles:[...], value }]  base-only V, for tests

Large tables are split into parts < 25 MiB so they satisfy the Cloudflare Workers
per-asset size limit; each part is a self-contained (indices, values) sub-list.

Run from repo root:
    uv run scripts/export_web_universal.py models/<run>/final_model.npz \
        --patterns core --alphabet 16
"""
from __future__ import annotations

import os
import sys
import json
import glob
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ntuple import library as lib
from src.ntuple.universal_value import UniversalNTuple

_PAT = {"core": lib.CORE, "default": lib.DEFAULT_PATTERNS, "specialist": lib.SPECIALIST_4X4}


def _random_board(H, W, rng, max_exp=11, fill=0.6):
    b = np.zeros((H, W), dtype=np.int8)
    cells = rng.random((H, W)) < fill
    b[cells] = rng.integers(1, max_exp + 1, size=int(cells.sum()), dtype=np.int8)
    return b


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", nargs="?", default=None, help="path to a *.npz checkpoint")
    ap.add_argument("--patterns", choices=list(_PAT), default="core")
    ap.add_argument("--alphabet", type=int, default=16)
    ap.add_argument("--out", default="web/static/model")
    ap.add_argument("--shapes", nargs="+",
                    default=["4x4", "5x5", "4x5", "5x4", "3x4", "6x6"])
    args = ap.parse_args()

    model = args.model
    if model is None:
        cands = sorted(glob.glob("models/universal_*/final_model.npz"))
        if not cands:
            raise SystemExit("no universal model found under models/universal_*/")
        model = cands[-1]

    # The browser value function is base-only and has no notion of stages, so a
    # multi-stage (M7) checkpoint cannot be exported faithfully: its value depends
    # on which stage a board is in, and shipping stage-0 tables would silently give
    # wrong late-game values. Reject it up front with an actionable message rather
    # than dying inside load() on the stage-config mismatch.
    with np.load(model, allow_pickle=False) as _d:
        saved_stages = _d["stages"].tolist() if "stages" in _d else []
    if saved_stages:
        raise SystemExit(
            f"{model} is a multi-stage checkpoint (stages={saved_stages}), which the "
            "browser value function does not support (it evaluates the base tables "
            "only). Export a model trained without --stages.")

    patterns = _PAT[args.patterns]
    if args.alphabet != 16:
        patterns = lib.with_alphabet(patterns, args.alphabet)
    # residual=False -> value() is base-only, exactly what the browser computes.
    net = UniversalNTuple(patterns=patterns, residual=False)
    net.load(model)
    print(f"loaded {model}: {len(patterns)} patterns, alphabet {net.alphabet}")

    os.makedirs(args.out, exist_ok=True)
    # Clear any stale part files from a previous export.
    for old in glob.glob(os.path.join(args.out, "lut*.bin")):
        os.remove(old)
    # <25 MiB/asset (Cloudflare limit): 8 bytes/entry -> ≤ ~3.1M entries; use 2M.
    MAX_PER_PART = 2_000_000
    parts, table_sizes = [], []
    for k, p in enumerate(net.patterns):
        size = p.table_size
        table = net.LUT[net.offsets[k]: net.offsets[k] + size]
        nz = np.nonzero(table)[0].astype(np.uint32)
        vals = table[nz].astype(np.float32)
        part_nnz = []
        for pi, start in enumerate(range(0, max(nz.size, 1), MAX_PER_PART)):
            idx_p = nz[start:start + MAX_PER_PART]
            val_p = vals[start:start + MAX_PER_PART]
            with open(os.path.join(args.out, f"lut{k}_{pi}.bin"), "wb") as f:
                f.write(idx_p.tobytes())
                f.write(val_p.tobytes())
            part_nnz.append(int(idx_p.size))
        parts.append(part_nnz)
        table_sizes.append(int(size))
        print(f"  {p.id}: {nz.size:,}/{size:,} nonzero in {len(part_nnz)} part(s) "
              f"({nz.size * 8 / 1e6:.1f} MB)")

    manifest = {
        "alphabet": int(net.alphabet),
        "patterns": [{"id": p.id, "cells": [list(c) for c in p.cells]}
                     for p in net.patterns],
        "tableSizes": table_sizes,
        "parts": parts,
        "shapes": args.shapes,
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f)

    # Golden: base-only V on random boards of each shape, for the TS parity test.
    rng = np.random.default_rng(0)
    golden = []
    for tok in args.shapes:
        H, W = map(int, tok.lower().split("x"))
        for _ in range(6):
            b = _random_board(H, W, rng)
            golden.append({"shape": [H, W],
                           "tiles": [(1 << int(e)) if e else 0 for e in b.reshape(-1)],
                           "value": float(net.value(b))})
    with open(os.path.join(args.out, "golden.json"), "w") as f:
        json.dump(golden, f)
    total = sum(sum(ps) for ps in parts) * 8 / 1e6
    print(f"wrote {args.out}/ — {total:.1f} MB of tables, {len(golden)} golden boards")


if __name__ == "__main__":
    main()
