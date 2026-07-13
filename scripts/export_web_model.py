"""Export the trained n-tuple LUTs to a compact sparse binary for the browser.

The dense LUT is ``[4, 16_777_216]`` float32 = 256 MiB, but only ~7% is non-zero.
We ship only the non-zero ``(index, value)`` pairs per table:

    web/static/model/lut{t}.bin  =  uint32[nnz] indices  ++  float32[nnz] values
    web/static/model/manifest.json = { tableSize, nTuples, nSyms, tupleLen, counts }

The browser worker allocates 4 dense ``Float32Array(16_777_216)`` and scatters the
pairs back in (``arr[idx] = val``), reproducing the exact table used in Python.

Run from the repo root:  ``uv run scripts/export_web_model.py``
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.agent.ntuple import INDEX_BASE, TUPLE_LEN, NTupleNetwork


def find_default_model() -> str:
    candidates = sorted(glob.glob("models/ntuple_2048_*/final_model.npz"))
    if not candidates:
        candidates = sorted(glob.glob("models/ntuple_2048_*/best_model.npz"))
    if not candidates:
        raise SystemExit("no trained n-tuple model found under models/ntuple_2048_*/")
    return candidates[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", nargs="?", default=None, help="path to a *.npz checkpoint")
    ap.add_argument("--out", default="web/static/model", help="output directory")
    args = ap.parse_args()

    model_path = args.model or find_default_model()
    print(f"[export] loading {model_path}")
    net = NTupleNetwork()
    net.load(model_path)

    os.makedirs(args.out, exist_ok=True)
    table_size = INDEX_BASE ** TUPLE_LEN
    counts, total_nnz, total_bytes = [], 0, 0
    for t in range(net.n_tuples):
        lut = net.LUT[t]
        assert lut.shape == (table_size,), lut.shape
        nz = np.nonzero(lut)[0].astype(np.uint32)
        vals = lut[nz].astype(np.float32)
        counts.append(int(nz.size))
        total_nnz += int(nz.size)
        path = os.path.join(args.out, f"lut{t}.bin")
        with open(path, "wb") as f:
            f.write(nz.tobytes())      # uint32[nnz] indices
            f.write(vals.tobytes())    # float32[nnz] values
        size = os.path.getsize(path)
        total_bytes += size
        print(f"[export] lut{t}: {nz.size:>9,} non-zero  ({size / 1e6:6.1f} MB)")

    manifest = {
        "tableSize": table_size,
        "nTuples": net.n_tuples,
        "nSyms": net.n_syms,
        "tupleLen": TUPLE_LEN,
        "indexBase": INDEX_BASE,
        "counts": counts,
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[export] manifest.json written to {args.out}")
    print(f"[export] total non-zero: {total_nnz:,}  ({total_bytes / 1e6:.1f} MB uncompressed)")


if __name__ == "__main__":
    main()
