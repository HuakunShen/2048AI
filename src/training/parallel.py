"""Lock-free parallel self-play training (Hogwild!) — use all CPU cores.

Single-process TD training pegs one core and leaves the rest idle. N-tuple TD
parallelises naturally with the **Hogwild!** recipe: many worker processes play
self-play games and apply TD updates to **one shared copy** of the lookup tables,
without locks. Concurrent read/modify/write races cause occasional lost updates,
but for sparse high-dimensional tables that is benign and TD still converges
(plan §5.5 "lock-free optimistic parallelism"; the deterministic single-process
path in :mod:`src.training.selfplay` remains the reproducible baseline).

Implementation notes:
  * Tables live in ``multiprocessing.RawArray`` (one physical copy in shared
    memory). Each worker wraps them as numpy views, so the numba kernels write
    straight into shared memory — no copying, no serialization on the hot path.
  * Fork start method: workers inherit the RawArrays passed as arguments.
  * Each worker has its own RNG seed and its own ``TransitionQuota`` so the shape
    mix is balanced even per-worker.
  * The main process holds a net bound to the same shared tables, so periodic
    evaluation and checkpointing see the live (racy) weights — fine for
    monitoring; the reproducible numbers come from a fresh greedy eval.
"""
from __future__ import annotations

import time
import multiprocessing as mp
from collections import deque, defaultdict
from typing import Dict, List, Tuple

import numpy as np

from src.ntuple.universal_value import UniversalNTuple
from src.ntuple import library as lib
from src.training.curriculum import TransitionQuota
from src.training.selfplay import play_game

_PATTERN_SETS = {"core": lambda: lib.CORE, "default": lambda: lib.DEFAULT_PATTERNS,
                 "specialist": lambda: lib.SPECIALIST_4X4}


def build_net(cfg: Dict) -> UniversalNTuple:
    """Construct a net from a plain-dict config (picklable across processes)."""
    patterns = _PATTERN_SETS[cfg["patterns"]]()
    if cfg["alphabet"] != 16:
        patterns = lib.with_alphabet(patterns, cfg["alphabet"])
    return UniversalNTuple(patterns=patterns, alpha=cfg["alpha"], tc=cfg["tc"],
                           residual=cfg["residual"], rho=cfg["rho"],
                           alpha_residual=cfg["alpha_residual"],
                           stages=cfg.get("stages"))


def _make_raws(net: UniversalNTuple) -> Dict:
    """Allocate shared RawArrays sized to the net's tables (× n_stages)."""
    n = net.n_stages * net.total
    raws = {"LUT": mp.RawArray("f", n)}
    if net.tc:
        raws["E"] = mp.RawArray("f", n)
        raws["A"] = mp.RawArray("f", n)
    if net.residual and net.res_total:
        raws["R"] = mp.RawArray("f", net.res_total)
        raws["RE"] = mp.RawArray("f", net.res_total)
        raws["RA"] = mp.RawArray("f", net.res_total)
    return raws


def _bind(net: UniversalNTuple, raws: Dict) -> UniversalNTuple:
    """Rebind a net's arrays onto shared RawArrays (in place)."""
    shape = (net.n_stages, net.total) if net.n_stages > 1 else (net.total,)
    net.LUT = np.frombuffer(raws["LUT"], dtype=np.float32).reshape(shape)
    if net.tc:
        net.E = np.frombuffer(raws["E"], dtype=np.float32).reshape(shape)
        net.A = np.frombuffer(raws["A"], dtype=np.float32).reshape(shape)
    if net.residual and net.res_total:
        net.R = np.frombuffer(raws["R"], dtype=np.float32)
        net.RE = np.frombuffer(raws["RE"], dtype=np.float32)
        net.RA = np.frombuffer(raws["RA"], dtype=np.float32)
    return net


def _worker(wid: int, cfg: Dict, raws: Dict, shapes: List[Tuple[int, int]],
            weights: List[float], seed: int, report_every: int,
            q: mp.Queue, stop):
    net = _bind(build_net(cfg), raws)
    quota = TransitionQuota(shapes, weights)
    rng = np.random.default_rng(seed)
    n, maxes = 0, defaultdict(list)
    while not stop.is_set():
        idx = quota.next_index()
        H, W = shapes[idx]
        _, mt, moves = play_game(net, H, W, rng, learn=True)
        quota.record(idx, moves)
        n += 1
        maxes[(H, W)].append(mt)
        if n >= report_every:
            q.put((n, {k: v for k, v in maxes.items()}))
            n, maxes = 0, defaultdict(list)
    q.put((n, {k: v for k, v in maxes.items()}))     # flush tail


def train_parallel(cfg: Dict, shapes, weights, workers: int, total_games: int,
                   eval_every: int, eval_cb=None, log=print, report_every: int = 25,
                   seed: int = 0) -> UniversalNTuple:
    """Run Hogwild self-play across ``workers`` processes. Returns the shared net."""
    ctx = mp.get_context("fork")
    net = build_net(cfg)
    raws = _make_raws(net)
    _bind(net, raws)                       # main net shares the same tables

    q = ctx.Queue()
    stop = ctx.Event()
    procs = [ctx.Process(target=_worker,
                         args=(i, cfg, raws, shapes, weights, seed + 1 + i,
                               report_every, q, stop), daemon=True)
             for i in range(workers)]
    for p in procs:
        p.start()

    recent = defaultdict(lambda: deque(maxlen=400))
    done = next_eval = 0
    next_eval = eval_every
    t0 = time.time()
    try:
        while done < total_games:
            n, maxes = q.get()
            done += n
            for shp, vals in maxes.items():
                recent[shp].extend(vals)
            if done // 400 != (done - n) // 400:      # ~ every 400 games
                rate = done / (time.time() - t0)
                parts = " ".join(f"{h}x{w}:{np.mean(recent[(h, w)]):.0f}"
                                 for h, w in shapes if recent[(h, w)])
                log(f"game {done:7d} | {rate:6.0f} g/s ({workers}w) | avg_max {parts}")
            if done >= next_eval and eval_cb is not None:
                next_eval += eval_every
                eval_cb(done, net)
    finally:
        stop.set()
        for p in procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
    return net
