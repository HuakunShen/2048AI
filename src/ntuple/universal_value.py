"""Universal N-Tuple value function (Milestone M3).

One model, every shape. The value of a board of *any* ``H×W`` is

    V(B) = Σ_k  g_k · u_k(B),     u_k(B) = (1 / |Π_k|) · Σ_{p ∈ Π_k} T_k(x_{k,p})

where ``Π_k`` is the set of placements of pattern ``k`` on this board (compiled by
:class:`~src.ntuple.pattern.PatternCompiler`) and ``T_k`` is a dense lookup table
**shared across all board shapes**. Two properties fall out of the construction:

* **Symmetry invariance** — placements include every orientation of the pattern,
  so ``V(B) = V(gB)`` for every board symmetry ``g`` (verified in tests).
* **Scale stability across sizes** — the ``1/|Π_k|`` placement mean keeps each
  pattern's contribution O(1) whether the board has 6 placements (4×4) or 42
  (8×8), so a large board's value doesn't inflate just from having more
  placements (plan §2.3, §5.1). A single update moves ``u_k`` by ``α·δ``
  independent of board size.

This first cut keeps ``g_k = 1`` and adds no residual/stage/calibrator head — the
shared base whose 4×4 parity and cross-size learning are the plan's Go/No-Go gate
(§10.2). The role ids the compiler produces are already threaded through so the
M4 residual head can be layered on without touching the hot path.

Tables live in one flat ``float32`` array with per-pattern offsets so the numba
kernels index without Python-object churn; TC accumulators ``E``/``A`` mirror it.
"""
from __future__ import annotations

from typing import List

import numpy as np
from numba import njit

from src.ntuple.pattern import PatternCompiler, N_ROLES
from src.ntuple import library as lib

MAX_EXPONENT = 15               # default clip: tiles to 2**15 for the 16-symbol alphabet
RESIDUAL_MAX_LEN = 4            # only small (L<=4) patterns get a per-value residual


@njit(cache=True)
def _pattern_value(exp, cells, pow_, lut, offset):
    """Mean of ``lut`` over one pattern's placements. ``exp`` is the flat board."""
    n, L = cells.shape
    s = 0.0
    for i in range(n):
        idx = 0
        for j in range(L):
            idx += exp[cells[i, j]] * pow_[j]
        s += lut[offset + idx]
    return s / n


@njit(cache=True)
def _pattern_update(exp, cells, pow_, lut, offset, step):
    """Add ``step`` to every active entry of one pattern (plain TD)."""
    n, L = cells.shape
    for i in range(n):
        idx = 0
        for j in range(L):
            idx += exp[cells[i, j]] * pow_[j]
        lut[offset + idx] += step


@njit(cache=True)
def _pattern_update_tc(exp, cells, pow_, lut, acc_e, acc_a, offset, alpha, delta):
    """Temporal-coherence update for one pattern.

    Per-entry adaptive step ``α·|E|/A·δ/|Π_k|``. The coherence ratio is computed
    from raw ``E``/``A`` (accumulating the shared TD error ``δ``), so it matches
    the classic per-weight TC; only the applied step carries the ``1/n`` placement
    normalisation (plan §5.1).
    """
    n, L = cells.shape
    step_scale = delta / n
    for i in range(n):
        idx = 0
        for j in range(L):
            idx += exp[cells[i, j]] * pow_[j]
        e = offset + idx
        a = acc_a[e]
        lr = (abs(acc_e[e]) / a) if a > 0.0 else 1.0
        lut[e] += alpha * lr * step_scale
        acc_e[e] += delta
        acc_a[e] += abs(delta)


@njit(cache=True)
def _pattern_value_res(exp, cells, pow_, roles, lut, base_off,
                       res, res_off, table_size, rho):
    """Placement mean of ``T_k(x) + rho·R_k(role, x)`` for a residual pattern."""
    n, L = cells.shape
    s = 0.0
    for i in range(n):
        idx = 0
        for j in range(L):
            idx += exp[cells[i, j]] * pow_[j]
        s += lut[base_off + idx] + rho * res[res_off + roles[i] * table_size + idx]
    return s / n


@njit(cache=True)
def _pattern_update_tc_res(exp, cells, pow_, roles, lut, acc_e, acc_a, base_off,
                           res, res_e, res_a, res_off, table_size,
                           alpha, alpha_res, rho, delta):
    """TC update of both the shared base table and the per-role residual table.

    The base moves with ``alpha`` as usual; the residual moves with ``alpha_res``
    (typically smaller, plan §5.2) scaled by ``rho`` so its effect on ``V``
    matches its coefficient in the value. Each keeps its own ``E``/``A``.
    """
    n, L = cells.shape
    step = delta / n
    for i in range(n):
        idx = 0
        for j in range(L):
            idx += exp[cells[i, j]] * pow_[j]
        b = base_off + idx
        a = acc_a[b]
        lr = (abs(acc_e[b]) / a) if a > 0.0 else 1.0
        lut[b] += alpha * lr * step
        acc_e[b] += delta
        acc_a[b] += abs(delta)

        e = res_off + roles[i] * table_size + idx
        ra = res_a[e]
        rlr = (abs(res_e[e]) / ra) if ra > 0.0 else 1.0
        res[e] += alpha_res * rlr * rho * step
        res_e[e] += delta
        res_a[e] += abs(delta)


@njit(cache=True)
def _pattern_value_ms(exp, cells, pow_, lut, acc_a, offset, stage):
    """Multi-stage placement mean with lazy weight promotion (plan §7.1).

    ``lut``/``acc_a`` are 2D ``[n_stages, total]``. An entry that has never been
    visited at ``stage`` (``acc_a[stage,e] == 0``) reads the deepest *visited*
    lower stage instead — so a high stage inherits the lower stage's learned
    value until it accumulates its own evidence.
    """
    n, L = cells.shape
    s = 0.0
    for i in range(n):
        idx = 0
        for j in range(L):
            idx += exp[cells[i, j]] * pow_[j]
        e = offset + idx
        st = stage
        while st > 0 and acc_a[st, e] == 0.0:
            st -= 1
        s += lut[st, e]
    return s / n


@njit(cache=True)
def _pattern_update_tc_ms(exp, cells, pow_, lut, acc_e, acc_a, offset, stage,
                          alpha, delta):
    """Multi-stage TC update. First visit to a stage-``s`` entry seeds it from the
    deepest visited lower stage (weight promotion), then it updates independently.
    """
    n, L = cells.shape
    step = delta / n
    for i in range(n):
        idx = 0
        for j in range(L):
            idx += exp[cells[i, j]] * pow_[j]
        e = offset + idx
        if stage > 0 and acc_a[stage, e] == 0.0:
            st = stage - 1
            while st > 0 and acc_a[st, e] == 0.0:
                st -= 1
            lut[stage, e] = lut[st, e]              # promote from lower stage
        a = acc_a[stage, e]
        lr = (abs(acc_e[stage, e]) / a) if a > 0.0 else 1.0
        lut[stage, e] += alpha * lr * step
        acc_e[stage, e] += delta
        acc_a[stage, e] += abs(delta)


def _flat_exp(board: np.ndarray, max_exp: int = MAX_EXPONENT) -> np.ndarray:
    """Flat int64 exponents clipped to ``max_exp`` for LUT indexing."""
    e = np.ascontiguousarray(board, dtype=np.int64).reshape(-1)
    return np.minimum(e, max_exp)


class UniversalNTuple:
    """Shape-agnostic n-tuple value function with shared tables + TC learning."""

    def __init__(self, patterns: List = None, alpha: float = 0.5, tc: bool = True,
                 residual: bool = False, rho: float = 0.25,
                 alpha_residual: float = 0.1, stages=None):
        if residual and not tc:
            raise ValueError("residual head currently requires tc=True")
        # Multi-stage weight promotion (plan §7.1): separate table sets per game
        # stage, split by max-tile exponent thresholds. Stage s inherits stage
        # s-1's weights on first visit, then updates independently.
        self.stage_thresholds = tuple(sorted(int(s) for s in stages)) if stages else ()
        self.n_stages = len(self.stage_thresholds) + 1
        if self.n_stages > 1 and not tc:
            raise ValueError("multi-stage requires tc=True")
        if self.n_stages > 1 and residual:
            raise ValueError("multi-stage + residual not supported together")
        self.patterns = list(patterns if patterns is not None else lib.DEFAULT_PATTERNS)
        self.compiler = PatternCompiler(self.patterns)
        self.alpha = alpha
        self.tc = tc
        self.residual = residual
        self.rho = rho
        self.alpha_residual = alpha_residual
        self.hash = self.compiler.hash
        # Tile alphabet (all patterns share one). alphabet=16 -> tiles up to 2^15
        # (32768); alphabet=18 -> up to 2^17 (131072), needed for the 4x4 endgame.
        self.alphabet = self.patterns[0].alphabet
        self.max_exponent = self.alphabet - 1
        sizes = np.array([p.table_size for p in self.patterns], dtype=np.int64)
        self.offsets = np.concatenate(([0], np.cumsum(sizes)[:-1])).astype(np.int64)
        self.total = int(sizes.sum())
        shape = (self.n_stages, self.total) if self.n_stages > 1 else (self.total,)
        self.LUT = np.zeros(shape, dtype=np.float32)
        if tc:
            self.E = np.zeros(shape, dtype=np.float32)
            self.A = np.zeros(shape, dtype=np.float32)
        else:
            self.E = self.A = None

        # M4 conditional residual: per-role tables for small (L<=4) patterns only,
        # laid out flat as [role*table_size + idx]. Large tables get no per-value
        # residual (plan §3.3 / §10.1: residual stays small).
        self.res_offsets = np.full(len(self.patterns), -1, dtype=np.int64)
        running = 0
        if residual:
            for k, p in enumerate(self.patterns):
                if p.length <= RESIDUAL_MAX_LEN:
                    self.res_offsets[k] = running
                    running += N_ROLES * p.table_size
        self.res_total = running
        self.R = np.zeros(running, dtype=np.float32)
        if residual and tc:
            self.RE = np.zeros(running, dtype=np.float32)
            self.RA = np.zeros(running, dtype=np.float32)
        else:
            self.RE = self.RA = None

    # -- evaluation -----------------------------------------------------
    def _pattern_v(self, exp, cp, k) -> float:
        ro = self.res_offsets[k]
        if self.residual and ro >= 0:
            return _pattern_value_res(exp, cp.cells, cp.pow, cp.roles, self.LUT,
                                      self.offsets[k], self.R, ro,
                                      self.patterns[k].table_size, self.rho)
        return _pattern_value(exp, cp.cells, cp.pow, self.LUT, self.offsets[k])

    def _stage(self, board: np.ndarray) -> int:
        """Game stage from the board's max-tile exponent (0 if no staging)."""
        if not self.stage_thresholds:
            return 0
        me = min(int(board.max()), self.max_exponent) if board.size else 0
        st = 0
        for thr in self.stage_thresholds:
            if me >= thr:
                st += 1
        return st

    def value(self, board: np.ndarray) -> float:
        exp = _flat_exp(board, self.max_exponent)
        cs = self.compiler.get(*board.shape)
        total = 0.0
        if self.n_stages > 1:
            stage = self._stage(board)
            for k, cp in enumerate(cs.compiled):
                if cp.n_instances:
                    total += _pattern_value_ms(exp, cp.cells, cp.pow, self.LUT,
                                               self.A, self.offsets[k], stage)
            return float(total)
        for k, cp in enumerate(cs.compiled):
            if cp.n_instances:
                total += self._pattern_v(exp, cp, k)
        return float(total)

    def update(self, board: np.ndarray, target: float) -> float:
        """Afterstate TD update of ``V(board)`` toward ``target``; returns old V."""
        exp = _flat_exp(board, self.max_exponent)
        cs = self.compiler.get(*board.shape)
        if self.n_stages > 1:
            stage = self._stage(board)
            v = 0.0
            for k, cp in enumerate(cs.compiled):
                if cp.n_instances:
                    v += _pattern_value_ms(exp, cp.cells, cp.pow, self.LUT,
                                           self.A, self.offsets[k], stage)
            delta = target - v
            for k, cp in enumerate(cs.compiled):
                if cp.n_instances:
                    _pattern_update_tc_ms(exp, cp.cells, cp.pow, self.LUT, self.E,
                                          self.A, self.offsets[k], stage,
                                          self.alpha, delta)
            return float(v)
        v = 0.0
        for k, cp in enumerate(cs.compiled):
            if cp.n_instances:
                v += self._pattern_v(exp, cp, k)
        delta = target - v
        for k, cp in enumerate(cs.compiled):
            if not cp.n_instances:
                continue
            ro = self.res_offsets[k]
            if self.residual and ro >= 0 and self.tc:
                _pattern_update_tc_res(
                    exp, cp.cells, cp.pow, cp.roles, self.LUT, self.E, self.A,
                    self.offsets[k], self.R, self.RE, self.RA, ro,
                    self.patterns[k].table_size, self.alpha, self.alpha_residual,
                    self.rho, delta)
            elif self.tc:
                _pattern_update_tc(exp, cp.cells, cp.pow, self.LUT, self.E, self.A,
                                   self.offsets[k], self.alpha, delta)
            else:
                _pattern_update(exp, cp.cells, cp.pow, self.LUT,
                                self.offsets[k], self.alpha * delta / cp.n_instances)
        return float(v)

    # -- persistence ----------------------------------------------------
    def save(self, path: str):
        schema = [{"id": p.id, "cells": list(p.cells), "alphabet": p.alphabet}
                  for p in self.patterns]
        # Multi-stage value() needs the visited markers (A) to do fallback reads,
        # so persist A when staged (compresses well — unvisited entries are 0).
        A_save = self.A if self.n_stages > 1 else np.zeros(0, dtype=np.float32)
        np.savez_compressed(
            path, LUT=self.LUT, offsets=self.offsets, alpha=self.alpha,
            tc=self.tc, hash=self.hash, residual=self.residual, rho=self.rho,
            R=self.R, res_offsets=self.res_offsets,
            stages=np.array(self.stage_thresholds, dtype=np.int64), A=A_save,
            schema=np.frombuffer(repr(schema).encode(), dtype=np.uint8))

    def load(self, path: str):
        data = np.load(path, allow_pickle=False)
        loaded_hash = str(data["hash"])
        if loaded_hash != self.hash:
            raise ValueError(
                f"pattern schema mismatch: checkpoint {loaded_hash} != model "
                f"{self.hash}; refusing to load (plan §9.2 fail-fast).")
        saved_stages = tuple(data["stages"].tolist()) if "stages" in data else ()
        if saved_stages != self.stage_thresholds:
            raise ValueError(
                f"stage config mismatch: checkpoint {saved_stages} != model "
                f"{self.stage_thresholds}; refusing to load.")
        self.LUT = data["LUT"].astype(np.float32)
        self.offsets = data["offsets"].astype(np.int64)
        self.alpha = float(data["alpha"])
        if self.n_stages > 1 and "A" in data and data["A"].size:
            self.A = data["A"].astype(np.float32)
        if "R" in data and self.residual:
            self.R = data["R"].astype(np.float32)
            self.res_offsets = data["res_offsets"].astype(np.int64)
