"""Pattern schema + compiler (Milestone M2).

A :class:`Pattern` is defined once in **relative coordinates** with ``(0, 0)`` as
its local origin. The compiler turns a set of patterns plus a concrete board
shape ``(H, W)`` into flat, allocation-free arrays that the universal value
function iterates on the hot path:

  * **placements** — every orientation (D4 of the pattern shape) at every legal
    translation on the board, as flat cell indices ``[n_inst, L]``. All
    placements of one pattern share a single lookup table, which is what makes
    ``V`` invariant to board symmetries and lets one model serve every shape.
  * **roles** — a symmetry-invariant position role per placement
    (corner / edge / near-edge / interior), so a later residual head can recover
    the corner/edge semantics that pure translation sharing throws away
    (plan §4.3). Computed and stored now; consumed by the M4 residual.
  * **schema hash** — a fingerprint of the pattern definitions so a checkpoint
    refuses to load against a changed pattern set (plan §9.2, §5.5).

The compiler caches per ``(pattern_set, H, W)`` so value evaluation never builds
Python objects (plan §4.4 "implementation requirement").
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from src.game.symmetry import orient_cells

DEFAULT_ALPHABET = 16           # empty + tiles 2..32768 (exponents 0..15)

# Position roles (symmetry-invariant); index into the residual head.
ROLE_CORNER, ROLE_EDGE, ROLE_NEAR_EDGE, ROLE_INTERIOR = 0, 1, 2, 3
N_ROLES = 4


@dataclass(frozen=True)
class Pattern:
    """A local n-tuple shape in relative coordinates."""
    id: str
    cells: Tuple[Tuple[int, int], ...]
    alphabet: int = DEFAULT_ALPHABET
    storage: str = "dense"

    @property
    def length(self) -> int:
        return len(self.cells)

    @property
    def table_size(self) -> int:
        return self.alphabet ** self.length


@dataclass
class CompiledPattern:
    """One pattern compiled for a specific board shape."""
    pattern: Pattern
    cells: np.ndarray               # [n_inst, L] flat board indices (int32)
    roles: np.ndarray               # [n_inst] int8 position role
    pow: np.ndarray                 # [L] int64 mixed-radix weights
    table_size: int

    @property
    def n_instances(self) -> int:
        return self.cells.shape[0]


def _role(min_r, max_r, min_c, max_c, H, W) -> int:
    """Symmetry-invariant role from the placement's distance to each boundary."""
    dh = min(min_r, H - 1 - max_r)      # rows to nearest horizontal edge
    dv = min(min_c, W - 1 - max_c)      # cols to nearest vertical edge
    if dh == 0 and dv == 0:
        return ROLE_CORNER
    if dh == 0 or dv == 0:
        return ROLE_EDGE
    if min(dh, dv) == 1:
        return ROLE_NEAR_EDGE
    return ROLE_INTERIOR


def compile_pattern(pattern: Pattern, H: int, W: int) -> CompiledPattern:
    """Compile one pattern for an ``H×W`` board."""
    L = pattern.length
    pow_ = (pattern.alphabet ** np.arange(L)).astype(np.int64)
    seen = set()
    cells_rows: List[List[int]] = []
    roles: List[int] = []
    for oriented in orient_cells(pattern.cells):        # [L, 2] each
        ph = int(oriented[:, 0].max()) + 1
        pw = int(oriented[:, 1].max()) + 1
        if ph > H or pw > W:
            continue                                    # orientation doesn't fit
        for dr in range(H - ph + 1):
            for dc in range(W - pw + 1):
                abs_rc = oriented + (dr, dc)
                flat = (abs_rc[:, 0] * W + abs_rc[:, 1]).astype(np.int32)
                key = tuple(flat.tolist())
                if key in seen:
                    continue
                seen.add(key)
                cells_rows.append(flat.tolist())
                roles.append(_role(int(abs_rc[:, 0].min()), int(abs_rc[:, 0].max()),
                                   int(abs_rc[:, 1].min()), int(abs_rc[:, 1].max()),
                                   H, W))
    cells = np.array(cells_rows, dtype=np.int32) if cells_rows \
        else np.zeros((0, L), dtype=np.int32)
    return CompiledPattern(pattern=pattern, cells=cells,
                           roles=np.array(roles, dtype=np.int8),
                           pow=pow_, table_size=pattern.table_size)


def schema_hash(patterns: List[Pattern]) -> str:
    """Stable fingerprint of a pattern set (order-independent per pattern id)."""
    parts = []
    for p in sorted(patterns, key=lambda q: q.id):
        parts.append(f"{p.id}:{p.alphabet}:{sorted(p.cells)}")
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


@dataclass
class CompiledPatternSet:
    """All patterns compiled for one shape, plus a per-shape cache."""
    patterns: List[Pattern]
    H: int
    W: int
    compiled: List[CompiledPattern]
    hash: str = field(default="")

    @classmethod
    def build(cls, patterns: List[Pattern], H: int, W: int) -> "CompiledPatternSet":
        return cls(patterns=list(patterns), H=H, W=W,
                   compiled=[compile_pattern(p, H, W) for p in patterns],
                   hash=schema_hash(patterns))


class PatternCompiler:
    """Lazily compiles + caches a pattern set for every board shape it sees."""

    def __init__(self, patterns: List[Pattern]):
        self.patterns = list(patterns)
        self.hash = schema_hash(self.patterns)
        self._cache: dict = {}

    def get(self, H: int, W: int) -> CompiledPatternSet:
        key = (H, W)
        if key not in self._cache:
            self._cache[key] = CompiledPatternSet.build(self.patterns, H, W)
        return self._cache[key]
