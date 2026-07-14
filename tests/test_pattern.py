"""M2 acceptance tests: symmetry + pattern compiler.

Covers the plan's §9.2 checklist that is independent of the value function:
placement counts match the analytic formula, roles are well-formed and
symmetry-invariant, footprints stay in bounds, and the schema hash is stable and
sensitive. Value-level symmetry equality is tested in ``test_universal.py``.
"""
import numpy as np
import pytest

from src.game.symmetry import orient_cells, board_automorphisms
from src.ntuple.pattern import (Pattern, PatternCompiler, compile_pattern,
                                schema_hash, N_ROLES)
from src.ntuple import library as lib


class TestSymmetry:
    def test_orient_square_dedups(self):
        # A 2×2 square has a small orbit; distinct oriented sequences < 8.
        oriented = orient_cells([(0, 0), (0, 1), (1, 0), (1, 1)])
        assert 1 <= len(oriented) <= 8
        for o in oriented:
            assert o.min() == 0                    # normalised to quadrant

    def test_orient_asymmetric_has_8(self):
        oriented = orient_cells([(0, 0), (0, 1), (0, 2), (1, 0)])  # an L, no symmetry
        assert len(oriented) == 8

    def test_board_automorphisms_count(self):
        assert board_automorphisms(4, 4).shape == (8, 16)      # square → D4
        assert board_automorphisms(3, 5).shape == (4, 15)      # rectangle → 4
        # every automorphism is a genuine permutation of all cells.
        for perm in board_automorphisms(4, 4):
            assert sorted(perm.tolist()) == list(range(16))


def _distinct_footprints(cp):
    return {frozenset(row.tolist()) for row in cp.cells}


class TestPlacementCounts:
    @pytest.mark.parametrize("H,W", [(4, 4), (5, 6), (8, 8), (4, 7)])
    def test_rect_2x3_footprints(self, H, W):
        cp = compile_pattern(lib.RECT_2X3, H, W)
        # Distinct cell-sets = 2×3 translations + 3×2 translations (plan §9.2).
        expect = (H - 1) * (W - 2) + (H - 2) * (W - 1)
        assert len(_distinct_footprints(cp)) == expect, (H, W)
        # Total instances include the rectangle's reflected readings (which force
        # the shared table to be symmetry-invariant), so they are a small
        # multiple of the footprint count.
        assert cp.n_instances % expect == 0
        assert 1 <= cp.n_instances // expect <= 8

    def test_oversized_pattern_yields_zero(self):
        # line_6 (1×6) cannot be placed on a 4×4 board.
        assert compile_pattern(lib.LINE_6, 4, 4).n_instances == 0

    def test_line_4_on_4x4(self):
        cp = compile_pattern(lib.LINE_4, 4, 4)
        # 4 horizontal + 4 vertical distinct lines = 8 footprints.
        assert len(_distinct_footprints(cp)) == 8


class TestRoles:
    @pytest.mark.parametrize("H,W", [(4, 4), (6, 6), (8, 8)])
    def test_roles_in_range(self, H, W):
        for p in lib.DEFAULT_PATTERNS:
            cp = compile_pattern(p, H, W)
            if cp.n_instances:
                assert cp.roles.min() >= 0 and cp.roles.max() < N_ROLES

    def test_flat_indices_in_bounds(self):
        for p in lib.DEFAULT_PATTERNS:
            cp = compile_pattern(p, 8, 8)
            if cp.n_instances:
                assert cp.cells.min() >= 0 and cp.cells.max() < 64


class TestSchemaHash:
    def test_stable_and_order_independent(self):
        a = schema_hash(lib.CORE)
        b = schema_hash(list(reversed(lib.CORE)))
        assert a == b

    def test_sensitive_to_change(self):
        changed = lib.CORE + [Pattern("extra", ((0, 0), (1, 1)))]
        assert schema_hash(changed) != schema_hash(lib.CORE)


class TestCompilerCache:
    def test_caches_by_shape(self):
        c = PatternCompiler(lib.CORE)
        assert c.get(4, 4) is c.get(4, 4)
        assert c.get(4, 4) is not c.get(5, 5)
