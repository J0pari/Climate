"""Mathematical witnesses for staged finite sheaf/cohomology realization."""
from __future__ import annotations

import unittest

from reference.sheaf_cohomology import (
    CellularSheafGF2,
    ConstantCellularSheafGF2,
    FiniteCover,
    FiniteSimplicialComplex,
)


class SheafCohomologyReferenceTests(unittest.TestCase):
    def test_interval_is_connected_and_has_no_one_cohomology(self):
        sheaf = ConstantCellularSheafGF2(
            FiniteSimplicialComplex.from_maximal_simplices([("a", "b")])
        )
        sheaf.verify_complex()
        self.assertEqual(sheaf.betti_numbers(), (1, 0))

    def test_two_isolated_points_have_two_components(self):
        sheaf = ConstantCellularSheafGF2(
            FiniteSimplicialComplex.from_maximal_simplices([("a",), ("b",)])
        )
        sheaf.verify_complex()
        self.assertEqual(sheaf.betti_numbers(), (2,))

    def test_triangle_boundary_has_one_independent_one_cycle(self):
        sheaf = ConstantCellularSheafGF2(
            FiniteSimplicialComplex.from_maximal_simplices(
                [("a", "b"), ("b", "c"), ("a", "c")]
            )
        )
        sheaf.verify_complex()
        self.assertEqual(sheaf.betti_numbers(), (1, 1))

    def test_filled_triangle_kills_the_boundary_cycle(self):
        sheaf = ConstantCellularSheafGF2(
            FiniteSimplicialComplex.from_maximal_simplices([("a", "b", "c")])
        )
        sheaf.verify_complex()
        self.assertEqual(sheaf.betti_numbers(), (1, 0, 0))

    def test_tetrahedron_boundary_has_two_dimensional_class(self):
        sheaf = ConstantCellularSheafGF2(
            FiniteSimplicialComplex.from_maximal_simplices(
                [
                    ("a", "b", "c"),
                    ("a", "b", "d"),
                    ("a", "c", "d"),
                    ("b", "c", "d"),
                ]
            )
        )
        sheaf.verify_complex()
        self.assertEqual(sheaf.betti_numbers(), (1, 0, 1))

    def test_declared_finite_cover_builds_exact_nerve(self):
        cover = FiniteCover.from_members(
            {
                "A": {"x", "y"},
                "B": {"y", "z"},
                "C": {"z"},
            }
        )
        nerve = cover.nerve()
        self.assertEqual(
            nerve.faces,
            frozenset(
                {
                    ("A",),
                    ("B",),
                    ("C",),
                    ("A", "B"),
                    ("B", "C"),
                }
            ),
        )
        self.assertEqual(ConstantCellularSheafGF2(nerve).betti_numbers(), (1, 0))

    def test_three_way_intersection_creates_filled_nerve_triangle(self):
        cover = FiniteCover.from_members(
            {"A": {"x"}, "B": {"x"}, "C": {"x"}}
        )
        nerve = cover.nerve()
        self.assertIn(("A", "B", "C"), nerve.faces)
        self.assertEqual(ConstantCellularSheafGF2(nerve).betti_numbers(), (1, 0, 0))

    def test_nonconstant_stalk_dimensions_form_real_block_coboundary(self):
        base = FiniteSimplicialComplex.from_maximal_simplices([("a", "b")])
        sheaf = CellularSheafGF2(
            base=base,
            stalk_dimensions={
                ("a",): 2,
                ("b",): 1,
                ("a", "b"): 1,
            },
            restrictions={
                (("a",), ("a", "b")): [[1, 0]],
                (("b",), ("a", "b")): [[1]],
            },
        )
        self.assertEqual(sheaf.coboundary_matrix(0), [[1, 0, 1]])
        self.assertEqual(sheaf.cohomology_dimensions(), (2, 0))
        self.assertEqual(sheaf.global_section_dimension(), 2)

    def test_restriction_composition_is_a_hard_invariant(self):
        base = FiniteSimplicialComplex.from_maximal_simplices([("a", "b", "c")])
        dimensions = {simplex: 1 for simplex in base.faces}
        restrictions = {
            (face, coface): [[1]]
            for face in base.faces
            for coface in base.faces
            if set(face) < set(coface)
        }
        restrictions[(("a",), ("a", "b", "c"))] = [[0]]
        with self.assertRaisesRegex(ValueError, "restriction composition fails"):
            CellularSheafGF2(base, dimensions, restrictions)

    def test_d_squared_zero_is_checked_for_generic_sheaf(self):
        base = FiniteSimplicialComplex.from_maximal_simplices(
            [("a", "b", "c", "d")]
        )
        sheaf = CellularSheafGF2.constant_rank_one(base)
        sheaf.verify_complex()
        for degree in range(base.dimension - 1):
            self.assertTrue(sheaf.d_squared_is_zero(degree))

    def test_global_section_compatibility_is_kernel_of_d0(self):
        base = FiniteSimplicialComplex.from_maximal_simplices([("a", "b")])
        sheaf = CellularSheafGF2.constant_rank_one(base)

        compatible = {("a",): (1,), ("b",): (1,)}
        incompatible = {("a",): (1,), ("b",): (0,)}

        self.assertEqual(sheaf.local_compatibility_residual(compatible), (0,))
        self.assertTrue(sheaf.is_compatible_local_assignment(compatible))
        self.assertEqual(sheaf.local_compatibility_residual(incompatible), (1,))
        self.assertFalse(sheaf.is_compatible_local_assignment(incompatible))
        self.assertEqual(sheaf.global_section_dimension(), 1)
        self.assertEqual(sheaf.global_section_dimension(), sheaf.cohomology_dimension(0))

    def test_nonconstant_restrictions_change_compatibility_semantics(self):
        base = FiniteSimplicialComplex.from_maximal_simplices([("a", "b")])
        sheaf = CellularSheafGF2(
            base=base,
            stalk_dimensions={
                ("a",): 2,
                ("b",): 1,
                ("a", "b"): 1,
            },
            restrictions={
                (("a",), ("a", "b")): [[1, 0]],
                (("b",), ("a", "b")): [[1]],
            },
        )
        self.assertTrue(
            sheaf.is_compatible_local_assignment({("a",): (1, 1), ("b",): (1,)})
        )
        self.assertFalse(
            sheaf.is_compatible_local_assignment({("a",): (0, 1), ("b",): (1,)})
        )

    def test_partial_or_malformed_local_assignment_fails_closed(self):
        base = FiniteSimplicialComplex.from_maximal_simplices([("a", "b")])
        sheaf = CellularSheafGF2.constant_rank_one(base)
        with self.assertRaisesRegex(ValueError, "surface mismatch"):
            sheaf.local_compatibility_residual({("a",): (1,)})
        with self.assertRaisesRegex(ValueError, "non-GF\(2\)"):
            sheaf.local_compatibility_residual({("a",): (1,), ("b",): (2,)})


if __name__ == "__main__":
    unittest.main()
