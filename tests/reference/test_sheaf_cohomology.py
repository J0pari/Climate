"""Mathematical witnesses for finite-complex cohomology realization."""
from __future__ import annotations

import unittest

from reference.sheaf_cohomology import ConstantCellularSheafGF2, FiniteSimplicialComplex


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

    def test_d_squared_zero_is_checked_in_every_supported_degree(self):
        complex_ = FiniteSimplicialComplex.from_maximal_simplices(
            [("a", "b", "c", "d")]
        )
        for degree in range(complex_.dimension - 1):
            self.assertTrue(complex_.d_squared_is_zero(degree))


if __name__ == "__main__":
    unittest.main()
