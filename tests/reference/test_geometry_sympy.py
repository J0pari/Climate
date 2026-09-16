"""Exact symbolic witnesses for the independent geometry reference."""
from __future__ import annotations

import unittest
from pathlib import Path

import sympy as sp

from reference.geometry_sympy import all_components_zero, compute_fixture, load_fixture_manifest


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "fixtures" / "geometry" / "analytic-fixtures.json"


class GeometryReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixtures = {
            item["fixture_id"]: item
            for item in load_fixture_manifest(MANIFEST)["fixtures"]
        }

    def test_flat_cartesian_has_zero_connection_and_curvature(self):
        result = compute_fixture(self.fixtures["flat.cartesian.2d"])
        self.assertTrue(all_components_zero(result.christoffel))
        self.assertTrue(all_components_zero(result.riemann))
        self.assertTrue(all_components_zero(result.ricci))
        self.assertEqual(sp.simplify(result.scalar_curvature), 0)

    def test_flat_polar_has_nonzero_connection_but_zero_curvature(self):
        result = compute_fixture(self.fixtures["flat.polar.2d"])
        self.assertFalse(all_components_zero(result.christoffel))
        self.assertTrue(all_components_zero(result.riemann))
        self.assertTrue(all_components_zero(result.ricci))
        self.assertEqual(sp.simplify(result.scalar_curvature), 0)

    def test_radius_two_sphere_has_expected_positive_curvature(self):
        result = compute_fixture(self.fixtures["sphere.radius2.2d"])
        self.assertEqual(sp.simplify(result.scalar_curvature), sp.Rational(1, 2))
        residual = (result.ricci - sp.Rational(1, 4) * result.metric).applyfunc(sp.simplify)
        self.assertTrue(all_components_zero(residual))

    def test_unit_poincare_disk_has_expected_negative_curvature(self):
        result = compute_fixture(self.fixtures["hyperbolic.poincare.radius1.2d"])
        self.assertEqual(sp.simplify(result.scalar_curvature), -2)
        residual = (result.ricci + result.metric).applyfunc(sp.simplify)
        self.assertTrue(all_components_zero(residual))

    def test_riemann_is_antisymmetric_in_derivative_indices(self):
        result = compute_fixture(self.fixtures["sphere.radius2.2d"])
        n = len(result.coordinates)
        for upper in range(n):
            for lower in range(n):
                for a in range(n):
                    for b in range(n):
                        residual = sp.simplify(
                            result.riemann[upper][lower][a][b]
                            + result.riemann[upper][lower][b][a]
                        )
                        self.assertEqual(residual, 0)

    def test_levi_civita_connection_is_torsion_free(self):
        result = compute_fixture(self.fixtures["hyperbolic.poincare.radius1.2d"])
        n = len(result.coordinates)
        for upper in range(n):
            for a in range(n):
                for b in range(n):
                    self.assertEqual(
                        sp.simplify(
                            result.christoffel[upper][a][b]
                            - result.christoffel[upper][b][a]
                        ),
                        0,
                    )


if __name__ == "__main__":
    unittest.main()
