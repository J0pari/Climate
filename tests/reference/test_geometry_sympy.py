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

    def test_nonlinear_flat_map_requires_inhomogeneous_connection_term(self):
        fixture = self.fixtures["flat.quadratic_shear.2d"]
        result = compute_fixture(fixture)
        u, v = result.coordinates

        old_coordinates = sp.Matrix([u, v + sp.Rational(1, 2) * u**2])
        jacobian_old_from_new = old_coordinates.jacobian((u, v))
        transformed_metric = sp.simplify(
            jacobian_old_from_new.T * sp.eye(2) * jacobian_old_from_new
        )
        self.assertTrue(all_components_zero(transformed_metric - result.metric))

        jacobian_new_from_old = sp.simplify(jacobian_old_from_new.inv())
        transformed_connection = [
            [[sp.S.Zero for _ in range(2)] for _ in range(2)]
            for _ in range(2)
        ]
        for upper in range(2):
            for lower_a in range(2):
                for lower_b in range(2):
                    value = sp.S.Zero
                    for old_coordinate in range(2):
                        value += (
                            jacobian_new_from_old[upper, old_coordinate]
                            * sp.diff(
                                old_coordinates[old_coordinate],
                                (u, v)[lower_a],
                                (u, v)[lower_b],
                            )
                        )
                    transformed_connection[upper][lower_a][lower_b] = sp.simplify(value)
                    self.assertEqual(
                        sp.simplify(
                            result.christoffel[upper][lower_a][lower_b] - value
                        ),
                        0,
                    )

        self.assertEqual(transformed_connection[1][0][0], 1)
        # The homogeneous tensor-like part is exactly zero because the source
        # Cartesian connection is zero.  Omitting the inhomogeneous second-
        # derivative term therefore predicts zero and is an exact negative control.
        homogeneous_only_v_uu = sp.S.Zero
        self.assertNotEqual(
            sp.simplify(result.christoffel[1][0][0] - homogeneous_only_v_uu),
            0,
        )
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

    def test_levi_civita_connection_is_metric_compatible(self):
        result = compute_fixture(self.fixtures["sphere.radius2.2d"])
        n = len(result.coordinates)
        for deriv in range(n):
            for i in range(n):
                for j in range(n):
                    covariant_derivative = sp.diff(
                        result.metric[i, j], result.coordinates[deriv]
                    )
                    for contracted in range(n):
                        covariant_derivative -= (
                            result.christoffel[contracted][deriv][i]
                            * result.metric[contracted, j]
                        )
                        covariant_derivative -= (
                            result.christoffel[contracted][deriv][j]
                            * result.metric[i, contracted]
                        )
                    normalized = sp.simplify(sp.expand_trig(covariant_derivative))
                    self.assertEqual(normalized, 0)

    def test_first_bianchi_identity_holds(self):
        result = compute_fixture(self.fixtures["sphere.radius2.2d"])
        n = len(result.coordinates)
        for upper in range(n):
            for j in range(n):
                for k in range(n):
                    for l in range(n):
                        cyclic = (
                            result.riemann[upper][j][k][l]
                            + result.riemann[upper][k][l][j]
                            + result.riemann[upper][l][j][k]
                        )
                        self.assertEqual(sp.simplify(cyclic), 0)


if __name__ == "__main__":
    unittest.main()