from __future__ import annotations

import unittest

import sympy as sp

from reference.noether_sympy import PointSymmetry, analyze_point_symmetry


class NoetherReferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.t = sp.symbols("t")
        self.m, self.k = sp.symbols("m k", nonzero=True)

    def test_time_translation_gives_harmonic_oscillator_energy(self) -> None:
        q, v, a = sp.symbols("q v a")
        lagrangian = self.m * v**2 / 2 - self.k * q**2 / 2
        report = analyze_point_symmetry(
            lagrangian,
            self.t,
            [q],
            [v],
            [a],
            PointSymmetry(tau=sp.Integer(1), eta=(sp.Integer(0),)),
        )

        expected_energy = self.m * v**2 / 2 + self.k * q**2 / 2
        self.assertEqual(sp.simplify(report.invariance_residual), 0)
        self.assertEqual(sp.simplify(report.charge + expected_energy), 0)
        self.assertEqual(sp.simplify(report.identity_residual), 0)

    def test_spatial_translation_gives_free_particle_momentum(self) -> None:
        q, v, a = sp.symbols("q v a")
        lagrangian = self.m * v**2 / 2
        report = analyze_point_symmetry(
            lagrangian,
            self.t,
            [q],
            [v],
            [a],
            PointSymmetry(tau=sp.Integer(0), eta=(sp.Integer(1),)),
        )

        self.assertEqual(sp.simplify(report.invariance_residual), 0)
        self.assertEqual(sp.simplify(report.charge - self.m * v), 0)
        self.assertEqual(sp.simplify(report.identity_residual), 0)

    def test_rotational_symmetry_gives_angular_momentum(self) -> None:
        x, y, vx, vy, ax, ay = sp.symbols("x y vx vy ax ay")
        lagrangian = (
            self.m * (vx**2 + vy**2) / 2
            - self.k * (x**2 + y**2) / 2
        )
        report = analyze_point_symmetry(
            lagrangian,
            self.t,
            [x, y],
            [vx, vy],
            [ax, ay],
            PointSymmetry(tau=sp.Integer(0), eta=(-y, x)),
        )

        angular_momentum = self.m * (x * vy - y * vx)
        self.assertEqual(sp.simplify(report.invariance_residual), 0)
        self.assertEqual(sp.simplify(report.charge - angular_momentum), 0)
        self.assertEqual(sp.simplify(report.identity_residual), 0)

    def test_galilean_boost_requires_boundary_term_and_yields_charge(self) -> None:
        q, v, a = sp.symbols("q v a")
        lagrangian = self.m * v**2 / 2
        report = analyze_point_symmetry(
            lagrangian,
            self.t,
            [q],
            [v],
            [a],
            PointSymmetry(
                tau=sp.Integer(0),
                eta=(self.t,),
                boundary=self.m * q,
            ),
        )

        expected = self.m * (self.t * v - q)
        self.assertEqual(sp.simplify(report.invariance_residual), 0)
        self.assertEqual(sp.simplify(report.charge - expected), 0)
        self.assertEqual(sp.simplify(report.identity_residual), 0)

    def test_non_symmetry_has_nonzero_invariance_but_noether_identity_still_closes(self) -> None:
        q, v, a = sp.symbols("q v a")
        lagrangian = self.m * v**2 / 2 - self.k * q**2 / 2
        report = analyze_point_symmetry(
            lagrangian,
            self.t,
            [q],
            [v],
            [a],
            PointSymmetry(tau=sp.Integer(0), eta=(q,)),
        )

        self.assertNotEqual(sp.simplify(report.invariance_residual), 0)
        self.assertEqual(sp.simplify(report.identity_residual), 0)

    def test_generalized_velocity_dependent_generator_is_not_silently_retyped(self) -> None:
        q, v, a = sp.symbols("q v a")
        lagrangian = self.m * v**2 / 2
        with self.assertRaisesRegex(ValueError, "point symmetry"):
            analyze_point_symmetry(
                lagrangian,
                self.t,
                [q],
                [v],
                [a],
                PointSymmetry(tau=sp.Integer(0), eta=(v,)),
            )


if __name__ == "__main__":
    unittest.main()
