from __future__ import annotations

import unittest

import sympy as sp

from reference.information_geometry_sympy import (
    NormalInformationGeometry,
    expected_outer_product,
    expected_vector,
    natural_gradient,
    negative_expected_hessian_mu_rho,
    pullback_metric,
)


class InformationGeometryReferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.geometry = NormalInformationGeometry.symbols()

    def assert_matrix_zero(self, matrix: sp.Matrix) -> None:
        self.assertTrue(all(sp.simplify(value) == 0 for value in matrix))

    def test_score_has_zero_expectation_in_both_coordinate_systems(self) -> None:
        g = self.geometry
        self.assert_matrix_zero(expected_vector(g.score_mu_sigma_standardized(), g.z))
        self.assert_matrix_zero(expected_vector(g.score_mu_rho_standardized(), g.z))

    def test_expected_score_outer_product_is_exact_fisher_metric(self) -> None:
        g = self.geometry
        sigma_fisher = expected_outer_product(g.score_mu_sigma_standardized(), g.z)
        rho_fisher = expected_outer_product(g.score_mu_rho_standardized(), g.z)
        self.assert_matrix_zero(sigma_fisher - g.fisher_mu_sigma())
        self.assert_matrix_zero(rho_fisher - g.fisher_mu_rho())

    def test_negative_expected_hessian_agrees_with_fisher_metric(self) -> None:
        g = self.geometry
        observed_definition = negative_expected_hessian_mu_rho(g)
        self.assert_matrix_zero(observed_definition - g.fisher_mu_rho())

    def test_fisher_metric_is_positive_definite_for_sigma_positive(self) -> None:
        g = self.geometry
        metric = g.fisher_mu_sigma()
        self.assertEqual(sp.simplify(metric[0, 0]), g.sigma**-2)
        self.assertEqual(sp.simplify(metric.det()), 2 * g.sigma**-4)
        self.assertTrue(sp.ask(sp.Q.positive(metric[0, 0])))
        self.assertTrue(sp.ask(sp.Q.positive(metric.det())))

    def test_log_scale_metric_is_pullback_not_a_new_ad_hoc_metric(self) -> None:
        g = self.geometry
        jacobian = g.jacobian_mu_rho_to_mu_sigma()
        sigma_metric_in_rho = g.fisher_mu_sigma().subs(g.sigma, sp.exp(g.rho))
        pulled_back = pullback_metric(sigma_metric_in_rho, jacobian)
        self.assert_matrix_zero(pulled_back - g.fisher_mu_rho())

    def test_natural_gradient_transforms_as_a_parameter_space_vector(self) -> None:
        g = self.geometry
        a, b = sp.symbols("a b", real=True)
        differential_sigma = sp.Matrix([a, b])
        jacobian = g.jacobian_mu_rho_to_mu_sigma()
        differential_rho = jacobian.T * differential_sigma

        sigma_metric = g.fisher_mu_sigma().subs(g.sigma, sp.exp(g.rho))
        natural_sigma = natural_gradient(sigma_metric, differential_sigma)
        natural_rho = natural_gradient(g.fisher_mu_rho(), differential_rho)

        self.assert_matrix_zero(jacobian * natural_rho - natural_sigma)

    def test_metric_inverse_is_cramer_rao_scale_for_n_independent_observations(self) -> None:
        g = self.geometry
        n = sp.symbols("n", positive=True)
        information_n = n * g.fisher_mu_sigma()
        bound = information_n.inv().applyfunc(sp.simplify)
        expected = sp.diag(g.sigma**2 / n, g.sigma**2 / (2 * n))
        self.assert_matrix_zero(bound - expected)


if __name__ == "__main__":
    unittest.main()
