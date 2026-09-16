from __future__ import annotations

import unittest

import sympy as sp

from reference.information_geometry_sympy import (
    NormalInformationGeometry,
    alpha_connection_lower_mu_rho,
    amari_chentsov_mu_rho,
    expected_outer_product,
    expected_vector,
    levi_civita_lower,
    natural_gradient,
    negative_expected_hessian_mu_rho,
    pullback_metric,
)


class InformationGeometryReferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.geometry = NormalInformationGeometry.symbols()

    def assert_matrix_zero(self, matrix: sp.Matrix) -> None:
        self.assertTrue(all(sp.simplify(value) == 0 for value in matrix))

    def assert_tensor3_zero(self, tensor) -> None:
        self.assertTrue(
            all(
                sp.simplify(value) == 0
                for plane in tensor
                for row in plane
                for value in row
            )
        )

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

    def test_alpha_connection_known_normal_components(self) -> None:
        g = self.geometry
        alpha = sp.symbols("alpha", real=True)
        gamma = alpha_connection_lower_mu_rho(g, alpha)
        scale = sp.exp(-2 * g.rho)

        expected = (
            (
                (sp.Integer(0), (1 - alpha) * scale),
                (-(alpha + 1) * scale, sp.Integer(0)),
            ),
            (
                (-(alpha + 1) * scale, sp.Integer(0)),
                (sp.Integer(0), -4 * alpha),
            ),
        )
        residual = tuple(
            tuple(
                tuple(sp.simplify(gamma[i][j][k] - expected[i][j][k]) for k in range(2))
                for j in range(2)
            )
            for i in range(2)
        )
        self.assert_tensor3_zero(residual)

    def test_alpha_zero_is_fisher_levi_civita_connection(self) -> None:
        g = self.geometry
        alpha_zero = alpha_connection_lower_mu_rho(g, sp.Integer(0))
        levi_civita = levi_civita_lower(g.fisher_mu_rho(), (g.mu, g.rho))
        residual = tuple(
            tuple(
                tuple(sp.simplify(alpha_zero[i][j][k] - levi_civita[i][j][k]) for k in range(2))
                for j in range(2)
            )
            for i in range(2)
        )
        self.assert_tensor3_zero(residual)

    def test_alpha_connection_is_torsion_free(self) -> None:
        g = self.geometry
        alpha = sp.symbols("alpha", real=True)
        gamma = alpha_connection_lower_mu_rho(g, alpha)
        residual = tuple(
            tuple(
                tuple(sp.simplify(gamma[i][j][k] - gamma[j][i][k]) for k in range(2))
                for j in range(2)
            )
            for i in range(2)
        )
        self.assert_tensor3_zero(residual)

    def test_dual_alpha_connections_reconstruct_metric_derivative(self) -> None:
        g = self.geometry
        alpha = sp.symbols("alpha", real=True)
        positive = alpha_connection_lower_mu_rho(g, alpha)
        negative = alpha_connection_lower_mu_rho(g, -alpha)
        metric = g.fisher_mu_rho()
        coordinates = (g.mu, g.rho)

        residual = tuple(
            tuple(
                tuple(
                    sp.simplify(
                        sp.diff(metric[j, k], coordinates[i])
                        - positive[i][j][k]
                        - negative[i][k][j]
                    )
                    for k in range(2)
                )
                for j in range(2)
            )
            for i in range(2)
        )
        self.assert_tensor3_zero(residual)

    def test_alpha_difference_is_controlled_by_amari_chentsov_tensor(self) -> None:
        g = self.geometry
        alpha = sp.symbols("alpha", real=True)
        gamma = alpha_connection_lower_mu_rho(g, alpha)
        gamma_zero = alpha_connection_lower_mu_rho(g, sp.Integer(0))
        cubic = amari_chentsov_mu_rho(g)

        residual = tuple(
            tuple(
                tuple(
                    sp.simplify(
                        gamma[i][j][k]
                        - gamma_zero[i][j][k]
                        + sp.Rational(1, 2) * alpha * cubic[i][j][k]
                    )
                    for k in range(2)
                )
                for j in range(2)
            )
            for i in range(2)
        )
        self.assert_tensor3_zero(residual)


if __name__ == "__main__":
    unittest.main()
