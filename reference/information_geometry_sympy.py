"""Exact information-geometry reference for a normal location-scale family.

This module keeps the statistical model explicit. For one observation

    X ~ Normal(mu, sigma^2),  sigma > 0,

it derives score functions, Fisher information, the negative expected Hessian,
Amari-Chentsov cubic tensor, and Amari alpha-connections. Coordinate changes
between `(mu, sigma)` and `(mu, rho=log sigma)` remain explicit.

No regularization is part of Fisher information or an alpha-connection itself;
damping or Tikhonov terms belong to a separate numerical policy layer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import sympy as sp

Tensor3 = tuple[tuple[tuple[sp.Expr, ...], ...], ...]


@dataclass(frozen=True)
class NormalInformationGeometry:
    mu: sp.Symbol
    sigma: sp.Symbol
    rho: sp.Symbol
    z: sp.Symbol

    @classmethod
    def symbols(cls) -> "NormalInformationGeometry":
        return cls(
            mu=sp.symbols("mu", real=True),
            sigma=sp.symbols("sigma", positive=True),
            rho=sp.symbols("rho", real=True),
            z=sp.symbols("z", real=True),
        )

    def score_mu_sigma_standardized(self) -> sp.Matrix:
        """Score expressed with X = mu + sigma*z and z~N(0,1)."""
        return sp.Matrix([self.z / self.sigma, (self.z**2 - 1) / self.sigma])

    def score_mu_rho_standardized(self) -> sp.Matrix:
        """Score in coordinates `(mu, rho=log sigma)`."""
        return sp.Matrix([self.z * sp.exp(-self.rho), self.z**2 - 1])

    def hessian_mu_rho_standardized(self) -> sp.Matrix:
        """Parameter Hessian of log p at fixed observation, then standardized.

        `z=(X-mu)/exp(rho)` is substituted only after differentiating with
        respect to the parameters at fixed X. Treating z as parameter-independent
        during differentiation would give the wrong statistical connection.
        """
        z, rho = self.z, self.rho
        return sp.Matrix(
            [
                [-sp.exp(-2 * rho), -2 * z * sp.exp(-rho)],
                [-2 * z * sp.exp(-rho), -2 * z**2],
            ]
        )

    def fisher_mu_sigma(self) -> sp.Matrix:
        return sp.diag(self.sigma**-2, 2 * self.sigma**-2)

    def fisher_mu_rho(self) -> sp.Matrix:
        return sp.diag(sp.exp(-2 * self.rho), sp.Integer(2))

    def jacobian_mu_rho_to_mu_sigma(self) -> sp.Matrix:
        """Jacobian d(mu,sigma)/d(mu,rho), with sigma=exp(rho)."""
        return sp.diag(sp.Integer(1), sp.exp(self.rho))


def standard_normal_expectation(expr: sp.Expr, z: sp.Symbol) -> sp.Expr:
    """Exact expectation of a polynomial under Z~N(0,1)."""
    polynomial = sp.Poly(sp.expand(expr), z)
    result = sp.Integer(0)
    for (power,), coefficient in polynomial.terms():
        if power % 2 == 1:
            moment = sp.Integer(0)
        elif power == 0:
            moment = sp.Integer(1)
        else:
            moment = sp.factorial2(power - 1)
        result += coefficient * moment
    return sp.simplify(result)


def expected_vector(vector: sp.Matrix, z: sp.Symbol) -> sp.Matrix:
    return vector.applyfunc(lambda expr: standard_normal_expectation(expr, z))


def expected_outer_product(score: sp.Matrix, z: sp.Symbol) -> sp.Matrix:
    return sp.Matrix(
        score.rows,
        score.rows,
        lambda i, j: standard_normal_expectation(score[i] * score[j], z),
    )


def negative_expected_hessian_mu_rho(geometry: NormalInformationGeometry) -> sp.Matrix:
    """Negative expected Hessian of log p in `(mu,rho)` coordinates."""
    return -geometry.hessian_mu_rho_standardized().applyfunc(
        lambda expr: standard_normal_expectation(expr, geometry.z)
    )


def _tensor3(dimension: int, fn: Callable[[int, int, int], sp.Expr]) -> Tensor3:
    return tuple(
        tuple(
            tuple(sp.simplify(fn(i, j, k)) for k in range(dimension))
            for j in range(dimension)
        )
        for i in range(dimension)
    )


def amari_chentsov_mu_rho(geometry: NormalInformationGeometry) -> Tensor3:
    """Amari-Chentsov cubic tensor T_ijk = E[s_i s_j s_k]."""
    score = geometry.score_mu_rho_standardized()
    z = geometry.z
    return _tensor3(
        score.rows,
        lambda i, j, k: standard_normal_expectation(score[i] * score[j] * score[k], z),
    )


def alpha_connection_lower_mu_rho(
    geometry: NormalInformationGeometry,
    alpha: sp.Expr,
) -> Tensor3:
    """Lowered Amari alpha-connection coefficients Γ^(α)_{ij,k}.

    Convention:

        Γ^(α)_{ij,k}
          = E[(∂_i∂_j l + (1-α)/2 ∂_i l ∂_j l) ∂_k l]

    for log likelihood `l`. The first two indices are therefore symmetric for
    this torsion-free statistical connection.
    """
    score = geometry.score_mu_rho_standardized()
    hessian = geometry.hessian_mu_rho_standardized()
    z = geometry.z
    factor = (1 - alpha) / 2
    return _tensor3(
        score.rows,
        lambda i, j, k: standard_normal_expectation(
            (hessian[i, j] + factor * score[i] * score[j]) * score[k],
            z,
        ),
    )


def levi_civita_lower(metric: sp.Matrix, coordinates: tuple[sp.Symbol, ...]) -> Tensor3:
    """Lowered Levi-Civita coefficients Γ_{ij,k} from a metric."""
    if metric.rows != metric.cols or metric.rows != len(coordinates):
        raise ValueError("metric dimension must match coordinate dimension")
    n = metric.rows
    return _tensor3(
        n,
        lambda i, j, k: sp.Rational(1, 2)
        * (
            sp.diff(metric[j, k], coordinates[i])
            + sp.diff(metric[i, k], coordinates[j])
            - sp.diff(metric[i, j], coordinates[k])
        ),
    )


def pullback_metric(metric: sp.Matrix, jacobian: sp.Matrix) -> sp.Matrix:
    return (jacobian.T * metric * jacobian).applyfunc(sp.simplify)


def natural_gradient(metric: sp.Matrix, differential: sp.Matrix) -> sp.Matrix:
    """Raise a covector with an exact Fisher metric."""
    if metric.rows != metric.cols:
        raise ValueError("metric must be square")
    if differential.cols != 1 or differential.rows != metric.rows:
        raise ValueError("objective differential dimension must match metric")
    return (metric.inv() * differential).applyfunc(sp.simplify)
