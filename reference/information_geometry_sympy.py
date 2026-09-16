"""Exact information-geometry reference for a normal location-scale family.

This module keeps the statistical model explicit. For one observation

    X ~ Normal(mu, sigma^2),  sigma > 0,

it derives the score, expected Fisher information, negative expected Hessian,
and coordinate transformation between `(mu, sigma)` and `(mu, rho=log sigma)`.
No regularization is part of Fisher information itself; damping or Tikhonov
terms belong to a separate numerical policy layer.
"""
from __future__ import annotations

from dataclasses import dataclass

import sympy as sp


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
    """Negative expected Hessian of log p in `(mu,rho)` coordinates.

    The Hessian is expressed after standardizing `X=mu+exp(rho) z`, so the
    expectation is exact through standard-normal moments rather than sampled
    observations.
    """
    z, rho = geometry.z, geometry.rho
    hessian = sp.Matrix(
        [
            [-sp.exp(-2 * rho), -2 * z * sp.exp(-rho)],
            [-2 * z * sp.exp(-rho), -2 * z**2],
        ]
    )
    return -hessian.applyfunc(lambda expr: standard_normal_expectation(expr, z))


def pullback_metric(metric: sp.Matrix, jacobian: sp.Matrix) -> sp.Matrix:
    return (jacobian.T * metric * jacobian).applyfunc(sp.simplify)


def natural_gradient(metric: sp.Matrix, differential: sp.Matrix) -> sp.Matrix:
    """Raise a covector with an exact Fisher metric."""
    if metric.rows != metric.cols:
        raise ValueError("metric must be square")
    if differential.cols != 1 or differential.rows != metric.rows:
        raise ValueError("objective differential dimension must match metric")
    return (metric.inv() * differential).applyfunc(sp.simplify)
