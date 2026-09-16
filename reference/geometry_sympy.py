"""Symbolic reference geometry for Climate verification fixtures.

This module is intentionally small and readable. It is an independent oracle
for the accelerated geometry path, not a production-scale implementation.

Convention:
    R^i_{jkl} = d_k Gamma^i_{lj} - d_l Gamma^i_{kj}
                 + Gamma^i_{km} Gamma^m_{lj}
                 - Gamma^i_{lm} Gamma^m_{kj}
    Ric_{jl} = R^i_{jil}
    R = g^{jl} Ric_{jl}

No climate interpretation is produced here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import sympy as sp


@dataclass(frozen=True)
class GeometryResult:
    coordinates: tuple[sp.Symbol, ...]
    metric: sp.Matrix
    inverse_metric: sp.Matrix
    christoffel: tuple[tuple[tuple[sp.Expr, ...], ...], ...]
    riemann: tuple[
        tuple[tuple[tuple[sp.Expr, ...], ...], ...], ...
    ]
    ricci: sp.Matrix
    scalar_curvature: sp.Expr


def _simplify(value: sp.Expr) -> sp.Expr:
    return sp.simplify(sp.trigsimp(sp.factor(value)))


def levi_civita_geometry(
    coordinates: Sequence[sp.Symbol],
    metric: sp.Matrix,
) -> GeometryResult:
    """Compute Levi-Civita connection and curvature symbolically.

    The implementation favors transparent index loops over clever tensor APIs so
    it can serve as an independent differential witness for optimized code.
    """
    coords = tuple(coordinates)
    n = len(coords)
    if metric.shape != (n, n):
        raise ValueError(f"metric shape {metric.shape} does not match {n} coordinates")
    if metric != metric.T:
        raise ValueError("metric must be symmetric")

    determinant = _simplify(metric.det())
    if determinant == 0:
        raise ValueError("metric is singular")

    inverse = metric.inv().applyfunc(_simplify)

    gamma = [
        [[sp.S.Zero for _ in range(n)] for _ in range(n)]
        for _ in range(n)
    ]
    half = sp.Rational(1, 2)
    for upper in range(n):
        for lower_a in range(n):
            for lower_b in range(n):
                value = sp.S.Zero
                for contracted in range(n):
                    value += inverse[upper, contracted] * (
                        sp.diff(metric[contracted, lower_b], coords[lower_a])
                        + sp.diff(metric[contracted, lower_a], coords[lower_b])
                        - sp.diff(metric[lower_a, lower_b], coords[contracted])
                    )
                gamma[upper][lower_a][lower_b] = _simplify(half * value)

    riemann = [
        [
            [[sp.S.Zero for _ in range(n)] for _ in range(n)]
            for _ in range(n)
        ]
        for _ in range(n)
    ]
    for upper in range(n):
        for lower in range(n):
            for deriv_a in range(n):
                for deriv_b in range(n):
                    value = (
                        sp.diff(gamma[upper][deriv_b][lower], coords[deriv_a])
                        - sp.diff(gamma[upper][deriv_a][lower], coords[deriv_b])
                    )
                    for contracted in range(n):
                        value += (
                            gamma[upper][deriv_a][contracted]
                            * gamma[contracted][deriv_b][lower]
                            - gamma[upper][deriv_b][contracted]
                            * gamma[contracted][deriv_a][lower]
                        )
                    riemann[upper][lower][deriv_a][deriv_b] = _simplify(value)

    ricci = sp.MutableDenseMatrix.zeros(n, n)
    for lower_a in range(n):
        for lower_b in range(n):
            ricci[lower_a, lower_b] = _simplify(
                sum(
                    riemann[contracted][lower_a][contracted][lower_b]
                    for contracted in range(n)
                )
            )
    ricci = sp.Matrix(ricci)

    scalar = sp.S.Zero
    for i in range(n):
        for j in range(n):
            scalar += inverse[i, j] * ricci[i, j]
    scalar = _simplify(scalar)

    return GeometryResult(
        coordinates=coords,
        metric=metric,
        inverse_metric=inverse,
        christoffel=tuple(
            tuple(tuple(component for component in row) for row in plane)
            for plane in gamma
        ),
        riemann=tuple(
            tuple(
                tuple(tuple(component for component in row) for row in plane)
                for plane in cube
            )
            for cube in riemann
        ),
        ricci=ricci,
        scalar_curvature=scalar,
    )


def parse_metric_fixture(fixture: dict) -> tuple[tuple[sp.Symbol, ...], sp.Matrix]:
    """Parse one repository analytic fixture into SymPy coordinates/metric."""
    names = fixture["coordinates"]
    coordinates = tuple(sp.Symbol(name, real=True) for name in names)
    local_dict = {symbol.name: symbol for symbol in coordinates}
    local_dict.update({
        "sin": sp.sin,
        "cos": sp.cos,
        "exp": sp.exp,
        "sqrt": sp.sqrt,
        "log": sp.log,
    })

    def parse(expression: str) -> sp.Expr:
        return sp.sympify(expression.replace("^", "**"), locals=local_dict)

    metric = sp.Matrix([
        [parse(expression) for expression in row]
        for row in fixture["metric"]
    ])
    return coordinates, metric


def load_fixture_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def compute_fixture(fixture: dict) -> GeometryResult:
    coordinates, metric = parse_metric_fixture(fixture)
    return levi_civita_geometry(coordinates, metric)


def all_components_zero(values) -> bool:
    """Recursively check a nested symbolic tensor for exact simplified zeros."""
    if isinstance(values, sp.MatrixBase):
        return all(_simplify(value) == 0 for value in values)
    if isinstance(values, (tuple, list)):
        return all(all_components_zero(value) for value in values)
    return _simplify(values) == 0
