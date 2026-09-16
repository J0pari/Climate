"""Exact symbolic Lie-bracket reference for finite-dimensional vector fields.

For vector fields X and Y in coordinates x^i, this module realizes

    [X,Y]^i = X^j ∂_j Y^i - Y^j ∂_j X^i.

It is a mathematical authority for infinitesimal-generator claims. A finite
composition score such as ||f(g(s)) - g(f(s))|| is a different object unless
`f` and `g` are explicitly parameterized flows and the infinitesimal limit is
controlled.
"""
from __future__ import annotations

from typing import Sequence

import sympy as sp


def _column(values: Sequence[sp.Expr]) -> sp.Matrix:
    vector = sp.Matrix(values)
    if vector.cols != 1:
        vector = vector.reshape(len(values), 1)
    return vector


def lie_bracket(
    coordinates: Sequence[sp.Symbol],
    left: Sequence[sp.Expr],
    right: Sequence[sp.Expr],
) -> sp.Matrix:
    """Return the exact coordinate Lie bracket `[left,right]`."""
    coords = tuple(coordinates)
    x = _column(left)
    y = _column(right)
    if len(coords) == 0:
        raise ValueError("at least one coordinate is required")
    if x.rows != len(coords) or y.rows != len(coords):
        raise ValueError("vector-field dimension must match coordinate dimension")
    return (y.jacobian(coords) * x - x.jacobian(coords) * y).applyfunc(sp.simplify)


def directional_derivative(
    coordinates: Sequence[sp.Symbol],
    vector_field: Sequence[sp.Expr],
    scalar: sp.Expr,
) -> sp.Expr:
    """Exact directional derivative X(f)."""
    coords = tuple(coordinates)
    x = _column(vector_field)
    if x.rows != len(coords):
        raise ValueError("vector-field dimension must match coordinate dimension")
    return sp.simplify(sum(x[i] * sp.diff(scalar, coords[i]) for i in range(len(coords))))


def is_zero_vector(vector: sp.Matrix) -> bool:
    return all(sp.simplify(component) == 0 for component in vector)
