"""Exact symbolic Noether reference for finite-dimensional point symmetries.

The climate repository contains exploratory symmetry language, but approximate
constancy of a diagnostic is not Noether's theorem. This module implements the
actual variational statement for a Lagrangian L(t, q, qdot):

    pr^(1) X(L) + L D_t(tau) = D_t(B)

for a point transformation X = tau d/dt + eta_i d/dq_i. When that invariance
residual vanishes, the Noether charge

    J = p_i eta_i - H tau - B

obeys the off-shell identity

    D_t J - R + (eta_i - qdot_i tau) E_i(L) = 0,

where R is the invariance residual and E_i(L) = dL/dq_i - D_t(dL/dqdot_i).
On Euler-Lagrange trajectories and for R=0, J is conserved.

This establishes the mathematics only. It does not imply that the forced,
dissipative climate system admits these exact symmetries.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import sympy as sp


@dataclass(frozen=True)
class PointSymmetry:
    tau: sp.Expr
    eta: tuple[sp.Expr, ...]
    boundary: sp.Expr = sp.Integer(0)


@dataclass(frozen=True)
class NoetherReport:
    invariance_residual: sp.Expr
    charge: sp.Expr
    euler_lagrange_residuals: tuple[sp.Expr, ...]
    characteristic: tuple[sp.Expr, ...]
    identity_residual: sp.Expr


def total_time_derivative(
    expr: sp.Expr,
    time: sp.Symbol,
    coordinates: Sequence[sp.Symbol],
    velocities: Sequence[sp.Symbol],
    accelerations: Sequence[sp.Symbol],
) -> sp.Expr:
    """Total derivative on first-jet expressions, represented on the second jet."""
    result = sp.diff(expr, time)
    for q, qdot, qddot in zip(coordinates, velocities, accelerations, strict=True):
        result += sp.diff(expr, q) * qdot + sp.diff(expr, qdot) * qddot
    return sp.simplify(result)


def analyze_point_symmetry(
    lagrangian: sp.Expr,
    time: sp.Symbol,
    coordinates: Sequence[sp.Symbol],
    velocities: Sequence[sp.Symbol],
    accelerations: Sequence[sp.Symbol],
    symmetry: PointSymmetry,
) -> NoetherReport:
    """Return the exact point-symmetry Noether residuals and charge.

    ``tau``, every component of ``eta``, and the boundary term are restricted to
    point-symmetry dependence on ``(t, q)``. The Lagrangian may depend on
    ``(t, q, qdot)`` but not acceleration. Unsupported generalized/contact
    symmetries fail explicitly rather than being silently interpreted as point
    symmetries.
    """
    coordinates = tuple(coordinates)
    velocities = tuple(velocities)
    accelerations = tuple(accelerations)

    if not coordinates:
        raise ValueError("Noether analysis requires at least one coordinate")
    if not (len(coordinates) == len(velocities) == len(accelerations) == len(symmetry.eta)):
        raise ValueError("coordinate, velocity, acceleration, and eta dimensions must match")
    if any(lagrangian.has(acceleration) for acceleration in accelerations):
        raise ValueError("Lagrangian must be first-order and may not depend on acceleration")

    point_expressions = (symmetry.tau, *symmetry.eta, symmetry.boundary)
    forbidden = (*velocities, *accelerations)
    for expr in point_expressions:
        if any(expr.has(symbol) for symbol in forbidden):
            raise ValueError("point symmetry tau/eta/boundary may depend only on t and q")

    dt_tau = total_time_derivative(
        symmetry.tau, time, coordinates, velocities, accelerations
    )

    invariance = symmetry.tau * sp.diff(lagrangian, time)
    for q, qdot, eta in zip(coordinates, velocities, symmetry.eta, strict=True):
        prolonged_eta = total_time_derivative(
            eta, time, coordinates, velocities, accelerations
        ) - qdot * dt_tau
        invariance += eta * sp.diff(lagrangian, q)
        invariance += prolonged_eta * sp.diff(lagrangian, qdot)
    invariance += lagrangian * dt_tau
    invariance -= total_time_derivative(
        symmetry.boundary, time, coordinates, velocities, accelerations
    )
    invariance = sp.simplify(sp.expand(invariance))

    momenta = tuple(sp.diff(lagrangian, qdot) for qdot in velocities)
    hamiltonian = sp.simplify(
        sum(momentum * qdot for momentum, qdot in zip(momenta, velocities, strict=True))
        - lagrangian
    )
    charge = sp.simplify(
        sum(momentum * eta for momentum, eta in zip(momenta, symmetry.eta, strict=True))
        - hamiltonian * symmetry.tau
        - symmetry.boundary
    )

    euler_lagrange = tuple(
        sp.simplify(
            sp.diff(lagrangian, q)
            - total_time_derivative(momentum, time, coordinates, velocities, accelerations)
        )
        for q, momentum in zip(coordinates, momenta, strict=True)
    )
    characteristic = tuple(
        sp.simplify(eta - qdot * symmetry.tau)
        for eta, qdot in zip(symmetry.eta, velocities, strict=True)
    )

    identity = total_time_derivative(
        charge, time, coordinates, velocities, accelerations
    ) - invariance
    identity += sum(
        char * residual
        for char, residual in zip(characteristic, euler_lagrange, strict=True)
    )
    identity = sp.simplify(sp.expand(identity))

    return NoetherReport(
        invariance_residual=invariance,
        charge=charge,
        euler_lagrange_residuals=euler_lagrange,
        characteristic=characteristic,
        identity_residual=identity,
    )
