"""Observation-uncertainty residuals for station identity sheaves.

This module is intentionally distinct from exact sheaf compatibility.  It
standardizes observed edge differences by caller-supplied observation standard
deviations under an explicit independent-error assumption.  It does not define
an approximate global section, a cohomology class, or a scientific significance
threshold.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import linalg as scipy_linalg

from src.station_sheaf import StationEdgeChunk, StationIdentitySheaf, StationSection, StationVariable


COVARIANCE_ASSUMPTION = "independent_station_observation_errors_diagonal"
CORRELATED_COVARIANCE_ASSUMPTION = "caller_supplied_observation_covariance"
STANDARDIZED_RESIDUAL_UNIT = "standard_deviation"
RANK_TOLERANCE_POLICY = "64_times_dimension_times_machine_epsilon_times_max_eigenvalue"


@dataclass(frozen=True)
class DiagonalObservationUncertainty:
    variables: tuple[StationVariable, ...]
    standard_deviation: np.ndarray
    observed: np.ndarray

    def __post_init__(self) -> None:
        variables = tuple(self.variables)
        if not variables or len({item.variable_id for item in variables}) != len(variables):
            raise ValueError("uncertainty variables must be non-empty and unique")
        sigma = np.asarray(self.standard_deviation, dtype=np.float64)
        observed = np.asarray(self.observed, dtype=bool)
        if sigma.ndim != 2 or observed.shape != sigma.shape:
            raise ValueError("standard_deviation and observed must share (station, variable) shape")
        if sigma.shape[1] != len(variables):
            raise ValueError("uncertainty variable dimension does not match variables")
        if np.any(~np.isfinite(sigma[observed])) or np.any(sigma[observed] <= 0.0):
            raise ValueError("observed standard deviations must be finite and positive")
        sigma = np.array(sigma, copy=True)
        observed = np.array(observed, copy=True)
        sigma.setflags(write=False)
        observed.setflags(write=False)
        object.__setattr__(self, "variables", variables)
        object.__setattr__(self, "standard_deviation", sigma)
        object.__setattr__(self, "observed", observed)

    @property
    def station_count(self) -> int:
        return int(self.standard_deviation.shape[0])


@dataclass(frozen=True)
class UncertaintyResidualReport:
    row_indices: np.ndarray
    standardized_residuals: np.ndarray
    structural_row_count: int
    raw_residual_energy: float
    chi_square: float
    degrees_of_freedom: int
    variable_units: tuple[str, ...]
    covariance_assumption: str = COVARIANCE_ASSUMPTION
    residual_unit: str = STANDARDIZED_RESIDUAL_UNIT

    def __post_init__(self) -> None:
        rows = np.asarray(self.row_indices, dtype=np.int64)
        residuals = np.asarray(self.standardized_residuals, dtype=np.float64)
        if rows.ndim != 1 or residuals.ndim != 1 or rows.size != residuals.size:
            raise ValueError("uncertainty residual rows and values must be matching one-dimensional arrays")
        if self.structural_row_count < 0 or self.degrees_of_freedom != rows.size:
            raise ValueError("invalid uncertainty residual dimensions")
        if rows.size and (int(rows.min()) < 0 or int(rows.max()) >= self.structural_row_count):
            raise ValueError("uncertainty residual row outside structural row space")
        if not np.all(np.isfinite(residuals)):
            raise ValueError("standardized residuals must be finite")
        if not np.isfinite(self.raw_residual_energy) or self.raw_residual_energy < 0.0:
            raise ValueError("raw residual energy must be finite and non-negative")
        if not np.isfinite(self.chi_square) or self.chi_square < 0.0:
            raise ValueError("chi-square statistic must be finite and non-negative")
        if not self.variable_units or any(not unit.strip() for unit in self.variable_units):
            raise ValueError("variable units must be explicit")
        rows = np.array(rows, copy=True)
        residuals = np.array(residuals, copy=True)
        rows.setflags(write=False)
        residuals.setflags(write=False)
        object.__setattr__(self, "row_indices", rows)
        object.__setattr__(self, "standardized_residuals", residuals)

    def reduced_chi_square(self) -> float | None:
        if self.degrees_of_freedom == 0:
            return None
        return self.chi_square / float(self.degrees_of_freedom)


def diagonal_uncertainty_residual(
    sheaf: StationIdentitySheaf,
    edges: StationEdgeChunk,
    section: StationSection,
    uncertainty: DiagonalObservationUncertainty,
) -> UncertaintyResidualReport:
    """Return uncertainty-standardized edge residuals without changing exact sheaf semantics."""
    if section.station_count != sheaf.station_count or section.variables != sheaf.variables:
        raise ValueError("section schema does not match sheaf")
    if uncertainty.station_count != sheaf.station_count or uncertainty.variables != sheaf.variables:
        raise ValueError("uncertainty schema does not match sheaf")
    if len(edges) and int(edges.head.max()) >= sheaf.station_count:
        raise ValueError("edge references station outside sheaf")

    structural_rows = len(edges) * sheaf.variable_count
    if len(edges) == 0:
        return UncertaintyResidualReport(
            row_indices=np.empty(0, dtype=np.int64),
            standardized_residuals=np.empty(0, dtype=np.float64),
            structural_row_count=0,
            raw_residual_energy=0.0,
            chi_square=0.0,
            degrees_of_freedom=0,
            variable_units=tuple(item.unit for item in sheaf.variables),
        )

    active = (
        section.observed[edges.tail]
        & section.observed[edges.head]
        & uncertainty.observed[edges.tail]
        & uncertainty.observed[edges.head]
    )
    difference = section.values[edges.head] - section.values[edges.tail]
    variance = (
        uncertainty.standard_deviation[edges.tail] ** 2
        + uncertainty.standard_deviation[edges.head] ** 2
    )
    flat = active.reshape(-1)
    active_difference = difference.reshape(-1)[flat]
    active_variance = variance.reshape(-1)[flat]
    if np.any(~np.isfinite(active_variance)) or np.any(active_variance <= 0.0):
        raise ValueError("active propagated edge variances must be finite and positive")
    standardized = active_difference / np.sqrt(active_variance)
    chi_square = float(standardized @ standardized)
    raw_energy = float(active_difference @ active_difference)
    return UncertaintyResidualReport(
        row_indices=np.flatnonzero(flat).astype(np.int64, copy=False),
        standardized_residuals=standardized,
        structural_row_count=structural_rows,
        raw_residual_energy=raw_energy,
        chi_square=chi_square,
        degrees_of_freedom=int(standardized.size),
        variable_units=tuple(item.unit for item in sheaf.variables),
    )


@dataclass(frozen=True)
class CorrelatedObservationUncertainty:
    """Full covariance over station-major, variable-minor observation coordinates."""

    variables: tuple[StationVariable, ...]
    covariance: np.ndarray
    observed: np.ndarray

    def __post_init__(self) -> None:
        variables = tuple(self.variables)
        if not variables or len({item.variable_id for item in variables}) != len(variables):
            raise ValueError("uncertainty variables must be non-empty and unique")
        covariance = np.asarray(self.covariance, dtype=np.float64)
        observed = np.asarray(self.observed, dtype=bool)
        if observed.ndim != 2 or observed.shape[1] != len(variables):
            raise ValueError("observed must have (station, variable) shape")
        coordinate_count = int(observed.size)
        if covariance.shape != (coordinate_count, coordinate_count):
            raise ValueError("covariance must span every station-variable observation coordinate")
        if not np.all(np.isfinite(covariance)):
            raise ValueError("observation covariance must be finite")
        scale = max(float(np.max(np.abs(covariance))), 1.0)
        symmetry_tolerance = 64.0 * np.finfo(np.float64).eps * scale
        if np.max(np.abs(covariance - covariance.T)) > symmetry_tolerance:
            raise ValueError("observation covariance must be symmetric up to numerical roundoff")
        active_diagonal = np.diag(covariance)[observed.reshape(-1)]
        if np.any(active_diagonal <= 0.0):
            raise ValueError("observed covariance diagonal entries must be positive")
        eigenvalues = scipy_linalg.eigvalsh(covariance, check_finite=False)
        max_eigenvalue = max(float(np.max(np.abs(eigenvalues))), 1.0)
        psd_tolerance = 64.0 * covariance.shape[0] * np.finfo(np.float64).eps * max_eigenvalue
        if float(np.min(eigenvalues)) < -psd_tolerance:
            raise ValueError("observation covariance must be positive semidefinite")
        covariance = np.array(covariance, copy=True)
        observed = np.array(observed, copy=True)
        covariance.setflags(write=False)
        observed.setflags(write=False)
        object.__setattr__(self, "variables", variables)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "observed", observed)

    @property
    def station_count(self) -> int:
        return int(self.observed.shape[0])


@dataclass(frozen=True)
class CorrelatedUncertaintyResidualReport:
    row_indices: np.ndarray
    raw_residuals: np.ndarray
    residual_covariance: np.ndarray
    whitened_residual_modes: np.ndarray
    retained_covariance_eigenvalues: np.ndarray
    structural_row_count: int
    chi_square: float
    degrees_of_freedom: int
    numerical_rank_tolerance: float
    row_units: tuple[str, ...]
    covariance_assumption: str = CORRELATED_COVARIANCE_ASSUMPTION
    rank_tolerance_policy: str = RANK_TOLERANCE_POLICY

    def __post_init__(self) -> None:
        rows = np.asarray(self.row_indices, dtype=np.int64)
        raw = np.asarray(self.raw_residuals, dtype=np.float64)
        covariance = np.asarray(self.residual_covariance, dtype=np.float64)
        whitened = np.asarray(self.whitened_residual_modes, dtype=np.float64)
        eigenvalues = np.asarray(self.retained_covariance_eigenvalues, dtype=np.float64)
        if rows.ndim != 1 or raw.shape != rows.shape:
            raise ValueError("correlated residual rows and values must be matching vectors")
        if covariance.shape != (rows.size, rows.size):
            raise ValueError("residual covariance shape does not match active residual rows")
        if whitened.ndim != 1 or eigenvalues.shape != whitened.shape:
            raise ValueError("whitened modes and retained covariance eigenvalues must match")
        if self.degrees_of_freedom != whitened.size or self.degrees_of_freedom < 1:
            raise ValueError("correlated residual degrees of freedom must match retained modes")
        if len(self.row_units) != rows.size or any(not unit.strip() for unit in self.row_units):
            raise ValueError("each active residual row must carry an explicit unit")
        if rows.size and (int(rows.min()) < 0 or int(rows.max()) >= self.structural_row_count):
            raise ValueError("correlated residual row outside structural row space")
        if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(covariance)):
            raise ValueError("correlated residual values and covariance must be finite")
        if not np.all(np.isfinite(whitened)) or not np.all(np.isfinite(eigenvalues)):
            raise ValueError("whitened residual modes must be finite")
        if np.any(eigenvalues <= 0.0):
            raise ValueError("retained residual covariance eigenvalues must be positive")
        if not np.isfinite(self.chi_square) or self.chi_square < 0.0:
            raise ValueError("correlated chi-square must be finite and non-negative")
        if not np.isfinite(self.numerical_rank_tolerance) or self.numerical_rank_tolerance < 0.0:
            raise ValueError("numerical rank tolerance must be finite and non-negative")
        for name, value in (
            ("row_indices", rows),
            ("raw_residuals", raw),
            ("residual_covariance", covariance),
            ("whitened_residual_modes", whitened),
            ("retained_covariance_eigenvalues", eigenvalues),
        ):
            copied = np.array(value, copy=True)
            copied.setflags(write=False)
            object.__setattr__(self, name, copied)

    def reduced_chi_square(self) -> float:
        return self.chi_square / float(self.degrees_of_freedom)


def correlated_uncertainty_residual(
    sheaf: StationIdentitySheaf,
    edges: StationEdgeChunk,
    section: StationSection,
    uncertainty: CorrelatedObservationUncertainty,
) -> CorrelatedUncertaintyResidualReport:
    """Propagate full observation covariance through the active sheaf residual map."""
    if section.station_count != sheaf.station_count or section.variables != sheaf.variables:
        raise ValueError("section schema does not match sheaf")
    if uncertainty.station_count != sheaf.station_count or uncertainty.variables != sheaf.variables:
        raise ValueError("uncertainty schema does not match sheaf")
    if len(edges) == 0:
        raise ValueError("correlated residual covariance requires at least one edge")
    if int(edges.head.max()) >= sheaf.station_count:
        raise ValueError("edge references station outside sheaf")

    active = (
        section.observed[edges.tail]
        & section.observed[edges.head]
        & uncertainty.observed[edges.tail]
        & uncertainty.observed[edges.head]
    )
    flat_active = active.reshape(-1)
    row_indices = np.flatnonzero(flat_active).astype(np.int64, copy=False)
    if row_indices.size == 0:
        raise ValueError("correlated residual covariance has no active observed rows")

    difference = section.values[edges.head] - section.values[edges.tail]
    raw_residuals = difference.reshape(-1)[flat_active]
    coboundary = sheaf.coboundary_0(edges)[row_indices].toarray()
    residual_covariance = coboundary @ uncertainty.covariance @ coboundary.T
    residual_covariance = 0.5 * (residual_covariance + residual_covariance.T)
    eigenvalues, eigenvectors = scipy_linalg.eigh(
        residual_covariance,
        check_finite=False,
    )
    max_eigenvalue = max(float(np.max(np.abs(eigenvalues))), 1.0)
    numerical_rank_tolerance = (
        64.0
        * residual_covariance.shape[0]
        * np.finfo(np.float64).eps
        * max_eigenvalue
    )
    if float(np.min(eigenvalues)) < -numerical_rank_tolerance:
        raise ValueError("propagated residual covariance is not positive semidefinite")
    retained = eigenvalues > numerical_rank_tolerance
    if not np.any(retained):
        raise ValueError("propagated residual covariance has no positive-variance mode")
    retained_eigenvalues = eigenvalues[retained]
    projected = eigenvectors[:, retained].T @ raw_residuals
    whitened = projected / np.sqrt(retained_eigenvalues)
    chi_square = float(whitened @ whitened)
    row_units = tuple(
        sheaf.variables[int(row % sheaf.variable_count)].unit
        for row in row_indices
    )
    return CorrelatedUncertaintyResidualReport(
        row_indices=row_indices,
        raw_residuals=raw_residuals,
        residual_covariance=residual_covariance,
        whitened_residual_modes=whitened,
        retained_covariance_eigenvalues=retained_eigenvalues,
        structural_row_count=len(edges) * sheaf.variable_count,
        chi_square=chi_square,
        degrees_of_freedom=int(retained_eigenvalues.size),
        numerical_rank_tolerance=numerical_rank_tolerance,
        row_units=row_units,
    )
