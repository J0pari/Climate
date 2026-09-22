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

from src.station_sheaf import StationEdgeChunk, StationIdentitySheaf, StationSection, StationVariable


COVARIANCE_ASSUMPTION = "independent_station_observation_errors_diagonal"
STANDARDIZED_RESIDUAL_UNIT = "standard_deviation"


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
