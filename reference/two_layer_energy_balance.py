#!/usr/bin/env python3
"""Exact linear reference for a standard two-layer global energy-balance model.

Climate owns the physical variables, units, reservoir exchanges, and balance
identities. SciPy owns the generic matrix exponential used to propagate the
constant-forcing linear system.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import eig, expm


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = ROOT / "fixtures" / "physics" / "two-layer-ebm-geoffroy-mean-v1.json"


@dataclass(frozen=True)
class TwoLayerParameters:
    surface_heat_capacity_w_yr_m2_k: float
    deep_heat_capacity_w_yr_m2_k: float
    ocean_heat_exchange_w_m2_k: float
    climate_feedback_w_m2_k: float

    def __post_init__(self) -> None:
        for name, value in (
            ("surface_heat_capacity_w_yr_m2_k", self.surface_heat_capacity_w_yr_m2_k),
            ("deep_heat_capacity_w_yr_m2_k", self.deep_heat_capacity_w_yr_m2_k),
            ("ocean_heat_exchange_w_m2_k", self.ocean_heat_exchange_w_m2_k),
            ("climate_feedback_w_m2_k", self.climate_feedback_w_m2_k),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


def load_fixture(path: Path = DEFAULT_FIXTURE) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parameters_from_fixture(fixture: dict[str, Any]) -> TwoLayerParameters:
    values = fixture["parameters"]
    return TwoLayerParameters(
        surface_heat_capacity_w_yr_m2_k=float(values["surface_heat_capacity_w_yr_m2_k"]),
        deep_heat_capacity_w_yr_m2_k=float(values["deep_heat_capacity_w_yr_m2_k"]),
        ocean_heat_exchange_w_m2_k=float(values["ocean_heat_exchange_w_m2_k"]),
        climate_feedback_w_m2_k=float(values["climate_feedback_w_m2_k"]),
    )


def initial_state_from_fixture(fixture: dict[str, Any]) -> np.ndarray:
    state = fixture["initial_state"]
    return _state_vector(
        [
            float(state["surface_temperature_anomaly_k"]),
            float(state["deep_temperature_anomaly_k"]),
        ]
    )


def forcing_from_fixture(fixture: dict[str, Any]) -> float:
    forcing = float(fixture["forcing"]["magnitude_w_m2"])
    if not math.isfinite(forcing):
        raise ValueError("forcing must be finite")
    return forcing


def _state_vector(state: np.ndarray | list[float] | tuple[float, float]) -> np.ndarray:
    value = np.asarray(state, dtype=float)
    if value.shape != (2,):
        raise ValueError(f"state must have shape (2,), got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("state must be finite")
    return value


def _finite_scalar(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def system_matrix(parameters: TwoLayerParameters) -> np.ndarray:
    c_surface = parameters.surface_heat_capacity_w_yr_m2_k
    c_deep = parameters.deep_heat_capacity_w_yr_m2_k
    gamma = parameters.ocean_heat_exchange_w_m2_k
    feedback = parameters.climate_feedback_w_m2_k
    return np.array(
        [
            [-(feedback + gamma) / c_surface, gamma / c_surface],
            [gamma / c_deep, -gamma / c_deep],
        ],
        dtype=float,
    )


def forcing_vector(parameters: TwoLayerParameters, forcing_w_m2: float) -> np.ndarray:
    forcing = _finite_scalar("forcing_w_m2", forcing_w_m2)
    return np.array(
        [forcing / parameters.surface_heat_capacity_w_yr_m2_k, 0.0], dtype=float
    )


def ocean_heat_uptake_w_m2(
    state: np.ndarray | list[float] | tuple[float, float],
    parameters: TwoLayerParameters,
) -> float:
    surface, deep = _state_vector(state)
    return float(parameters.ocean_heat_exchange_w_m2_k * (surface - deep))


def toa_imbalance_w_m2(
    state: np.ndarray | list[float] | tuple[float, float],
    parameters: TwoLayerParameters,
    forcing_w_m2: float,
) -> float:
    surface, _ = _state_vector(state)
    forcing = _finite_scalar("forcing_w_m2", forcing_w_m2)
    return float(forcing - parameters.climate_feedback_w_m2_k * surface)


def tendency_k_per_year(
    state: np.ndarray | list[float] | tuple[float, float],
    parameters: TwoLayerParameters,
    forcing_w_m2: float,
) -> np.ndarray:
    value = _state_vector(state)
    tendency = system_matrix(parameters) @ value + forcing_vector(parameters, forcing_w_m2)
    if not np.isfinite(tendency).all():
        raise RuntimeError("two-layer tendency became non-finite")
    return tendency


def reservoir_storage_fluxes_w_m2(
    state: np.ndarray | list[float] | tuple[float, float],
    parameters: TwoLayerParameters,
    forcing_w_m2: float,
) -> np.ndarray:
    tendency = tendency_k_per_year(state, parameters, forcing_w_m2)
    return np.array(
        [
            parameters.surface_heat_capacity_w_yr_m2_k * tendency[0],
            parameters.deep_heat_capacity_w_yr_m2_k * tendency[1],
        ],
        dtype=float,
    )


def total_heat_content_w_yr_m2(
    state: np.ndarray | list[float] | tuple[float, float],
    parameters: TwoLayerParameters,
) -> float:
    surface, deep = _state_vector(state)
    return float(
        parameters.surface_heat_capacity_w_yr_m2_k * surface
        + parameters.deep_heat_capacity_w_yr_m2_k * deep
    )


def energy_budget_residual_w_m2(
    state: np.ndarray | list[float] | tuple[float, float],
    parameters: TwoLayerParameters,
    forcing_w_m2: float,
) -> float:
    storage = reservoir_storage_fluxes_w_m2(state, parameters, forcing_w_m2)
    external = toa_imbalance_w_m2(state, parameters, forcing_w_m2)
    return float(storage.sum() - external)


def equilibrium_state_k(parameters: TwoLayerParameters, forcing_w_m2: float) -> np.ndarray:
    forcing = _finite_scalar("forcing_w_m2", forcing_w_m2)
    temperature = forcing / parameters.climate_feedback_w_m2_k
    return np.array([temperature, temperature], dtype=float)


def advance_constant_forcing(
    state: np.ndarray | list[float] | tuple[float, float],
    parameters: TwoLayerParameters,
    forcing_w_m2: float,
    dt_years: float,
) -> np.ndarray:
    value = _state_vector(state)
    dt = _finite_scalar("dt_years", dt_years)
    if dt < 0.0:
        raise ValueError("dt_years must be nonnegative")
    equilibrium = equilibrium_state_k(parameters, forcing_w_m2)
    advanced = equilibrium + expm(system_matrix(parameters) * dt) @ (value - equilibrium)
    if not np.isfinite(advanced).all():
        raise RuntimeError("two-layer exact advance became non-finite")
    return np.asarray(advanced, dtype=float)


def trajectory_constant_forcing(
    initial_state: np.ndarray | list[float] | tuple[float, float],
    parameters: TwoLayerParameters,
    forcing_w_m2: float,
    times_years: np.ndarray | list[float],
) -> np.ndarray:
    initial = _state_vector(initial_state)
    times = np.asarray(times_years, dtype=float)
    if times.ndim != 1 or times.size == 0:
        raise ValueError("times_years must be a nonempty one-dimensional array")
    if not np.isfinite(times).all() or np.any(times < 0.0):
        raise ValueError("times_years must be finite and nonnegative")
    if np.any(np.diff(times) < 0.0):
        raise ValueError("times_years must be nondecreasing")

    equilibrium = equilibrium_state_k(parameters, forcing_w_m2)
    anomaly = initial - equilibrium
    matrix = system_matrix(parameters)
    trajectory = np.vstack([equilibrium + expm(matrix * time) @ anomaly for time in times])
    if not np.isfinite(trajectory).all():
        raise RuntimeError("two-layer trajectory became non-finite")
    return trajectory


def flux_coordinates_w_m2(
    state: np.ndarray | list[float] | tuple[float, float],
    parameters: TwoLayerParameters,
    forcing_w_m2: float,
) -> np.ndarray:
    return np.array(
        [
            toa_imbalance_w_m2(state, parameters, forcing_w_m2),
            ocean_heat_uptake_w_m2(state, parameters),
        ],
        dtype=float,
    )


def state_from_flux_coordinates_k(
    flux_coordinates: np.ndarray | list[float] | tuple[float, float],
    parameters: TwoLayerParameters,
    forcing_w_m2: float,
) -> np.ndarray:
    flux = np.asarray(flux_coordinates, dtype=float)
    if flux.shape != (2,) or not np.isfinite(flux).all():
        raise ValueError("flux_coordinates must be a finite shape-(2,) vector")
    forcing = _finite_scalar("forcing_w_m2", forcing_w_m2)
    toa_imbalance, ocean_heat_uptake = flux
    surface = (forcing - toa_imbalance) / parameters.climate_feedback_w_m2_k
    deep = surface - ocean_heat_uptake / parameters.ocean_heat_exchange_w_m2_k
    return np.array([surface, deep], dtype=float)


def mode_basis(parameters: TwoLayerParameters) -> tuple[np.ndarray, np.ndarray]:
    eigenvalues, eigenvectors = eig(system_matrix(parameters))
    if np.max(np.abs(eigenvalues.imag)) > 1e-12 or np.max(np.abs(eigenvectors.imag)) > 1e-12:
        raise RuntimeError("expected real decay modes for the stable two-layer reference")
    eigenvalues = eigenvalues.real
    basis = eigenvectors.real
    if np.any(eigenvalues >= 0.0):
        raise RuntimeError("two-layer reference must have strictly decaying modes")

    timescales = -1.0 / eigenvalues
    order = np.argsort(timescales)
    timescales = timescales[order]
    basis = basis[:, order]

    for column in range(basis.shape[1]):
        vector = basis[:, column]
        pivot = int(np.argmax(np.abs(vector)))
        if vector[pivot] < 0.0:
            basis[:, column] = -vector

    if not np.isfinite(basis).all() or abs(float(np.linalg.det(basis))) <= 1e-12:
        raise RuntimeError("two-layer modal basis is singular or non-finite")
    return timescales, basis


def modal_coordinates(
    state: np.ndarray | list[float] | tuple[float, float],
    parameters: TwoLayerParameters,
    forcing_w_m2: float,
) -> np.ndarray:
    value = _state_vector(state)
    equilibrium = equilibrium_state_k(parameters, forcing_w_m2)
    _, basis = mode_basis(parameters)
    coordinates = np.linalg.solve(basis, value - equilibrium)
    if not np.isfinite(coordinates).all():
        raise RuntimeError("modal coordinates became non-finite")
    return coordinates


def state_from_modal_coordinates_k(
    coordinates: np.ndarray | list[float] | tuple[float, float],
    parameters: TwoLayerParameters,
    forcing_w_m2: float,
) -> np.ndarray:
    modal = np.asarray(coordinates, dtype=float)
    if modal.shape != (2,) or not np.isfinite(modal).all():
        raise ValueError("modal coordinates must be a finite shape-(2,) vector")
    equilibrium = equilibrium_state_k(parameters, forcing_w_m2)
    _, basis = mode_basis(parameters)
    return np.asarray(equilibrium + basis @ modal, dtype=float)


def summarize_fixture(fixture: dict[str, Any]) -> dict[str, Any]:
    parameters = parameters_from_fixture(fixture)
    forcing = forcing_from_fixture(fixture)
    initial = initial_state_from_fixture(fixture)
    timescales, _ = mode_basis(parameters)
    equilibrium = equilibrium_state_k(parameters, forcing)
    state_1000 = advance_constant_forcing(initial, parameters, forcing, 1000.0)
    return {
        "fixture_id": fixture["fixture_id"],
        "forcing_w_m2": forcing,
        "equilibrium_temperature_anomaly_k": float(equilibrium[0]),
        "fast_timescale_years": float(timescales[0]),
        "slow_timescale_years": float(timescales[1]),
        "state_after_1000_years_k": state_1000.tolist(),
        "initial_energy_budget_residual_w_m2": energy_budget_residual_w_m2(
            initial, parameters, forcing
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    summary = summarize_fixture(load_fixture(args.fixture))
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        for key, value in summary.items():
            print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
