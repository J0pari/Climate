#!/usr/bin/env python3
"""Representation-dynamics reference for the two-layer energy-balance model.

The physical evolution and coordinate maps come from the exact two-layer EBM
reference. PyDMD supplies the generic data-driven linear-system identification
used as an independent witness that full-rank representations retain the same
thermal decay modes.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))

import numpy as np
from pydmd import DMD

from reference.two_layer_energy_balance import (
    advance_constant_forcing,
    equilibrium_state_k,
    flux_coordinates_w_m2,
    forcing_from_fixture,
    load_fixture as load_ebm_fixture,
    modal_coordinates,
    mode_basis,
    parameters_from_fixture,
)


DEFAULT_FIXTURE = ROOT / "fixtures" / "physics" / "two-layer-ebm-geoffroy-mean-v1.json"


def _sample_temperature_pairs(
    fixture: dict[str, Any], *, dt_years: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    parameters = parameters_from_fixture(fixture)
    forcing = forcing_from_fixture(fixture)
    equilibrium = equilibrium_state_k(parameters, forcing)
    dt = float(dt_years)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt_years must be finite and positive")

    perturbations = np.array(
        [
            [-0.45, -0.30],
            [-0.45, 0.30],
            [-0.20, -0.10],
            [-0.20, 0.25],
            [0.15, -0.30],
            [0.15, 0.15],
            [0.35, -0.10],
            [0.35, 0.30],
            [0.50, -0.25],
            [0.50, 0.20],
        ],
        dtype=float,
    )
    states_x = equilibrium + perturbations
    states_y = np.vstack(
        [
            advance_constant_forcing(state, parameters, forcing, dt)
            for state in states_x
        ]
    )
    return states_x, states_y


def _temperature_anomaly_view(
    states: np.ndarray, fixture: dict[str, Any]
) -> np.ndarray:
    parameters = parameters_from_fixture(fixture)
    forcing = forcing_from_fixture(fixture)
    equilibrium = equilibrium_state_k(parameters, forcing)
    return np.asarray(states - equilibrium, dtype=float)


def _flux_view(states: np.ndarray, fixture: dict[str, Any]) -> np.ndarray:
    parameters = parameters_from_fixture(fixture)
    forcing = forcing_from_fixture(fixture)
    return np.vstack(
        [flux_coordinates_w_m2(state, parameters, forcing) for state in states]
    )


def _modal_view(states: np.ndarray, fixture: dict[str, Any]) -> np.ndarray:
    parameters = parameters_from_fixture(fixture)
    forcing = forcing_from_fixture(fixture)
    return np.vstack(
        [modal_coordinates(state, parameters, forcing) for state in states]
    )


def representation_pairs(
    fixture: dict[str, Any], *, dt_years: float = 1.0
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    states_x, states_y = _sample_temperature_pairs(fixture, dt_years=dt_years)
    temperature_x = _temperature_anomaly_view(states_x, fixture)
    temperature_y = _temperature_anomaly_view(states_y, fixture)
    flux_x = _flux_view(states_x, fixture)
    flux_y = _flux_view(states_y, fixture)
    modal_x = _modal_view(states_x, fixture)
    modal_y = _modal_view(states_y, fixture)

    return {
        "temperature_state": (temperature_x, temperature_y),
        "heat_flux": (flux_x, flux_y),
        "thermal_modes": (modal_x, modal_y),
        "temperature_plus_flux": (
            np.column_stack([temperature_x, flux_x]),
            np.column_stack([temperature_y, flux_y]),
        ),
        "all_full_rank_views": (
            np.column_stack([temperature_x, flux_x, modal_x]),
            np.column_stack([temperature_y, flux_y, modal_y]),
        ),
        "surface_temperature_scalar": (
            temperature_x[:, :1],
            temperature_y[:, :1],
        ),
        "toa_imbalance_scalar": (
            flux_x[:, :1],
            flux_y[:, :1],
        ),
    }


def _fit_dmd(
    x_samples: np.ndarray,
    y_samples: np.ndarray,
    *,
    rank: int,
) -> tuple[np.ndarray, float]:
    x = np.asarray(x_samples, dtype=float)
    y = np.asarray(y_samples, dtype=float)
    if x.shape != y.shape or x.ndim != 2 or x.shape[0] < 2:
        raise ValueError("DMD samples must be matched two-dimensional arrays")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("DMD samples must be finite")
    if rank < 1 or rank > min(x.shape):
        raise ValueError("DMD rank is incompatible with sample matrix shape")

    model = DMD(svd_rank=rank, exact=True, tlsq_rank=0)
    model.fit(x.T, y.T)
    eigenvalues = np.asarray(model.eigs)
    predicted = np.asarray(model.predict(x.T)).T
    denominator = max(float(np.linalg.norm(y)), np.finfo(float).tiny)
    relative_error = float(np.linalg.norm(predicted - y) / denominator)
    if not np.isfinite(relative_error) or not np.isfinite(eigenvalues).all():
        raise RuntimeError("DMD produced non-finite diagnostics")
    return eigenvalues, relative_error


def decay_timescales_years(eigenvalues: np.ndarray, dt_years: float) -> np.ndarray:
    dt = float(dt_years)
    values = np.asarray(eigenvalues)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("eigenvalues must be a nonempty vector")
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt_years must be finite and positive")
    if np.max(np.abs(values.imag)) > 1e-10:
        raise ValueError("reference decay modes are expected to be real")
    real = values.real
    if np.any(real <= 0.0) or np.any(real >= 1.0):
        raise ValueError("reference decay eigenvalues must lie strictly between zero and one")
    timescales = -dt / np.log(real)
    return np.sort(np.asarray(timescales, dtype=float))


def _representation_structure(samples: np.ndarray) -> dict[str, Any]:
    x = np.asarray(samples, dtype=float)
    rank = int(np.linalg.matrix_rank(x))
    dimension = int(x.shape[1])
    structure: dict[str, Any] = {
        "dimension": dimension,
        "matrix_rank": rank,
        "redundant_dimension_count": dimension - rank,
    }
    if rank < dimension:
        structure["condition_number_status"] = "rank_deficient"
        structure["condition_number_detail"] = (
            f"sample matrix has structural rank {rank} below observed dimension {dimension}"
        )
        return structure

    condition = float(np.linalg.cond(x))
    if not math.isfinite(condition):
        structure["condition_number_status"] = "undefined"
        structure["condition_number_detail"] = (
            "condition number became non-finite despite full structural rank"
        )
        return structure

    structure["condition_number_status"] = "finite"
    structure["condition_number"] = condition
    return structure


def analyze_representations(
    fixture: dict[str, Any], *, dt_years: float = 1.0
) -> dict[str, Any]:
    parameters = parameters_from_fixture(fixture)
    exact_timescales, _ = mode_basis(parameters)
    pairs = representation_pairs(fixture, dt_years=dt_years)

    full_names = (
        "temperature_state",
        "heat_flux",
        "thermal_modes",
        "temperature_plus_flux",
        "all_full_rank_views",
    )
    result: dict[str, Any] = {
        "fixture_id": fixture["fixture_id"],
        "dt_years": float(dt_years),
        "exact_timescales_years": exact_timescales.tolist(),
        "implementation_versions": {
            "pydmd": importlib.metadata.version("pydmd"),
            "numpy": importlib.metadata.version("numpy"),
            "scipy": importlib.metadata.version("scipy"),
        },
        "representations": {},
    }

    for name in full_names:
        x, y = pairs[name]
        eigenvalues, prediction_error = _fit_dmd(x, y, rank=2)
        timescales = decay_timescales_years(eigenvalues, dt_years)
        relative_timescale_error = float(
            np.max(np.abs(timescales - exact_timescales) / exact_timescales)
        )
        result["representations"][name] = {
            **_representation_structure(x),
            "dmd_timescales_years": timescales.tolist(),
            "max_relative_timescale_error": relative_timescale_error,
            "one_step_relative_error": prediction_error,
        }

    for name in ("surface_temperature_scalar", "toa_imbalance_scalar"):
        x, y = pairs[name]
        eigenvalues, prediction_error = _fit_dmd(x, y, rank=1)
        timescales = decay_timescales_years(eigenvalues, dt_years)
        result["representations"][name] = {
            **_representation_structure(x),
            "dmd_timescales_years": timescales.tolist(),
            "one_step_relative_error": prediction_error,
            "resolved_mode_count": 1,
        }

    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--dt-years", type=float, default=1.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = analyze_representations(
        load_ebm_fixture(args.fixture), dt_years=args.dt_years
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"exact timescales: {result['exact_timescales_years']}")
        for name, diagnostics in result["representations"].items():
            print(f"{name}: {diagnostics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
