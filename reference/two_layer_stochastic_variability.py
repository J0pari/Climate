#!/usr/bin/env python3
"""Seeded stochastic internal-variability control for the two-layer EBM.

A piecewise-constant AR(1) heat-exchange perturbation is applied with equal and
opposite storage fluxes to the surface and deep reservoirs.  It therefore
redistributes heat internally while the total-column budget remains governed by
the declared external forcing and top-of-atmosphere feedback.

The finite-interval state update is exact up to SciPy's matrix exponential for
the declared linear system and interval-constant forcing/exchange flux.
"""
from __future__ import annotations

import argparse
import hashlib
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
from scipy.linalg import expm

from reference.two_layer_energy_balance import (
    TwoLayerParameters,
    equilibrium_state_k,
    load_fixture as load_ebm_fixture,
    parameters_from_fixture,
    system_matrix,
)


DEFAULT_EBM_FIXTURE = (
    ROOT / "fixtures" / "physics" / "two-layer-ebm-geoffroy-mean-v1.json"
)
DEFAULT_STOCHASTIC_FIXTURE = (
    ROOT / "fixtures" / "physics" / "two-layer-ebm-stochastic-internal-variability-v1.json"
)


def load_stochastic_fixture(
    path: Path = DEFAULT_STOCHASTIC_FIXTURE,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported stochastic-variability fixture schema")
    dt = float(payload["dt_years"])
    burn = payload["burn_in_steps"]
    samples = payload["sample_steps"]
    forcing = float(payload["external_forcing_w_m2"])
    rho = float(payload["internal_exchange_ar1_rho"])
    sigma = float(payload["internal_exchange_stationary_std_w_m2"])
    lag = payload["lag_steps"]
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt_years must be finite and positive")
    if not isinstance(burn, int) or burn < 0:
        raise ValueError("burn_in_steps must be a nonnegative integer")
    if not isinstance(samples, int) or samples < 128:
        raise ValueError("sample_steps must be an integer >= 128")
    if not math.isfinite(forcing):
        raise ValueError("external_forcing_w_m2 must be finite")
    if not math.isfinite(rho) or abs(rho) >= 1.0:
        raise ValueError("internal_exchange_ar1_rho must have magnitude < 1")
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError(
            "internal_exchange_stationary_std_w_m2 must be finite and positive"
        )
    if not isinstance(payload.get("seed"), int):
        raise ValueError("stochastic fixture seed must be an integer")
    if not isinstance(lag, int) or lag < 1 or lag >= samples:
        raise ValueError("lag_steps must be an integer inside the retained trajectory")
    return payload


def _step_matrix(
    parameters: TwoLayerParameters,
    dt_years: float,
) -> np.ndarray:
    dt = float(dt_years)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt_years must be finite and positive")
    matrix = np.zeros((4, 4), dtype=float)
    matrix[:2, :2] = system_matrix(parameters)
    matrix[0, 2] = 1.0 / parameters.surface_heat_capacity_w_yr_m2_k
    matrix[0, 3] = 1.0 / parameters.surface_heat_capacity_w_yr_m2_k
    matrix[1, 3] = -1.0 / parameters.deep_heat_capacity_w_yr_m2_k
    step = expm(matrix * dt)
    if not np.isfinite(step).all():
        raise RuntimeError("stochastic exact step matrix became non-finite")
    return step


def _trajectory_digest(states: np.ndarray, exchange: np.ndarray) -> str:
    state_values = np.asarray(states, dtype="<f8")
    exchange_values = np.asarray(exchange, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(b"two-layer-stochastic-state-k\0")
    digest.update(state_values.tobytes(order="C"))
    digest.update(b"internal-exchange-w-m2\0")
    digest.update(exchange_values.tobytes(order="C"))
    return "sha256:" + digest.hexdigest()


def generate_stochastic_trajectory(
    ebm_fixture: dict[str, Any],
    stochastic_fixture: dict[str, Any],
) -> dict[str, Any]:
    parameters = parameters_from_fixture(ebm_fixture)
    dt = float(stochastic_fixture["dt_years"])
    burn = int(stochastic_fixture["burn_in_steps"])
    sample_steps = int(stochastic_fixture["sample_steps"])
    forcing = float(stochastic_fixture["external_forcing_w_m2"])
    rho = float(stochastic_fixture["internal_exchange_ar1_rho"])
    stationary_std = float(
        stochastic_fixture["internal_exchange_stationary_std_w_m2"]
    )
    seed = int(stochastic_fixture["seed"])

    step = _step_matrix(parameters, dt)
    state = equilibrium_state_k(parameters, forcing)
    internal_exchange = 0.0
    innovation_std = stationary_std * math.sqrt(1.0 - rho * rho)
    rng = np.random.default_rng(seed)

    states = np.empty((sample_steps, 2), dtype=float)
    exchange = np.empty(sample_steps, dtype=float)
    budget_residual = np.empty(sample_steps, dtype=float)
    direct_internal_storage = np.empty(sample_steps, dtype=float)

    total_steps = burn + sample_steps
    retained = 0
    for index in range(total_steps):
        internal_exchange = (
            rho * internal_exchange + innovation_std * float(rng.normal())
        )
        augmented = np.array(
            [state[0], state[1], forcing, internal_exchange],
            dtype=float,
        )
        state = np.asarray((step @ augmented)[:2], dtype=float)
        if not np.isfinite(state).all():
            raise RuntimeError("stochastic two-layer state became non-finite")

        if index < burn:
            continue

        tendency = system_matrix(parameters) @ state + np.array(
            [
                (forcing + internal_exchange)
                / parameters.surface_heat_capacity_w_yr_m2_k,
                -internal_exchange / parameters.deep_heat_capacity_w_yr_m2_k,
            ],
            dtype=float,
        )
        storage = (
            parameters.surface_heat_capacity_w_yr_m2_k * tendency[0]
            + parameters.deep_heat_capacity_w_yr_m2_k * tendency[1]
        )
        external = forcing - parameters.climate_feedback_w_m2_k * state[0]
        direct_internal = (
            parameters.surface_heat_capacity_w_yr_m2_k
            * (
                internal_exchange
                / parameters.surface_heat_capacity_w_yr_m2_k
            )
            + parameters.deep_heat_capacity_w_yr_m2_k
            * (
                -internal_exchange
                / parameters.deep_heat_capacity_w_yr_m2_k
            )
        )

        states[retained] = state
        exchange[retained] = internal_exchange
        budget_residual[retained] = storage - external
        direct_internal_storage[retained] = direct_internal
        retained += 1

    if retained != sample_steps:
        raise RuntimeError("stochastic retained sample count is inconsistent")

    return {
        "states_k": states,
        "internal_exchange_w_m2": exchange,
        "budget_residual_w_m2": budget_residual,
        "direct_internal_storage_w_m2": direct_internal_storage,
        "trajectory_digest": _trajectory_digest(states, exchange),
    }


def _state_statistics(states: np.ndarray, lag_steps: int) -> dict[str, Any]:
    values = np.asarray(states, dtype=float)
    if (
        values.ndim != 2
        or values.shape[1] != 2
        or values.shape[0] <= lag_steps
        or not np.isfinite(values).all()
    ):
        raise ValueError("state statistics require a finite shape-(n,2) trajectory")
    mean = np.mean(values, axis=0)
    centered = values - mean
    covariance = centered.T @ centered / values.shape[0]
    lag_covariance = (
        centered[:-lag_steps].T @ centered[lag_steps:]
        / (values.shape[0] - lag_steps)
    )
    return {
        "mean_state_k": mean.tolist(),
        "state_covariance_k2": covariance.tolist(),
        "lag_covariance_k2": lag_covariance.tolist(),
        "surface_variance_k2": float(covariance[0, 0]),
        "deep_variance_k2": float(covariance[1, 1]),
    }


def summarize_stochastic_variability(
    ebm_fixture: dict[str, Any],
    stochastic_fixture: dict[str, Any],
) -> dict[str, Any]:
    generated = generate_stochastic_trajectory(ebm_fixture, stochastic_fixture)
    states = generated["states_k"]
    exchange = generated["internal_exchange_w_m2"]
    lag_steps = int(stochastic_fixture["lag_steps"])
    statistics = _state_statistics(states, lag_steps)

    exchange_centered = exchange - float(np.mean(exchange))
    realized_std = float(np.sqrt(np.mean(exchange_centered * exchange_centered)))
    realized_lag_correlation = float(
        np.dot(exchange_centered[:-1], exchange_centered[1:])
        / math.sqrt(
            float(np.dot(exchange_centered[:-1], exchange_centered[:-1]))
            * float(np.dot(exchange_centered[1:], exchange_centered[1:]))
        )
    )

    return {
        "fixture_id": stochastic_fixture["fixture_id"],
        "ebm_fixture_id": ebm_fixture["fixture_id"],
        "dt_years": float(stochastic_fixture["dt_years"]),
        "burn_in_steps": int(stochastic_fixture["burn_in_steps"]),
        "sample_steps": int(stochastic_fixture["sample_steps"]),
        "external_forcing_w_m2": float(stochastic_fixture["external_forcing_w_m2"]),
        "internal_exchange_ar1_rho": float(
            stochastic_fixture["internal_exchange_ar1_rho"]
        ),
        "internal_exchange_stationary_std_w_m2": float(
            stochastic_fixture["internal_exchange_stationary_std_w_m2"]
        ),
        "seed": int(stochastic_fixture["seed"]),
        "lag_steps": lag_steps,
        "trajectory_digest": generated["trajectory_digest"],
        "max_abs_total_budget_residual_w_m2": float(
            np.max(np.abs(generated["budget_residual_w_m2"]))
        ),
        "max_abs_direct_internal_storage_w_m2": float(
            np.max(np.abs(generated["direct_internal_storage_w_m2"]))
        ),
        "realized_internal_exchange_std_w_m2": realized_std,
        "realized_internal_exchange_lag1_correlation": realized_lag_correlation,
        "statistics": statistics,
        "implementation_versions": {
            "numpy": importlib.metadata.version("numpy"),
            "scipy": importlib.metadata.version("scipy"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ebm-fixture", type=Path, default=DEFAULT_EBM_FIXTURE)
    parser.add_argument(
        "--stochastic-fixture",
        type=Path,
        default=DEFAULT_STOCHASTIC_FIXTURE,
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = summarize_stochastic_variability(
        load_ebm_fixture(args.ebm_fixture),
        load_stochastic_fixture(args.stochastic_fixture),
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    else:
        print(f"trajectory_digest: {result['trajectory_digest']}")
        print(
            "max budget residual W/m2: "
            f"{result['max_abs_total_budget_residual_w_m2']:.6e}"
        )
        print(f"state covariance K2: {result['statistics']['state_covariance_k2']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
