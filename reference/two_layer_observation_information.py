#!/usr/bin/env python3
"""Observation-induced state and modal Fisher geometry for the two-layer EBM.

The reference keeps three responsibilities separate:

* the two-layer EBM owns the physical state, fluxes, and exact thermal modes;
* the observation-control fixture owns declared synthetic noise scales;
* NumPy supplies generic matrix algebra for an independent reference
  implementation of Gaussian-mean Fisher pullbacks and coordinate changes.

The synthetic noise magnitudes are structural controls, not calibrated observing
system uncertainties.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))

import numpy as np

from reference.two_layer_energy_balance import (
    DEFAULT_FIXTURE as DEFAULT_EBM_FIXTURE,
    load_fixture as load_ebm_fixture,
    mode_basis,
    parameters_from_fixture,
)


DEFAULT_OBSERVATION_FIXTURE = (
    ROOT / "fixtures" / "physics" / "two-layer-ebm-observation-control-v1.json"
)

OBSERVATION_SUBSETS: dict[str, tuple[str, ...]] = {
    "surface_only": ("surface_temperature",),
    "toa_only": ("toa_imbalance",),
    "ocean_only": ("ocean_heat_uptake",),
    "surface_plus_toa": ("surface_temperature", "toa_imbalance"),
    "surface_plus_ocean": ("surface_temperature", "ocean_heat_uptake"),
    "toa_plus_ocean": ("toa_imbalance", "ocean_heat_uptake"),
    "all_channels": (
        "surface_temperature",
        "toa_imbalance",
        "ocean_heat_uptake",
    ),
}

COORDINATE_PROBES = np.array(
    [
        [0.10, -0.05],
        [0.00, 0.10],
        [0.08, 0.03],
    ],
    dtype=float,
)


def load_observation_fixture(path: Path = DEFAULT_OBSERVATION_FIXTURE) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        fixture = json.load(handle)
    if fixture.get("schema_version") != 1:
        raise ValueError("unsupported observation-control fixture schema_version")
    if fixture.get("provenance", {}).get("kind") != "synthetic_structural_control":
        raise ValueError("observation-control fixture must declare synthetic structural provenance")
    channels = fixture.get("channels")
    if not isinstance(channels, dict) or not channels:
        raise ValueError("observation-control fixture must declare channels")
    return fixture


def _observation_row(
    channel_id: str,
    climate_feedback: float,
    ocean_heat_exchange: float,
) -> np.ndarray:
    if channel_id == "surface_temperature":
        return np.array([1.0, 0.0], dtype=float)
    if channel_id == "toa_imbalance":
        return np.array([-climate_feedback, 0.0], dtype=float)
    if channel_id == "ocean_heat_uptake":
        return np.array([ocean_heat_exchange, -ocean_heat_exchange], dtype=float)
    raise ValueError(f"unknown two-layer observation channel: {channel_id}")


def observation_jacobian(
    ebm_fixture: dict[str, Any],
    channel_ids: Iterable[str],
) -> np.ndarray:
    parameters = parameters_from_fixture(ebm_fixture)
    ids = tuple(channel_ids)
    if not ids:
        raise ValueError("at least one observation channel is required")
    rows = np.vstack(
        [
            _observation_row(
                channel_id,
                parameters.climate_feedback_w_m2_k,
                parameters.ocean_heat_exchange_w_m2_k,
            )
            for channel_id in ids
        ]
    )
    if not np.isfinite(rows).all():
        raise RuntimeError("observation Jacobian became non-finite")
    return rows


def flux_coordinate_jacobian(ebm_fixture: dict[str, Any]) -> np.ndarray:
    """Jacobian from temperature perturbations to `(TOA imbalance, ocean uptake)`."""
    parameters = parameters_from_fixture(ebm_fixture)
    matrix = np.array(
        [
            [-parameters.climate_feedback_w_m2_k, 0.0],
            [
                parameters.ocean_heat_exchange_w_m2_k,
                -parameters.ocean_heat_exchange_w_m2_k,
            ],
        ],
        dtype=float,
    )
    if not np.isfinite(matrix).all() or abs(float(np.linalg.det(matrix))) <= 1e-14:
        raise RuntimeError("two-layer flux coordinate map is singular or non-finite")
    return matrix


def observation_noise_stddev(
    observation_fixture: dict[str, Any],
    channel_ids: Iterable[str],
    *,
    noise_multiplier: float = 1.0,
) -> np.ndarray:
    multiplier = float(noise_multiplier)
    if not math.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError("noise_multiplier must be finite and positive")
    channels = observation_fixture["channels"]
    values: list[float] = []
    for channel_id in channel_ids:
        if channel_id not in channels:
            raise ValueError(f"missing observation-control channel: {channel_id}")
        sigma = float(channels[channel_id]["standard_deviation"])
        if not math.isfinite(sigma) or sigma <= 0.0:
            raise ValueError(f"invalid standard deviation for channel {channel_id}")
        values.append(multiplier * sigma)
    return np.asarray(values, dtype=float)


def gaussian_mean_fisher_reference(
    mean_jacobian: np.ndarray,
    observation_stddev: np.ndarray,
) -> dict[str, Any]:
    jacobian = np.asarray(mean_jacobian, dtype=float)
    noise = np.asarray(observation_stddev, dtype=float)
    if jacobian.ndim != 2 or jacobian.shape[0] == 0 or jacobian.shape[1] == 0:
        raise ValueError("mean_jacobian must be a nonempty matrix")
    if noise.shape != (jacobian.shape[0],):
        raise ValueError("observation_stddev must match Jacobian rows")
    if not np.isfinite(jacobian).all():
        raise ValueError("mean_jacobian must be finite")
    if not np.isfinite(noise).all() or np.any(noise <= 0.0):
        raise ValueError("observation standard deviations must be finite and positive")

    whitened = jacobian / noise[:, None]
    fisher = whitened.T @ whitened
    singular_values = np.linalg.svd(whitened, compute_uv=False)
    tolerance = (
        np.finfo(float).eps
        * max(whitened.shape)
        * float(singular_values[0])
        if singular_values.size
        else 0.0
    )
    rank = int(np.count_nonzero(singular_values > tolerance))
    nullity = int(jacobian.shape[1] - rank)
    condition_number = math.inf if nullity else float(np.linalg.cond(fisher))
    return {
        "fisher": fisher,
        "whitened_jacobian": whitened,
        "rank": rank,
        "nullity": nullity,
        "condition_number_2": condition_number,
    }


def metric_in_coordinates(
    state_metric: np.ndarray,
    state_from_coordinate_jacobian: np.ndarray,
) -> np.ndarray:
    """Express a state-space bilinear form in another local coordinate chart.

    If `delta_x = A delta_y`, then `G_y = A^T G_x A`.
    """
    metric = np.asarray(state_metric, dtype=float)
    jacobian = np.asarray(state_from_coordinate_jacobian, dtype=float)
    if metric.ndim != 2 or metric.shape[0] != metric.shape[1]:
        raise ValueError("state_metric must be square")
    if jacobian.ndim != 2 or jacobian.shape[0] != metric.shape[0]:
        raise ValueError("coordinate Jacobian must map into the state dimension")
    if not np.isfinite(metric).all() or not np.isfinite(jacobian).all():
        raise ValueError("metric and coordinate Jacobian must be finite")
    transformed = jacobian.T @ metric @ jacobian
    if not np.isfinite(transformed).all():
        raise RuntimeError("coordinate-transformed metric became non-finite")
    return transformed


def squared_metric_length(vector: np.ndarray, metric: np.ndarray) -> float:
    displacement = np.asarray(vector, dtype=float)
    bilinear = np.asarray(metric, dtype=float)
    if displacement.ndim != 1 or bilinear.shape != (displacement.size, displacement.size):
        raise ValueError("metric length dimensions do not agree")
    if not np.isfinite(displacement).all() or not np.isfinite(bilinear).all():
        raise ValueError("metric length inputs must be finite")
    value = float(displacement @ bilinear @ displacement)
    if value < -1e-12:
        raise ValueError("squared metric length must be nonnegative")
    return max(value, 0.0)


def _modal_diagnostics(fisher_state: np.ndarray, modal_basis: np.ndarray) -> dict[str, Any]:
    modal = metric_in_coordinates(fisher_state, modal_basis)
    diagonal = np.diag(modal)
    if np.any(diagonal < -1e-12):
        raise RuntimeError("modal Fisher diagonal must be nonnegative")
    fast_information = float(max(diagonal[0], 0.0))
    slow_information = float(max(diagonal[1], 0.0))
    denominator = math.sqrt(fast_information * slow_information)
    coupling = 0.0 if denominator == 0.0 else float(modal[0, 1] / denominator)
    return {
        "fisher": modal,
        "fast_mode_information": fast_information,
        "slow_mode_information": slow_information,
        "normalized_cross_mode_coupling": coupling,
    }


def coordinate_covariance_diagnostics(
    ebm_fixture: dict[str, Any],
    state_metric: np.ndarray,
) -> dict[str, Any]:
    parameters = parameters_from_fixture(ebm_fixture)
    _, modal_basis = mode_basis(parameters)
    state_to_flux = flux_coordinate_jacobian(ebm_fixture)
    flux_to_state = np.linalg.inv(state_to_flux)
    modal_to_state = modal_basis
    state_to_modal = np.linalg.inv(modal_basis)

    flux_metric = metric_in_coordinates(state_metric, flux_to_state)
    modal_metric = metric_in_coordinates(state_metric, modal_to_state)

    probes = []
    max_fisher_disagreement = 0.0
    euclidean_changes = []
    for state_delta in COORDINATE_PROBES:
        flux_delta = state_to_flux @ state_delta
        modal_delta = state_to_modal @ state_delta
        fisher_lengths = {
            "temperature_state": squared_metric_length(state_delta, state_metric),
            "heat_flux": squared_metric_length(flux_delta, flux_metric),
            "thermal_modes": squared_metric_length(modal_delta, modal_metric),
        }
        values = np.asarray(list(fisher_lengths.values()), dtype=float)
        disagreement = float(values.max() - values.min())
        max_fisher_disagreement = max(max_fisher_disagreement, disagreement)

        raw_euclidean = {
            "temperature_state": float(state_delta @ state_delta),
            "heat_flux": float(flux_delta @ flux_delta),
            "thermal_modes": float(modal_delta @ modal_delta),
        }
        euclidean_values = np.asarray(list(raw_euclidean.values()), dtype=float)
        euclidean_changes.append(float(euclidean_values.max() - euclidean_values.min()))
        probes.append(
            {
                "temperature_delta": state_delta.tolist(),
                "heat_flux_delta": flux_delta.tolist(),
                "thermal_mode_delta": modal_delta.tolist(),
                "fisher_squared_length": fisher_lengths,
                "raw_coordinate_euclidean_squared_norm": raw_euclidean,
                "fisher_max_absolute_disagreement": disagreement,
            }
        )

    millikelvin_from_kelvin = np.diag([1000.0, 1000.0])
    millikelvin_to_kelvin = np.linalg.inv(millikelvin_from_kelvin)
    millikelvin_metric = metric_in_coordinates(state_metric, millikelvin_to_kelvin)
    unit_probe_kelvin = COORDINATE_PROBES[0]
    unit_probe_millikelvin = millikelvin_from_kelvin @ unit_probe_kelvin
    unit_fisher_kelvin = squared_metric_length(unit_probe_kelvin, state_metric)
    unit_fisher_millikelvin = squared_metric_length(
        unit_probe_millikelvin, millikelvin_metric
    )

    return {
        "flux_coordinate_jacobian": state_to_flux.tolist(),
        "flux_fisher": flux_metric.tolist(),
        "modal_fisher": modal_metric.tolist(),
        "probes": probes,
        "max_fisher_squared_length_disagreement": max_fisher_disagreement,
        "min_raw_euclidean_squared_norm_spread": min(euclidean_changes),
        "unit_rescaling": {
            "kelvin_to_millikelvin_scale": 1000.0,
            "fisher_squared_length_kelvin": unit_fisher_kelvin,
            "fisher_squared_length_millikelvin": unit_fisher_millikelvin,
            "raw_euclidean_squared_norm_kelvin": float(
                unit_probe_kelvin @ unit_probe_kelvin
            ),
            "raw_euclidean_squared_norm_millikelvin": float(
                unit_probe_millikelvin @ unit_probe_millikelvin
            ),
        },
    }


def analyze_observation_geometry(
    ebm_fixture: dict[str, Any],
    observation_fixture: dict[str, Any],
    *,
    noise_multiplier: float = 1.0,
) -> dict[str, Any]:
    if observation_fixture.get("ebm_fixture_id") != ebm_fixture.get("fixture_id"):
        raise ValueError("observation-control fixture references a different EBM fixture")
    parameters = parameters_from_fixture(ebm_fixture)
    timescales, basis = mode_basis(parameters)
    subsets: dict[str, Any] = {}

    for subset_name, channel_ids in OBSERVATION_SUBSETS.items():
        jacobian = observation_jacobian(ebm_fixture, channel_ids)
        noise = observation_noise_stddev(
            observation_fixture,
            channel_ids,
            noise_multiplier=noise_multiplier,
        )
        state = gaussian_mean_fisher_reference(jacobian, noise)
        modal = _modal_diagnostics(state["fisher"], basis)
        subsets[subset_name] = {
            "channels": list(channel_ids),
            "state_fisher": state["fisher"].tolist(),
            "state_rank": state["rank"],
            "state_nullity": state["nullity"],
            "state_condition_number_2": state["condition_number_2"],
            "modal_fisher": modal["fisher"].tolist(),
            "fast_mode_information": modal["fast_mode_information"],
            "slow_mode_information": modal["slow_mode_information"],
            "normalized_cross_mode_coupling": modal[
                "normalized_cross_mode_coupling"
            ],
        }

    all_channel_metric = np.asarray(subsets["all_channels"]["state_fisher"], dtype=float)
    return {
        "ebm_fixture_id": ebm_fixture["fixture_id"],
        "observation_fixture_id": observation_fixture["fixture_id"],
        "noise_multiplier": float(noise_multiplier),
        "thermal_timescales_years": timescales.tolist(),
        "modal_basis": basis.tolist(),
        "subsets": subsets,
        "coordinate_covariance": coordinate_covariance_diagnostics(
            ebm_fixture,
            all_channel_metric,
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ebm-fixture", type=Path, default=DEFAULT_EBM_FIXTURE)
    parser.add_argument(
        "--observation-fixture", type=Path, default=DEFAULT_OBSERVATION_FIXTURE
    )
    parser.add_argument("--noise-multiplier", type=float, default=1.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = analyze_observation_geometry(
        load_ebm_fixture(args.ebm_fixture),
        load_observation_fixture(args.observation_fixture),
        noise_multiplier=args.noise_multiplier,
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        for name, diagnostics in result["subsets"].items():
            print(f"{name}: {diagnostics}")
        print(f"coordinate_covariance: {result['coordinate_covariance']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
