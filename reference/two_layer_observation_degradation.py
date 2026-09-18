#!/usr/bin/env python3
"""Observation-degradation rung for the forced two-layer EBM benchmark.

The underlying physical trajectories, discovery/confirmation forcing split, and
controlled-linear fitting method are inherited unchanged from the existing
forced-OOD rung. This module changes only the observation map.

Each representation declares a linear observation matrix plus named Gaussian
noise components. Reusing one component in multiple channels creates an exact
redundant measurement. A fixed Moore-Penrose inverse of the noiseless
observation matrix provides an explicit, non-learned state reconstruction
diagnostic. It does not claim that an unobserved state component is inferable
from one instantaneous measurement.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))

import numpy as np

from reference.two_layer_energy_balance import load_fixture as load_ebm_fixture
from reference.two_layer_forced_representation import (
    discovery_transition_samples,
    fit_controlled_linear_map,
    load_training_fixture,
    protocol_transition_samples,
)
from reference.two_layer_forcing_protocols import load_protocol_fixture


DEFAULT_EBM_FIXTURE = (
    ROOT / "fixtures" / "physics" / "two-layer-ebm-geoffroy-mean-v1.json"
)
DEFAULT_PROTOCOL_FIXTURE = (
    ROOT / "fixtures" / "physics" / "two-layer-ebm-forcing-protocols-v1.json"
)
DEFAULT_TRAINING_FIXTURE = (
    ROOT / "fixtures" / "physics" / "two-layer-ebm-forced-representation-v1.json"
)
DEFAULT_OBSERVATION_FIXTURE = (
    ROOT / "fixtures" / "physics" / "two-layer-ebm-observation-degradation-v1.json"
)


def load_observation_fixture(
    path: Path = DEFAULT_OBSERVATION_FIXTURE,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported observation-degradation fixture schema")
    representations = payload.get("representations")
    if not isinstance(representations, list) or len(representations) < 2:
        raise ValueError("observation fixture requires multiple representations")
    ids = [item.get("representation_id") for item in representations]
    if any(not isinstance(item, str) or not item for item in ids):
        raise ValueError("every observation representation requires an id")
    if len(ids) != len(set(ids)):
        raise ValueError("observation representation ids must be unique")
    noise_std = float(payload["noise_std_k"])
    rcond = float(payload["pseudoinverse_rcond"])
    if not math.isfinite(noise_std) or noise_std <= 0.0:
        raise ValueError("noise_std_k must be finite and positive")
    if not math.isfinite(rcond) or rcond <= 0.0:
        raise ValueError("pseudoinverse_rcond must be finite and positive")
    for key in ("discovery_noise_seed", "confirmation_noise_seed"):
        if not isinstance(payload.get(key), int):
            raise ValueError(f"{key} must be an integer")
    for representation in representations:
        _representation_matrix(representation)
    return payload


def _representation_matrix(spec: Mapping[str, Any]) -> np.ndarray:
    channels = spec.get("channels")
    if not isinstance(channels, list) or not channels:
        raise ValueError("observation representation requires channels")
    rows: list[list[float]] = []
    for channel in channels:
        if not isinstance(channel, dict):
            raise ValueError("observation channel must be an object")
        weights = channel.get("state_weights")
        if not isinstance(weights, list) or len(weights) != 2:
            raise ValueError("state_weights must contain two coefficients")
        row = [float(value) for value in weights]
        if not all(math.isfinite(value) for value in row):
            raise ValueError("observation weights must be finite")
        component = channel.get("noise_component")
        if component is not None and (
            not isinstance(component, str) or not component.strip()
        ):
            raise ValueError("noise_component must be null or a non-empty string")
        rows.append(row)
    matrix = np.asarray(rows, dtype=float)
    if not np.any(matrix):
        raise ValueError("observation matrix must contain a nonzero state weight")
    return matrix


def _noise_component_names(observation_fixture: Mapping[str, Any]) -> tuple[str, ...]:
    names = {
        channel["noise_component"]
        for representation in observation_fixture["representations"]
        for channel in representation["channels"]
        if channel.get("noise_component") is not None
    }
    return tuple(sorted(names))


def _draw_noise(
    rng: np.random.Generator,
    count: int,
    component_names: tuple[str, ...],
) -> dict[str, np.ndarray]:
    if count < 1:
        raise ValueError("noise draw count must be positive")
    return {
        name: rng.normal(0.0, 1.0, size=count)
        for name in component_names
    }


def _observe(
    spec: Mapping[str, Any],
    states: np.ndarray,
    noise: Mapping[str, np.ndarray],
    noise_std_k: float,
) -> np.ndarray:
    values = np.asarray(states, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError("physical states must be finite shape-(n,2) arrays")
    matrix = _representation_matrix(spec)
    observed = values @ matrix.T
    for column, channel in enumerate(spec["channels"]):
        component = channel.get("noise_component")
        if component is None:
            continue
        try:
            sample = np.asarray(noise[component], dtype=float)
        except KeyError as exc:
            raise ValueError(f"missing declared noise component {component!r}") from exc
        if sample.shape != (values.shape[0],) or not np.isfinite(sample).all():
            raise ValueError("noise component does not match physical sample count")
        observed[:, column] += noise_std_k * sample
    if not np.isfinite(observed).all():
        raise RuntimeError("observation map produced non-finite values")
    return observed


def _reconstruct(
    spec: Mapping[str, Any],
    observations: np.ndarray,
    *,
    rcond: float,
) -> np.ndarray:
    matrix = _representation_matrix(spec)
    reconstruction = np.linalg.pinv(matrix, rcond=rcond)
    observed = np.asarray(observations, dtype=float)
    if observed.ndim != 2 or observed.shape[1] != matrix.shape[0]:
        raise ValueError("observation array does not match representation dimension")
    states = observed @ reconstruction.T
    if not np.isfinite(states).all():
        raise RuntimeError("fixed observation reconstruction became non-finite")
    return states


def _predict_observations(
    fit: Mapping[str, Any],
    observations_x: np.ndarray,
    forcing_start: np.ndarray,
    forcing_rate: np.ndarray,
) -> np.ndarray:
    design = np.column_stack(
        [
            np.asarray(observations_x, dtype=float),
            np.asarray(forcing_start, dtype=float),
            np.asarray(forcing_rate, dtype=float),
        ]
    )
    predicted = design @ np.asarray(fit["weights"], dtype=float)
    if not np.isfinite(predicted).all():
        raise RuntimeError("controlled observation prediction became non-finite")
    return predicted


def _relative_error(predicted: np.ndarray, expected: np.ndarray) -> float:
    predicted = np.asarray(predicted, dtype=float)
    expected = np.asarray(expected, dtype=float)
    if predicted.shape != expected.shape or not np.isfinite(predicted).all():
        raise ValueError("relative-error arrays must be finite and shape matched")
    denominator = max(float(np.linalg.norm(expected)), np.finfo(float).tiny)
    return float(np.linalg.norm(predicted - expected) / denominator)


def analyze_observation_degradation(
    ebm_fixture: dict[str, Any],
    protocol_fixture: dict[str, Any],
    training_fixture: dict[str, Any],
    observation_fixture: dict[str, Any],
) -> dict[str, Any]:
    representations = observation_fixture["representations"]
    representation_ids = [item["representation_id"] for item in representations]
    required = {
        "clean_temperature_state",
        "noisy_temperature_state",
        "noisy_surface_scalar",
        "redundant_noisy_surface_pair",
    }
    if set(representation_ids) != required or len(representation_ids) != len(required):
        raise ValueError(
            "observation-degradation v1 representation identities changed "
            "without a fixture version"
        )

    discovery = discovery_transition_samples(
        ebm_fixture, protocol_fixture, training_fixture
    )
    dt = float(training_fixture["transition_dt_years"])
    noise_std = float(observation_fixture["noise_std_k"])
    rcond = float(observation_fixture["pseudoinverse_rcond"])
    component_names = _noise_component_names(observation_fixture)

    discovery_rng = np.random.default_rng(
        int(observation_fixture["discovery_noise_seed"])
    )
    discovery_noise_x = _draw_noise(
        discovery_rng, discovery["states_x"].shape[0], component_names
    )
    discovery_noise_y = _draw_noise(
        discovery_rng, discovery["states_y"].shape[0], component_names
    )

    protocol_map = {
        item["protocol_id"]: item for item in protocol_fixture.get("protocols", [])
    }
    confirmation_ids = training_fixture.get("confirmation_protocol_ids")
    if not isinstance(confirmation_ids, list) or not confirmation_ids:
        raise ValueError("confirmation protocol identities are required")
    confirmation_rng = np.random.default_rng(
        int(observation_fixture["confirmation_noise_seed"])
    )
    confirmation: dict[str, dict[str, Any]] = {}
    for protocol_id in confirmation_ids:
        try:
            protocol = protocol_map[protocol_id]
        except KeyError as exc:
            raise ValueError(
                f"confirmation protocol does not resolve: {protocol_id}"
            ) from exc
        samples = protocol_transition_samples(
            protocol, ebm_fixture, dt_years=dt
        )
        confirmation[protocol_id] = {
            "samples": samples,
            "noise_x": _draw_noise(
                confirmation_rng, samples["states_x"].shape[0], component_names
            ),
            "noise_y": _draw_noise(
                confirmation_rng, samples["states_y"].shape[0], component_names
            ),
        }

    result: dict[str, Any] = {
        "fixture_id": observation_fixture["fixture_id"],
        "ebm_fixture_id": ebm_fixture["fixture_id"],
        "forcing_protocol_fixture_id": protocol_fixture["fixture_id"],
        "training_fixture_id": training_fixture["fixture_id"],
        "transition_dt_years": dt,
        "noise_std_k": noise_std,
        "discovery_noise_seed": int(observation_fixture["discovery_noise_seed"]),
        "confirmation_noise_seed": int(
            observation_fixture["confirmation_noise_seed"]
        ),
        "implementation_versions": {
            "numpy": importlib.metadata.version("numpy"),
            "scipy": importlib.metadata.version("scipy"),
        },
        "representations": {},
    }

    for spec in representations:
        representation_id = spec["representation_id"]
        matrix = _representation_matrix(spec)
        structural_rank = int(np.linalg.matrix_rank(matrix))

        train_x = _observe(
            spec, discovery["states_x"], discovery_noise_x, noise_std
        )
        train_y = _observe(
            spec, discovery["states_y"], discovery_noise_y, noise_std
        )
        fit = fit_controlled_linear_map(
            train_x,
            train_y,
            discovery["forcing_start_w_m2"],
            discovery["forcing_rate_w_m2_per_year"],
        )
        discovery_reconstruction = _reconstruct(
            spec, train_x, rcond=rcond
        )
        discovery_predicted_observations = _predict_observations(
            fit,
            train_x,
            discovery["forcing_start_w_m2"],
            discovery["forcing_rate_w_m2_per_year"],
        )
        discovery_predicted_state = _reconstruct(
            spec, discovery_predicted_observations, rcond=rcond
        )

        reconstructed_confirmation: list[np.ndarray] = []
        reconstruction_targets: list[np.ndarray] = []
        predicted_confirmation: list[np.ndarray] = []
        prediction_targets: list[np.ndarray] = []
        per_protocol: dict[str, Any] = {}
        for protocol_id in confirmation_ids:
            item = confirmation[protocol_id]
            samples = item["samples"]
            observed_x = _observe(
                spec, samples["states_x"], item["noise_x"], noise_std
            )
            reconstructed_x = _reconstruct(spec, observed_x, rcond=rcond)
            predicted_observed_y = _predict_observations(
                fit,
                observed_x,
                samples["forcing_start_w_m2"],
                samples["forcing_rate_w_m2_per_year"],
            )
            predicted_state_y = _reconstruct(
                spec, predicted_observed_y, rcond=rcond
            )
            state_reconstruction_error = _relative_error(
                reconstructed_x, samples["states_x"]
            )
            forced_state_prediction_error = _relative_error(
                predicted_state_y, samples["states_y"]
            )
            per_protocol[protocol_id] = {
                "sample_count": int(samples["states_x"].shape[0]),
                "state_reconstruction_relative_error": state_reconstruction_error,
                "forced_state_prediction_relative_error": forced_state_prediction_error,
            }
            reconstructed_confirmation.append(reconstructed_x)
            reconstruction_targets.append(samples["states_x"])
            predicted_confirmation.append(predicted_state_y)
            prediction_targets.append(samples["states_y"])

        confirmation_state_reconstruction_error = _relative_error(
            np.vstack(reconstructed_confirmation),
            np.vstack(reconstruction_targets),
        )
        confirmation_forced_state_prediction_error = _relative_error(
            np.vstack(predicted_confirmation),
            np.vstack(prediction_targets),
        )
        result["representations"][representation_id] = {
            "observation_dimension": int(matrix.shape[0]),
            "structural_observation_rank": structural_rank,
            "redundant_dimension_count": int(matrix.shape[0] - structural_rank),
            "design_dimension": int(fit["design_dimension"]),
            "design_rank": int(fit["design_rank"]),
            "condition_number_status": fit["condition_number_status"],
            "condition_number": fit["condition_number"],
            "discovery_state_reconstruction_relative_error": _relative_error(
                discovery_reconstruction, discovery["states_x"]
            ),
            "discovery_forced_state_prediction_relative_error": _relative_error(
                discovery_predicted_state, discovery["states_y"]
            ),
            "confirmation_state_reconstruction_relative_error": (
                confirmation_state_reconstruction_error
            ),
            "confirmation_forced_state_prediction_relative_error": (
                confirmation_forced_state_prediction_error
            ),
            "protocols": per_protocol,
        }

    surface = result["representations"]["noisy_surface_scalar"]
    redundant = result["representations"]["redundant_noisy_surface_pair"]
    noisy_full = result["representations"]["noisy_temperature_state"]
    clean = result["representations"]["clean_temperature_state"]
    result["redundant_structural_rank_gain"] = int(
        redundant["structural_observation_rank"]
        - surface["structural_observation_rank"]
    )
    result["redundant_vs_surface_state_reconstruction_error_delta"] = float(
        redundant["confirmation_state_reconstruction_relative_error"]
        - surface["confirmation_state_reconstruction_relative_error"]
    )
    result["redundant_vs_surface_forced_prediction_error_delta"] = float(
        redundant["confirmation_forced_state_prediction_relative_error"]
        - surface["confirmation_forced_state_prediction_relative_error"]
    )
    result["noisy_full_vs_surface_forced_prediction_error_gap"] = float(
        surface["confirmation_forced_state_prediction_relative_error"]
        - noisy_full["confirmation_forced_state_prediction_relative_error"]
    )
    result["clean_control_forced_prediction_relative_error"] = float(
        clean["confirmation_forced_state_prediction_relative_error"]
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ebm-fixture", type=Path, default=DEFAULT_EBM_FIXTURE)
    parser.add_argument(
        "--protocol-fixture", type=Path, default=DEFAULT_PROTOCOL_FIXTURE
    )
    parser.add_argument(
        "--training-fixture", type=Path, default=DEFAULT_TRAINING_FIXTURE
    )
    parser.add_argument(
        "--observation-fixture", type=Path, default=DEFAULT_OBSERVATION_FIXTURE
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = analyze_observation_degradation(
        load_ebm_fixture(args.ebm_fixture),
        load_protocol_fixture(args.protocol_fixture),
        load_training_fixture(args.training_fixture),
        load_observation_fixture(args.observation_fixture),
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    else:
        for name, diagnostics in result["representations"].items():
            print(
                f"{name}: reconstruct="
                f"{diagnostics['confirmation_state_reconstruction_relative_error']:.6e} "
                "forced="
                f"{diagnostics['confirmation_forced_state_prediction_relative_error']:.6e} "
                f"rank={diagnostics['structural_observation_rank']}/"
                f"{diagnostics['observation_dimension']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
