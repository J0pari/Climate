#!/usr/bin/env python3
"""Discovery-trained controlled-linear representation benchmark for the two-layer EBM.

The model class is intentionally ordinary: ordinary least squares identifies a
fixed-step controlled linear map from representation coordinates plus forcing
start/rate controls. Training forcing endpoints are sampled only from the
fixture's discovery domain. Confirmation uses immutable forcing protocols that
leave that domain.

This benchmark is designed to let representations lose. The full two-layer
state is Markov-closed under the declared controls and should extrapolate
essentially exactly because the physical reference is linear. Surface
temperature alone omits the deep-reservoir state and therefore cannot become
Markov-closed merely by exposing the same forcing controls.
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

from reference.two_layer_energy_balance import (
    equilibrium_state_k,
    load_fixture as load_ebm_fixture,
    parameters_from_fixture,
)
from reference.two_layer_forcing_protocols import (
    advance_affine_forcing,
    load_protocol_fixture,
)

DEFAULT_EBM_FIXTURE = ROOT / "fixtures" / "physics" / "two-layer-ebm-geoffroy-mean-v1.json"
DEFAULT_PROTOCOL_FIXTURE = ROOT / "fixtures" / "physics" / "two-layer-ebm-forcing-protocols-v1.json"
DEFAULT_TRAINING_FIXTURE = ROOT / "fixtures" / "physics" / "two-layer-ebm-forced-representation-v1.json"


def load_training_fixture(path: Path = DEFAULT_TRAINING_FIXTURE) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported forced-representation fixture schema")
    return payload


def _finite_pair(name: str, value: Any) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must contain exactly two endpoints")
    lower, upper = float(value[0]), float(value[1])
    if not math.isfinite(lower) or not math.isfinite(upper) or not lower < upper:
        raise ValueError(f"{name} must contain increasing finite endpoints")
    return lower, upper


def _representation(name: str, states: np.ndarray) -> np.ndarray:
    values = np.asarray(states, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError("physical state samples must be finite shape-(n,2) arrays")
    if name == "temperature_state":
        return values
    if name == "surface_temperature_scalar":
        return values[:, :1]
    raise ValueError(f"unknown forced-representation view {name!r}")


def discovery_transition_samples(
    ebm_fixture: dict[str, Any],
    protocol_fixture: dict[str, Any],
    training_fixture: dict[str, Any],
) -> dict[str, np.ndarray]:
    parameters = parameters_from_fixture(ebm_fixture)
    count = int(training_fixture["training_sample_count"])
    if count < 32:
        raise ValueError("training_sample_count must be at least 32")
    dt = float(training_fixture["transition_dt_years"])
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("transition_dt_years must be finite and positive")
    state_lower, state_upper = _finite_pair("state_bounds_k", training_fixture["state_bounds_k"])
    forcing_lower, forcing_upper = _finite_pair(
        "discovery_forcing_domain_w_m2",
        protocol_fixture["discovery_forcing_domain_w_m2"],
    )

    rng = np.random.default_rng(int(training_fixture["seed"]))
    states_x = rng.uniform(state_lower, state_upper, size=(count, 2))
    forcing_start = rng.uniform(forcing_lower, forcing_upper, size=count)
    forcing_end = rng.uniform(forcing_lower, forcing_upper, size=count)
    states_y = np.vstack(
        [
            advance_affine_forcing(state, parameters, start, end, dt)
            for state, start, end in zip(states_x, forcing_start, forcing_end)
        ]
    )
    forcing_rate = (forcing_end - forcing_start) / dt

    if (
        np.min(forcing_start) < forcing_lower
        or np.max(forcing_start) > forcing_upper
        or np.min(forcing_end) < forcing_lower
        or np.max(forcing_end) > forcing_upper
    ):
        raise RuntimeError("discovery sampler escaped its declared forcing domain")

    return {
        "states_x": states_x,
        "states_y": states_y,
        "forcing_start_w_m2": forcing_start,
        "forcing_rate_w_m2_per_year": forcing_rate,
    }


def fit_controlled_linear_map(
    x_representation: np.ndarray,
    y_representation: np.ndarray,
    forcing_start_w_m2: np.ndarray,
    forcing_rate_w_m2_per_year: np.ndarray,
) -> dict[str, Any]:
    x = np.asarray(x_representation, dtype=float)
    y = np.asarray(y_representation, dtype=float)
    forcing_start = np.asarray(forcing_start_w_m2, dtype=float)
    forcing_rate = np.asarray(forcing_rate_w_m2_per_year, dtype=float)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("representation transitions must be matched two-dimensional arrays")
    if forcing_start.shape != (x.shape[0],) or forcing_rate.shape != (x.shape[0],):
        raise ValueError("forcing controls must match transition sample count")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("representation transitions must be finite")

    design = np.column_stack([x, forcing_start, forcing_rate])
    weights, _, rank, singular_values = np.linalg.lstsq(design, y, rcond=None)
    predicted = design @ weights
    denominator = max(float(np.linalg.norm(y)), np.finfo(float).tiny)
    relative_error = float(np.linalg.norm(predicted - y) / denominator)
    max_abs_error = float(np.max(np.abs(predicted - y)))
    singular_values = np.asarray(singular_values, dtype=float)
    if singular_values.size == 0 or not np.isfinite(singular_values).all():
        raise RuntimeError("least-squares fit returned invalid singular values")
    tolerance = (
        np.finfo(float).eps
        * max(design.shape)
        * float(np.max(singular_values))
    )
    numerical_rank = int(np.count_nonzero(singular_values > tolerance))
    if numerical_rank != int(rank):
        raise RuntimeError("least-squares and explicit numerical rank diagnostics disagree")
    if numerical_rank < design.shape[1]:
        condition_status = "rank_deficient"
        condition_number = None
    else:
        condition_status = "finite"
        condition_number = float(singular_values[0] / singular_values[-1])
        if not math.isfinite(condition_number):
            raise RuntimeError("full-rank design produced non-finite condition number")

    return {
        "weights": np.asarray(weights, dtype=float),
        "design_dimension": int(design.shape[1]),
        "design_rank": numerical_rank,
        "rank_tolerance": float(tolerance),
        "condition_number_status": condition_status,
        "condition_number": condition_number,
        "training_relative_error": relative_error,
        "training_max_abs_error": max_abs_error,
    }


def protocol_transition_samples(
    protocol: dict[str, Any],
    ebm_fixture: dict[str, Any],
    *,
    dt_years: float,
) -> dict[str, np.ndarray]:
    parameters = parameters_from_fixture(ebm_fixture)
    dt = float(dt_years)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt_years must be finite and positive")

    state = equilibrium_state_k(
        parameters, float(protocol["initial_equilibrium_forcing_w_m2"])
    )
    states_x: list[np.ndarray] = []
    states_y: list[np.ndarray] = []
    forcing_start_values: list[float] = []
    forcing_rate_values: list[float] = []

    segments = protocol.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("confirmation protocol requires segments")
    for segment_index, segment in enumerate(segments):
        duration = float(segment["duration_years"])
        steps_float = duration / dt
        steps = int(round(steps_float))
        if steps < 1 or not math.isclose(steps_float, steps, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"protocol segment {segment_index} duration must be an integer multiple of dt"
            )
        segment_start = float(segment["forcing_start_w_m2"])
        segment_end = float(segment["forcing_end_w_m2"])
        if not math.isfinite(segment_start) or not math.isfinite(segment_end):
            raise ValueError("protocol forcing must be finite")

        for step in range(steps):
            fraction_start = step / steps
            fraction_end = (step + 1) / steps
            forcing_start = segment_start + fraction_start * (segment_end - segment_start)
            forcing_end = segment_start + fraction_end * (segment_end - segment_start)
            next_state = advance_affine_forcing(
                state, parameters, forcing_start, forcing_end, dt
            )
            states_x.append(np.array(state, copy=True))
            states_y.append(np.array(next_state, copy=True))
            forcing_start_values.append(forcing_start)
            forcing_rate_values.append((forcing_end - forcing_start) / dt)
            state = next_state

    return {
        "states_x": np.vstack(states_x),
        "states_y": np.vstack(states_y),
        "forcing_start_w_m2": np.asarray(forcing_start_values, dtype=float),
        "forcing_rate_w_m2_per_year": np.asarray(forcing_rate_values, dtype=float),
    }


def _predict(
    fit: dict[str, Any],
    representation_x: np.ndarray,
    forcing_start: np.ndarray,
    forcing_rate: np.ndarray,
) -> np.ndarray:
    design = np.column_stack(
        [
            np.asarray(representation_x, dtype=float),
            np.asarray(forcing_start, dtype=float),
            np.asarray(forcing_rate, dtype=float),
        ]
    )
    predicted = design @ np.asarray(fit["weights"], dtype=float)
    if not np.isfinite(predicted).all():
        raise RuntimeError("controlled linear prediction became non-finite")
    return predicted


def analyze_forced_representations(
    ebm_fixture: dict[str, Any],
    protocol_fixture: dict[str, Any],
    training_fixture: dict[str, Any],
) -> dict[str, Any]:
    discovery = discovery_transition_samples(
        ebm_fixture, protocol_fixture, training_fixture
    )
    dt = float(training_fixture["transition_dt_years"])
    representations = training_fixture.get("representations")
    if representations != ["temperature_state", "surface_temperature_scalar"]:
        raise ValueError("benchmark representation identities changed without a fixture version")

    protocol_map = {
        item["protocol_id"]: item for item in protocol_fixture.get("protocols", [])
    }
    confirmation_ids = training_fixture.get("confirmation_protocol_ids")
    if not isinstance(confirmation_ids, list) or not confirmation_ids:
        raise ValueError("confirmation protocol identities are required")
    for protocol_id in confirmation_ids:
        if protocol_id not in protocol_map:
            raise ValueError(f"confirmation protocol does not resolve: {protocol_id}")

    result: dict[str, Any] = {
        "fixture_id": training_fixture["fixture_id"],
        "ebm_fixture_id": ebm_fixture["fixture_id"],
        "forcing_protocol_fixture_id": protocol_fixture["fixture_id"],
        "transition_dt_years": dt,
        "training_sample_count": int(training_fixture["training_sample_count"]),
        "implementation_versions": {
            "numpy": importlib.metadata.version("numpy"),
            "scipy": importlib.metadata.version("scipy"),
        },
        "representations": {},
    }

    for representation_name in representations:
        train_x = _representation(representation_name, discovery["states_x"])
        train_y = _representation(representation_name, discovery["states_y"])
        fit = fit_controlled_linear_map(
            train_x,
            train_y,
            discovery["forcing_start_w_m2"],
            discovery["forcing_rate_w_m2_per_year"],
        )
        confirmation_predictions: list[np.ndarray] = []
        confirmation_targets: list[np.ndarray] = []
        per_protocol: dict[str, Any] = {}
        for protocol_id in confirmation_ids:
            samples = protocol_transition_samples(
                protocol_map[protocol_id], ebm_fixture, dt_years=dt
            )
            rep_x = _representation(representation_name, samples["states_x"])
            rep_y = _representation(representation_name, samples["states_y"])
            predicted = _predict(
                fit,
                rep_x,
                samples["forcing_start_w_m2"],
                samples["forcing_rate_w_m2_per_year"],
            )
            denominator = max(float(np.linalg.norm(rep_y)), np.finfo(float).tiny)
            relative_error = float(np.linalg.norm(predicted - rep_y) / denominator)
            max_abs_error = float(np.max(np.abs(predicted - rep_y)))
            per_protocol[protocol_id] = {
                "sample_count": int(rep_y.shape[0]),
                "relative_error": relative_error,
                "max_abs_error": max_abs_error,
                "forcing_min_w_m2": float(np.min(samples["forcing_start_w_m2"])),
                "forcing_max_w_m2": float(np.max(samples["forcing_start_w_m2"])),
            }
            confirmation_predictions.append(predicted)
            confirmation_targets.append(rep_y)

        all_predictions = np.vstack(confirmation_predictions)
        all_targets = np.vstack(confirmation_targets)
        denominator = max(float(np.linalg.norm(all_targets)), np.finfo(float).tiny)
        confirmation_relative_error = float(
            np.linalg.norm(all_predictions - all_targets) / denominator
        )
        confirmation_max_abs_error = float(
            np.max(np.abs(all_predictions - all_targets))
        )
        result["representations"][representation_name] = {
            "representation_dimension": int(train_x.shape[1]),
            "design_dimension": fit["design_dimension"],
            "design_rank": fit["design_rank"],
            "rank_tolerance": fit["rank_tolerance"],
            "condition_number_status": fit["condition_number_status"],
            "condition_number": fit["condition_number"],
            "training_relative_error": fit["training_relative_error"],
            "training_max_abs_error": fit["training_max_abs_error"],
            "confirmation_relative_error": confirmation_relative_error,
            "confirmation_max_abs_error": confirmation_max_abs_error,
            "protocols": per_protocol,
        }

    full = result["representations"]["temperature_state"]
    scalar = result["representations"]["surface_temperature_scalar"]
    result["closure_gap_confirmation_relative_error"] = float(
        scalar["confirmation_relative_error"] - full["confirmation_relative_error"]
    )
    discovery_domain = protocol_fixture["discovery_forcing_domain_w_m2"]
    confirmation_domain = protocol_fixture["confirmation_forcing_domain_w_m2"]
    result["held_out_absolute_forcing_margin_w_m2"] = float(
        max(abs(float(confirmation_domain[0])), abs(float(confirmation_domain[1])))
        - max(abs(float(discovery_domain[0])), abs(float(discovery_domain[1])))
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ebm-fixture", type=Path, default=DEFAULT_EBM_FIXTURE)
    parser.add_argument("--protocol-fixture", type=Path, default=DEFAULT_PROTOCOL_FIXTURE)
    parser.add_argument("--training-fixture", type=Path, default=DEFAULT_TRAINING_FIXTURE)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = analyze_forced_representations(
        load_ebm_fixture(args.ebm_fixture),
        load_protocol_fixture(args.protocol_fixture),
        load_training_fixture(args.training_fixture),
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    else:
        for name, diagnostics in result["representations"].items():
            print(
                f"{name}: train={diagnostics['training_relative_error']:.6e} "
                f"confirm={diagnostics['confirmation_relative_error']:.6e}"
            )
        print(
            "closure gap: "
            f"{result['closure_gap_confirmation_relative_error']:.6e}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
