#!/usr/bin/env python3
"""Exact forcing-protocol reference for the linear two-layer EBM.

The forcing within each segment is affine in time. Propagation is exact up to
the matrix exponential by augmenting the physical state with forcing and a
constant coordinate. This gives one implementation path for held forcing,
ramps, overshoots, and reversals without step-size integration error.

The discovery/confirmation forcing domains in the fixture are benchmark
semantics for later fitted methods. This exact reference is not "trained" and
does not itself make an out-of-distribution generalization claim.
"""
from __future__ import annotations

import argparse
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
    advance_constant_forcing,
    energy_budget_residual_w_m2,
    equilibrium_state_k,
    load_fixture as load_ebm_fixture,
    parameters_from_fixture,
    system_matrix,
    total_heat_content_w_yr_m2,
)

DEFAULT_EBM_FIXTURE = ROOT / "fixtures" / "physics" / "two-layer-ebm-geoffroy-mean-v1.json"
DEFAULT_PROTOCOL_FIXTURE = ROOT / "fixtures" / "physics" / "two-layer-ebm-forcing-protocols-v1.json"


def load_protocol_fixture(path: Path = DEFAULT_PROTOCOL_FIXTURE) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported forcing-protocol fixture schema")
    return payload


def _finite(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _finite_state(state: np.ndarray | list[float]) -> np.ndarray:
    value = np.asarray(state, dtype=float)
    if value.shape != (2,) or not np.isfinite(value).all():
        raise ValueError("state must be a finite shape-(2,) vector")
    return value


def advance_affine_forcing(
    state: np.ndarray | list[float],
    parameters: TwoLayerParameters,
    forcing_start_w_m2: float,
    forcing_end_w_m2: float,
    dt_years: float,
) -> np.ndarray:
    """Advance exactly through one finite-duration affine forcing segment."""
    value = _finite_state(state)
    start = _finite("forcing_start_w_m2", forcing_start_w_m2)
    end = _finite("forcing_end_w_m2", forcing_end_w_m2)
    dt = _finite("dt_years", dt_years)
    if dt <= 0.0:
        raise ValueError("dt_years must be finite and positive for an affine segment")

    rate = (end - start) / dt
    matrix = np.zeros((4, 4), dtype=float)
    matrix[:2, :2] = system_matrix(parameters)
    matrix[0, 2] = 1.0 / parameters.surface_heat_capacity_w_yr_m2_k
    matrix[2, 3] = rate

    augmented = np.array([value[0], value[1], start, 1.0], dtype=float)
    propagated = expm(matrix * dt) @ augmented
    forcing_error = abs(float(propagated[2]) - end)
    forcing_scale = max(abs(start), abs(end), 1.0)
    if forcing_error > 64.0 * np.finfo(float).eps * forcing_scale:
        raise RuntimeError("affine forcing augmentation failed to reproduce segment endpoint")
    result = np.asarray(propagated[:2], dtype=float)
    if not np.isfinite(result).all():
        raise RuntimeError("affine forcing propagation became non-finite")
    return result


def _validate_domain(name: str, domain: Any) -> tuple[float, float]:
    if not isinstance(domain, list) or len(domain) != 2:
        raise ValueError(f"{name} must contain two endpoints")
    lower = _finite(f"{name}[0]", domain[0])
    upper = _finite(f"{name}[1]", domain[1])
    if not lower < upper:
        raise ValueError(f"{name} endpoints must be increasing")
    return lower, upper


def _validate_protocol(protocol: dict[str, Any]) -> None:
    if not isinstance(protocol.get("protocol_id"), str) or not protocol["protocol_id"]:
        raise ValueError("forcing protocol requires a non-empty protocol_id")
    _finite("initial_equilibrium_forcing_w_m2", protocol["initial_equilibrium_forcing_w_m2"])
    segments = protocol.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError(f"{protocol['protocol_id']} requires at least one forcing segment")
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise ValueError(f"{protocol['protocol_id']} segment {index} must be an object")
        duration = _finite("duration_years", segment["duration_years"])
        if duration <= 0.0:
            raise ValueError(f"{protocol['protocol_id']} segment {index} duration must be positive")
        _finite("forcing_start_w_m2", segment["forcing_start_w_m2"])
        _finite("forcing_end_w_m2", segment["forcing_end_w_m2"])


def simulate_protocol(
    protocol: dict[str, Any],
    parameters: TwoLayerParameters,
    *,
    samples_per_segment: int,
) -> dict[str, Any]:
    _validate_protocol(protocol)
    if not isinstance(samples_per_segment, int) or samples_per_segment < 2:
        raise ValueError("samples_per_segment must be an integer >= 2")

    initial_forcing = float(protocol["initial_equilibrium_forcing_w_m2"])
    state = equilibrium_state_k(parameters, initial_forcing)
    current_time = 0.0
    times: list[float] = []
    forcings: list[float] = []
    states: list[list[float]] = []
    segment_indices: list[int] = []

    def append_sample(time: float, forcing: float, sample_state: np.ndarray, segment: int) -> None:
        times.append(float(time))
        forcings.append(float(forcing))
        states.append([float(sample_state[0]), float(sample_state[1])])
        segment_indices.append(int(segment))

    segments = protocol["segments"]
    for segment_index, segment in enumerate(segments):
        duration = float(segment["duration_years"])
        forcing_start = float(segment["forcing_start_w_m2"])
        forcing_end = float(segment["forcing_end_w_m2"])
        segment_initial = np.array(state, copy=True)

        # A boundary sample is post-jump in forcing and pre-evolution in state.
        append_sample(current_time, forcing_start, segment_initial, segment_index)
        for sample_index in range(1, samples_per_segment + 1):
            fraction = sample_index / samples_per_segment
            local_time = duration * fraction
            local_forcing = forcing_start + fraction * (forcing_end - forcing_start)
            sample_state = advance_affine_forcing(
                segment_initial,
                parameters,
                forcing_start,
                local_forcing,
                local_time,
            )
            append_sample(
                current_time + local_time,
                local_forcing,
                sample_state,
                segment_index,
            )
        state = np.asarray(states[-1], dtype=float)
        current_time += duration

    state_array = np.asarray(states, dtype=float)
    forcing_array = np.asarray(forcings, dtype=float)
    residuals = np.array(
        [
            energy_budget_residual_w_m2(sample_state, parameters, forcing)
            for sample_state, forcing in zip(state_array, forcing_array)
        ],
        dtype=float,
    )
    if not np.isfinite(residuals).all():
        raise RuntimeError("forcing protocol produced non-finite energy-budget diagnostics")

    return {
        "protocol_id": protocol["protocol_id"],
        "time_years": times,
        "forcing_w_m2": forcings,
        "state_k": states,
        "segment_index": segment_indices,
        "max_abs_energy_budget_residual_w_m2": float(np.max(np.abs(residuals))),
        "peak_surface_temperature_anomaly_k": float(np.max(state_array[:, 0])),
        "minimum_surface_temperature_anomaly_k": float(np.min(state_array[:, 0])),
        "terminal_state_k": states[-1],
        "terminal_total_heat_content_w_yr_m2": total_heat_content_w_yr_m2(
            state_array[-1], parameters
        ),
    }


def analyze_protocols(
    ebm_fixture: dict[str, Any],
    protocol_fixture: dict[str, Any],
) -> dict[str, Any]:
    parameters = parameters_from_fixture(ebm_fixture)
    discovery = _validate_domain(
        "discovery_forcing_domain_w_m2",
        protocol_fixture["discovery_forcing_domain_w_m2"],
    )
    confirmation = _validate_domain(
        "confirmation_forcing_domain_w_m2",
        protocol_fixture["confirmation_forcing_domain_w_m2"],
    )
    if confirmation[0] > discovery[0] or confirmation[1] < discovery[1]:
        raise ValueError("confirmation forcing domain must contain the discovery domain")

    samples_per_segment = int(protocol_fixture["samples_per_segment"])
    protocols = protocol_fixture.get("protocols")
    if not isinstance(protocols, list) or not protocols:
        raise ValueError("forcing-protocol fixture must declare protocols")

    results: dict[str, Any] = {}
    ids: set[str] = set()
    all_forcings: list[float] = []
    max_budget_residual = 0.0
    for protocol in protocols:
        if not isinstance(protocol, dict):
            raise ValueError("forcing protocol must be an object")
        protocol_id = str(protocol.get("protocol_id", ""))
        if protocol_id in ids:
            raise ValueError(f"duplicate forcing protocol {protocol_id}")
        ids.add(protocol_id)
        result = simulate_protocol(
            protocol,
            parameters,
            samples_per_segment=samples_per_segment,
        )
        results[protocol_id] = result
        all_forcings.extend(result["forcing_w_m2"])
        max_budget_residual = max(
            max_budget_residual,
            float(result["max_abs_energy_budget_residual_w_m2"]),
        )

    if "constant_control" not in results:
        raise ValueError("forcing-protocol fixture requires constant_control")
    control = results["constant_control"]
    initial = equilibrium_state_k(parameters, 0.0)
    constant_error = 0.0
    for time, forcing, observed in zip(
        control["time_years"], control["forcing_w_m2"], control["state_k"]
    ):
        if not math.isclose(float(forcing), 1.0, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("constant_control must hold forcing at 1 W/m2")
        expected = advance_constant_forcing(initial, parameters, 1.0, float(time))
        constant_error = max(
            constant_error,
            float(np.max(np.abs(np.asarray(observed, dtype=float) - expected))),
        )

    observed_min = min(all_forcings)
    observed_max = max(all_forcings)
    if observed_min < confirmation[0] - 1e-12 or observed_max > confirmation[1] + 1e-12:
        raise ValueError("protocol forcing leaves the declared confirmation domain")

    discovery_abs = max(abs(discovery[0]), abs(discovery[1]))
    confirmation_abs = max(abs(confirmation[0]), abs(confirmation[1]))
    return {
        "fixture_id": protocol_fixture["fixture_id"],
        "ebm_fixture_id": ebm_fixture["fixture_id"],
        "discovery_forcing_domain_w_m2": list(discovery),
        "confirmation_forcing_domain_w_m2": list(confirmation),
        "held_out_absolute_forcing_margin_w_m2": confirmation_abs - discovery_abs,
        "constant_forcing_equivalence_max_abs_state_error_k": constant_error,
        "max_abs_energy_budget_residual_w_m2": max_budget_residual,
        "protocols": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ebm-fixture", type=Path, default=DEFAULT_EBM_FIXTURE)
    parser.add_argument("--protocol-fixture", type=Path, default=DEFAULT_PROTOCOL_FIXTURE)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = analyze_protocols(
        load_ebm_fixture(args.ebm_fixture),
        load_protocol_fixture(args.protocol_fixture),
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    else:
        print(f"fixture_id: {result['fixture_id']}")
        print(
            "constant equivalence max abs state error K: "
            f"{result['constant_forcing_equivalence_max_abs_state_error_k']:.3e}"
        )
        print(
            "max abs energy budget residual W/m2: "
            f"{result['max_abs_energy_budget_residual_w_m2']:.3e}"
        )
        for protocol_id, values in result["protocols"].items():
            print(
                f"{protocol_id}: terminal={values['terminal_state_k']} "
                f"peak_surface={values['peak_surface_temperature_anomaly_k']:.6f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
