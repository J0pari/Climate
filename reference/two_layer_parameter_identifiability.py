#!/usr/bin/env python3
"""Local parameter-sensitivity rank benchmark for the two-layer EBM.

This rung separates structural parameter degeneracy from numerical derivative
failure.  It perturbs the four positive EBM parameters multiplicatively in
log coordinates and compares two observation designs:

* equilibrium two-temperature states at several fixed forcing levels;
* transient two-temperature trajectories under the existing held-out forcing
  protocols.

At equilibrium the two-layer model depends on climate feedback but not on
either heat capacity or ocean exchange, so three sensitivity columns are
exactly zero.  Transient forcing excites storage and exchange timescales and can
locally distinguish additional parameter directions.

Full column rank here is a local sensitivity statement at one declared
parameter point under one declared excitation set.  It is not global
identifiability, observational calibration, or a posterior uncertainty claim.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))

import numpy as np

from reference.two_layer_energy_balance import (
    TwoLayerParameters,
    equilibrium_state_k,
    load_fixture as load_ebm_fixture,
    parameters_from_fixture,
)
from reference.two_layer_forcing_protocols import (
    load_protocol_fixture,
    simulate_protocol,
)


DEFAULT_EBM_FIXTURE = (
    ROOT / "fixtures" / "physics" / "two-layer-ebm-geoffroy-mean-v1.json"
)
DEFAULT_PROTOCOL_FIXTURE = (
    ROOT / "fixtures" / "physics" / "two-layer-ebm-forcing-protocols-v1.json"
)
DEFAULT_PARAMETER_FIXTURE = (
    ROOT / "fixtures" / "physics" / "two-layer-ebm-parameter-identifiability-v1.json"
)

PARAMETER_IDS = (
    "surface_heat_capacity_w_yr_m2_k",
    "deep_heat_capacity_w_yr_m2_k",
    "ocean_heat_exchange_w_m2_k",
    "climate_feedback_w_m2_k",
)


def load_parameter_fixture(
    path: Path = DEFAULT_PARAMETER_FIXTURE,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported parameter-identifiability fixture schema")
    if payload.get("parameter_ids") != list(PARAMETER_IDS):
        raise ValueError(
            "parameter-identifiability v1 parameter identities changed "
            "without a fixture version"
        )
    step = float(payload["log_parameter_step"])
    half_multiplier = float(payload["convergence_half_step_multiplier"])
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("log_parameter_step must be finite and positive")
    if (
        not math.isfinite(half_multiplier)
        or half_multiplier <= 0.0
        or half_multiplier >= 1.0
    ):
        raise ValueError(
            "convergence_half_step_multiplier must be finite and in (0, 1)"
        )
    forcings = payload.get("equilibrium_forcing_w_m2")
    if (
        not isinstance(forcings, list)
        or len(forcings) < 2
        or not all(math.isfinite(float(value)) for value in forcings)
        or not any(float(value) != 0.0 for value in forcings)
    ):
        raise ValueError("equilibrium_forcing_w_m2 must contain finite forcing probes")
    protocols = payload.get("transient_protocol_ids")
    if (
        not isinstance(protocols, list)
        or not protocols
        or not all(isinstance(value, str) and value for value in protocols)
        or len(protocols) != len(set(protocols))
    ):
        raise ValueError("transient_protocol_ids must be unique non-empty strings")
    return payload


def _parameter_vector(parameters: TwoLayerParameters) -> np.ndarray:
    values = np.array(
        [
            parameters.surface_heat_capacity_w_yr_m2_k,
            parameters.deep_heat_capacity_w_yr_m2_k,
            parameters.ocean_heat_exchange_w_m2_k,
            parameters.climate_feedback_w_m2_k,
        ],
        dtype=float,
    )
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        raise ValueError("two-layer sensitivity parameters must be finite and positive")
    return values


def _parameters_from_vector(values: np.ndarray) -> TwoLayerParameters:
    vector = np.asarray(values, dtype=float)
    if vector.shape != (4,) or not np.isfinite(vector).all() or np.any(vector <= 0.0):
        raise ValueError("parameter vector must contain four finite positive values")
    return TwoLayerParameters(
        surface_heat_capacity_w_yr_m2_k=float(vector[0]),
        deep_heat_capacity_w_yr_m2_k=float(vector[1]),
        ocean_heat_exchange_w_m2_k=float(vector[2]),
        climate_feedback_w_m2_k=float(vector[3]),
    )


def _equilibrium_outputs(
    parameters: TwoLayerParameters,
    forcing_values: list[float],
) -> np.ndarray:
    output = np.concatenate(
        [
            equilibrium_state_k(parameters, float(forcing))
            for forcing in forcing_values
        ]
    )
    if not np.isfinite(output).all():
        raise RuntimeError("equilibrium sensitivity output became non-finite")
    return output


def _transient_outputs(
    parameters: TwoLayerParameters,
    protocol_fixture: dict[str, Any],
    protocol_ids: list[str],
) -> np.ndarray:
    protocol_map = {
        item["protocol_id"]: item for item in protocol_fixture.get("protocols", [])
    }
    samples_per_segment = int(protocol_fixture["samples_per_segment"])
    outputs: list[np.ndarray] = []
    for protocol_id in protocol_ids:
        try:
            protocol = protocol_map[protocol_id]
        except KeyError as exc:
            raise ValueError(
                f"parameter-identifiability protocol does not resolve: {protocol_id}"
            ) from exc
        result = simulate_protocol(
            protocol,
            parameters,
            samples_per_segment=samples_per_segment,
        )
        states = np.asarray(result["state_k"], dtype=float)
        if states.ndim != 2 or states.shape[1] != 2 or not np.isfinite(states).all():
            raise RuntimeError("transient sensitivity protocol emitted invalid states")
        outputs.append(states.reshape(-1))
    if not outputs:
        raise ValueError("transient parameter sensitivity requires protocols")
    return np.concatenate(outputs)


def centered_log_parameter_jacobian(
    parameters: TwoLayerParameters,
    output_function: Callable[[TwoLayerParameters], np.ndarray],
    *,
    log_step: float,
) -> np.ndarray:
    step = float(log_step)
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("log_step must be finite and positive")
    base = _parameter_vector(parameters)
    columns: list[np.ndarray] = []
    expected_shape: tuple[int, ...] | None = None
    for index in range(base.size):
        plus = np.array(base, copy=True)
        minus = np.array(base, copy=True)
        plus[index] *= math.exp(step)
        minus[index] *= math.exp(-step)
        high = np.asarray(output_function(_parameters_from_vector(plus)), dtype=float)
        low = np.asarray(output_function(_parameters_from_vector(minus)), dtype=float)
        if high.ndim != 1 or high.shape != low.shape:
            raise ValueError("sensitivity outputs must be matched one-dimensional arrays")
        if expected_shape is None:
            expected_shape = high.shape
        elif high.shape != expected_shape:
            raise ValueError("sensitivity output dimension changed across perturbations")
        derivative = (high - low) / (2.0 * step)
        if not np.isfinite(derivative).all():
            raise RuntimeError("parameter sensitivity derivative became non-finite")
        columns.append(derivative)
    return np.column_stack(columns)


def _matrix_diagnostics(matrix: np.ndarray) -> dict[str, Any]:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(PARAMETER_IDS):
        raise ValueError("parameter sensitivity matrix must have four columns")
    if not np.isfinite(values).all():
        raise ValueError("parameter sensitivity matrix must be finite")
    singular_values = np.linalg.svd(values, compute_uv=False)
    largest = float(singular_values[0]) if singular_values.size else 0.0
    tolerance = (
        np.finfo(float).eps * max(values.shape) * largest
        if largest > 0.0
        else 0.0
    )
    rank = int(np.count_nonzero(singular_values > tolerance))
    if rank == values.shape[1]:
        condition_status = "finite"
        condition_number = float(singular_values[0] / singular_values[-1])
        if not math.isfinite(condition_number):
            raise RuntimeError("full-rank sensitivity matrix has non-finite condition number")
    else:
        condition_status = "rank_deficient"
        condition_number = None

    column_norms = np.linalg.norm(values, axis=0)
    exactly_zero = [
        PARAMETER_IDS[index]
        for index, norm in enumerate(column_norms)
        if float(norm) == 0.0
    ]
    return {
        "output_dimension": int(values.shape[0]),
        "parameter_dimension": int(values.shape[1]),
        "rank": rank,
        "nullity": int(values.shape[1] - rank),
        "rank_tolerance": float(tolerance),
        "singular_values": [float(value) for value in singular_values],
        "condition_number_status": condition_status,
        "condition_number": condition_number,
        "column_norms": {
            parameter_id: float(column_norms[index])
            for index, parameter_id in enumerate(PARAMETER_IDS)
        },
        "exactly_zero_sensitivity_parameters": exactly_zero,
    }


def _relative_frobenius_disagreement(
    primary: np.ndarray,
    refined: np.ndarray,
) -> float:
    left = np.asarray(primary, dtype=float)
    right = np.asarray(refined, dtype=float)
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("sensitivity matrices must be finite and shape matched")
    denominator = max(float(np.linalg.norm(right)), np.finfo(float).tiny)
    return float(np.linalg.norm(left - right) / denominator)


def analyze_parameter_identifiability(
    ebm_fixture: dict[str, Any],
    protocol_fixture: dict[str, Any],
    parameter_fixture: dict[str, Any],
) -> dict[str, Any]:
    parameters = parameters_from_fixture(ebm_fixture)
    step = float(parameter_fixture["log_parameter_step"])
    half_step = step * float(parameter_fixture["convergence_half_step_multiplier"])
    equilibrium_forcings = [
        float(value) for value in parameter_fixture["equilibrium_forcing_w_m2"]
    ]
    protocol_ids = list(parameter_fixture["transient_protocol_ids"])

    equilibrium_output = lambda candidate: _equilibrium_outputs(
        candidate, equilibrium_forcings
    )
    transient_output = lambda candidate: _transient_outputs(
        candidate, protocol_fixture, protocol_ids
    )

    equilibrium = centered_log_parameter_jacobian(
        parameters, equilibrium_output, log_step=step
    )
    equilibrium_refined = centered_log_parameter_jacobian(
        parameters, equilibrium_output, log_step=half_step
    )
    transient = centered_log_parameter_jacobian(
        parameters, transient_output, log_step=step
    )
    transient_refined = centered_log_parameter_jacobian(
        parameters, transient_output, log_step=half_step
    )

    equilibrium_diagnostics = _matrix_diagnostics(equilibrium_refined)
    transient_diagnostics = _matrix_diagnostics(transient_refined)

    return {
        "fixture_id": parameter_fixture["fixture_id"],
        "ebm_fixture_id": ebm_fixture["fixture_id"],
        "forcing_protocol_fixture_id": protocol_fixture["fixture_id"],
        "parameter_ids": list(PARAMETER_IDS),
        "log_parameter_step": step,
        "refined_log_parameter_step": half_step,
        "equilibrium_forcing_w_m2": equilibrium_forcings,
        "transient_protocol_ids": protocol_ids,
        "implementation_versions": {
            "numpy": importlib.metadata.version("numpy"),
            "scipy": importlib.metadata.version("scipy"),
        },
        "equilibrium": equilibrium_diagnostics,
        "transient": transient_diagnostics,
        "local_rank_gain": int(
            transient_diagnostics["rank"] - equilibrium_diagnostics["rank"]
        ),
        "equilibrium_invisible_parameter_count": len(
            equilibrium_diagnostics["exactly_zero_sensitivity_parameters"]
        ),
        "equilibrium_step_consistency_relative_frobenius": (
            _relative_frobenius_disagreement(equilibrium, equilibrium_refined)
        ),
        "transient_step_consistency_relative_frobenius": (
            _relative_frobenius_disagreement(transient, transient_refined)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ebm-fixture", type=Path, default=DEFAULT_EBM_FIXTURE)
    parser.add_argument(
        "--protocol-fixture", type=Path, default=DEFAULT_PROTOCOL_FIXTURE
    )
    parser.add_argument(
        "--parameter-fixture", type=Path, default=DEFAULT_PARAMETER_FIXTURE
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = analyze_parameter_identifiability(
        load_ebm_fixture(args.ebm_fixture),
        load_protocol_fixture(args.protocol_fixture),
        load_parameter_fixture(args.parameter_fixture),
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    else:
        print(
            "equilibrium rank: "
            f"{result['equilibrium']['rank']}/{result['equilibrium']['parameter_dimension']}"
        )
        print(
            "transient rank: "
            f"{result['transient']['rank']}/{result['transient']['parameter_dimension']}"
        )
        print(f"local rank gain: {result['local_rank_gain']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
