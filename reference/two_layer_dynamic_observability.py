#!/usr/bin/env python3
"""Dynamic observability reference for the linear two-layer EBM.

This module deliberately distinguishes two questions that are easy to conflate:

* instantaneous Fisher geometry asks which state directions are identified by
  observations at one instant;
* dynamical observability asks whether measurements across time identify the
  initial state when the linear evolution law is known.

For independent Gaussian measurements at declared sample times ``t_k``, the
initial-state mean Jacobian stacks ``H exp(A t_k)`` and the corresponding Fisher
matrix is the exact Gaussian pullback for that sampled experiment. The Kalman
observability matrix is reported separately as a noise-independent structural
rank witness.

The observation-noise fixture is synthetic and the dynamics are the exact
linear two-layer control. Nothing here establishes real observing-system
adequacy or climate-state observability.
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
from scipy.linalg import expm

from reference.two_layer_energy_balance import (
    DEFAULT_FIXTURE as DEFAULT_EBM_FIXTURE,
    load_fixture as load_ebm_fixture,
    parameters_from_fixture,
    system_matrix,
)
from reference.two_layer_observation_information import (
    DEFAULT_OBSERVATION_FIXTURE,
    OBSERVATION_SUBSETS,
    gaussian_mean_fisher_reference,
    load_observation_fixture,
    observation_jacobian,
    observation_noise_stddev,
)


def _sample_times(sample_times_years: Iterable[float]) -> np.ndarray:
    times = np.asarray(tuple(sample_times_years), dtype=float)
    if times.ndim != 1 or times.size == 0:
        raise ValueError("sample_times_years must be a nonempty one-dimensional sequence")
    if not np.isfinite(times).all() or np.any(times < 0.0):
        raise ValueError("sample_times_years must be finite and nonnegative")
    return times


def linear_observability_report(
    system: np.ndarray,
    observation: np.ndarray,
) -> dict[str, Any]:
    """Return the finite-dimensional Kalman observability matrix and rank.

    For an ``n``-state LTI system this stacks ``H A^k`` for ``k=0..n-1``.
    Numerical rank uses a machine-precision and scale-derived tolerance; it is a
    structural numerical witness, not a scientific significance threshold.
    """
    matrix = np.asarray(system, dtype=float)
    sensor = np.asarray(observation, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("system must be a nonempty square matrix")
    if sensor.ndim != 2 or sensor.shape[0] == 0 or sensor.shape[1] != matrix.shape[0]:
        raise ValueError("observation must be a nonempty matrix with one column per state")
    if not np.isfinite(matrix).all() or not np.isfinite(sensor).all():
        raise ValueError("system and observation matrices must be finite")

    state_dimension = matrix.shape[0]
    blocks = [sensor @ np.linalg.matrix_power(matrix, power) for power in range(state_dimension)]
    observability = np.vstack(blocks)
    singular_values = np.linalg.svd(observability, compute_uv=False)
    tolerance = (
        np.finfo(float).eps
        * max(observability.shape)
        * float(singular_values[0])
        if singular_values.size
        else 0.0
    )
    rank = int(np.count_nonzero(singular_values > tolerance))
    return {
        "observability_matrix": observability,
        "singular_values": singular_values,
        "rank": rank,
        "nullity": int(state_dimension - rank),
        "tolerance": float(tolerance),
    }


def two_layer_observability_report(
    ebm_fixture: dict[str, Any],
    channel_ids: Iterable[str],
) -> dict[str, Any]:
    parameters = parameters_from_fixture(ebm_fixture)
    return linear_observability_report(
        system_matrix(parameters),
        observation_jacobian(ebm_fixture, channel_ids),
    )


def sampled_initial_state_jacobian(
    ebm_fixture: dict[str, Any],
    channel_ids: Iterable[str],
    sample_times_years: Iterable[float],
) -> np.ndarray:
    """Jacobian of sampled observations with respect to the initial state."""
    times = _sample_times(sample_times_years)
    parameters = parameters_from_fixture(ebm_fixture)
    generator = system_matrix(parameters)
    instantaneous = observation_jacobian(ebm_fixture, channel_ids)
    blocks = [instantaneous @ expm(generator * float(time)) for time in times]
    stacked = np.vstack(blocks)
    if not np.isfinite(stacked).all():
        raise RuntimeError("sampled initial-state Jacobian became non-finite")
    return stacked


def sampled_initial_state_fisher(
    ebm_fixture: dict[str, Any],
    observation_fixture: dict[str, Any],
    channel_ids: Iterable[str],
    sample_times_years: Iterable[float],
    *,
    noise_multiplier: float = 1.0,
) -> dict[str, Any]:
    """Fisher information for the initial state from independent sampled observations.

    The same declared per-channel noise scale is used at every sample time and
    measurement errors are assumed independent across channels and sample times.
    Duplicated times therefore add precision but cannot add a new row direction.
    """
    times = _sample_times(sample_times_years)
    ids = tuple(channel_ids)
    jacobian = sampled_initial_state_jacobian(ebm_fixture, ids, times)
    per_time_noise = observation_noise_stddev(
        observation_fixture,
        ids,
        noise_multiplier=noise_multiplier,
    )
    repeated_noise = np.tile(per_time_noise, times.size)
    report = gaussian_mean_fisher_reference(jacobian, repeated_noise)
    return {
        **report,
        "sample_times_years": times,
        "channel_ids": ids,
        "measurement_count": int(jacobian.shape[0]),
    }


def analyze_dynamic_observability(
    ebm_fixture: dict[str, Any],
    observation_fixture: dict[str, Any],
    *,
    sample_times_years: Iterable[float] = (0.0, 10.0),
    noise_multiplier: float = 1.0,
) -> dict[str, Any]:
    if observation_fixture.get("ebm_fixture_id") != ebm_fixture.get("fixture_id"):
        raise ValueError("observation-control fixture references a different EBM fixture")
    times = _sample_times(sample_times_years)
    subsets: dict[str, Any] = {}
    for subset_name, channel_ids in OBSERVATION_SUBSETS.items():
        instantaneous = sampled_initial_state_fisher(
            ebm_fixture,
            observation_fixture,
            channel_ids,
            (0.0,),
            noise_multiplier=noise_multiplier,
        )
        sampled = sampled_initial_state_fisher(
            ebm_fixture,
            observation_fixture,
            channel_ids,
            times,
            noise_multiplier=noise_multiplier,
        )
        structural = two_layer_observability_report(ebm_fixture, channel_ids)
        subsets[subset_name] = {
            "channels": list(channel_ids),
            "instantaneous_rank": instantaneous["rank"],
            "instantaneous_nullity": instantaneous["nullity"],
            "sampled_rank": sampled["rank"],
            "sampled_nullity": sampled["nullity"],
            "sampled_fisher": sampled["fisher"].tolist(),
            "sampled_condition_number_2": sampled["condition_number_2"],
            "kalman_observability_rank": structural["rank"],
            "kalman_observability_nullity": structural["nullity"],
        }

    return {
        "ebm_fixture_id": ebm_fixture["fixture_id"],
        "observation_fixture_id": observation_fixture["fixture_id"],
        "sample_times_years": times.tolist(),
        "noise_multiplier": float(noise_multiplier),
        "assumptions": [
            "linear time-invariant two-layer dynamics with known parameters",
            "known deterministic forcing does not contribute to the initial-state Jacobian",
            "independent Gaussian measurement errors across channels and sample times",
            "no process noise or model discrepancy",
        ],
        "subsets": subsets,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ebm-fixture", type=Path, default=DEFAULT_EBM_FIXTURE)
    parser.add_argument(
        "--observation-fixture", type=Path, default=DEFAULT_OBSERVATION_FIXTURE
    )
    parser.add_argument("--sample-time", type=float, action="append", dest="sample_times")
    parser.add_argument("--noise-multiplier", type=float, default=1.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    sample_times = args.sample_times if args.sample_times is not None else [0.0, 10.0]

    result = analyze_dynamic_observability(
        load_ebm_fixture(args.ebm_fixture),
        load_observation_fixture(args.observation_fixture),
        sample_times_years=sample_times,
        noise_multiplier=args.noise_multiplier,
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        for name, diagnostics in result["subsets"].items():
            print(f"{name}: {diagnostics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
