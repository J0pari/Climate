#!/usr/bin/env python3
"""Pareto observation-design reference for the two-layer EBM structural control.

This module separates four authorities:

* the two-layer EBM owns the physical state and exact thermal modes;
* the observation fixture owns synthetic noise scales;
* the design fixture owns synthetic channel costs;
* this reference enumerates channel subsets and reports nondominated information/cost tradeoffs.

The result is deliberately a Pareto frontier, not a single preferred observing
system. Synthetic costs and noise scales are structural controls only.
"""
from __future__ import annotations

import argparse
from itertools import combinations
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
from reference.two_layer_observation_information import (
    DEFAULT_OBSERVATION_FIXTURE,
    gaussian_mean_fisher_reference,
    load_observation_fixture,
    metric_in_coordinates,
    observation_jacobian,
    observation_noise_stddev,
)


DEFAULT_DESIGN_FIXTURE = (
    ROOT / "fixtures" / "physics" / "two-layer-ebm-observation-design-control-v1.json"
)


def load_design_fixture(path: Path = DEFAULT_DESIGN_FIXTURE) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        fixture = json.load(handle)
    if fixture.get("schema_version") != 1:
        raise ValueError("unsupported observation-design fixture schema_version")
    if fixture.get("provenance", {}).get("kind") != "synthetic_structural_control":
        raise ValueError("observation-design fixture must declare synthetic structural provenance")
    costs = fixture.get("channel_costs")
    if not isinstance(costs, dict) or not costs:
        raise ValueError("observation-design fixture must declare channel_costs")
    for channel_id, raw_cost in costs.items():
        cost = float(raw_cost)
        if not channel_id or not math.isfinite(cost) or cost <= 0.0:
            raise ValueError("every observation-design channel cost must be finite and positive")
    return fixture


def _all_nonempty_subsets(channel_ids: Iterable[str]) -> tuple[tuple[str, ...], ...]:
    channels = tuple(sorted(channel_ids))
    if not channels:
        raise ValueError("at least one observation channel is required")
    return tuple(
        subset
        for size in range(1, len(channels) + 1)
        for subset in combinations(channels, size)
    )


def _modal_information(ebm_fixture: dict[str, Any], fisher_state: np.ndarray) -> dict[str, Any]:
    parameters = parameters_from_fixture(ebm_fixture)
    _, basis = mode_basis(parameters)
    modal = metric_in_coordinates(fisher_state, basis)
    diagonal = np.diag(modal)
    if np.any(diagonal < -1e-12):
        raise RuntimeError("modal Fisher diagonal must be nonnegative")
    fast_information = float(max(diagonal[0], 0.0))
    slow_information = float(max(diagonal[1], 0.0))
    denominator = math.sqrt(fast_information * slow_information)
    coupling = 0.0 if denominator == 0.0 else float(modal[0, 1] / denominator)
    return {
        "modal_fisher": modal,
        "fast_mode_information": fast_information,
        "slow_mode_information": slow_information,
        "normalized_cross_mode_coupling": coupling,
    }


def _close(a: float, b: float) -> float:
    return 1e-12 * max(1.0, abs(a), abs(b))


def design_dominates(candidate: dict[str, Any], other: dict[str, Any]) -> bool:
    """Return whether candidate weakly improves every Pareto objective and one strictly.

    Objectives are lower synthetic cost and higher structural rank, fast-mode
    information, and slow-mode information. Cross-mode coupling and condition
    number remain descriptive because neither has a universal monotone meaning.
    """
    cost_tol = _close(float(candidate["cost"]), float(other["cost"]))
    fast_tol = _close(
        float(candidate["fast_mode_information"]),
        float(other["fast_mode_information"]),
    )
    slow_tol = _close(
        float(candidate["slow_mode_information"]),
        float(other["slow_mode_information"]),
    )

    not_worse = (
        float(candidate["cost"]) <= float(other["cost"]) + cost_tol
        and int(candidate["state_rank"]) >= int(other["state_rank"])
        and float(candidate["fast_mode_information"])
        >= float(other["fast_mode_information"]) - fast_tol
        and float(candidate["slow_mode_information"])
        >= float(other["slow_mode_information"]) - slow_tol
    )
    strictly_better = (
        float(candidate["cost"]) < float(other["cost"]) - cost_tol
        or int(candidate["state_rank"]) > int(other["state_rank"])
        or float(candidate["fast_mode_information"])
        > float(other["fast_mode_information"]) + fast_tol
        or float(candidate["slow_mode_information"])
        > float(other["slow_mode_information"]) + slow_tol
    )
    return bool(not_worse and strictly_better)


def analyze_observation_design(
    ebm_fixture: dict[str, Any],
    observation_fixture: dict[str, Any],
    design_fixture: dict[str, Any],
    *,
    noise_multiplier: float = 1.0,
    cost_multiplier: float = 1.0,
) -> dict[str, Any]:
    if observation_fixture.get("ebm_fixture_id") != ebm_fixture.get("fixture_id"):
        raise ValueError("observation-control fixture references a different EBM fixture")
    if design_fixture.get("observation_fixture_id") != observation_fixture.get("fixture_id"):
        raise ValueError("observation-design fixture references a different observation fixture")

    noise_scale = float(noise_multiplier)
    cost_scale = float(cost_multiplier)
    if not math.isfinite(noise_scale) or noise_scale <= 0.0:
        raise ValueError("noise_multiplier must be finite and positive")
    if not math.isfinite(cost_scale) or cost_scale <= 0.0:
        raise ValueError("cost_multiplier must be finite and positive")

    observation_channels = set(observation_fixture["channels"])
    cost_channels = set(design_fixture["channel_costs"])
    if cost_channels != observation_channels:
        missing = sorted(observation_channels - cost_channels)
        extra = sorted(cost_channels - observation_channels)
        raise ValueError(
            f"observation-design channel costs must exactly cover observation channels; missing={missing}, extra={extra}"
        )

    designs: list[dict[str, Any]] = []
    for subset in _all_nonempty_subsets(observation_channels):
        jacobian = observation_jacobian(ebm_fixture, subset)
        noise = observation_noise_stddev(
            observation_fixture,
            subset,
            noise_multiplier=noise_scale,
        )
        fisher = gaussian_mean_fisher_reference(jacobian, noise)
        modal = _modal_information(ebm_fixture, fisher["fisher"])
        cost = cost_scale * sum(float(design_fixture["channel_costs"][channel]) for channel in subset)
        designs.append(
            {
                "design_id": "+".join(subset),
                "channels": list(subset),
                "cost": float(cost),
                "state_rank": int(fisher["rank"]),
                "state_nullity": int(fisher["nullity"]),
                "state_condition_number_2": float(fisher["condition_number_2"]),
                "state_fisher": fisher["fisher"].tolist(),
                "modal_fisher": modal["modal_fisher"].tolist(),
                "fast_mode_information": modal["fast_mode_information"],
                "slow_mode_information": modal["slow_mode_information"],
                "normalized_cross_mode_coupling": modal[
                    "normalized_cross_mode_coupling"
                ],
            }
        )

    frontier = [
        design
        for design in designs
        if not any(
            design_dominates(other, design)
            for other in designs
            if other["design_id"] != design["design_id"]
        )
    ]
    frontier.sort(key=lambda item: (item["cost"], item["design_id"]))

    return {
        "ebm_fixture_id": ebm_fixture["fixture_id"],
        "observation_fixture_id": observation_fixture["fixture_id"],
        "design_fixture_id": design_fixture["fixture_id"],
        "noise_multiplier": noise_scale,
        "cost_multiplier": cost_scale,
        "objectives": {
            "minimize": ["synthetic_cost"],
            "maximize": [
                "state_rank",
                "fast_mode_information",
                "slow_mode_information",
            ],
            "descriptive_only": [
                "state_condition_number_2",
                "normalized_cross_mode_coupling",
            ],
        },
        "designs": designs,
        "pareto_design_ids": [item["design_id"] for item in frontier],
        "interpretation_boundary": (
            "This is a synthetic structural information-design control. Pareto membership under equal arbitrary costs and synthetic noise does not recommend a real observing system."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ebm-fixture", type=Path, default=DEFAULT_EBM_FIXTURE)
    parser.add_argument(
        "--observation-fixture", type=Path, default=DEFAULT_OBSERVATION_FIXTURE
    )
    parser.add_argument("--design-fixture", type=Path, default=DEFAULT_DESIGN_FIXTURE)
    parser.add_argument("--noise-multiplier", type=float, default=1.0)
    parser.add_argument("--cost-multiplier", type=float, default=1.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = analyze_observation_design(
        load_ebm_fixture(args.ebm_fixture),
        load_observation_fixture(args.observation_fixture),
        load_design_fixture(args.design_fixture),
        noise_multiplier=args.noise_multiplier,
        cost_multiplier=args.cost_multiplier,
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print("pareto designs:", ", ".join(result["pareto_design_ids"]))
        for design in result["designs"]:
            print(design)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
