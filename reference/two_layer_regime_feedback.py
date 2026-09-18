#!/usr/bin/env python3
"""Exact local two-regime feedback control for the two-layer EBM.

The benchmark declares cold and warm surface-temperature regimes with distinct
positive climate-feedback coefficients.  Each sampled finite transition must
remain at least the fixture-declared margin from the threshold at every
validation substep.  Because accepted transitions never traverse the feedback
discontinuity, each transition is exactly the standard linear two-layer EBM
with its regime-fixed feedback coefficient.

This module owns only the physical sample construction and regime-admissibility
checks.  Representation fitting lives in a separate method.
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
    load_fixture as load_ebm_fixture,
    parameters_from_fixture,
    system_matrix,
)


DEFAULT_EBM_FIXTURE = (
    ROOT / "fixtures" / "physics" / "two-layer-ebm-geoffroy-mean-v1.json"
)
DEFAULT_REGIME_FIXTURE = (
    ROOT / "fixtures" / "physics" / "two-layer-ebm-regime-feedback-v1.json"
)
REGIME_IDS = ("cold", "warm")


def _finite_pair(name: str, value: Any) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must contain exactly two endpoints")
    lower, upper = float(value[0]), float(value[1])
    if not math.isfinite(lower) or not math.isfinite(upper) or not lower < upper:
        raise ValueError(f"{name} endpoints must be finite and increasing")
    return lower, upper


def load_regime_fixture(path: Path = DEFAULT_REGIME_FIXTURE) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported regime-feedback fixture schema")
    threshold = float(payload["surface_temperature_threshold_k"])
    margin = float(payload["minimum_regime_margin_k"])
    dt = float(payload["transition_dt_years"])
    substeps = payload["validation_substeps"]
    if not math.isfinite(threshold):
        raise ValueError("surface_temperature_threshold_k must be finite")
    if not math.isfinite(margin) or margin <= 0.0:
        raise ValueError("minimum_regime_margin_k must be finite and positive")
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("transition_dt_years must be finite and positive")
    if not isinstance(substeps, int) or substeps < 8:
        raise ValueError("validation_substeps must be an integer >= 8")
    regimes = payload.get("regimes")
    if not isinstance(regimes, dict) or tuple(regimes) != REGIME_IDS:
        raise ValueError("regime-feedback v1 regime identities changed")
    for regime_id, expected_side in (("cold", "below"), ("warm", "above")):
        item = regimes[regime_id]
        feedback = float(item["climate_feedback_w_m2_k"])
        if not math.isfinite(feedback) or feedback <= 0.0:
            raise ValueError("regime feedback coefficients must be finite and positive")
        if item.get("side") != expected_side:
            raise ValueError(f"{regime_id} regime has unexpected threshold side")
    for split in ("discovery", "confirmation"):
        count = payload[f"{split}_sample_count_per_regime"]
        seeds = payload[f"{split}_seeds"]
        domains = payload[f"{split}_domains"]
        if not isinstance(count, int) or count < 32:
            raise ValueError(f"{split} sample count must be an integer >= 32")
        if not isinstance(seeds, dict) or set(seeds) != set(REGIME_IDS):
            raise ValueError(f"{split} seeds must cover both regimes")
        if not all(isinstance(seeds[key], int) for key in REGIME_IDS):
            raise ValueError(f"{split} seeds must be integers")
        if not isinstance(domains, dict) or set(domains) != set(REGIME_IDS):
            raise ValueError(f"{split} domains must cover both regimes")
        for regime_id in REGIME_IDS:
            domain = domains[regime_id]
            _finite_pair(
                f"{split}.{regime_id}.surface_temperature_k",
                domain["surface_temperature_k"],
            )
            _finite_pair(
                f"{split}.{regime_id}.deep_temperature_k",
                domain["deep_temperature_k"],
            )
            _finite_pair(
                f"{split}.{regime_id}.forcing_w_m2",
                domain["forcing_w_m2"],
            )
    return payload


def _parameters_for_regime(
    base: TwoLayerParameters,
    fixture: dict[str, Any],
    regime_id: str,
) -> TwoLayerParameters:
    if regime_id not in REGIME_IDS:
        raise ValueError(f"unknown regime {regime_id!r}")
    feedback = float(fixture["regimes"][regime_id]["climate_feedback_w_m2_k"])
    return TwoLayerParameters(
        surface_heat_capacity_w_yr_m2_k=base.surface_heat_capacity_w_yr_m2_k,
        deep_heat_capacity_w_yr_m2_k=base.deep_heat_capacity_w_yr_m2_k,
        ocean_heat_exchange_w_m2_k=base.ocean_heat_exchange_w_m2_k,
        climate_feedback_w_m2_k=feedback,
    )


def _propagate_batch(
    states: np.ndarray,
    forcing: np.ndarray,
    parameters: TwoLayerParameters,
    time_years: float,
) -> np.ndarray:
    values = np.asarray(states, dtype=float)
    controls = np.asarray(forcing, dtype=float)
    if (
        values.ndim != 2
        or values.shape[1] != 2
        or controls.shape != (values.shape[0],)
        or not np.isfinite(values).all()
        or not np.isfinite(controls).all()
    ):
        raise ValueError("batch propagation inputs are invalid")
    time = float(time_years)
    if not math.isfinite(time) or time < 0.0:
        raise ValueError("batch propagation time must be finite and nonnegative")
    matrix = np.zeros((3, 3), dtype=float)
    matrix[:2, :2] = system_matrix(parameters)
    matrix[0, 2] = 1.0 / parameters.surface_heat_capacity_w_yr_m2_k
    augmented = np.column_stack([values, controls])
    propagated = augmented @ expm(matrix * time).T
    result = np.asarray(propagated[:, :2], dtype=float)
    if not np.isfinite(result).all():
        raise RuntimeError("regime batch propagation became non-finite")
    return result


def _regime_margin(
    surface_values: np.ndarray,
    threshold: float,
    regime_id: str,
) -> np.ndarray:
    if regime_id == "cold":
        return threshold - np.asarray(surface_values, dtype=float)
    if regime_id == "warm":
        return np.asarray(surface_values, dtype=float) - threshold
    raise ValueError(f"unknown regime {regime_id!r}")


def _sample_one_regime(
    ebm_fixture: dict[str, Any],
    regime_fixture: dict[str, Any],
    *,
    split: str,
    regime_id: str,
) -> dict[str, Any]:
    base = parameters_from_fixture(ebm_fixture)
    parameters = _parameters_for_regime(base, regime_fixture, regime_id)
    count = int(regime_fixture[f"{split}_sample_count_per_regime"])
    seed = int(regime_fixture[f"{split}_seeds"][regime_id])
    domain = regime_fixture[f"{split}_domains"][regime_id]
    surface_bounds = _finite_pair(
        "surface_temperature_k", domain["surface_temperature_k"]
    )
    deep_bounds = _finite_pair(
        "deep_temperature_k", domain["deep_temperature_k"]
    )
    forcing_bounds = _finite_pair("forcing_w_m2", domain["forcing_w_m2"])
    dt = float(regime_fixture["transition_dt_years"])
    threshold = float(regime_fixture["surface_temperature_threshold_k"])
    required_margin = float(regime_fixture["minimum_regime_margin_k"])
    validation_substeps = int(regime_fixture["validation_substeps"])
    times = np.linspace(0.0, dt, validation_substeps + 1)

    rng = np.random.default_rng(seed)
    states_x = np.column_stack(
        [
            rng.uniform(*surface_bounds, size=count),
            rng.uniform(*deep_bounds, size=count),
        ]
    )
    forcing = rng.uniform(*forcing_bounds, size=count)
    states_y = np.empty_like(states_x)
    minimum_margin = math.inf

    for time in times:
        propagated = _propagate_batch(
            states_x,
            forcing,
            parameters,
            float(time),
        )
        margins = _regime_margin(propagated[:, 0], threshold, regime_id)
        local_minimum = float(np.min(margins))
        if local_minimum < required_margin:
            raise RuntimeError(
                f"{split} {regime_id} transition violates regime margin: "
                f"{local_minimum} < {required_margin}"
            )
        minimum_margin = min(minimum_margin, local_minimum)
        if math.isclose(float(time), dt, rel_tol=0.0, abs_tol=1e-15):
            states_y[:] = propagated

    label = 0 if regime_id == "cold" else 1
    return {
        "states_x": states_x,
        "forcing_w_m2": forcing,
        "states_y": states_y,
        "labels": np.full(count, label, dtype=np.int64),
        "minimum_regime_margin_k": float(minimum_margin),
    }


def _split_samples(
    ebm_fixture: dict[str, Any],
    regime_fixture: dict[str, Any],
    split: str,
) -> dict[str, Any]:
    if split not in {"discovery", "confirmation"}:
        raise ValueError("split must be discovery or confirmation")
    pieces = [
        _sample_one_regime(
            ebm_fixture,
            regime_fixture,
            split=split,
            regime_id=regime_id,
        )
        for regime_id in REGIME_IDS
    ]
    return {
        "states_x": np.vstack([item["states_x"] for item in pieces]),
        "forcing_w_m2": np.concatenate(
            [item["forcing_w_m2"] for item in pieces]
        ),
        "states_y": np.vstack([item["states_y"] for item in pieces]),
        "labels": np.concatenate([item["labels"] for item in pieces]),
        "minimum_regime_margin_k": min(
            item["minimum_regime_margin_k"] for item in pieces
        ),
    }


def _sample_digest(discovery: dict[str, Any], confirmation: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for split_name, split in (
        ("discovery", discovery),
        ("confirmation", confirmation),
    ):
        digest.update(split_name.encode("utf-8"))
        digest.update(b"\0")
        for key in ("states_x", "forcing_w_m2", "states_y"):
            digest.update(key.encode("utf-8"))
            digest.update(b"\0")
            digest.update(np.asarray(split[key], dtype="<f8").tobytes(order="C"))
        digest.update(b"labels\0")
        digest.update(np.asarray(split["labels"], dtype="<i8").tobytes(order="C"))
    return "sha256:" + digest.hexdigest()


def generate_regime_feedback_samples(
    ebm_fixture: dict[str, Any],
    regime_fixture: dict[str, Any],
) -> dict[str, Any]:
    discovery = _split_samples(ebm_fixture, regime_fixture, "discovery")
    confirmation = _split_samples(ebm_fixture, regime_fixture, "confirmation")
    return {
        "discovery": discovery,
        "confirmation": confirmation,
        "sample_digest": _sample_digest(discovery, confirmation),
        "minimum_regime_margin_k": min(
            discovery["minimum_regime_margin_k"],
            confirmation["minimum_regime_margin_k"],
        ),
    }


def summarize_regime_feedback(
    ebm_fixture: dict[str, Any],
    regime_fixture: dict[str, Any],
) -> dict[str, Any]:
    samples = generate_regime_feedback_samples(ebm_fixture, regime_fixture)
    return {
        "fixture_id": regime_fixture["fixture_id"],
        "ebm_fixture_id": ebm_fixture["fixture_id"],
        "transition_dt_years": float(regime_fixture["transition_dt_years"]),
        "surface_temperature_threshold_k": float(
            regime_fixture["surface_temperature_threshold_k"]
        ),
        "required_regime_margin_k": float(
            regime_fixture["minimum_regime_margin_k"]
        ),
        "observed_minimum_regime_margin_k": float(
            samples["minimum_regime_margin_k"]
        ),
        "sample_digest": samples["sample_digest"],
        "discovery_sample_count": int(samples["discovery"]["states_x"].shape[0]),
        "confirmation_sample_count": int(
            samples["confirmation"]["states_x"].shape[0]
        ),
        "regime_feedback_w_m2_k": {
            regime_id: float(
                regime_fixture["regimes"][regime_id][
                    "climate_feedback_w_m2_k"
                ]
            )
            for regime_id in REGIME_IDS
        },
        "implementation_versions": {
            "numpy": importlib.metadata.version("numpy"),
            "scipy": importlib.metadata.version("scipy"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ebm-fixture", type=Path, default=DEFAULT_EBM_FIXTURE)
    parser.add_argument(
        "--regime-fixture",
        type=Path,
        default=DEFAULT_REGIME_FIXTURE,
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = summarize_regime_feedback(
        load_ebm_fixture(args.ebm_fixture),
        load_regime_fixture(args.regime_fixture),
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    else:
        print(f"sample_digest: {result['sample_digest']}")
        print(
            "minimum regime margin K: "
            f"{result['observed_minimum_regime_margin_k']:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
