#!/usr/bin/env python3
"""Representation test for the two-layer EBM regime-feedback control.

The same ordinary least-squares solver is used for both candidate feature
spaces.  The raw global feature map is [surface, deep, forcing].  The
regime-gated map uses two disjoint copies of those same three features selected
by the fixture-declared regime label.  No intercept or regularization is used.

A swapped-regime confirmation control preserves feature dimension while giving
the gated model the wrong semantic label.  This distinguishes useful regime
semantics from simply adding coordinates.
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

from reference.two_layer_energy_balance import load_fixture as load_ebm_fixture
from reference.two_layer_regime_feedback import (
    DEFAULT_REGIME_FIXTURE,
    generate_regime_feedback_samples,
    load_regime_fixture,
)


DEFAULT_EBM_FIXTURE = (
    ROOT / "fixtures" / "physics" / "two-layer-ebm-geoffroy-mean-v1.json"
)


def _raw_design(states: np.ndarray, forcing: np.ndarray) -> np.ndarray:
    values = np.asarray(states, dtype=float)
    controls = np.asarray(forcing, dtype=float)
    if (
        values.ndim != 2
        or values.shape[1] != 2
        or controls.shape != (values.shape[0],)
        or not np.isfinite(values).all()
        or not np.isfinite(controls).all()
    ):
        raise ValueError("raw regime design inputs are invalid")
    return np.column_stack([values, controls])


def _gated_design(
    states: np.ndarray,
    forcing: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    raw = _raw_design(states, forcing)
    regime_labels = np.asarray(labels)
    if regime_labels.shape != (raw.shape[0],):
        raise ValueError("regime labels must match sample count")
    if not np.all(np.isin(regime_labels, [0, 1])):
        raise ValueError("regime labels must be 0 or 1")
    cold = (regime_labels == 0).astype(float)[:, None]
    warm = (regime_labels == 1).astype(float)[:, None]
    return np.column_stack([raw * cold, raw * warm])


def _fit(design: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    matrix = np.asarray(design, dtype=float)
    output = np.asarray(targets, dtype=float)
    if (
        matrix.ndim != 2
        or output.ndim != 2
        or output.shape[0] != matrix.shape[0]
        or output.shape[1] != 2
        or not np.isfinite(matrix).all()
        or not np.isfinite(output).all()
    ):
        raise ValueError("regime fit arrays are invalid")
    weights, _, rank, singular_values = np.linalg.lstsq(matrix, output, rcond=None)
    predicted = matrix @ weights
    denominator = max(float(np.linalg.norm(output)), np.finfo(float).tiny)
    relative_error = float(np.linalg.norm(predicted - output) / denominator)
    singular_values = np.asarray(singular_values, dtype=float)
    tolerance = (
        np.finfo(float).eps
        * max(matrix.shape)
        * float(np.max(singular_values))
    )
    numerical_rank = int(np.count_nonzero(singular_values > tolerance))
    if numerical_rank != int(rank):
        raise RuntimeError("least-squares and explicit regime ranks disagree")
    if numerical_rank < matrix.shape[1]:
        condition_status = "rank_deficient"
        condition_number = None
    else:
        condition_status = "finite"
        condition_number = float(singular_values[0] / singular_values[-1])
        if not math.isfinite(condition_number):
            raise RuntimeError("full-rank regime design has non-finite condition number")
    return {
        "weights": np.asarray(weights, dtype=float),
        "design_dimension": int(matrix.shape[1]),
        "design_rank": numerical_rank,
        "rank_tolerance": float(tolerance),
        "condition_number_status": condition_status,
        "condition_number": condition_number,
        "training_relative_error": relative_error,
    }


def _relative_prediction_error(
    predicted: np.ndarray,
    target: np.ndarray,
) -> float:
    left = np.asarray(predicted, dtype=float)
    right = np.asarray(target, dtype=float)
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("prediction arrays must be finite and shape matched")
    denominator = max(float(np.linalg.norm(right)), np.finfo(float).tiny)
    return float(np.linalg.norm(left - right) / denominator)


def analyze_regime_representations(
    ebm_fixture: dict[str, Any],
    regime_fixture: dict[str, Any],
) -> dict[str, Any]:
    samples = generate_regime_feedback_samples(ebm_fixture, regime_fixture)
    discovery = samples["discovery"]
    confirmation = samples["confirmation"]

    raw_discovery = _raw_design(
        discovery["states_x"], discovery["forcing_w_m2"]
    )
    raw_confirmation = _raw_design(
        confirmation["states_x"], confirmation["forcing_w_m2"]
    )
    gated_discovery = _gated_design(
        discovery["states_x"],
        discovery["forcing_w_m2"],
        discovery["labels"],
    )
    gated_confirmation = _gated_design(
        confirmation["states_x"],
        confirmation["forcing_w_m2"],
        confirmation["labels"],
    )

    raw_fit = _fit(raw_discovery, discovery["states_y"])
    gated_fit = _fit(gated_discovery, discovery["states_y"])

    raw_confirmation_prediction = raw_confirmation @ raw_fit["weights"]
    gated_confirmation_prediction = gated_confirmation @ gated_fit["weights"]
    swapped_confirmation_design = _gated_design(
        confirmation["states_x"],
        confirmation["forcing_w_m2"],
        1 - confirmation["labels"],
    )
    swapped_prediction = swapped_confirmation_design @ gated_fit["weights"]

    raw_confirmation_error = _relative_prediction_error(
        raw_confirmation_prediction, confirmation["states_y"]
    )
    gated_confirmation_error = _relative_prediction_error(
        gated_confirmation_prediction, confirmation["states_y"]
    )
    swapped_confirmation_error = _relative_prediction_error(
        swapped_prediction, confirmation["states_y"]
    )

    return {
        "fixture_id": regime_fixture["fixture_id"],
        "ebm_fixture_id": ebm_fixture["fixture_id"],
        "sample_digest": samples["sample_digest"],
        "observed_minimum_regime_margin_k": float(
            samples["minimum_regime_margin_k"]
        ),
        "raw_global": {
            key: value
            for key, value in raw_fit.items()
            if key != "weights"
        }
        | {"confirmation_relative_error": raw_confirmation_error},
        "regime_gated": {
            key: value
            for key, value in gated_fit.items()
            if key != "weights"
        }
        | {"confirmation_relative_error": gated_confirmation_error},
        "swapped_regime_confirmation_relative_error": swapped_confirmation_error,
        "confirmation_error_reduction": float(
            raw_confirmation_error - gated_confirmation_error
        ),
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
    result = analyze_regime_representations(
        load_ebm_fixture(args.ebm_fixture),
        load_regime_fixture(args.regime_fixture),
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    else:
        print(
            "raw confirmation error: "
            f"{result['raw_global']['confirmation_relative_error']:.6e}"
        )
        print(
            "gated confirmation error: "
            f"{result['regime_gated']['confirmation_relative_error']:.6e}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
