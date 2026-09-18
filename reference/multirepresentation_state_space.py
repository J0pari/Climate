#!/usr/bin/env python3
"""Linear state-space baseline over temporal structural-world trajectories.

World relationships and observation maps come only from
multirepresentation_worlds.py. The role-based dynamics fixture supplies time
ordering and held-out trajectory seeds. Standard raw/PCA/CCA/factor
representations are fitted on discovery views, then the same generic PyDMD
adapter fits one-step dynamics in each representation.

Where the world authority exposes a shared evaluation target, the existing
matched-information k-nearest-neighbor target probe is trained on discovery
representations and used to decode DMD-predicted confirmation next states.
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
from sklearn.neighbors import KNeighborsRegressor

from reference.dmd_common import (
    fit_exact_dmd,
    predict_exact_dmd,
    relative_prediction_error,
)
from reference.multirepresentation_baselines import (
    discovery_fitted_representations,
)
from reference.multirepresentation_common import abs_spearman
from reference.multirepresentation_structure_evaluation import (
    DEFAULT_EVALUATION_FIXTURE,
    load_evaluation_fixture,
)
from reference.multirepresentation_worlds import (
    DEFAULT_DYNAMICS_FIXTURE,
    DEFAULT_FIXTURE as DEFAULT_WORLD_FIXTURE,
    bind_world_fixture_authority,
    generate_world_trajectories,
    load_dynamics_fixture,
    load_fixture as load_world_fixture,
    world_set_digest,
)


def _target_forecast_metrics(
    discovery_representation: np.ndarray,
    predicted_confirmation_next: np.ndarray,
    discovery_target: np.ndarray,
    confirmation_target_next: np.ndarray,
    probe_config: dict[str, Any],
) -> dict[str, float]:
    train = np.asarray(discovery_representation, dtype=float)
    predicted_state = np.asarray(predicted_confirmation_next, dtype=float)
    train_target = np.asarray(discovery_target, dtype=float)
    expected_target = np.asarray(confirmation_target_next, dtype=float)
    if (
        train.ndim != 2
        or predicted_state.ndim != 2
        or train.shape[0] != train_target.shape[0]
        or predicted_state.shape[0] != expected_target.shape[0]
        or not np.isfinite(train).all()
        or not np.isfinite(predicted_state).all()
        or not np.isfinite(train_target).all()
        or not np.isfinite(expected_target).all()
    ):
        raise ValueError("state-space target probe inputs are invalid")
    scale = float(np.std(expected_target))
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("confirmation target scale must be positive and finite")
    probe = KNeighborsRegressor(
        n_neighbors=int(probe_config["n_neighbors"]),
        weights=str(probe_config["weights"]),
        metric=str(probe_config["metric"]),
        p=int(probe_config["p"]),
    )
    probe.fit(train, train_target)
    predicted_target = np.asarray(probe.predict(predicted_state), dtype=float)
    if predicted_target.shape != expected_target.shape:
        raise RuntimeError("state-space target probe returned an unexpected shape")
    rmse = float(np.sqrt(np.mean((predicted_target - expected_target) ** 2)))
    return {
        "confirmation_target_normalized_rmse": rmse / scale,
        "confirmation_target_abs_spearman": abs_spearman(
            predicted_target,
            expected_target,
        ),
    }


def analyze_state_space_baselines(
    world_fixture: dict[str, Any],
    dynamics_fixture: dict[str, Any],
    evaluation_fixture: dict[str, Any],
    *,
    world_fixture_path: Path = DEFAULT_WORLD_FIXTURE,
) -> dict[str, Any]:
    bind_world_fixture_authority(
        world_fixture,
        dynamics_fixture,
        world_fixture_path=world_fixture_path,
    )
    bind_world_fixture_authority(
        world_fixture,
        evaluation_fixture,
        world_fixture_path=world_fixture_path,
    )
    discovery = generate_world_trajectories(
        world_fixture,
        dynamics_fixture,
        world_fixture_path=world_fixture_path,
    )
    confirmation = generate_world_trajectories(
        world_fixture,
        dynamics_fixture,
        confirmation=True,
        world_fixture_path=world_fixture_path,
    )
    if set(discovery) != set(confirmation):
        raise RuntimeError("state-space discovery and confirmation worlds differ")

    baseline_config = evaluation_fixture["coordinate_baselines"]
    probe_config = evaluation_fixture["matched_information_probe"]
    requested = list(probe_config["representations"])
    worlds: dict[str, Any] = {}

    for world_id in sorted(discovery):
        discovery_world = discovery[world_id]
        confirmation_world = confirmation[world_id]
        representations = discovery_fitted_representations(
            discovery_world,
            confirmation_world,
            baseline_config,
        )
        if set(representations) != set(requested):
            raise RuntimeError("state-space representation identities changed")

        target_name = discovery_world.ground_truth.get(
            "shared_evaluation_target_name"
        )
        per_representation: dict[str, Any] = {}
        for representation_id in requested:
            discovery_rep, confirmation_rep = representations[representation_id]
            train_x = discovery_rep[:-1]
            train_y = discovery_rep[1:]
            confirmation_x = confirmation_rep[:-1]
            confirmation_y = confirmation_rep[1:]
            rank = int(np.linalg.matrix_rank(train_x))
            if rank < 1:
                raise RuntimeError(
                    f"{world_id} {representation_id} has zero discovery rank"
                )
            fit = fit_exact_dmd(train_x, train_y, rank=rank)
            confirmation_prediction = predict_exact_dmd(fit, confirmation_x)
            item: dict[str, Any] = {
                "representation_dimension": int(discovery_rep.shape[1]),
                "discovery_state_rank": rank,
                "training_relative_error": float(fit.training_relative_error),
                "confirmation_relative_error": relative_prediction_error(
                    confirmation_prediction,
                    confirmation_y,
                ),
                "eigenvalue_count": int(fit.eigenvalues.size),
            }
            if target_name is not None:
                if (
                    not isinstance(target_name, str)
                    or target_name not in discovery_world.targets
                    or target_name not in confirmation_world.targets
                ):
                    raise ValueError("authoritative state-space target is inconsistent")
                item.update(
                    _target_forecast_metrics(
                        discovery_rep,
                        confirmation_prediction,
                        discovery_world.targets[target_name],
                        confirmation_world.targets[target_name][1:],
                        probe_config,
                    )
                )
            per_representation[representation_id] = item

        worlds[world_id] = {
            "relationship": discovery_world.relationship,
            "shared_evaluation_target_name": target_name,
            "representations": per_representation,
        }

    return {
        "dynamics_fixture_id": dynamics_fixture["fixture_id"],
        "world_fixture_id": world_fixture["fixture_id"],
        "evaluation_fixture_id": evaluation_fixture["fixture_id"],
        "discovery_trajectory_digest": world_set_digest(discovery),
        "confirmation_trajectory_digest": world_set_digest(confirmation),
        "dt": float(dynamics_fixture["dt"]),
        "worlds": worlds,
        "implementation_versions": {
            "numpy": importlib.metadata.version("numpy"),
            "pydmd": importlib.metadata.version("pydmd"),
            "scikit-learn": importlib.metadata.version("scikit-learn"),
            "scipy": importlib.metadata.version("scipy"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-fixture", type=Path, default=DEFAULT_WORLD_FIXTURE)
    parser.add_argument(
        "--dynamics-fixture",
        type=Path,
        default=DEFAULT_DYNAMICS_FIXTURE,
    )
    parser.add_argument(
        "--evaluation-fixture",
        type=Path,
        default=DEFAULT_EVALUATION_FIXTURE,
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = analyze_state_space_baselines(
        load_world_fixture(args.world_fixture),
        load_dynamics_fixture(args.dynamics_fixture),
        load_evaluation_fixture(args.evaluation_fixture),
        world_fixture_path=args.world_fixture,
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    else:
        for world_id, item in result["worlds"].items():
            raw = item["representations"]["raw_concat"]
            print(
                f"{world_id}: raw confirmation relative error="
                f"{raw['confirmation_relative_error']:.6e}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
