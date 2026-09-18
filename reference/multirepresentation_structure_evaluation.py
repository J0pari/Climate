#!/usr/bin/env python3
"""Evaluate multirepresentation methods against the authoritative world zoo.

Ground truth remains exclusively in structural-worlds-v1.json and
multirepresentation_worlds.py.  This module owns evaluation only: it binds to
the exact world-fixture Git blob, derives a confirmation realization by a
fixture-declared seed offset, runs a conservative nonlinear dependence test
that may abstain, and compares established coordinate baselines on the held-out
confirmation realization.

No world-id-to-truth table exists here.  Expected selector behavior and recovery
targets are read from each generated StructuralWorld's authoritative
ground_truth record.
"""
from __future__ import annotations

import argparse
import hashlib
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
from sklearn.decomposition import FactorAnalysis
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler

from reference.multirepresentation_common import (
    abs_spearman,
    concat_pca,
    concat_spectral,
    jointly_smooth,
    linear_cca,
)
from reference.multirepresentation_worlds import (
    DEFAULT_FIXTURE as DEFAULT_WORLD_FIXTURE,
    StructuralWorld,
    generate_worlds,
    load_fixture as load_world_fixture,
)


DEFAULT_EVALUATION_FIXTURE = (
    ROOT
    / "fixtures"
    / "multirepresentation"
    / "structural-world-evaluation-v1.json"
)
SUPPORTED_DECISIONS = {
    "shared_coordinate_supported",
    "abstain_no_cross_view_evidence",
}


def _git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode("utf-8")
    return hashlib.sha1(header + data).hexdigest()


def load_evaluation_fixture(
    path: Path = DEFAULT_EVALUATION_FIXTURE,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported structural-world evaluation fixture schema")
    offset = payload.get("confirmation_seed_offset")
    if not isinstance(offset, int) or offset <= 0:
        raise ValueError("confirmation_seed_offset must be a positive integer")

    selector = payload.get("dependence_selector")
    if not isinstance(selector, dict):
        raise ValueError("dependence_selector policy is required")
    neighbors = selector.get("n_neighbors")
    permutations = selector.get("permutations")
    random_state = selector.get("random_state")
    alpha = float(selector.get("alpha"))
    if not isinstance(neighbors, int) or neighbors < 2:
        raise ValueError("dependence selector n_neighbors must be >= 2")
    if not isinstance(permutations, int) or permutations < 31:
        raise ValueError("dependence selector permutations must be >= 31")
    if not isinstance(random_state, int):
        raise ValueError("dependence selector random_state must be an integer")
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("dependence selector alpha must be in (0,1)")
    if selector.get("decision_when_rejected") not in SUPPORTED_DECISIONS:
        raise ValueError("invalid rejected-null selector decision")
    if selector.get("decision_when_not_rejected") not in SUPPORTED_DECISIONS:
        raise ValueError("invalid non-rejected selector decision")
    return payload


def _bind_world_authority(
    world_fixture_path: Path,
    world_fixture: dict[str, Any],
    evaluation_fixture: dict[str, Any],
) -> None:
    if world_fixture.get("fixture_id") != evaluation_fixture.get("world_fixture_id"):
        raise ValueError("evaluation policy resolved a different world fixture identity")
    observed_blob = _git_blob_sha(world_fixture_path)
    expected_blob = evaluation_fixture.get("world_fixture_git_blob_sha")
    if observed_blob != expected_blob:
        raise ValueError(
            "authoritative structural-world fixture bytes changed without "
            "a new evaluation-policy binding"
        )


def _standardize(view: np.ndarray) -> np.ndarray:
    values = np.asarray(view, dtype=float)
    if values.ndim != 2 or values.shape[0] < 4 or not np.isfinite(values).all():
        raise ValueError("dependence evaluation requires finite 2-D views")
    return StandardScaler().fit_transform(values)


def _pairwise_mutual_information_statistic(
    view_a: np.ndarray,
    view_b: np.ndarray,
    *,
    n_neighbors: int,
    random_state: int,
) -> float:
    a = _standardize(view_a)
    b = _standardize(view_b)
    scores: list[float] = []
    for column in range(b.shape[1]):
        values = mutual_info_regression(
            a,
            b[:, column],
            discrete_features=False,
            n_neighbors=n_neighbors,
            random_state=random_state,
        )
        scores.extend(float(value) for value in values)
    for column in range(a.shape[1]):
        values = mutual_info_regression(
            b,
            a[:, column],
            discrete_features=False,
            n_neighbors=n_neighbors,
            random_state=random_state,
        )
        scores.extend(float(value) for value in values)
    if not scores or not np.isfinite(scores).all():
        raise RuntimeError("mutual-information statistic became non-finite")
    return max(scores)


def dependence_selector(
    world: StructuralWorld,
    policy: dict[str, Any],
) -> dict[str, Any]:
    n_neighbors = int(policy["n_neighbors"])
    permutations = int(policy["permutations"])
    random_state = int(policy["random_state"])
    alpha = float(policy["alpha"])
    observed = _pairwise_mutual_information_statistic(
        world.view_a,
        world.view_b,
        n_neighbors=n_neighbors,
        random_state=random_state,
    )
    rng = np.random.default_rng(random_state)
    exceedances = 0
    for _ in range(permutations):
        permuted_b = world.view_b[rng.permutation(world.view_b.shape[0])]
        null_value = _pairwise_mutual_information_statistic(
            world.view_a,
            permuted_b,
            n_neighbors=n_neighbors,
            random_state=random_state,
        )
        if null_value >= observed:
            exceedances += 1
    pvalue = float((1 + exceedances) / (permutations + 1))
    rejected = pvalue <= alpha
    decision = (
        policy["decision_when_rejected"]
        if rejected
        else policy["decision_when_not_rejected"]
    )
    return {
        "statistic": observed,
        "permutation_pvalue": pvalue,
        "permutations": permutations,
        "alpha": alpha,
        "decision": decision,
    }


def _factor_analysis_coordinate(
    view_a: np.ndarray,
    view_b: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    joined = np.column_stack([_standardize(view_a), _standardize(view_b)])
    coordinate = FactorAnalysis(
        n_components=1,
        svd_method=str(config["svd_method"]),
    ).fit_transform(joined)[:, 0]
    if not np.isfinite(coordinate).all():
        raise RuntimeError("factor analysis returned non-finite values")
    return coordinate


def _coordinate_methods(
    world: StructuralWorld,
    config: dict[str, Any],
) -> dict[str, Callable[[], np.ndarray]]:
    spectral = config["spectral"]
    jsf = config["jsf"]
    factor = config["factor_analysis"]
    return {
        "concat_pca": lambda: concat_pca(world.view_a, world.view_b),
        "linear_cca": lambda: linear_cca(world.view_a, world.view_b),
        "concat_spectral": lambda: concat_spectral(
            world.view_a,
            world.view_b,
            n_neighbors=int(spectral["n_neighbors"]),
            random_state=int(spectral["random_state"]),
        ),
        "factor_analysis": lambda: _factor_analysis_coordinate(
            world.view_a,
            world.view_b,
            factor,
        ),
        "jointly_smooth": lambda: jointly_smooth(
            world.view_a,
            world.view_b,
            jsf,
        ),
    }


def _expected_selector_decision(world: StructuralWorld) -> str:
    target_name = world.ground_truth.get("shared_evaluation_target_name")
    if target_name is None:
        return "abstain_no_cross_view_evidence"
    if not isinstance(target_name, str) or target_name not in world.targets:
        raise ValueError(
            f"{world.world_id} has an invalid authoritative shared evaluation target"
        )
    return "shared_coordinate_supported"


def _evaluate_coordinate_baselines(
    world: StructuralWorld,
    config: dict[str, Any],
) -> dict[str, Any]:
    target_name = world.ground_truth.get("shared_evaluation_target_name")
    target = None if target_name is None else world.targets[str(target_name)]
    results: dict[str, Any] = {}
    for method_id, call in _coordinate_methods(world, config).items():
        try:
            coordinate = np.asarray(call(), dtype=float)
            if coordinate.shape != (world.view_a.shape[0],):
                raise ValueError("coordinate method returned an unexpected shape")
            item: dict[str, Any] = {
                "status": "ok",
                "emits_coordinate": True,
            }
            if target is not None:
                item["shared_target_abs_spearman"] = abs_spearman(
                    coordinate,
                    np.asarray(target, dtype=float),
                )
            results[method_id] = item
        except Exception as error:  # method failure is benchmark output, not substitution
            results[method_id] = {
                "status": "failed",
                "emits_coordinate": False,
                "error_type": type(error).__name__,
            }
    return results


def _sample_digest(worlds: dict[str, StructuralWorld]) -> str:
    digest = hashlib.sha256()
    for world_id in sorted(worlds):
        world = worlds[world_id]
        digest.update(world_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(np.asarray(world.view_a, dtype="<f8").tobytes(order="C"))
        digest.update(np.asarray(world.view_b, dtype="<f8").tobytes(order="C"))
    return "sha256:" + digest.hexdigest()


def evaluate_structural_worlds(
    world_fixture: dict[str, Any],
    evaluation_fixture: dict[str, Any],
    *,
    world_fixture_path: Path = DEFAULT_WORLD_FIXTURE,
) -> dict[str, Any]:
    _bind_world_authority(
        world_fixture_path,
        world_fixture,
        evaluation_fixture,
    )
    discovery = generate_worlds(world_fixture)
    confirmation = generate_worlds(
        world_fixture,
        seed_offset=int(evaluation_fixture["confirmation_seed_offset"]),
    )
    if set(discovery) != set(confirmation):
        raise RuntimeError("discovery and confirmation world identities differ")

    selector_policy = evaluation_fixture["dependence_selector"]
    baseline_config = evaluation_fixture["coordinate_baselines"]
    world_results: dict[str, Any] = {}
    discovery_calibrated: list[bool] = []
    confirmation_calibrated: list[bool] = []

    for world_id in sorted(discovery):
        discovery_world = discovery[world_id]
        confirmation_world = confirmation[world_id]
        if (
            discovery_world.relationship != confirmation_world.relationship
            or discovery_world.static_identifiability
            != confirmation_world.static_identifiability
            or discovery_world.ground_truth != confirmation_world.ground_truth
        ):
            raise RuntimeError(
                f"confirmation semantics changed for structural world {world_id}"
            )

        expected = _expected_selector_decision(discovery_world)
        discovery_selector = dependence_selector(
            discovery_world,
            selector_policy,
        )
        confirmation_selector = dependence_selector(
            confirmation_world,
            selector_policy,
        )
        discovery_ok = discovery_selector["decision"] == expected
        confirmation_ok = confirmation_selector["decision"] == expected
        discovery_calibrated.append(discovery_ok)
        confirmation_calibrated.append(confirmation_ok)

        world_results[world_id] = {
            "relationship": discovery_world.relationship,
            "static_identifiability": discovery_world.static_identifiability,
            "shared_evaluation_target_name": discovery_world.ground_truth.get(
                "shared_evaluation_target_name"
            ),
            "expected_selector_decision": expected,
            "discovery_selector": discovery_selector,
            "confirmation_selector": confirmation_selector,
            "discovery_selector_calibrated": discovery_ok,
            "confirmation_selector_calibrated": confirmation_ok,
            "confirmation_coordinate_baselines": _evaluate_coordinate_baselines(
                confirmation_world,
                baseline_config,
            ),
        }

    return {
        "fixture_id": evaluation_fixture["fixture_id"],
        "world_fixture_id": world_fixture["fixture_id"],
        "world_fixture_git_blob_sha": evaluation_fixture[
            "world_fixture_git_blob_sha"
        ],
        "confirmation_seed_offset": int(
            evaluation_fixture["confirmation_seed_offset"]
        ),
        "discovery_sample_digest": _sample_digest(discovery),
        "confirmation_sample_digest": _sample_digest(confirmation),
        "selector_summary": {
            "all_discovery_decisions_calibrated": all(discovery_calibrated),
            "all_confirmation_decisions_calibrated": all(confirmation_calibrated),
        },
        "worlds": world_results,
        "implementation_versions": {
            "datafold": importlib.metadata.version("datafold"),
            "numpy": importlib.metadata.version("numpy"),
            "scikit-learn": importlib.metadata.version("scikit-learn"),
            "scipy": importlib.metadata.version("scipy"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--world-fixture",
        type=Path,
        default=DEFAULT_WORLD_FIXTURE,
    )
    parser.add_argument(
        "--evaluation-fixture",
        type=Path,
        default=DEFAULT_EVALUATION_FIXTURE,
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    world_fixture = load_world_fixture(args.world_fixture)
    evaluation_fixture = load_evaluation_fixture(args.evaluation_fixture)
    result = evaluate_structural_worlds(
        world_fixture,
        evaluation_fixture,
        world_fixture_path=args.world_fixture,
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    else:
        for world_id, item in result["worlds"].items():
            print(
                f"{world_id}: expected={item['expected_selector_decision']} "
                f"discovery={item['discovery_selector']['decision']} "
                f"confirmation={item['confirmation_selector']['decision']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
