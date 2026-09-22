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
from sklearn.feature_selection import mutual_info_regression
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.preprocessing import StandardScaler

from reference.multirepresentation_baselines import (
    discovery_fitted_representations,
    factor_analysis_coordinate,
)
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
    bind_world_fixture_authority,
    generate_worlds,
    load_fixture as load_world_fixture,
    world_set_digest,
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

    dimension_selector = payload.get("dimension_selector")
    if not isinstance(dimension_selector, dict):
        raise ValueError("dimension_selector policy is required")
    if dimension_selector.get("family") != "two_nearest_neighbor_integer_likelihood":
        raise ValueError("dimension selector family is unsupported")
    if dimension_selector.get("standardization") != "per_observed_coordinate":
        raise ValueError("dimension selector standardization policy is unsupported")
    if dimension_selector.get("neighbor_search") != "brute_euclidean":
        raise ValueError("dimension selector neighbor-search policy is unsupported")
    if dimension_selector.get("shared_dimension_rule") != "view_a_plus_view_b_minus_joint":
        raise ValueError("dimension selector shared-dimension rule is unsupported")

    structure_selector = payload.get("observational_structure_selector")
    if not isinstance(structure_selector, dict):
        raise ValueError("observational_structure_selector policy is required")
    if structure_selector.get("family") != "dependence_plus_intrinsic_dimension":
        raise ValueError("observational structure selector family is unsupported")
    expected_structure_decisions = {
        "no_cross_view_evidence_decision": "abstain_no_cross_view_evidence",
        "shared_only_decision": "shared_only_dimension_structure_supported",
        "shared_plus_private_decision":
            "shared_plus_private_dimension_structure_supported",
        "conflict_decision": "abstain_dependence_dimension_conflict",
    }
    for field, expected in expected_structure_decisions.items():
        if structure_selector.get(field) != expected:
            raise ValueError(
                f"observational structure selector {field} policy is unsupported"
            )

    probe = payload.get("matched_information_probe")
    if not isinstance(probe, dict):
        raise ValueError("matched_information_probe policy is required")
    probe_neighbors = probe.get("n_neighbors")
    if not isinstance(probe_neighbors, int) or probe_neighbors < 2:
        raise ValueError("matched-information n_neighbors must be >= 2")
    if probe.get("weights") not in {"uniform", "distance"}:
        raise ValueError("matched-information weights policy is unsupported")
    if probe.get("metric") != "minkowski" or int(probe.get("p", 0)) < 1:
        raise ValueError("matched-information metric policy is unsupported")
    expected_probe_representations = [
        "raw_concat",
        "concat_pca",
        "linear_cca",
        "factor_analysis",
    ]
    if probe.get("representations") != expected_probe_representations:
        raise ValueError(
            "matched-information representation identities changed without "
            "an evaluation fixture version"
        )
    return payload


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


def _twonn_integer_dimension(view: np.ndarray) -> dict[str, Any]:
    values = _standardize(view)
    if values.shape[1] < 1:
        raise ValueError("dimension selection requires at least one observed coordinate")
    distances = NearestNeighbors(
        n_neighbors=3,
        algorithm="brute",
        metric="euclidean",
    ).fit(values).kneighbors(
        values,
        return_distance=True,
    )[0][:, 1:3]
    if (
        distances.shape != (values.shape[0], 2)
        or not np.isfinite(distances).all()
        or np.any(distances[:, 0] <= 0.0)
        or np.any(distances[:, 1] < distances[:, 0])
    ):
        raise ValueError("two-nearest-neighbor distances are invalid")
    ratios = distances[:, 1] / distances[:, 0]
    log_ratio_sum = float(np.log(ratios).sum())
    if not math.isfinite(log_ratio_sum) or log_ratio_sum <= 0.0:
        raise ValueError("two-nearest-neighbor likelihood is degenerate")

    sample_count = int(values.shape[0])
    candidate_log_likelihoods: dict[str, float] = {}
    best_dimension = None
    best_log_likelihood = -math.inf
    for dimension in range(1, values.shape[1] + 1):
        log_likelihood = (
            sample_count * math.log(float(dimension))
            - (dimension + 1.0) * log_ratio_sum
        )
        candidate_log_likelihoods[str(dimension)] = log_likelihood
        if log_likelihood > best_log_likelihood:
            best_log_likelihood = log_likelihood
            best_dimension = dimension
    if best_dimension is None:
        raise RuntimeError("dimension likelihood produced no candidate")
    return {
        "selected_dimension": int(best_dimension),
        "continuous_mle": float(sample_count / log_ratio_sum),
        "ambient_dimension": int(values.shape[1]),
        "sample_count": sample_count,
        "candidate_log_likelihoods": candidate_log_likelihoods,
    }


def select_world_dimensions(
    world: StructuralWorld,
    policy: dict[str, Any],
) -> dict[str, Any]:
    if policy.get("family") != "two_nearest_neighbor_integer_likelihood":
        raise ValueError("dimension selector family is unsupported")
    if policy.get("standardization") != "per_observed_coordinate":
        raise ValueError("dimension selector standardization policy is unsupported")
    if policy.get("neighbor_search") != "brute_euclidean":
        raise ValueError("dimension selector neighbor-search policy is unsupported")
    if policy.get("shared_dimension_rule") != "view_a_plus_view_b_minus_joint":
        raise ValueError("dimension selector shared-dimension rule is unsupported")

    view_a = _twonn_integer_dimension(world.view_a)
    view_b = _twonn_integer_dimension(world.view_b)
    joint = _twonn_integer_dimension(np.column_stack([world.view_a, world.view_b]))
    selected_shared = (
        int(view_a["selected_dimension"])
        + int(view_b["selected_dimension"])
        - int(joint["selected_dimension"])
    )
    if selected_shared < 0 or selected_shared > min(
        int(view_a["selected_dimension"]),
        int(view_b["selected_dimension"]),
    ):
        return {
            "status": "inconsistent_dimension_decomposition",
            "view_a": view_a,
            "view_b": view_b,
            "joint": joint,
        }
    return {
        "status": "selected",
        "view_a": view_a,
        "view_b": view_b,
        "joint": joint,
        "selected_shared_dimension": selected_shared,
        "selected_private_dimensions": [
            int(view_a["selected_dimension"]) - selected_shared,
            int(view_b["selected_dimension"]) - selected_shared,
        ],
    }


def coarse_observational_structure(
    dependence: dict[str, Any],
    dimensions: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    """Compose only observationally supported structure without generative labels."""
    if policy.get("family") != "dependence_plus_intrinsic_dimension":
        raise ValueError("observational structure selector family is unsupported")
    dependence_decision = dependence.get("decision")
    if dependence_decision == "abstain_no_cross_view_evidence":
        return {
            "status": "abstained",
            "decision": policy["no_cross_view_evidence_decision"],
        }
    if dependence_decision != "shared_coordinate_supported":
        raise ValueError("dependence selector emitted an unsupported decision")
    if dimensions.get("status") != "selected":
        return {
            "status": "abstained",
            "decision": policy["conflict_decision"],
        }

    shared = int(dimensions["selected_shared_dimension"])
    private = [int(value) for value in dimensions["selected_private_dimensions"]]
    if len(private) != 2 or any(value < 0 for value in private):
        raise ValueError("dimension selector emitted invalid private dimensions")
    if shared <= 0:
        return {
            "status": "abstained",
            "decision": policy["conflict_decision"],
        }
    if private == [0, 0]:
        decision = policy["shared_only_decision"]
    else:
        decision = policy["shared_plus_private_decision"]
    return {
        "status": "supported",
        "decision": decision,
        "selected_shared_dimension": shared,
        "selected_private_dimensions": private,
    }


def _expected_coarse_observational_structure(world: StructuralWorld) -> str:
    target_name = world.ground_truth.get("shared_evaluation_target_name")
    if target_name is None:
        return "abstain_no_cross_view_evidence"
    shared = int(world.ground_truth["shared_dimension"])
    private = [int(value) for value in world.ground_truth["private_dimensions"]]
    if shared <= 0:
        raise ValueError(
            f"{world.world_id} has a shared target but no authoritative shared dimension"
        )
    if private == [0, 0]:
        return "shared_only_dimension_structure_supported"
    return "shared_plus_private_dimension_structure_supported"


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
        "factor_analysis": lambda: factor_analysis_coordinate(
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


def _matched_information_probe(
    discovery: StructuralWorld,
    confirmation: StructuralWorld,
    baseline_config: dict[str, Any],
    probe_config: dict[str, Any],
) -> dict[str, Any] | None:
    target_name = discovery.ground_truth.get("shared_evaluation_target_name")
    if target_name is None:
        return None
    if (
        not isinstance(target_name, str)
        or target_name not in discovery.targets
        or target_name not in confirmation.targets
    ):
        raise ValueError("authoritative shared evaluation target is inconsistent")

    train_target = np.asarray(discovery.targets[target_name], dtype=float)
    test_target = np.asarray(confirmation.targets[target_name], dtype=float)
    if (
        train_target.shape != (discovery.view_a.shape[0],)
        or test_target.shape != (confirmation.view_a.shape[0],)
        or not np.isfinite(train_target).all()
        or not np.isfinite(test_target).all()
    ):
        raise ValueError("matched-information targets are invalid")
    target_scale = float(np.std(test_target))
    if not math.isfinite(target_scale) or target_scale <= 0.0:
        raise ValueError("confirmation target scale must be finite and positive")

    representations = discovery_fitted_representations(
        discovery,
        confirmation,
        baseline_config,
    )
    requested = list(probe_config["representations"])
    if set(representations) != set(requested):
        raise ValueError("matched-information representation policy is inconsistent")

    results: dict[str, Any] = {}
    for name in requested:
        train, test = representations[name]
        probe = KNeighborsRegressor(
            n_neighbors=int(probe_config["n_neighbors"]),
            weights=str(probe_config["weights"]),
            metric=str(probe_config["metric"]),
            p=int(probe_config["p"]),
        )
        probe.fit(train, train_target)
        predicted = np.asarray(probe.predict(test), dtype=float)
        if predicted.shape != test_target.shape or not np.isfinite(predicted).all():
            raise RuntimeError(f"matched-information probe failed for {name}")
        rmse = float(np.sqrt(np.mean((predicted - test_target) ** 2)))
        results[name] = {
            "representation_dimension": int(test.shape[1]),
            "confirmation_normalized_rmse": rmse / target_scale,
            "confirmation_abs_spearman": abs_spearman(predicted, test_target),
        }
    return results


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


def evaluate_structural_worlds(
    world_fixture: dict[str, Any],
    evaluation_fixture: dict[str, Any],
    *,
    world_fixture_path: Path = DEFAULT_WORLD_FIXTURE,
) -> dict[str, Any]:
    bind_world_fixture_authority(
        world_fixture,
        evaluation_fixture,
        world_fixture_path=world_fixture_path,
    )
    discovery = generate_worlds(world_fixture)
    confirmation = generate_worlds(
        world_fixture,
        seed_offset=int(evaluation_fixture["confirmation_seed_offset"]),
    )
    if set(discovery) != set(confirmation):
        raise RuntimeError("discovery and confirmation world identities differ")

    selector_policy = evaluation_fixture["dependence_selector"]
    dimension_policy = evaluation_fixture["dimension_selector"]
    structure_policy = evaluation_fixture["observational_structure_selector"]
    baseline_config = evaluation_fixture["coordinate_baselines"]
    probe_config = evaluation_fixture["matched_information_probe"]
    world_results: dict[str, Any] = {}
    discovery_calibrated: list[bool] = []
    confirmation_calibrated: list[bool] = []
    discovery_dimensions_recovered: list[bool] = []
    confirmation_dimensions_recovered: list[bool] = []
    discovery_structure_calibrated: list[bool] = []
    confirmation_structure_calibrated: list[bool] = []

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
        discovery_dimensions = select_world_dimensions(
            discovery_world,
            dimension_policy,
        )
        confirmation_dimensions = select_world_dimensions(
            confirmation_world,
            dimension_policy,
        )
        expected_dimensions = {
            "shared_dimension": int(discovery_world.ground_truth["shared_dimension"]),
            "private_dimensions": [
                int(value)
                for value in discovery_world.ground_truth["private_dimensions"]
            ],
        }
        discovery_dimension_ok = (
            discovery_dimensions.get("status") == "selected"
            and discovery_dimensions.get("selected_shared_dimension")
            == expected_dimensions["shared_dimension"]
            and discovery_dimensions.get("selected_private_dimensions")
            == expected_dimensions["private_dimensions"]
        )
        confirmation_dimension_ok = (
            confirmation_dimensions.get("status") == "selected"
            and confirmation_dimensions.get("selected_shared_dimension")
            == expected_dimensions["shared_dimension"]
            and confirmation_dimensions.get("selected_private_dimensions")
            == expected_dimensions["private_dimensions"]
        )
        expected_structure = _expected_coarse_observational_structure(
            discovery_world
        )
        discovery_structure = coarse_observational_structure(
            discovery_selector,
            discovery_dimensions,
            structure_policy,
        )
        confirmation_structure = coarse_observational_structure(
            confirmation_selector,
            confirmation_dimensions,
            structure_policy,
        )
        discovery_structure_ok = (
            discovery_structure["decision"] == expected_structure
        )
        confirmation_structure_ok = (
            confirmation_structure["decision"] == expected_structure
        )
        discovery_ok = discovery_selector["decision"] == expected
        confirmation_ok = confirmation_selector["decision"] == expected
        discovery_calibrated.append(discovery_ok)
        confirmation_calibrated.append(confirmation_ok)
        discovery_dimensions_recovered.append(discovery_dimension_ok)
        confirmation_dimensions_recovered.append(confirmation_dimension_ok)
        discovery_structure_calibrated.append(discovery_structure_ok)
        confirmation_structure_calibrated.append(confirmation_structure_ok)

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
            "expected_dimensions": expected_dimensions,
            "discovery_dimension_selection": discovery_dimensions,
            "confirmation_dimension_selection": confirmation_dimensions,
            "discovery_dimensions_recovered": discovery_dimension_ok,
            "confirmation_dimensions_recovered": confirmation_dimension_ok,
            "expected_coarse_observational_structure": expected_structure,
            "discovery_coarse_observational_structure": discovery_structure,
            "confirmation_coarse_observational_structure": confirmation_structure,
            "discovery_coarse_structure_calibrated": discovery_structure_ok,
            "confirmation_coarse_structure_calibrated": confirmation_structure_ok,
            "confirmation_coordinate_baselines": _evaluate_coordinate_baselines(
                confirmation_world,
                baseline_config,
            ),
            "matched_information_probe": _matched_information_probe(
                discovery_world,
                confirmation_world,
                baseline_config,
                probe_config,
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
        "discovery_sample_digest": world_set_digest(discovery),
        "confirmation_sample_digest": world_set_digest(confirmation),
        "selector_summary": {
            "all_discovery_decisions_calibrated": all(discovery_calibrated),
            "all_confirmation_decisions_calibrated": all(confirmation_calibrated),
        },
        "dimension_summary": {
            "all_discovery_dimensions_recovered": all(
                discovery_dimensions_recovered
            ),
            "all_confirmation_dimensions_recovered": all(
                confirmation_dimensions_recovered
            ),
        },
        "coarse_structure_summary": {
            "all_discovery_decisions_calibrated": all(
                discovery_structure_calibrated
            ),
            "all_confirmation_decisions_calibrated": all(
                confirmation_structure_calibrated
            ),
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
