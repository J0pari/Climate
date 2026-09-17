#!/usr/bin/env python3
"""Reference experiment for a common coordinate across heterogeneous views.

The Climate-specific contribution here is the synthetic scientific question and its
controls. PCA, CCA, spectral embedding, and jointly smooth functions are delegated
to maintained numerical libraries.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = ROOT / "fixtures" / "multirepresentation" / "shared-effective-coordinate-v1.json"


@dataclass(frozen=True)
class SyntheticViews:
    view_a: np.ndarray
    view_b: np.ndarray
    common_coordinate: np.ndarray
    observation_nuisance: np.ndarray


def load_fixture(path: Path = DEFAULT_FIXTURE) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _uniform(rng: np.random.Generator, bounds: list[float], count: int) -> np.ndarray:
    if len(bounds) != 2 or not bounds[0] < bounds[1]:
        raise ValueError(f"invalid latent bounds: {bounds!r}")
    return rng.uniform(float(bounds[0]), float(bounds[1]), size=count)


def generate_views(fixture: dict[str, Any]) -> SyntheticViews:
    count = int(fixture["sample_count"])
    if count < 64:
        raise ValueError("multirepresentation fixture requires at least 64 samples")

    rng = np.random.default_rng(int(fixture["seed"]))
    domain = fixture["latent_domain"]
    background = _uniform(rng, domain["background"], count)
    coupling = _uniform(rng, domain["coupling"], count)
    nuisance = _uniform(rng, domain["observation_nuisance"], count)

    common = background + coupling**2
    radius = (nuisance + common / 2.0 + 2.0 / 3.0) / 2.0
    angle = 2.0 * math.pi * nuisance

    view_a = np.column_stack([background, coupling])
    view_b = np.column_stack([radius * np.cos(angle), radius * np.sin(angle)])

    for name, value in (
        ("view_a", view_a),
        ("view_b", view_b),
        ("common_coordinate", common),
        ("observation_nuisance", nuisance),
    ):
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values")

    return SyntheticViews(
        view_a=view_a,
        view_b=view_b,
        common_coordinate=common,
        observation_nuisance=nuisance,
    )


def _scale(view: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(view)


def concat_pca(view_a: np.ndarray, view_b: np.ndarray) -> np.ndarray:
    joined = np.column_stack([_scale(view_a), _scale(view_b)])
    return PCA(n_components=1, svd_solver="full").fit_transform(joined)[:, 0]


def linear_cca(view_a: np.ndarray, view_b: np.ndarray) -> np.ndarray:
    a = _scale(view_a)
    b = _scale(view_b)
    cca = CCA(n_components=1, scale=False, max_iter=2000, tol=1e-10)
    score_a, score_b = cca.fit_transform(a, b)
    score_a = StandardScaler().fit_transform(score_a)[:, 0]
    score_b = StandardScaler().fit_transform(score_b)[:, 0]
    if float(np.dot(score_a, score_b)) < 0.0:
        score_b = -score_b
    return 0.5 * (score_a + score_b)


def concat_spectral(
    view_a: np.ndarray, view_b: np.ndarray, *, n_neighbors: int, random_state: int
) -> np.ndarray:
    joined = np.column_stack([_scale(view_a), _scale(view_b)])
    embedding = SpectralEmbedding(
        n_components=1,
        affinity="nearest_neighbors",
        n_neighbors=n_neighbors,
        eigen_solver="arpack",
        random_state=random_state,
    )
    return embedding.fit_transform(joined)[:, 0]


def jointly_smooth(
    view_a: np.ndarray, view_b: np.ndarray, method_config: dict[str, Any]
) -> np.ndarray:
    try:
        import datafold.pcfold as pfold
        from datafold.dynfold.jsf import JointlySmoothFunctions
    except ImportError as exc:  # fail closed rather than silently substitute an implementation
        raise RuntimeError("datafold is required for the JSF method identity") from exc

    a = _scale(view_a)
    b = _scale(view_b)
    joined = np.column_stack([a, b])
    split = a.shape[1]

    kernel_a = pfold.kernels.ContinuousNNKernel(
        k_neighbor=int(method_config["k_neighbor"]),
        delta=float(method_config["delta"]),
    )
    kernel_b = pfold.kernels.ContinuousNNKernel(
        k_neighbor=int(method_config["k_neighbor"]),
        delta=float(method_config["delta"]),
    )
    model = JointlySmoothFunctions(
        data_splits=[
            ("resolved_subsystem_state", kernel_a, slice(0, split)),
            ("nonlinear_observation", kernel_b, slice(split, joined.shape[1])),
        ],
        n_kernel_eigenvectors=int(method_config["n_kernel_eigenvectors"]),
        n_jointly_smooth_functions=int(method_config["n_jointly_smooth_functions"]),
        kernel_eigenvalue_cut_off=float(method_config["kernel_eigenvalue_cut_off"]),
        eigenvector_tolerance=float(method_config["eigenvector_tolerance"]),
    )
    vectors = np.asarray(model.fit_transform(joined), dtype=float)
    if vectors.ndim != 2 or vectors.shape[1] < 2:
        raise RuntimeError("JSF did not return a nonconstant candidate coordinate")
    coordinate = vectors[:, 1]
    if not np.isfinite(coordinate).all():
        raise RuntimeError("JSF returned non-finite values")
    return coordinate


def abs_spearman(coordinate: np.ndarray, target: np.ndarray) -> float:
    if coordinate.ndim != 1 or target.ndim != 1 or coordinate.shape != target.shape:
        raise ValueError("Spearman inputs must be matched one-dimensional arrays")
    statistic = float(spearmanr(coordinate, target).statistic)
    if not math.isfinite(statistic):
        raise ValueError("Spearman statistic is non-finite")
    return abs(statistic)


def run_experiment(fixture: dict[str, Any]) -> dict[str, Any]:
    views = generate_views(fixture)
    spectral_config = fixture["methods"]["spectral_baseline"]
    jsf_config = fixture["methods"]["jsf_kernel"]

    coordinates = {
        "concat_pca": concat_pca(views.view_a, views.view_b),
        "linear_cca": linear_cca(views.view_a, views.view_b),
        "concat_spectral": concat_spectral(
            views.view_a,
            views.view_b,
            n_neighbors=int(spectral_config["n_neighbors"]),
            random_state=int(spectral_config["random_state"]),
        ),
        "jointly_smooth": jointly_smooth(views.view_a, views.view_b, jsf_config),
    }

    metrics: dict[str, dict[str, float]] = {}
    for name, coordinate in coordinates.items():
        metrics[name] = {
            "shared_abs_spearman": abs_spearman(coordinate, views.common_coordinate),
            "nuisance_abs_spearman": abs_spearman(coordinate, views.observation_nuisance),
        }

    shuffle_rng = np.random.default_rng(int(fixture["methods"]["shuffle_control_seed"]))
    shuffled_b = views.view_b[shuffle_rng.permutation(views.view_b.shape[0])]
    shuffled_coordinate = jointly_smooth(views.view_a, shuffled_b, jsf_config)
    shuffled_shared = abs_spearman(shuffled_coordinate, views.common_coordinate)

    guards = fixture["evaluation"]["acceptance_guards"]
    baseline_best = max(
        metrics[name]["shared_abs_spearman"]
        for name in ("concat_pca", "linear_cca", "concat_spectral")
    )
    jsf_shared = metrics["jointly_smooth"]["shared_abs_spearman"]
    jsf_nuisance = metrics["jointly_smooth"]["nuisance_abs_spearman"]
    advantage = jsf_shared - baseline_best

    checks = {
        "jsf_recovers_shared_coordinate": jsf_shared
        >= float(guards["jsf_shared_abs_spearman_min"]),
        "jsf_rejects_observation_nuisance": jsf_nuisance
        <= float(guards["jsf_nuisance_abs_spearman_max"]),
        "jsf_beats_best_baseline": advantage
        >= float(guards["jsf_advantage_over_best_baseline_min"]),
        "shuffle_breaks_cross_view_structure": shuffled_shared
        <= float(guards["shuffled_correspondence_shared_abs_spearman_max"]),
    }

    return {
        "fixture_id": fixture["fixture_id"],
        "sample_count": int(fixture["sample_count"]),
        "implementation_versions": {
            "datafold": importlib.metadata.version("datafold"),
            "scikit-learn": importlib.metadata.version("scikit-learn"),
            "numpy": importlib.metadata.version("numpy"),
            "scipy": importlib.metadata.version("scipy"),
        },
        "metrics": metrics,
        "controls": {
            "shuffled_correspondence_shared_abs_spearman": shuffled_shared,
            "best_baseline_shared_abs_spearman": baseline_best,
            "jsf_advantage_over_best_baseline": advantage,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = run_experiment(load_fixture(args.fixture))
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        for method, metrics in result["metrics"].items():
            print(
                f"{method}: shared={metrics['shared_abs_spearman']:.6f} "
                f"nuisance={metrics['nuisance_abs_spearman']:.6f}"
            )
        print(
            "shuffled correspondence: "
            f"shared={result['controls']['shuffled_correspondence_shared_abs_spearman']:.6f}"
        )
        print(f"passed={result['passed']}")

    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
