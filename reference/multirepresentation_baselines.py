#!/usr/bin/env python3
"""Shared fitted baseline representations for multirepresentation benchmarks.

These helpers contain no world truth. They fit standard raw/linear latent
representations on discovery views and apply the resulting transforms to a
separate confirmation world.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.preprocessing import StandardScaler

from reference.multirepresentation_worlds import StructuralWorld


def standardized_concat_coordinate_inputs(
    view_a: np.ndarray,
    view_b: np.ndarray,
) -> np.ndarray:
    a = StandardScaler().fit_transform(np.asarray(view_a, dtype=float))
    b = StandardScaler().fit_transform(np.asarray(view_b, dtype=float))
    joined = np.column_stack([a, b])
    if not np.isfinite(joined).all():
        raise RuntimeError("standardized concatenation became non-finite")
    return joined


def factor_analysis_coordinate(
    view_a: np.ndarray,
    view_b: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    joined = standardized_concat_coordinate_inputs(view_a, view_b)
    coordinate = FactorAnalysis(
        n_components=1,
        svd_method=str(config["svd_method"]),
    ).fit_transform(joined)[:, 0]
    if not np.isfinite(coordinate).all():
        raise RuntimeError("factor analysis returned non-finite values")
    return coordinate


def discovery_fitted_representations(
    discovery: StructuralWorld,
    confirmation: StructuralWorld,
    baseline_config: dict[str, Any],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if discovery.world_id != confirmation.world_id:
        raise ValueError("discovery and confirmation world identities differ")
    scaler_a = StandardScaler().fit(discovery.view_a)
    scaler_b = StandardScaler().fit(discovery.view_b)
    discovery_a = scaler_a.transform(discovery.view_a)
    discovery_b = scaler_b.transform(discovery.view_b)
    confirmation_a = scaler_a.transform(confirmation.view_a)
    confirmation_b = scaler_b.transform(confirmation.view_b)
    discovery_raw = np.column_stack([discovery_a, discovery_b])
    confirmation_raw = np.column_stack([confirmation_a, confirmation_b])

    pca = PCA(n_components=1, svd_solver="full")
    discovery_pca = pca.fit_transform(discovery_raw)
    confirmation_pca = pca.transform(confirmation_raw)

    factor_config = baseline_config["factor_analysis"]
    factor = FactorAnalysis(
        n_components=1,
        svd_method=str(factor_config["svd_method"]),
    )
    discovery_factor = factor.fit_transform(discovery_raw)
    confirmation_factor = factor.transform(confirmation_raw)

    cca = CCA(n_components=1, scale=False, max_iter=2000, tol=1e-10)
    discovery_cca_a, discovery_cca_b = cca.fit_transform(
        discovery_a,
        discovery_b,
    )
    confirmation_cca_a, confirmation_cca_b = cca.transform(
        confirmation_a,
        confirmation_b,
    )
    score_scaler_a = StandardScaler().fit(discovery_cca_a)
    score_scaler_b = StandardScaler().fit(discovery_cca_b)
    discovery_score_a = score_scaler_a.transform(discovery_cca_a)[:, 0]
    discovery_score_b = score_scaler_b.transform(discovery_cca_b)[:, 0]
    confirmation_score_a = score_scaler_a.transform(confirmation_cca_a)[:, 0]
    confirmation_score_b = score_scaler_b.transform(confirmation_cca_b)[:, 0]
    if float(np.dot(discovery_score_a, discovery_score_b)) < 0.0:
        discovery_score_b = -discovery_score_b
        confirmation_score_b = -confirmation_score_b
    discovery_cca = (0.5 * (discovery_score_a + discovery_score_b))[:, None]
    confirmation_cca = (0.5 * (confirmation_score_a + confirmation_score_b))[:, None]

    representations = {
        "raw_concat": (discovery_raw, confirmation_raw),
        "concat_pca": (discovery_pca, confirmation_pca),
        "linear_cca": (discovery_cca, confirmation_cca),
        "factor_analysis": (discovery_factor, confirmation_factor),
    }
    for name, (train, test) in representations.items():
        if (
            train.ndim != 2
            or test.ndim != 2
            or train.shape[0] != discovery.view_a.shape[0]
            or test.shape[0] != confirmation.view_a.shape[0]
            or not np.isfinite(train).all()
            or not np.isfinite(test).all()
        ):
            raise RuntimeError(f"invalid discovery-fitted representation {name}")
    return representations
