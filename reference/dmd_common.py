#!/usr/bin/env python3
"""Shared PyDMD adapter for Climate reference benchmarks.

This module owns only generic DMD fitting and prediction semantics. Scientific
method identities remain task-specific and declare their own assumptions,
rank-selection policy, targets, and interpretation boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from pydmd import DMD


@dataclass(frozen=True)
class ExactDMDFit:
    model: DMD
    eigenvalues: np.ndarray
    rank: int
    training_relative_error: float


def relative_prediction_error(
    predicted: np.ndarray,
    expected: np.ndarray,
) -> float:
    left = np.asarray(predicted, dtype=float)
    right = np.asarray(expected, dtype=float)
    if (
        left.shape != right.shape
        or left.ndim != 2
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
    ):
        raise ValueError("DMD prediction arrays must be finite and shape matched")
    denominator = max(float(np.linalg.norm(right)), np.finfo(float).tiny)
    value = float(np.linalg.norm(left - right) / denominator)
    if not math.isfinite(value):
        raise RuntimeError("DMD relative prediction error became non-finite")
    return value


def fit_exact_dmd(
    x_samples: np.ndarray,
    y_samples: np.ndarray,
    *,
    rank: int,
) -> ExactDMDFit:
    x = np.asarray(x_samples, dtype=float)
    y = np.asarray(y_samples, dtype=float)
    if x.shape != y.shape or x.ndim != 2 or x.shape[0] < 2:
        raise ValueError("DMD samples must be matched two-dimensional arrays")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("DMD samples must be finite")
    if not isinstance(rank, int) or rank < 1 or rank > min(x.shape):
        raise ValueError("DMD rank is incompatible with sample matrix shape")

    model = DMD(svd_rank=rank, exact=True, tlsq_rank=0)
    model.fit(x.T, y.T)
    eigenvalues = np.asarray(model.eigs)
    predicted = np.asarray(model.predict(x.T)).T
    training_error = relative_prediction_error(predicted, y)
    if (
        eigenvalues.ndim != 1
        or eigenvalues.size != rank
        or not np.isfinite(eigenvalues).all()
    ):
        raise RuntimeError("DMD produced invalid eigenvalues")
    eigenvalues = np.array(eigenvalues, copy=True)
    eigenvalues.setflags(write=False)
    return ExactDMDFit(
        model=model,
        eigenvalues=eigenvalues,
        rank=rank,
        training_relative_error=training_error,
    )


def predict_exact_dmd(
    fit: ExactDMDFit,
    x_samples: np.ndarray,
) -> np.ndarray:
    x = np.asarray(x_samples, dtype=float)
    if x.ndim != 2 or x.shape[1] != fit.model.modes.shape[0]:
        raise ValueError("DMD prediction samples do not match fitted state dimension")
    if not np.isfinite(x).all():
        raise ValueError("DMD prediction samples must be finite")
    predicted = np.asarray(fit.model.predict(x.T)).T
    if predicted.shape != x.shape or not np.isfinite(predicted).all():
        raise RuntimeError("DMD prediction produced invalid values")
    return predicted
