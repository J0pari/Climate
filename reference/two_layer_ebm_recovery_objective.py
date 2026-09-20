#!/usr/bin/env python3
"""Climate-owned objective semantics for two-layer EBM parameter recovery.

This module owns only the climate-specific mapping from positive two-layer EBM
parameters to declared temperature observations, the fixed-variance Gaussian
residual/likelihood semantics, and common log-parameter bounds. Generic
optimization remains owned by external libraries.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from reference.two_layer_energy_balance import TwoLayerParameters
from reference.two_layer_forcing_protocols import simulate_protocol
from reference.two_layer_parameter_identifiability import (
    PARAMETER_IDS,
    centered_log_parameter_jacobian,
)

LOG_TWO_PI = math.log(2.0 * math.pi)


def parameter_vector(parameters: TwoLayerParameters) -> np.ndarray:
    return np.array(
        [
            parameters.surface_heat_capacity_w_yr_m2_k,
            parameters.deep_heat_capacity_w_yr_m2_k,
            parameters.ocean_heat_exchange_w_m2_k,
            parameters.climate_feedback_w_m2_k,
        ],
        dtype=float,
    )


def parameters_from_log(log_parameters: np.ndarray) -> TwoLayerParameters:
    values = np.exp(np.asarray(log_parameters, dtype=float))
    if values.shape != (4,) or not np.isfinite(values).all() or np.any(values <= 0.0):
        raise ValueError("log-parameter vector must resolve to four finite positive values")
    return TwoLayerParameters(
        surface_heat_capacity_w_yr_m2_k=float(values[0]),
        deep_heat_capacity_w_yr_m2_k=float(values[1]),
        ocean_heat_exchange_w_m2_k=float(values[2]),
        climate_feedback_w_m2_k=float(values[3]),
    )


def log_parameter_bounds(
    reference_parameters: TwoLayerParameters,
    bound_factor: float,
) -> tuple[np.ndarray, np.ndarray]:
    factor = float(bound_factor)
    if not math.isfinite(factor) or factor <= 1.0:
        raise ValueError("log-parameter bound factor must be finite and > 1")
    center = np.log(parameter_vector(reference_parameters))
    width = math.log(factor)
    return center - width, center + width


def protocol_temperature_outputs(
    parameters: TwoLayerParameters,
    protocol_fixture: dict[str, Any],
    protocol_ids: list[str],
    *,
    samples_per_segment: int,
) -> np.ndarray:
    protocols = {
        item["protocol_id"]: item for item in protocol_fixture.get("protocols", [])
    }
    if len(protocols) != len(protocol_fixture.get("protocols", [])):
        raise ValueError("forcing protocol fixture contains duplicate IDs")

    outputs: list[np.ndarray] = []
    for protocol_id in protocol_ids:
        if protocol_id not in protocols:
            raise ValueError("recovery protocol does not resolve: " + protocol_id)
        run = simulate_protocol(
            protocols[protocol_id],
            parameters,
            samples_per_segment=samples_per_segment,
        )
        states = np.asarray(run["state_k"], dtype=float)
        if states.ndim != 2 or states.shape[1] != 2 or not np.isfinite(states).all():
            raise RuntimeError("recovery protocol emitted invalid temperature states")
        outputs.append(states.reshape(-1))
    if not outputs:
        raise ValueError("recovery objective requires at least one protocol")
    return np.concatenate(outputs)


def standardized_gaussian_temperature_residual(
    log_parameters: np.ndarray,
    observed: np.ndarray,
    protocol_fixture: dict[str, Any],
    protocol_ids: list[str],
    *,
    samples_per_segment: int,
    noise_std_k: float,
) -> np.ndarray:
    noise = float(noise_std_k)
    if not math.isfinite(noise) or noise <= 0.0:
        raise ValueError("observation noise standard deviation must be finite and positive")
    observation = np.asarray(observed, dtype=float)
    if observation.ndim != 1 or not np.isfinite(observation).all():
        raise ValueError("observed temperature vector must be finite and one-dimensional")
    predicted = protocol_temperature_outputs(
        parameters_from_log(log_parameters),
        protocol_fixture,
        protocol_ids,
        samples_per_segment=samples_per_segment,
    )
    if predicted.shape != observation.shape:
        raise ValueError("observed temperature vector shape does not match protocol output")
    residual = (predicted - observation) / noise
    if not np.isfinite(residual).all():
        raise RuntimeError("standardized recovery residual became non-finite")
    return residual


def standardized_gaussian_temperature_jacobian(
    log_parameters: np.ndarray,
    observed_size: int,
    protocol_fixture: dict[str, Any],
    protocol_ids: list[str],
    *,
    samples_per_segment: int,
    noise_std_k: float,
    log_step: float,
) -> np.ndarray:
    noise = float(noise_std_k)
    if not math.isfinite(noise) or noise <= 0.0:
        raise ValueError("observation noise standard deviation must be finite and positive")

    parameters = parameters_from_log(log_parameters)

    def raw_output(candidate: TwoLayerParameters) -> np.ndarray:
        return protocol_temperature_outputs(
            candidate,
            protocol_fixture,
            protocol_ids,
            samples_per_segment=samples_per_segment,
        )

    derivative = centered_log_parameter_jacobian(
        parameters,
        raw_output,
        log_step=float(log_step),
    ) / noise
    if derivative.shape != (int(observed_size), len(PARAMETER_IDS)):
        raise RuntimeError("recovery Jacobian shape does not match objective observations")
    if not np.isfinite(derivative).all():
        raise RuntimeError("recovery Jacobian became non-finite")
    return derivative


def gaussian_nll_per_observation(
    physical_residual_k: np.ndarray,
    noise_std_k: float,
) -> float:
    noise = float(noise_std_k)
    if not math.isfinite(noise) or noise <= 0.0:
        raise ValueError("observation noise standard deviation must be finite and positive")
    residual = np.asarray(physical_residual_k, dtype=float)
    if residual.ndim != 1 or not np.isfinite(residual).all():
        raise ValueError("physical residual must be finite and one-dimensional")
    standardized = residual / noise
    return float(
        0.5
        * np.mean(
            standardized * standardized + LOG_TWO_PI + 2.0 * math.log(noise)
        )
    )
