#!/usr/bin/env python3
"""Preregistered information-geometry recovery benchmark for the two-layer EBM.

Climate owns the likelihood, train/confirmation split, parameter coordinates,
controls, and interpretation. Existing two-layer EBM references own climate
physics and centered log-parameter sensitivities. SciPy owns generic TRF and
L-BFGS-B optimization implementations.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.optimize import least_squares, minimize

from reference.two_layer_energy_balance import (
    TwoLayerParameters,
    load_fixture as load_ebm_fixture,
    parameters_from_fixture,
)
from reference.two_layer_forcing_protocols import (
    load_protocol_fixture,
    simulate_protocol,
)
from reference.two_layer_parameter_identifiability import (
    PARAMETER_IDS,
    centered_log_parameter_jacobian,
)

DEFAULT_EBM_FIXTURE = ROOT / "fixtures" / "physics" / "two-layer-ebm-geoffroy-mean-v1.json"
DEFAULT_PROTOCOL_FIXTURE = ROOT / "fixtures" / "physics" / "two-layer-ebm-forcing-protocols-v1.json"
DEFAULT_RECOVERY_FIXTURE = ROOT / "fixtures" / "physics" / "two-layer-ebm-information-geometry-recovery-v1.json"

LOG_TWO_PI = math.log(2.0 * math.pi)


@dataclass
class EvalCounter:
    residual_calls: int = 0
    jacobian_calls: int = 0


def load_recovery_fixture(path: Path = DEFAULT_RECOVERY_FIXTURE) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported information-geometry recovery fixture schema")
    if payload.get("parameter_ids") != list(PARAMETER_IDS):
        raise ValueError("recovery fixture parameter ordering changed without a version")
    if payload.get("trial_pairing") != "cartesian_product_dataset_seeds_x_start_log_parameter_offsets":
        raise ValueError("recovery fixture trial pairing is not the preregistered Cartesian product")
    return payload


def _parameter_vector(parameters: TwoLayerParameters) -> np.ndarray:
    return np.array([
        parameters.surface_heat_capacity_w_yr_m2_k,
        parameters.deep_heat_capacity_w_yr_m2_k,
        parameters.ocean_heat_exchange_w_m2_k,
        parameters.climate_feedback_w_m2_k,
    ], dtype=float)


def _parameters_from_log(log_parameters: np.ndarray) -> TwoLayerParameters:
    values = np.exp(np.asarray(log_parameters, dtype=float))
    if values.shape != (4,) or not np.isfinite(values).all() or np.any(values <= 0.0):
        raise ValueError("log-parameter vector must resolve to four finite positive values")
    return TwoLayerParameters(
        surface_heat_capacity_w_yr_m2_k=float(values[0]),
        deep_heat_capacity_w_yr_m2_k=float(values[1]),
        ocean_heat_exchange_w_m2_k=float(values[2]),
        climate_feedback_w_m2_k=float(values[3]),
    )


def _protocol_map(protocol_fixture: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = {item["protocol_id"]: item for item in protocol_fixture.get("protocols", [])}
    if len(result) != len(protocol_fixture.get("protocols", [])):
        raise ValueError("forcing protocol fixture contains duplicate IDs")
    return result


def _temperature_outputs(
    parameters: TwoLayerParameters,
    protocol_fixture: dict[str, Any],
    protocol_ids: list[str],
    *,
    samples_per_segment: int,
) -> np.ndarray:
    protocols = _protocol_map(protocol_fixture)
    outputs: list[np.ndarray] = []
    for protocol_id in protocol_ids:
        if protocol_id not in protocols:
            raise ValueError("recovery protocol does not resolve: " + protocol_id)
        run = simulate_protocol(
            protocols[protocol_id], parameters, samples_per_segment=samples_per_segment
        )
        states = np.asarray(run["state_k"], dtype=float)
        if states.ndim != 2 or states.shape[1] != 2 or not np.isfinite(states).all():
            raise RuntimeError("recovery protocol emitted invalid temperature states")
        outputs.append(states.reshape(-1))
    if not outputs:
        raise ValueError("recovery benchmark requires at least one protocol")
    return np.concatenate(outputs)


def _nll_per_observation(residual: np.ndarray, noise_std: float) -> float:
    standardized = np.asarray(residual, dtype=float) / noise_std
    return float(0.5 * np.mean(standardized * standardized + LOG_TWO_PI + 2.0 * math.log(noise_std)))


def _log_parameter_rmse(log_parameters: np.ndarray, truth_log: np.ndarray) -> float:
    delta = np.asarray(log_parameters, dtype=float) - np.asarray(truth_log, dtype=float)
    return float(np.sqrt(np.mean(delta * delta)))


def _make_problem(
    observed: np.ndarray,
    protocol_fixture: dict[str, Any],
    protocol_ids: list[str],
    *,
    samples_per_segment: int,
    noise_std: float,
    log_step: float,
    counter: EvalCounter,
) -> tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    def raw_output(candidate: TwoLayerParameters) -> np.ndarray:
        return _temperature_outputs(
            candidate,
            protocol_fixture,
            protocol_ids,
            samples_per_segment=samples_per_segment,
        )

    def residual(log_parameters: np.ndarray) -> np.ndarray:
        counter.residual_calls += 1
        predicted = raw_output(_parameters_from_log(log_parameters))
        values = (predicted - observed) / noise_std
        if not np.isfinite(values).all():
            raise RuntimeError("standardized recovery residual became non-finite")
        return values

    def jacobian(log_parameters: np.ndarray) -> np.ndarray:
        counter.jacobian_calls += 1
        parameters = _parameters_from_log(log_parameters)
        derivative = centered_log_parameter_jacobian(
            parameters, raw_output, log_step=log_step
        ) / noise_std
        if derivative.shape != (observed.size, 4) or not np.isfinite(derivative).all():
            raise RuntimeError("recovery Jacobian is invalid")
        return derivative

    return residual, jacobian


def _natural_gradient(
    residual: Callable[[np.ndarray], np.ndarray],
    jacobian: Callable[[np.ndarray], np.ndarray],
    start: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    policy: dict[str, Any],
    *,
    fixed_fisher: bool,
) -> dict[str, Any]:
    x = np.clip(np.asarray(start, dtype=float), lower, upper)
    c1 = float(policy["armijo_c1"])
    max_iterations = int(policy["max_iterations"])
    max_backtracks = int(policy["max_backtracks"])
    tolerance = float(policy["parameter_step_tolerance"])
    initial_fisher: np.ndarray | None = None
    converged = False
    status = "max_iterations"
    iterations = 0

    for iteration in range(max_iterations):
        iterations = iteration + 1
        r = residual(x)
        j = jacobian(x)
        gradient = j.T @ r
        fisher = j.T @ j
        if initial_fisher is None:
            initial_fisher = np.array(fisher, copy=True)
        matrix = initial_fisher if fixed_fisher else fisher
        try:
            direction = -np.linalg.solve(matrix, gradient)
        except np.linalg.LinAlgError:
            status = "fisher_singular"
            break
        if not np.isfinite(direction).all():
            status = "nonfinite_direction"
            break
        if float(np.linalg.norm(direction)) <= tolerance:
            converged = True
            status = "parameter_step_tolerance"
            break

        objective = 0.5 * float(r @ r)
        slope = float(gradient @ direction)
        if not math.isfinite(slope) or slope >= 0.0:
            status = "non_descent_direction"
            break
        accepted = False
        for backtrack in range(max_backtracks + 1):
            scale = 0.5 ** backtrack
            candidate = np.clip(x + scale * direction, lower, upper)
            step = candidate - x
            if float(np.linalg.norm(step)) <= tolerance:
                converged = True
                status = "projected_step_tolerance"
                x = candidate
                accepted = True
                break
            candidate_r = residual(candidate)
            candidate_objective = 0.5 * float(candidate_r @ candidate_r)
            if candidate_objective <= objective + c1 * float(gradient @ step):
                x = candidate
                accepted = True
                break
        if converged:
            break
        if not accepted:
            status = "line_search_failed"
            break
    return {
        "log_parameters": x,
        "converged": converged,
        "status": status,
        "iterations": iterations,
    }


def _run_method(
    method_id: str,
    observed_discovery: np.ndarray,
    observed_confirmation: np.ndarray,
    protocol_fixture: dict[str, Any],
    recovery_fixture: dict[str, Any],
    start: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    truth_log: np.ndarray,
) -> dict[str, Any]:
    noise_std = float(recovery_fixture["observation_noise_std_k"])
    samples_per_segment = int(recovery_fixture["samples_per_segment"])
    log_step = float(recovery_fixture["centered_log_jacobian_step"])
    discovery_ids = list(recovery_fixture["discovery_protocol_ids"])
    confirmation_ids = list(recovery_fixture["confirmation_protocol_ids"])
    counter = EvalCounter()
    residual, jacobian = _make_problem(
        observed_discovery,
        protocol_fixture,
        discovery_ids,
        samples_per_segment=samples_per_segment,
        noise_std=noise_std,
        log_step=log_step,
        counter=counter,
    )

    if method_id == "information_geometry.natural_gradient.armijo_v1":
        fitted = _natural_gradient(
            residual, jacobian, start, lower, upper,
            recovery_fixture["natural_gradient_policy"], fixed_fisher=False,
        )
        x = fitted["log_parameters"]
        converged = bool(fitted["converged"])
        status = str(fitted["status"])
        iterations = int(fitted["iterations"])
    elif method_id == "information_geometry.fixed_initial_fisher.armijo_v1":
        fitted = _natural_gradient(
            residual, jacobian, start, lower, upper,
            recovery_fixture["natural_gradient_policy"], fixed_fisher=True,
        )
        x = fitted["log_parameters"]
        converged = bool(fitted["converged"])
        status = str(fitted["status"])
        iterations = int(fitted["iterations"])
    elif method_id == "optimization.scipy_trf_least_squares.jac_scaled_v1":
        policy = recovery_fixture["scipy_trf_policy"]
        fit = least_squares(
            residual,
            start,
            jac=jacobian,
            bounds=(lower, upper),
            method="trf",
            x_scale="jac",
            max_nfev=int(policy["max_function_evaluations"]),
            ftol=float(policy["ftol"]),
            xtol=float(policy["xtol"]),
            gtol=float(policy["gtol"]),
        )
        x = np.asarray(fit.x, dtype=float)
        converged = bool(fit.success)
        status = str(fit.message)
        iterations = int(getattr(fit, "njev", 0) or 0)
    elif method_id == "optimization.scipy_lbfgsb.exact_gradient_v1":
        policy = recovery_fixture["scipy_lbfgsb_policy"]
        def objective(value: np.ndarray) -> float:
            r = residual(value)
            return 0.5 * float(r @ r)
        def gradient(value: np.ndarray) -> np.ndarray:
            r = residual(value)
            j = jacobian(value)
            return j.T @ r
        fit = minimize(
            objective,
            start,
            jac=gradient,
            method="L-BFGS-B",
            bounds=list(zip(lower, upper, strict=True)),
            options={
                "maxiter": int(policy["max_iterations"]),
                "ftol": float(policy["ftol"]),
                "gtol": float(policy["gtol"]),
                "maxls": int(policy["max_line_search_steps"]),
            },
        )
        x = np.asarray(fit.x, dtype=float)
        converged = bool(fit.success)
        status = str(fit.message)
        iterations = int(fit.nit)
    else:
        raise ValueError("unknown recovery method: " + method_id)

    fitted_parameters = _parameters_from_log(x)
    predicted_confirmation = _temperature_outputs(
        fitted_parameters,
        protocol_fixture,
        confirmation_ids,
        samples_per_segment=samples_per_segment,
    )
    heldout_residual = predicted_confirmation - observed_confirmation
    return {
        "method_id": method_id,
        "converged": converged,
        "status": status,
        "iterations": iterations,
        "residual_evaluations": int(counter.residual_calls),
        "jacobian_evaluations": int(counter.jacobian_calls),
        "log_parameter_rmse": _log_parameter_rmse(x, truth_log),
        "heldout_nll_per_observation": _nll_per_observation(heldout_residual, noise_std),
        "fitted_parameters": {
            parameter_id: float(value)
            for parameter_id, value in zip(PARAMETER_IDS, np.exp(x), strict=True)
        },
    }


def _median(records: list[dict[str, Any]], field: str) -> float:
    return float(np.median([float(item[field]) for item in records]))


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "trial_count": len(records),
        "converged_count": int(sum(bool(item["converged"]) for item in records)),
        "convergence_rate": float(np.mean([bool(item["converged"]) for item in records])),
        "heldout_nll_median": _median(records, "heldout_nll_per_observation"),
        "log_parameter_rmse_median": _median(records, "log_parameter_rmse"),
        "residual_evaluations_median": _median(records, "residual_evaluations"),
        "jacobian_evaluations_median": _median(records, "jacobian_evaluations"),
    }


def _gauss_newton_equivalence(
    observed: np.ndarray,
    protocol_fixture: dict[str, Any],
    recovery_fixture: dict[str, Any],
    starts: list[np.ndarray],
) -> dict[str, Any]:
    noise_std = float(recovery_fixture["observation_noise_std_k"])
    counter = EvalCounter()
    residual, jacobian = _make_problem(
        observed,
        protocol_fixture,
        list(recovery_fixture["discovery_protocol_ids"]),
        samples_per_segment=int(recovery_fixture["samples_per_segment"]),
        noise_std=noise_std,
        log_step=float(recovery_fixture["centered_log_jacobian_step"]),
        counter=counter,
    )
    relative: list[float] = []
    for start in starts:
        r = residual(start)
        j = jacobian(start)
        gradient = j.T @ r
        fisher = j.T @ j
        natural = -np.linalg.solve(fisher, gradient)
        gauss_newton, *_ = np.linalg.lstsq(j, -r, rcond=None)
        denominator = max(float(np.linalg.norm(gauss_newton)), np.finfo(float).tiny)
        relative.append(float(np.linalg.norm(natural - gauss_newton) / denominator))
    return {
        "sample_count": len(relative),
        "max_relative_step_difference": float(max(relative)),
        "threshold": float(recovery_fixture["interpretation_thresholds"]["gauss_newton_step_max_relative_difference"]),
    }


def analyze_recovery(
    ebm_fixture: dict[str, Any],
    protocol_fixture: dict[str, Any],
    recovery_fixture: dict[str, Any],
) -> dict[str, Any]:
    truth = parameters_from_fixture(ebm_fixture)
    truth_log = np.log(_parameter_vector(truth))
    bound = math.log(float(recovery_fixture["log_parameter_bound_factor"]))
    lower = truth_log - bound
    upper = truth_log + bound
    starts = [truth_log + np.asarray(offset, dtype=float) for offset in recovery_fixture["start_log_parameter_offsets"]]
    if not all(np.all((start >= lower) & (start <= upper)) for start in starts):
        raise ValueError("preregistered optimizer start falls outside common parameter bounds")

    discovery_ids = list(recovery_fixture["discovery_protocol_ids"])
    confirmation_ids = list(recovery_fixture["confirmation_protocol_ids"])
    samples_per_segment = int(recovery_fixture["samples_per_segment"])
    truth_discovery = _temperature_outputs(
        truth, protocol_fixture, discovery_ids, samples_per_segment=samples_per_segment
    )
    truth_confirmation = _temperature_outputs(
        truth, protocol_fixture, confirmation_ids, samples_per_segment=samples_per_segment
    )
    noise_std = float(recovery_fixture["observation_noise_std_k"])

    method_ids = [
        "information_geometry.natural_gradient.armijo_v1",
        "optimization.scipy_trf_least_squares.jac_scaled_v1",
        "optimization.scipy_lbfgsb.exact_gradient_v1",
        "information_geometry.fixed_initial_fisher.armijo_v1",
    ]
    method_records: dict[str, list[dict[str, Any]]] = {method_id: [] for method_id in method_ids}
    trials: list[dict[str, Any]] = []
    first_observed_discovery: np.ndarray | None = None

    for seed in recovery_fixture["dataset_seeds"]:
        rng = np.random.default_rng(int(seed))
        observed_discovery = truth_discovery + rng.normal(0.0, noise_std, size=truth_discovery.shape)
        observed_confirmation = truth_confirmation + rng.normal(0.0, noise_std, size=truth_confirmation.shape)
        if first_observed_discovery is None:
            first_observed_discovery = np.array(observed_discovery, copy=True)
        for start_index, start in enumerate(starts):
            methods: dict[str, Any] = {}
            for method_id in method_ids:
                result = _run_method(
                    method_id,
                    observed_discovery,
                    observed_confirmation,
                    protocol_fixture,
                    recovery_fixture,
                    start,
                    lower,
                    upper,
                    truth_log,
                )
                methods[method_id] = result
                method_records[method_id].append(result)
            trials.append({
                "dataset_seed": int(seed),
                "start_index": start_index,
                "methods": methods,
            })

    if first_observed_discovery is None:
        raise ValueError("recovery fixture declares no dataset seeds")
    summaries = {method_id: _summary(records) for method_id, records in method_records.items()}
    natural = summaries["information_geometry.natural_gradient.armijo_v1"]
    comparator = summaries["optimization.scipy_trf_least_squares.jac_scaled_v1"]
    thresholds = recovery_fixture["interpretation_thresholds"]
    nll_delta = natural["heldout_nll_median"] - comparator["heldout_nll_median"]
    rmse_delta = natural["log_parameter_rmse_median"] - comparator["log_parameter_rmse_median"]
    nll_equivalent = abs(nll_delta) <= float(thresholds["heldout_nll_median_abs_delta_equivalence"])
    rmse_equivalent = abs(rmse_delta) <= float(thresholds["log_parameter_rmse_median_abs_delta_equivalence"])
    convergence_not_worse = natural["convergence_rate"] >= comparator["convergence_rate"]
    lower_cost = natural["residual_evaluations_median"] < comparator["residual_evaluations_median"]
    better_outcome = (
        nll_delta < -float(thresholds["heldout_nll_median_abs_delta_equivalence"])
        or rmse_delta < -float(thresholds["log_parameter_rmse_median_abs_delta_equivalence"])
    )
    distinct_advantage = bool(convergence_not_worse and (better_outcome or (nll_equivalent and rmse_equivalent and lower_cost)))
    equivalence = _gauss_newton_equivalence(
        first_observed_discovery, protocol_fixture, recovery_fixture, starts
    )

    return {
        "fixture_id": recovery_fixture["fixture_id"],
        "ebm_fixture_id": ebm_fixture["fixture_id"],
        "forcing_protocol_fixture_id": protocol_fixture["fixture_id"],
        "trial_count": len(trials),
        "dataset_seeds": [int(value) for value in recovery_fixture["dataset_seeds"]],
        "start_count": len(starts),
        "implementation_versions": {
            "numpy": importlib.metadata.version("numpy"),
            "scipy": importlib.metadata.version("scipy"),
        },
        "summaries": summaries,
        "primary_comparison": {
            "candidate_method_id": "information_geometry.natural_gradient.armijo_v1",
            "comparator_method_id": "optimization.scipy_trf_least_squares.jac_scaled_v1",
            "heldout_nll_median_delta_candidate_minus_comparator": float(nll_delta),
            "log_parameter_rmse_median_delta_candidate_minus_comparator": float(rmse_delta),
            "heldout_nll_equivalent": bool(nll_equivalent),
            "log_parameter_rmse_equivalent": bool(rmse_equivalent),
            "candidate_convergence_not_worse": bool(convergence_not_worse),
            "candidate_lower_residual_evaluation_cost": bool(lower_cost),
            "distinct_information_geometry_advantage_supported": distinct_advantage,
        },
        "gauss_newton_equivalence": equivalence,
        "trials": trials,
        "interpretation_rule": recovery_fixture["confirmation_interpretation"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ebm-fixture", type=Path, default=DEFAULT_EBM_FIXTURE)
    parser.add_argument("--protocol-fixture", type=Path, default=DEFAULT_PROTOCOL_FIXTURE)
    parser.add_argument("--recovery-fixture", type=Path, default=DEFAULT_RECOVERY_FIXTURE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = analyze_recovery(
        load_ebm_fixture(args.ebm_fixture),
        load_protocol_fixture(args.protocol_fixture),
        load_recovery_fixture(args.recovery_fixture),
    )
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    if args.json or not args.output:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
