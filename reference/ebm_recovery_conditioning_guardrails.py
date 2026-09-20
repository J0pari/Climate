#!/usr/bin/env python3
"""Climate-facing numerical conditioning witness for two-layer EBM recovery.

This is a bounded independent reference/oracle. Climate-specific EBM objective
and sensitivity semantics are reused from existing reference modules; NumPy owns
generic SVD measurement. Canonical production-facing conditioning policy remains
in the Rust src/numerics.rs surface.
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

from reference.information_geometry_ebm_recovery import load_recovery_fixture
from reference.two_layer_ebm_recovery_objective import (
    parameter_vector,
    parameters_from_log,
    protocol_temperature_outputs,
)
from reference.two_layer_energy_balance import (
    TwoLayerParameters,
    equilibrium_state_k,
    load_fixture as load_ebm_fixture,
    parameters_from_fixture,
)
from reference.two_layer_forcing_protocols import load_protocol_fixture
from reference.two_layer_parameter_identifiability import (
    PARAMETER_IDS,
    centered_log_parameter_jacobian,
    load_parameter_fixture,
)

DEFAULT_EBM_FIXTURE = ROOT / "fixtures" / "physics" / "two-layer-ebm-geoffroy-mean-v1.json"
DEFAULT_PROTOCOL_FIXTURE = ROOT / "fixtures" / "physics" / "two-layer-ebm-forcing-protocols-v1.json"
DEFAULT_RECOVERY_FIXTURE = ROOT / "fixtures" / "physics" / "two-layer-ebm-information-geometry-recovery-v1.json"
DEFAULT_IDENTIFIABILITY_FIXTURE = ROOT / "fixtures" / "physics" / "two-layer-ebm-parameter-identifiability-v1.json"
DEFAULT_CONDITIONING_FIXTURE = ROOT / "fixtures" / "numerics" / "ebm-recovery-conditioning-guardrails-v1.json"


def load_conditioning_fixture(path: Path = DEFAULT_CONDITIONING_FIXTURE) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported EBM conditioning-guardrail fixture schema")
    if payload.get("verification_policy", {}).get("semantics") != "numerical_test_policy_only":
        raise ValueError("conditioning threshold must remain explicitly numerical-test-only")
    maximum = float(payload["verification_policy"]["max_condition_number_2"])
    if not math.isfinite(maximum) or maximum < 1.0:
        raise ValueError("verification condition-number policy must be finite and >= 1")
    return payload


def conditioning_report(matrix: np.ndarray) -> dict[str, Any]:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("conditioning matrix must be nonempty and two-dimensional")
    if not np.isfinite(values).all():
        raise ValueError("conditioning matrix must be finite")

    singular_values = np.linalg.svd(values, compute_uv=False)
    sigma_max = float(singular_values[0])
    sigma_min = float(singular_values[-1])
    tolerance = float(np.finfo(float).eps * max(values.shape) * sigma_max)
    rank = int(np.count_nonzero(singular_values > tolerance))
    full_rank = min(values.shape)
    rank_deficient = rank < full_rank
    condition_number = None if rank_deficient else float(sigma_max / sigma_min)
    return {
        "rows": int(values.shape[0]),
        "cols": int(values.shape[1]),
        "sigma_max": sigma_max,
        "sigma_min": sigma_min,
        "numerical_rank": rank,
        "numerical_rank_tolerance": tolerance,
        "condition_number_2": condition_number,
        "rank_deficient": rank_deficient,
    }


def apply_verification_policy(report: dict[str, Any], maximum: float) -> str:
    if bool(report["rank_deficient"]):
        return "rank_deficient"
    condition = report["condition_number_2"]
    if condition is None:
        raise RuntimeError("full-rank conditioning report omitted condition number")
    return "condition_number_exceeded" if float(condition) > maximum else "accepted"


def _recovery_jacobian(
    parameters: TwoLayerParameters,
    protocol_fixture: dict[str, Any],
    recovery_fixture: dict[str, Any],
    *,
    noise_std_k: float,
) -> np.ndarray:
    protocol_ids = list(recovery_fixture["discovery_protocol_ids"])
    samples_per_segment = int(recovery_fixture["samples_per_segment"])

    def output(candidate: TwoLayerParameters) -> np.ndarray:
        return protocol_temperature_outputs(
            candidate,
            protocol_fixture,
            protocol_ids,
            samples_per_segment=samples_per_segment,
        )

    raw = centered_log_parameter_jacobian(
        parameters,
        output,
        log_step=float(recovery_fixture["centered_log_jacobian_step"]),
    )
    noise = float(noise_std_k)
    if not math.isfinite(noise) or noise <= 0.0:
        raise ValueError("noise standard deviation must be finite and positive")
    return raw / noise


def _equilibrium_jacobian(
    parameters: TwoLayerParameters,
    identifiability_fixture: dict[str, Any],
) -> np.ndarray:
    forcings = [float(value) for value in identifiability_fixture["equilibrium_forcing_w_m2"]]

    def output(candidate: TwoLayerParameters) -> np.ndarray:
        return np.concatenate(
            [equilibrium_state_k(candidate, forcing) for forcing in forcings]
        )

    return centered_log_parameter_jacobian(
        parameters,
        output,
        log_step=float(identifiability_fixture["log_parameter_step"]),
    )


def _relative_spread(values: list[float]) -> float:
    reference = float(values[0])
    scale = max(abs(reference), np.finfo(float).tiny)
    return float(max(abs(value - reference) for value in values) / scale)


def analyze_conditioning_guardrails(
    ebm_fixture: dict[str, Any],
    protocol_fixture: dict[str, Any],
    recovery_fixture: dict[str, Any],
    identifiability_fixture: dict[str, Any],
    conditioning_fixture: dict[str, Any],
) -> dict[str, Any]:
    authorities = conditioning_fixture["inherited_authorities"]
    observed_ids = {
        "ebm_fixture": ebm_fixture["fixture_id"],
        "forcing_fixture": protocol_fixture["fixture_id"],
        "recovery_fixture": recovery_fixture["fixture_id"],
        "identifiability_fixture": identifiability_fixture["fixture_id"],
    }
    if observed_ids != authorities:
        raise ValueError(
            "conditioning fixture authority identities do not match loaded inputs"
        )

    truth = parameters_from_fixture(ebm_fixture)
    truth_log = np.log(parameter_vector(truth))
    recovery_points: list[tuple[str, TwoLayerParameters]] = [("truth", truth)]
    for index, offset in enumerate(recovery_fixture["start_log_parameter_offsets"]):
        recovery_points.append(
            (
                f"start_{index}",
                parameters_from_log(truth_log + np.asarray(offset, dtype=float)),
            )
        )

    policy_maximum = float(
        conditioning_fixture["verification_policy"]["max_condition_number_2"]
    )
    base_noise = float(recovery_fixture["observation_noise_std_k"])
    point_reports: list[dict[str, Any]] = []
    fisher_rel_errors: list[float] = []
    truth_standardized_jacobian: np.ndarray | None = None

    for label, parameters in recovery_points:
        jacobian = _recovery_jacobian(
            parameters,
            protocol_fixture,
            recovery_fixture,
            noise_std_k=base_noise,
        )
        if truth_standardized_jacobian is None:
            truth_standardized_jacobian = np.array(jacobian, copy=True)
        report = conditioning_report(jacobian)
        status = apply_verification_policy(report, policy_maximum)
        if report["condition_number_2"] is None:
            fisher_relative_error = None
        else:
            fisher_report = conditioning_report(jacobian.T @ jacobian)
            if fisher_report["condition_number_2"] is None:
                raise RuntimeError("full-rank recovery Jacobian produced singular Fisher matrix")
            expected = float(report["condition_number_2"]) ** 2
            fisher_relative_error = abs(
                float(fisher_report["condition_number_2"]) - expected
            ) / expected
            fisher_rel_errors.append(float(fisher_relative_error))
        point_reports.append(
            {
                "label": label,
                "report": report,
                "verification_policy_status": status,
                "fisher_condition_squared_relative_error": fisher_relative_error,
            }
        )

    if truth_standardized_jacobian is None:
        raise RuntimeError("recovery conditioning analysis produced no truth Jacobian")

    noise_conditions: list[dict[str, Any]] = []
    for noise in conditioning_fixture["uniform_noise_std_k"]:
        report = conditioning_report(
            _recovery_jacobian(
                truth,
                protocol_fixture,
                recovery_fixture,
                noise_std_k=float(noise),
            )
        )
        if report["condition_number_2"] is None:
            raise RuntimeError("uniform noise scaling made recovery Jacobian rank deficient")
        noise_conditions.append(
            {"noise_std_k": float(noise), "condition_number_2": float(report["condition_number_2"])}
        )

    matrix_scale_conditions: list[dict[str, Any]] = []
    for factor in conditioning_fixture["matrix_scale_factors"]:
        report = conditioning_report(truth_standardized_jacobian * float(factor))
        if report["condition_number_2"] is None:
            raise RuntimeError("finite matrix scaling made recovery Jacobian rank deficient")
        matrix_scale_conditions.append(
            {"scale_factor": float(factor), "condition_number_2": float(report["condition_number_2"])}
        )

    equilibrium = conditioning_report(_equilibrium_jacobian(truth, identifiability_fixture))
    equilibrium_status = apply_verification_policy(equilibrium, policy_maximum)

    control = conditioning_fixture["near_collinear_control"]
    source_index = PARAMETER_IDS.index(control["source_parameter_id"])
    replaced_index = PARAMETER_IDS.index(control["replaced_parameter_id"])
    near_collinear = np.array(truth_standardized_jacobian, copy=True)
    near_collinear[:, replaced_index] = (
        near_collinear[:, source_index]
        + float(control["retained_original_multiplier"])
        * truth_standardized_jacobian[:, replaced_index]
    )
    near_report = conditioning_report(near_collinear)
    near_status = apply_verification_policy(near_report, policy_maximum)

    checks = conditioning_fixture["confirmation_checks"]
    noise_values = [item["condition_number_2"] for item in noise_conditions]
    matrix_scale_values = [
        item["condition_number_2"] for item in matrix_scale_conditions
    ]
    actual_checks = {
        "recovery_points_full_rank": all(
            not item["report"]["rank_deficient"] for item in point_reports
        ),
        "recovery_points_policy_accepted": all(
            item["verification_policy_status"] == "accepted" for item in point_reports
        ),
        "equilibrium_only_rank": int(equilibrium["numerical_rank"]),
        "equilibrium_only_nullity": int(equilibrium["cols"] - equilibrium["numerical_rank"]),
        "equilibrium_rank_deficient_condition_number_must_be_unavailable": (
            equilibrium["rank_deficient"]
            and equilibrium["condition_number_2"] is None
            and equilibrium_status == "rank_deficient"
        ),
        "near_collinear_control_full_rank": not near_report["rank_deficient"],
        "near_collinear_control_policy_rejected": (
            near_status == "condition_number_exceeded"
        ),
        "uniform_noise_scaling_condition_number_relative_spread": _relative_spread(
            noise_values
        ),
        "matrix_scaling_condition_number_relative_spread": _relative_spread(
            matrix_scale_values
        ),
        "fisher_condition_equals_jacobian_condition_squared_max_relative_error": (
            max(fisher_rel_errors) if fisher_rel_errors else math.inf
        ),
    }

    pass_flags = {
        "recovery_points_full_rank": actual_checks["recovery_points_full_rank"]
        == bool(checks["recovery_points_full_rank"]),
        "recovery_points_policy_accepted": actual_checks[
            "recovery_points_policy_accepted"
        ]
        == bool(checks["recovery_points_policy_accepted"]),
        "equilibrium_only_rank": actual_checks["equilibrium_only_rank"]
        == int(checks["equilibrium_only_rank"]),
        "equilibrium_only_nullity": actual_checks["equilibrium_only_nullity"]
        == int(checks["equilibrium_only_nullity"]),
        "equilibrium_rank_deficient_condition_number_must_be_unavailable": actual_checks[
            "equilibrium_rank_deficient_condition_number_must_be_unavailable"
        ]
        == bool(
            checks[
                "equilibrium_rank_deficient_condition_number_must_be_unavailable"
            ]
        ),
        "near_collinear_control_full_rank": actual_checks[
            "near_collinear_control_full_rank"
        ]
        == bool(checks["near_collinear_control_full_rank"]),
        "near_collinear_control_policy_rejected": actual_checks[
            "near_collinear_control_policy_rejected"
        ]
        == bool(checks["near_collinear_control_policy_rejected"]),
        "uniform_noise_scaling_condition_number_relative_tolerance": actual_checks[
            "uniform_noise_scaling_condition_number_relative_spread"
        ]
        <= float(checks["uniform_noise_scaling_condition_number_relative_tolerance"]),
        "matrix_scaling_condition_number_relative_tolerance": actual_checks[
            "matrix_scaling_condition_number_relative_spread"
        ]
        <= float(checks["matrix_scaling_condition_number_relative_tolerance"]),
        "fisher_condition_equals_jacobian_condition_squared_relative_tolerance": actual_checks[
            "fisher_condition_equals_jacobian_condition_squared_max_relative_error"
        ]
        <= float(
            checks[
                "fisher_condition_equals_jacobian_condition_squared_relative_tolerance"
            ]
        ),
    }

    return {
        "fixture_id": conditioning_fixture["fixture_id"],
        "implementation_versions": {
            "numpy": importlib.metadata.version("numpy"),
            "scipy": importlib.metadata.version("scipy"),
        },
        "verification_policy": conditioning_fixture["verification_policy"],
        "recovery_points": point_reports,
        "uniform_noise_scaling": noise_conditions,
        "matrix_scaling": matrix_scale_conditions,
        "equilibrium_only": {
            "report": equilibrium,
            "verification_policy_status": equilibrium_status,
        },
        "near_collinear_control": {
            "report": near_report,
            "verification_policy_status": near_status,
        },
        "actual_checks": actual_checks,
        "pass_flags": pass_flags,
        "all_checks_passed": all(pass_flags.values()),
        "interpretation_boundary": conditioning_fixture["interpretation_boundary"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ebm-fixture", type=Path, default=DEFAULT_EBM_FIXTURE)
    parser.add_argument("--protocol-fixture", type=Path, default=DEFAULT_PROTOCOL_FIXTURE)
    parser.add_argument("--recovery-fixture", type=Path, default=DEFAULT_RECOVERY_FIXTURE)
    parser.add_argument(
        "--identifiability-fixture", type=Path, default=DEFAULT_IDENTIFIABILITY_FIXTURE
    )
    parser.add_argument(
        "--conditioning-fixture", type=Path, default=DEFAULT_CONDITIONING_FIXTURE
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = analyze_conditioning_guardrails(
        load_ebm_fixture(args.ebm_fixture),
        load_protocol_fixture(args.protocol_fixture),
        load_recovery_fixture(args.recovery_fixture),
        load_parameter_fixture(args.identifiability_fixture),
        load_conditioning_fixture(args.conditioning_fixture),
    )
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    if args.json or not args.output:
        print(encoded, end="")
    return 0 if result["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
