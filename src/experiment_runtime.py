#!/usr/bin/env python3
"""Canonical local CPU experiment runtime for Climate's common evidence spine.

The registered two-layer EBM benchmark ladder executes through one runtime:
representation dynamics, exact forcing protocols, discovery-trained forced OOD,
observation degradation, local parameter-sensitivity rank, seeded stochastic
statistics preservation, and local regime-dependent feedback. The runtime owns
identity resolution, immutable input checks, subprocess receipts,
content-addressed artifacts, typed metric results, and contract-shaped
run/outcome records. Scientific methods remain in their own modules and are
invoked without changing their semantics.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = ROOT / "experiments" / "multirepresentation-ebm-dynamics.v1.json"
METHOD_REGISTRY = ROOT / "methods" / "registry.json"
CONTRACT = ROOT / "contracts" / "climate.cue"

EBM_DYNAMICS_EXPERIMENT = "multirepresentation.ebm_dynamics.v1"
EBM_FORCING_EXPERIMENT = "physics.two_layer_ebm.forcing_protocols.v1"
EBM_FORCED_OOD_EXPERIMENT = "multirepresentation.ebm_forced_ood.v1"
EBM_OBSERVATION_DEGRADATION_EXPERIMENT = (
    "multirepresentation.ebm_observation_degradation.v1"
)
EBM_PARAMETER_IDENTIFIABILITY_EXPERIMENT = (
    "physics.two_layer_ebm.parameter_identifiability.v1"
)
EBM_STOCHASTIC_STATISTICS_EXPERIMENT = (
    "multirepresentation.ebm_stochastic_statistics.v1"
)
EBM_REGIME_FEEDBACK_EXPERIMENT = (
    "multirepresentation.ebm_regime_feedback.v1"
)
EBM_BASELINE_METHOD = "physics.two_layer_ebm.exact_modes_v1"
EBM_DYNAMICS_CANDIDATE_METHOD = "dynamics.dmd.pydmd_v1"
EBM_FORCING_CANDIDATE_METHOD = "physics.two_layer_ebm.affine_forcing_v1"
EBM_FORCED_OOD_CANDIDATE_METHOD = "dynamics.affine_control_lstsq.numpy_v1"
EBM_PARAMETER_SENSITIVITY_METHOD = (
    "physics.two_layer_ebm.log_parameter_sensitivity_v1"
)
EBM_STOCHASTIC_BASELINE_METHOD = (
    "physics.two_layer_ebm.stochastic_internal_exchange_v1"
)
EBM_STOCHASTIC_STATISTICS_METHOD = (
    "multirepresentation.fixed_decode_statistics.numpy_v1"
)
EBM_REGIME_BASELINE_METHOD = "physics.two_layer_ebm.regime_feedback_local_v1"
EBM_REGIME_GATED_METHOD = "dynamics.regime_gated_lstsq.numpy_v1"
RUNNABLE_MATURITIES = {
    "runnable", "verified", "validated", "replicated", "decision-eligible"
}


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _write_json(path: Path, payload: Any) -> bytes:
    data = _canonical_json_bytes(payload)
    path.write_bytes(data)
    return data


def _method_map() -> dict[str, dict[str, Any]]:
    methods = _load_json(METHOD_REGISTRY).get("methods")
    if not isinstance(methods, list):
        raise ValueError("method registry has no methods list")
    result: dict[str, dict[str, Any]] = {}
    for item in methods:
        if not isinstance(item, dict) or not isinstance(item.get("method_id"), str):
            raise ValueError("method registry contains malformed descriptor")
        method_id = item["method_id"]
        if method_id in result:
            raise ValueError(f"duplicate method descriptor {method_id}")
        result[method_id] = item
    return result


def _resolved_method_build(method_id: str, descriptor: Mapping[str, Any]) -> str:
    identity = descriptor.get("build_identity")
    if not isinstance(identity, dict):
        raise ValueError(f"{method_id} has no build identity policy")
    if identity.get("policy") != "source_digest_at_run":
        raise ValueError(f"{method_id} has unsupported build identity policy")
    sources = identity.get("sources")
    if not isinstance(sources, list) or not sources or not all(isinstance(item, str) and item for item in sources):
        raise ValueError(f"{method_id} has no executable source binding")
    digest = hashlib.sha256()
    for relative in sources:
        path = ROOT / relative
        if not path.is_file():
            raise ValueError(f"method source does not exist: {relative}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return f"source-sha256:{digest.hexdigest()}"


def _resolve_configuration(experiment: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    configuration = experiment.get("configuration", {})
    if not isinstance(configuration, dict):
        raise ValueError("experiment configuration must be an object")
    resolved: dict[str, Any] = {}
    records: dict[str, dict[str, Any]] = {}
    for group in ("kernel_parameters", "numerical_policies", "data_policies", "execution_policies"):
        refs = configuration.get(group, [])
        if not isinstance(refs, list):
            raise ValueError(f"configuration group {group} must be a list")
        if not refs:
            continue
        resolved_refs: list[dict[str, Any]] = []
        for ref in refs:
            if not isinstance(ref, dict) or not isinstance(ref.get("record_path"), str):
                raise ValueError(f"malformed configuration reference in {group}")
            relative = ref["record_path"]
            path = (ROOT / relative).resolve()
            try:
                path.relative_to(ROOT.resolve())
            except ValueError as exc:
                raise ValueError(f"configuration path escapes repository: {relative}") from exc
            if not path.is_file():
                raise ValueError(f"configuration record does not exist: {relative}")
            if _sha256_file(path) != ref.get("digest"):
                raise ValueError(f"configuration digest mismatch for {relative}")
            record = _load_json(path)
            for field in ("configuration_id", "semantic_version", "kind", "owner", "provenance"):
                if record.get(field) != ref.get(field):
                    raise ValueError(f"configuration {relative} disagrees on {field}")
            records[ref["configuration_id"]] = record
            resolved_refs.append(dict(ref))
        resolved[group] = resolved_refs
    return resolved, records


def _resolve_datasets(
    experiment: Mapping[str, Any],
) -> list[tuple[dict[str, Any], Path]]:
    datasets = experiment.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("experiment requires at least one dataset reference")
    resolved: list[tuple[dict[str, Any], Path]] = []
    ids: set[str] = set()
    for raw in datasets:
        if not isinstance(raw, dict):
            raise ValueError("dataset reference must be an object")
        dataset = dict(raw)
        dataset_id = dataset.get("id")
        if not isinstance(dataset_id, str) or not dataset_id:
            raise ValueError("dataset reference requires a non-empty id")
        if dataset_id in ids:
            raise ValueError(f"duplicate dataset reference {dataset_id}")
        ids.add(dataset_id)
        citation = dataset.get("citation")
        if not isinstance(citation, str) or not citation.startswith("fixtures/"):
            raise ValueError("dataset citation must resolve to an immutable fixture path")
        path = (ROOT / citation).resolve()
        try:
            path.relative_to(ROOT.resolve())
        except ValueError as exc:
            raise ValueError("dataset fixture escapes repository") from exc
        if not path.is_file():
            raise ValueError(f"dataset fixture does not exist: {citation}")
        if _sha256_file(path) != dataset.get("digest"):
            raise ValueError(f"dataset digest mismatch for {citation}")
        resolved.append((dataset, path))
    return resolved


def _require_methods(
    experiment: Mapping[str, Any],
    methods: Mapping[str, Mapping[str, Any]],
    *,
    baseline_method: str,
    candidate_method: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if experiment.get("baseline_methods") != [baseline_method]:
        raise ValueError(f"experiment requires baseline method {baseline_method}")
    if experiment.get("candidate_methods") != [candidate_method]:
        raise ValueError(f"experiment requires candidate method {candidate_method}")
    baseline = methods.get(baseline_method)
    candidate = methods.get(candidate_method)
    for method_id, descriptor in (
        (baseline_method, baseline),
        (candidate_method, candidate),
    ):
        if descriptor is None:
            raise ValueError(f"experiment method does not resolve: {method_id}")
        if descriptor.get("maturity") not in RUNNABLE_MATURITIES:
            raise ValueError(f"experiment method is not runnable: {method_id}")
    return baseline, candidate


class MethodProcessFailure(RuntimeError):
    """A method subprocess failed after its provenance receipt was captured."""

    def __init__(
        self,
        *,
        method_id: str,
        backend_id: str,
        receipt: Mapping[str, Any],
    ) -> None:
        self.method_id = method_id
        self.backend_id = backend_id
        self.receipt = dict(receipt)
        detail = str(self.receipt.get("failure_detail", "method execution failed"))
        super().__init__(f"{method_id}: {detail}")


def _run_process(
    command: Sequence[str],
    *,
    method_id: str,
    backend_id: str,
) -> dict[str, Any]:
    if not method_id or not backend_id:
        raise ValueError("method_id and backend_id must be explicit for process provenance")
    argv = [str(item) for item in command]
    started = datetime.now(timezone.utc)
    try:
        process = subprocess.run(
            argv,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as exc:
        ended = datetime.now(timezone.utc)
        receipt = {
            "argv": argv,
            "command": shlex.join(argv),
            "started_at": started.isoformat(),
            "ended_at": ended.isoformat(),
            "stdout_digest": _sha256_bytes(b""),
            "stderr_digest": _sha256_bytes(b""),
            "failure_class": "implementation_unavailable",
            "failure_stage": "process_launch",
            "failure_detail": (
                f"subprocess could not be launched: {type(exc).__name__}: {exc}"
            ),
        }
        raise MethodProcessFailure(
            method_id=method_id,
            backend_id=backend_id,
            receipt=receipt,
        ) from exc

    ended = datetime.now(timezone.utc)
    receipt: dict[str, Any] = {
        "argv": argv,
        "command": shlex.join(argv),
        "exit_code": int(process.returncode),
        "started_at": started.isoformat(),
        "ended_at": ended.isoformat(),
        "stdout_digest": _sha256_bytes(process.stdout),
        "stderr_digest": _sha256_bytes(process.stderr),
    }
    if process.returncode != 0:
        receipt.update(
            {
                "failure_class": "process_failed",
                "failure_stage": "process_exit",
                "failure_detail": (
                    f"subprocess exited with status {process.returncode}"
                ),
            }
        )
        raise MethodProcessFailure(
            method_id=method_id,
            backend_id=backend_id,
            receipt=receipt,
        )

    try:
        payload = json.loads(process.stdout.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("method process JSON output must be an object")
        _canonical_json_bytes(payload)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        receipt.update(
            {
                "failure_class": "output_invalid",
                "failure_stage": "output_decode",
                "failure_detail": (
                    f"method process output is not valid canonical JSON: "
                    f"{type(exc).__name__}: {exc}"
                ),
            }
        )
        raise MethodProcessFailure(
            method_id=method_id,
            backend_id=backend_id,
            receipt=receipt,
        ) from exc

    receipt["payload"] = payload
    return receipt


def _artifact_ref(artifact_id: str, schema: str, filename: str, data: bytes) -> dict[str, Any]:
    return {
        "artifact_id": artifact_id,
        "digest": _sha256_bytes(data),
        "media_type": "application/json",
        "schema": schema,
        "bytes": len(data),
        "uri": filename,
    }


def _metric_definitions(experiment: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    definitions: dict[str, dict[str, Any]] = {}
    for group in ("primary_metrics", "secondary_metrics"):
        for metric in experiment.get(group, []):
            if not isinstance(metric, dict) or not isinstance(metric.get("metric_id"), str):
                raise ValueError(f"malformed metric definition in {group}")
            metric_id = metric["metric_id"]
            if metric_id in definitions:
                raise ValueError(f"duplicate metric definition {metric_id}")
            definitions[metric_id] = dict(metric)
    return definitions


def _finite_metric(
    definition: Mapping[str, Any],
    value: float | int,
    reference_population: str | None = None,
) -> dict[str, Any]:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"finite metric {definition['metric_id']} became non-finite")
    result: dict[str, Any] = {"metric": dict(definition), "status": "finite", "value": numeric}
    if reference_population:
        result["reference_population"] = reference_population
    return result


def _derive_ebm_dynamics_metrics(experiment: Mapping[str, Any], candidate: Mapping[str, Any]) -> list[dict[str, Any]]:
    definitions = _metric_definitions(experiment)
    representations = candidate.get("representations")
    if not isinstance(representations, dict):
        raise ValueError("candidate output lacks representation diagnostics")
    full_names = (
        "temperature_state", "heat_flux", "thermal_modes",
        "temperature_plus_flux", "all_full_rank_views",
    )
    scalar_names = ("surface_temperature_scalar", "toa_imbalance_scalar")
    concat_names = ("temperature_plus_flux", "all_full_rank_views")
    full = [representations[name] for name in full_names]
    scalar = [representations[name] for name in scalar_names]
    concat = [representations[name] for name in concat_names]
    results = [
        _finite_metric(
            definitions["multirepresentation.ebm.full_rank.max_relative_timescale_error"],
            max(float(item["max_relative_timescale_error"]) for item in full),
        ),
        _finite_metric(
            definitions["multirepresentation.ebm.full_rank.resolved_mode_count_min"],
            min(len(item["dmd_timescales_years"]) for item in full),
        ),
        _finite_metric(
            definitions["multirepresentation.ebm.scalar.resolved_mode_count_max"],
            max(int(item["resolved_mode_count"]) for item in scalar),
        ),
        _finite_metric(
            definitions["multirepresentation.ebm.scalar.one_step_relative_error_min"],
            min(float(item["one_step_relative_error"]) for item in scalar),
        ),
        _finite_metric(
            definitions["multirepresentation.ebm.concat.redundant_dimension_count_min"],
            min(int(item["redundant_dimension_count"]) for item in concat),
        ),
    ]
    condition_definition = definitions["multirepresentation.ebm.concat.condition_number"]
    for name in concat_names:
        item = representations[name]
        status = item.get("condition_number_status")
        if status == "finite":
            results.append(_finite_metric(condition_definition, float(item["condition_number"]), name))
        elif status in {"rank_deficient", "undefined"}:
            detail = item.get("condition_number_detail")
            if not isinstance(detail, str) or not detail:
                raise ValueError(f"{name} condition diagnostic lacks detail")
            results.append({
                "metric": dict(condition_definition),
                "status": status,
                "detail": detail,
                "reference_population": name,
            })
        else:
            raise ValueError(f"{name} has unknown condition_number_status {status!r}")
    return results


def _derive_forcing_metrics(
    experiment: Mapping[str, Any], candidate: Mapping[str, Any]
) -> list[dict[str, Any]]:
    definitions = _metric_definitions(experiment)
    return [
        _finite_metric(
            definitions[
                "physics.two_layer_ebm.affine_forcing.constant_equivalence_max_abs_state_error_k"
            ],
            float(candidate["constant_forcing_equivalence_max_abs_state_error_k"]),
        ),
        _finite_metric(
            definitions[
                "physics.two_layer_ebm.affine_forcing.max_abs_energy_budget_residual_w_m2"
            ],
            float(candidate["max_abs_energy_budget_residual_w_m2"]),
        ),
        _finite_metric(
            definitions[
                "physics.two_layer_ebm.ood.held_out_absolute_forcing_margin_w_m2"
            ],
            float(candidate["held_out_absolute_forcing_margin_w_m2"]),
        ),
    ]


def _derive_forced_ood_metrics(
    experiment: Mapping[str, Any], candidate: Mapping[str, Any]
) -> list[dict[str, Any]]:
    definitions = _metric_definitions(experiment)
    representations = candidate.get("representations")
    if not isinstance(representations, dict):
        raise ValueError("forced OOD candidate output lacks representation diagnostics")
    full = representations["temperature_state"]
    scalar = representations["surface_temperature_scalar"]
    return [
        _finite_metric(
            definitions[
                "multirepresentation.ebm.ood.full_state_confirmation_relative_error"
            ],
            float(full["confirmation_relative_error"]),
        ),
        _finite_metric(
            definitions[
                "multirepresentation.ebm.ood.surface_scalar_confirmation_relative_error"
            ],
            float(scalar["confirmation_relative_error"]),
        ),
        _finite_metric(
            definitions[
                "multirepresentation.ebm.ood.closure_gap_confirmation_relative_error"
            ],
            float(candidate["closure_gap_confirmation_relative_error"]),
        ),
        _finite_metric(
            definitions[
                "multirepresentation.ebm.ood.full_state_training_relative_error"
            ],
            float(full["training_relative_error"]),
        ),
        _finite_metric(
            definitions[
                "multirepresentation.ebm.ood.surface_scalar_training_relative_error"
            ],
            float(scalar["training_relative_error"]),
        ),
        _finite_metric(
            definitions[
                "multirepresentation.ebm.ood.held_out_absolute_forcing_margin_w_m2"
            ],
            float(candidate["held_out_absolute_forcing_margin_w_m2"]),
        ),
    ]


def _derive_observation_degradation_metrics(
    experiment: Mapping[str, Any], candidate: Mapping[str, Any]
) -> list[dict[str, Any]]:
    definitions = _metric_definitions(experiment)
    representations = candidate.get("representations")
    if not isinstance(representations, dict):
        raise ValueError(
            "observation-degradation candidate output lacks representation diagnostics"
        )
    ordered = (
        "clean_temperature_state",
        "noisy_temperature_state",
        "noisy_surface_scalar",
        "redundant_noisy_surface_pair",
    )
    if set(representations) != set(ordered):
        raise ValueError(
            "observation-degradation candidate representation identities changed"
        )

    results: list[dict[str, Any]] = []
    for name in ordered:
        item = representations[name]
        results.extend(
            [
                _finite_metric(
                    definitions[
                        "multirepresentation.ebm.observation.state_reconstruction_relative_error"
                    ],
                    float(item["confirmation_state_reconstruction_relative_error"]),
                    name,
                ),
                _finite_metric(
                    definitions[
                        "multirepresentation.ebm.observation.ood_forced_state_prediction_relative_error"
                    ],
                    float(item["confirmation_forced_state_prediction_relative_error"]),
                    name,
                ),
                _finite_metric(
                    definitions[
                        "multirepresentation.ebm.observation.structural_observation_rank"
                    ],
                    int(item["structural_observation_rank"]),
                    name,
                ),
                _finite_metric(
                    definitions[
                        "multirepresentation.ebm.observation.observation_dimension"
                    ],
                    int(item["observation_dimension"]),
                    name,
                ),
                _finite_metric(
                    definitions[
                        "multirepresentation.ebm.observation.redundant_dimension_count"
                    ],
                    int(item["redundant_dimension_count"]),
                    name,
                ),
            ]
        )
    results.extend(
        [
            _finite_metric(
                definitions[
                    "multirepresentation.ebm.observation.redundant_structural_rank_gain"
                ],
                int(candidate["redundant_structural_rank_gain"]),
            ),
            _finite_metric(
                definitions[
                    "multirepresentation.ebm.observation.redundant_vs_surface_state_reconstruction_error_delta"
                ],
                float(
                    candidate[
                        "redundant_vs_surface_state_reconstruction_error_delta"
                    ]
                ),
            ),
            _finite_metric(
                definitions[
                    "multirepresentation.ebm.observation.redundant_vs_surface_forced_prediction_error_delta"
                ],
                float(
                    candidate[
                        "redundant_vs_surface_forced_prediction_error_delta"
                    ]
                ),
            ),
            _finite_metric(
                definitions[
                    "multirepresentation.ebm.observation.noisy_full_vs_surface_forced_prediction_error_gap"
                ],
                float(
                    candidate[
                        "noisy_full_vs_surface_forced_prediction_error_gap"
                    ]
                ),
            ),
            _finite_metric(
                definitions[
                    "multirepresentation.ebm.observation.clean_control_forced_prediction_relative_error"
                ],
                float(candidate["clean_control_forced_prediction_relative_error"]),
            ),
        ]
    )
    return results


def _derive_parameter_identifiability_metrics(
    experiment: Mapping[str, Any], candidate: Mapping[str, Any]
) -> list[dict[str, Any]]:
    definitions = _metric_definitions(experiment)
    equilibrium = candidate.get("equilibrium")
    transient = candidate.get("transient")
    if not isinstance(equilibrium, dict) or not isinstance(transient, dict):
        raise ValueError(
            "parameter-identifiability candidate lacks sensitivity diagnostics"
        )
    condition = transient.get("condition_number")
    if transient.get("condition_number_status") != "finite" or condition is None:
        raise ValueError(
            "parameter-identifiability transient sensitivity must be full rank"
        )
    return [
        _finite_metric(
            definitions[
                "physics.two_layer_ebm.parameter.equilibrium_local_sensitivity_rank"
            ],
            int(equilibrium["rank"]),
        ),
        _finite_metric(
            definitions[
                "physics.two_layer_ebm.parameter.transient_local_sensitivity_rank"
            ],
            int(transient["rank"]),
        ),
        _finite_metric(
            definitions["physics.two_layer_ebm.parameter.local_rank_gain"],
            int(candidate["local_rank_gain"]),
        ),
        _finite_metric(
            definitions[
                "physics.two_layer_ebm.parameter.equilibrium_invisible_parameter_count"
            ],
            int(candidate["equilibrium_invisible_parameter_count"]),
        ),
        _finite_metric(
            definitions[
                "physics.two_layer_ebm.parameter.transient_condition_number"
            ],
            float(condition),
        ),
        _finite_metric(
            definitions[
                "physics.two_layer_ebm.parameter.equilibrium_step_consistency_relative_frobenius"
            ],
            float(candidate["equilibrium_step_consistency_relative_frobenius"]),
        ),
        _finite_metric(
            definitions[
                "physics.two_layer_ebm.parameter.transient_step_consistency_relative_frobenius"
            ],
            float(candidate["transient_step_consistency_relative_frobenius"]),
        ),
    ]


def _derive_stochastic_statistics_metrics(
    experiment: Mapping[str, Any], candidate: Mapping[str, Any]
) -> list[dict[str, Any]]:
    definitions = _metric_definitions(experiment)
    representations = candidate.get("representations")
    if not isinstance(representations, dict):
        raise ValueError("stochastic statistics candidate lacks representations")
    ordered = (
        "temperature_state",
        "surface_temperature_scalar",
        "redundant_surface_pair",
    )
    if set(representations) != set(ordered):
        raise ValueError("stochastic statistics representation identities changed")

    results: list[dict[str, Any]] = []
    for name in ordered:
        item = representations[name]
        results.extend(
            [
                _finite_metric(
                    definitions[
                        "multirepresentation.ebm.stochastic.state_covariance_relative_frobenius_error"
                    ],
                    float(item["state_covariance_relative_frobenius_error"]),
                    name,
                ),
                _finite_metric(
                    definitions[
                        "multirepresentation.ebm.stochastic.lag_covariance_relative_frobenius_error"
                    ],
                    float(item["lag_covariance_relative_frobenius_error"]),
                    name,
                ),
                _finite_metric(
                    definitions[
                        "multirepresentation.ebm.stochastic.deep_variance_relative_error"
                    ],
                    float(item["deep_variance_relative_error"]),
                    name,
                ),
                _finite_metric(
                    definitions[
                        "multirepresentation.ebm.stochastic.surface_variance_relative_error"
                    ],
                    float(item["surface_variance_relative_error"]),
                    name,
                ),
                _finite_metric(
                    definitions[
                        "multirepresentation.ebm.stochastic.structural_observation_rank"
                    ],
                    int(item["structural_observation_rank"]),
                    name,
                ),
            ]
        )

    physical_budget = candidate.get("physical_budget")
    if not isinstance(physical_budget, dict):
        raise ValueError("stochastic statistics candidate lacks physical budget")
    results.extend(
        [
            _finite_metric(
                definitions[
                    "multirepresentation.ebm.stochastic.full_state_covariance_error"
                ],
                float(candidate["full_state_covariance_error"]),
            ),
            _finite_metric(
                definitions[
                    "multirepresentation.ebm.stochastic.full_state_lag_covariance_error"
                ],
                float(candidate["full_state_lag_covariance_error"]),
            ),
            _finite_metric(
                definitions[
                    "multirepresentation.ebm.stochastic.redundant_vs_surface_covariance_error_delta"
                ],
                float(candidate["redundant_vs_surface_covariance_error_delta"]),
            ),
            _finite_metric(
                definitions[
                    "multirepresentation.ebm.stochastic.redundant_vs_surface_lag_covariance_error_delta"
                ],
                float(candidate["redundant_vs_surface_lag_covariance_error_delta"]),
            ),
            _finite_metric(
                definitions[
                    "physics.two_layer_ebm.stochastic.max_abs_total_budget_residual_w_m2"
                ],
                float(physical_budget["max_abs_total_budget_residual_w_m2"]),
            ),
            _finite_metric(
                definitions[
                    "physics.two_layer_ebm.stochastic.max_abs_direct_internal_storage_w_m2"
                ],
                float(physical_budget["max_abs_direct_internal_storage_w_m2"]),
            ),
        ]
    )
    return results


def _derive_regime_feedback_metrics(
    experiment: Mapping[str, Any], candidate: Mapping[str, Any]
) -> list[dict[str, Any]]:
    definitions = _metric_definitions(experiment)
    raw = candidate.get("raw_global")
    gated = candidate.get("regime_gated")
    if not isinstance(raw, dict) or not isinstance(gated, dict):
        raise ValueError("regime-feedback candidate lacks representation diagnostics")
    return [
        _finite_metric(
            definitions[
                "multirepresentation.ebm.regime.raw_global_confirmation_relative_error"
            ],
            float(raw["confirmation_relative_error"]),
        ),
        _finite_metric(
            definitions[
                "multirepresentation.ebm.regime.gated_confirmation_relative_error"
            ],
            float(gated["confirmation_relative_error"]),
        ),
        _finite_metric(
            definitions[
                "multirepresentation.ebm.regime.confirmation_error_reduction"
            ],
            float(candidate["confirmation_error_reduction"]),
        ),
        _finite_metric(
            definitions[
                "multirepresentation.ebm.regime.raw_global_training_relative_error"
            ],
            float(raw["training_relative_error"]),
        ),
        _finite_metric(
            definitions[
                "multirepresentation.ebm.regime.gated_training_relative_error"
            ],
            float(gated["training_relative_error"]),
        ),
        _finite_metric(
            definitions["multirepresentation.ebm.regime.raw_design_rank"],
            int(raw["design_rank"]),
        ),
        _finite_metric(
            definitions["multirepresentation.ebm.regime.gated_design_rank"],
            int(gated["design_rank"]),
        ),
        _finite_metric(
            definitions[
                "multirepresentation.ebm.regime.swapped_label_confirmation_relative_error"
            ],
            float(candidate["swapped_regime_confirmation_relative_error"]),
        ),
        _finite_metric(
            definitions[
                "physics.two_layer_ebm.regime.observed_minimum_regime_margin_k"
            ],
            float(candidate["observed_minimum_regime_margin_k"]),
        ),
    ]


def _validated_seeds(experiment: Mapping[str, Any]) -> list[int]:
    seeds = experiment.get("seeds", [])
    if not isinstance(seeds, list) or not all(isinstance(seed, int) for seed in seeds):
        raise ValueError("experiment seeds must be an integer list")
    return list(seeds)


def _scoped_run_id(run_scope: str, experiment_id: str, method_id: str) -> str:
    for name, value in (
        ("run_scope", run_scope),
        ("experiment_id", experiment_id),
        ("method_id", method_id),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be non-empty for run identity")
    return f"{run_scope}.{experiment_id}.{method_id}"


def _execution_identity(method_id: str, implementation_build: str, backend_id: str) -> dict[str, Any]:
    return {
        "method_id": method_id,
        "implementation_id": method_id,
        "implementation_build": implementation_build,
        "backend_id": backend_id,
        "precision": "FP64",
        "resource_class": "R1_portable_cpu",
    }


def _run_manifest(
    *,
    run_id: str,
    experiment_id: str,
    experiment_spec_digest: str,
    revision: str,
    method_builds: Mapping[str, str],
    execution_identity: Mapping[str, Any],
    dataset_digests: Sequence[str],
    resolved_configuration: Mapping[str, Any],
    seeds: Sequence[int],
    libraries: Mapping[str, str],
    receipt: Mapping[str, Any],
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not (
        isinstance(experiment_spec_digest, str)
        and experiment_spec_digest.startswith("sha256:")
        and len(experiment_spec_digest) == 71
    ):
        raise ValueError("experiment_spec_digest must be a SHA-256 identity")

    failure_class = receipt.get("failure_class")
    if failure_class is None:
        execution = {
            "requested": dict(execution_identity),
            "status": "eligible",
            "resolved": dict(execution_identity),
        }
        scientific_output_eligible = True
    else:
        failure_stage = receipt.get("failure_stage")
        failure_detail = receipt.get("failure_detail")
        if not isinstance(failure_stage, str) or not failure_stage:
            raise ValueError("failed process receipt requires failure_stage")
        if not isinstance(failure_detail, str) or not failure_detail:
            raise ValueError("failed process receipt requires failure_detail")
        execution = {
            "requested": dict(execution_identity),
            "status": "ineligible",
            "failure": str(failure_class),
        }
        scientific_output_eligible = False

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "experiment_id": experiment_id,
        "experiment_spec_digest": experiment_spec_digest,
        "repository_revision": revision,
        "producer_build": revision,
        "contract_fingerprint": hashlib.sha256(CONTRACT.read_bytes()).hexdigest(),
        "method_builds": dict(method_builds),
        "execution": execution,
        "scientific_output_eligible": scientific_output_eligible,
        "resolved_dataset_digests": list(dataset_digests),
        "resolved_configuration": dict(resolved_configuration),
        "seeds": list(seeds),
        "environment": {
            "os": platform.platform(),
            "arch": platform.machine(),
            "toolchains": {"python": platform.python_version()},
            "libraries": dict(libraries),
        },
        "hardware": {"cpu": platform.processor() or platform.machine() or "unknown-cpu"},
        "commands": [receipt["command"]],
        "started_at": receipt["started_at"],
        "ended_at": receipt["ended_at"],
        "stdout_digest": receipt["stdout_digest"],
        "stderr_digest": receipt["stderr_digest"],
        "artifacts": [dict(item) for item in artifacts],
    }
    if "exit_code" in receipt:
        manifest["exit_code"] = int(receipt["exit_code"])
    if failure_class is not None:
        manifest["failure_stage"] = receipt["failure_stage"]
        manifest["failure_detail"] = receipt["failure_detail"]
    return manifest


def _finalize_two_run_outcome(
    *,
    output_dir: Path,
    experiment_id: str,
    baseline_run: Mapping[str, Any],
    candidate_run: Mapping[str, Any],
    artifacts: Sequence[Mapping[str, Any]],
    metrics: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Materialize the common two-run outcome envelope.

    Family adapters own scientific evaluation, artifact construction, and metric
    semantics. This helper owns only the repository-wide receipt filenames and
    ExperimentOutcome envelope shared by the current CPU benchmark ladder.
    """
    baseline = dict(baseline_run)
    candidate = dict(candidate_run)
    artifact_records = [dict(item) for item in artifacts]
    metric_records = [dict(item) for item in metrics]
    _write_json(output_dir / "run-baseline.json", baseline)
    _write_json(output_dir / "run-candidate.json", candidate)

    outcome = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "runs": [baseline, candidate],
        "artifacts": artifact_records,
        "metrics": metric_records,
        "evidence": [],
    }
    _write_json(output_dir / "outcome.json", outcome)
    return outcome


def _run_ebm_dynamics_adapter(
    *,
    experiment: Mapping[str, Any],
    output_dir: Path,
    repository_revision: str,
    run_scope: str,
    methods: Mapping[str, Mapping[str, Any]],
    resolved_configuration: Mapping[str, Any],
    configuration_records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    baseline_descriptor, candidate_descriptor = _require_methods(
        experiment,
        methods,
        baseline_method=EBM_BASELINE_METHOD,
        candidate_method=EBM_DYNAMICS_CANDIDATE_METHOD,
    )
    datasets = _resolve_datasets(experiment)
    if len(datasets) != 1:
        raise ValueError("EBM dynamics adapter requires exactly one dataset")
    dataset, fixture_path = datasets[0]

    numerical_refs = resolved_configuration.get("numerical_policies", [])
    if len(numerical_refs) != 1:
        raise ValueError("EBM dynamics adapter requires exactly one numerical policy")
    settings = configuration_records[numerical_refs[0]["configuration_id"]]["settings"]
    dt_years = float(settings["transition_dt_years"])
    if not math.isfinite(dt_years) or dt_years <= 0.0:
        raise ValueError("transition_dt_years must be finite and positive")

    baseline_build = _resolved_method_build(
        EBM_BASELINE_METHOD, baseline_descriptor
    )
    candidate_build = _resolved_method_build(
        EBM_DYNAMICS_CANDIDATE_METHOD, candidate_descriptor
    )
    baseline_receipt = _run_process(
        (
            sys.executable,
            str(ROOT / "reference" / "two_layer_energy_balance.py"),
            "--fixture",
            str(fixture_path),
            "--json",
        ),
        method_id=EBM_BASELINE_METHOD,
        backend_id="scipy",
    )
    candidate_receipt = _run_process(
        (
            sys.executable,
            str(ROOT / "reference" / "two_layer_representation_dynamics.py"),
            "--fixture",
            str(fixture_path),
            "--dt-years",
            repr(dt_years),
            "--json",
        ),
        method_id=EBM_DYNAMICS_CANDIDATE_METHOD,
        backend_id="pydmd",
    )
    baseline_payload = baseline_receipt["payload"]
    candidate_payload = candidate_receipt["payload"]

    expected = [
        float(baseline_payload["fast_timescale_years"]),
        float(baseline_payload["slow_timescale_years"]),
    ]
    observed = [float(value) for value in candidate_payload["exact_timescales_years"]]
    if len(observed) != 2 or any(
        not math.isclose(left, right, rel_tol=2e-13, abs_tol=2e-13)
        for left, right in zip(expected, observed)
    ):
        raise RuntimeError(
            "candidate evaluation and exact baseline disagree on control timescales"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_bytes = _write_json(
        output_dir / "baseline-exact-modes.json", baseline_payload
    )
    candidate_bytes = _write_json(
        output_dir / "candidate-representation-dynamics.json", candidate_payload
    )
    baseline_artifact = _artifact_ref(
        "multirepresentation.ebm_dynamics.exact_modes",
        "thermal_decay_timescales/v1",
        "baseline-exact-modes.json",
        baseline_bytes,
    )
    candidate_artifact = _artifact_ref(
        "multirepresentation.ebm_dynamics.representation_dynamics",
        "ebm_representation_dynamics/v1",
        "candidate-representation-dynamics.json",
        candidate_bytes,
    )
    metrics = _derive_ebm_dynamics_metrics(experiment, candidate_payload)
    metric_bytes = _write_json(
        output_dir / "metric-results.json",
        {"schema_version": 1, "metrics": metrics},
    )
    metric_artifact = _artifact_ref(
        "multirepresentation.ebm_dynamics.metric_results",
        "metric_results/v1",
        "metric-results.json",
        metric_bytes,
    )

    libraries = {
        "numpy": importlib.metadata.version("numpy"),
        "scipy": importlib.metadata.version("scipy"),
        "pydmd": importlib.metadata.version("pydmd"),
    }
    seeds = _validated_seeds(experiment)
    baseline_identity = _execution_identity(
        EBM_BASELINE_METHOD, baseline_build, "scipy"
    )
    candidate_identity = _execution_identity(
        EBM_DYNAMICS_CANDIDATE_METHOD, candidate_build, "pydmd"
    )
    baseline_run = _run_manifest(
        run_id=_scoped_run_id(run_scope, EBM_DYNAMICS_EXPERIMENT, EBM_BASELINE_METHOD),
        experiment_id=EBM_DYNAMICS_EXPERIMENT,
        experiment_spec_digest=str(experiment["_runtime_spec_digest"]),
        revision=repository_revision,
        method_builds={EBM_BASELINE_METHOD: baseline_build},
        execution_identity=baseline_identity,
        dataset_digests=[dataset["digest"]],
        resolved_configuration=resolved_configuration,
        seeds=seeds,
        libraries={"numpy": libraries["numpy"], "scipy": libraries["scipy"]},
        receipt=baseline_receipt,
        artifacts=[baseline_artifact],
    )
    candidate_run = _run_manifest(
        run_id=_scoped_run_id(
            run_scope, EBM_DYNAMICS_EXPERIMENT, EBM_DYNAMICS_CANDIDATE_METHOD
        ),
        experiment_id=EBM_DYNAMICS_EXPERIMENT,
        experiment_spec_digest=str(experiment["_runtime_spec_digest"]),
        revision=repository_revision,
        method_builds={
            EBM_BASELINE_METHOD: baseline_build,
            EBM_DYNAMICS_CANDIDATE_METHOD: candidate_build,
        },
        execution_identity=candidate_identity,
        dataset_digests=[dataset["digest"]],
        resolved_configuration=resolved_configuration,
        seeds=seeds,
        libraries=libraries,
        receipt=candidate_receipt,
        artifacts=[candidate_artifact, metric_artifact],
    )
    return _finalize_two_run_outcome(
        output_dir=output_dir,
        experiment_id=EBM_DYNAMICS_EXPERIMENT,
        baseline_run=baseline_run,
        candidate_run=candidate_run,
        artifacts=[baseline_artifact, candidate_artifact, metric_artifact],
        metrics=metrics,
    )


def _run_ebm_forcing_adapter(
    *,
    experiment: Mapping[str, Any],
    output_dir: Path,
    repository_revision: str,
    run_scope: str,
    methods: Mapping[str, Mapping[str, Any]],
    resolved_configuration: Mapping[str, Any],
    configuration_records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    baseline_descriptor, candidate_descriptor = _require_methods(
        experiment,
        methods,
        baseline_method=EBM_BASELINE_METHOD,
        candidate_method=EBM_FORCING_CANDIDATE_METHOD,
    )
    datasets = _resolve_datasets(experiment)
    if len(datasets) != 2:
        raise ValueError("EBM forcing adapter requires exactly two datasets")
    by_id = {dataset["id"]: (dataset, path) for dataset, path in datasets}
    try:
        base_dataset, ebm_fixture_path = by_id[
            "physics.two_layer_ebm.geoffroy_mean.v1"
        ]
        protocol_dataset, protocol_fixture_path = by_id[
            "physics.two_layer_ebm.forcing_protocols.v1"
        ]
    except KeyError as exc:
        raise ValueError("EBM forcing adapter datasets do not match registered identities") from exc

    baseline_build = _resolved_method_build(
        EBM_BASELINE_METHOD, baseline_descriptor
    )
    candidate_build = _resolved_method_build(
        EBM_FORCING_CANDIDATE_METHOD, candidate_descriptor
    )
    baseline_receipt = _run_process(
        (
            sys.executable,
            str(ROOT / "reference" / "two_layer_energy_balance.py"),
            "--fixture",
            str(ebm_fixture_path),
            "--json",
        ),
        method_id=EBM_BASELINE_METHOD,
        backend_id="scipy",
    )
    candidate_receipt = _run_process(
        (
            sys.executable,
            str(ROOT / "reference" / "two_layer_forcing_protocols.py"),
            "--ebm-fixture",
            str(ebm_fixture_path),
            "--protocol-fixture",
            str(protocol_fixture_path),
            "--json",
        ),
        method_id=EBM_FORCING_CANDIDATE_METHOD,
        backend_id="scipy",
    )
    baseline_payload = baseline_receipt["payload"]
    candidate_payload = candidate_receipt["payload"]
    if candidate_payload.get("ebm_fixture_id") != baseline_payload.get("fixture_id"):
        raise RuntimeError("forcing candidate and baseline resolved different EBM fixtures")
    if float(candidate_payload["held_out_absolute_forcing_margin_w_m2"]) <= 0.0:
        raise RuntimeError("confirmation forcing domain does not extend discovery domain")

    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_bytes = _write_json(
        output_dir / "baseline-exact-modes.json", baseline_payload
    )
    candidate_bytes = _write_json(
        output_dir / "candidate-forcing-protocols.json", candidate_payload
    )
    baseline_artifact = _artifact_ref(
        "physics.two_layer_ebm.forcing_protocols.baseline",
        "thermal_decay_timescales/v1",
        "baseline-exact-modes.json",
        baseline_bytes,
    )
    candidate_artifact = _artifact_ref(
        "physics.two_layer_ebm.forcing_protocols.responses",
        "two_layer_forcing_response/v1",
        "candidate-forcing-protocols.json",
        candidate_bytes,
    )
    metrics = _derive_forcing_metrics(experiment, candidate_payload)
    metric_bytes = _write_json(
        output_dir / "metric-results.json",
        {"schema_version": 1, "metrics": metrics},
    )
    metric_artifact = _artifact_ref(
        "physics.two_layer_ebm.forcing_protocols.metric_results",
        "metric_results/v1",
        "metric-results.json",
        metric_bytes,
    )

    libraries = {
        "numpy": importlib.metadata.version("numpy"),
        "scipy": importlib.metadata.version("scipy"),
    }
    seeds = _validated_seeds(experiment)
    baseline_identity = _execution_identity(
        EBM_BASELINE_METHOD, baseline_build, "scipy"
    )
    candidate_identity = _execution_identity(
        EBM_FORCING_CANDIDATE_METHOD, candidate_build, "scipy"
    )
    baseline_run = _run_manifest(
        run_id=_scoped_run_id(run_scope, EBM_FORCING_EXPERIMENT, EBM_BASELINE_METHOD),
        experiment_id=EBM_FORCING_EXPERIMENT,
        experiment_spec_digest=str(experiment["_runtime_spec_digest"]),
        revision=repository_revision,
        method_builds={EBM_BASELINE_METHOD: baseline_build},
        execution_identity=baseline_identity,
        dataset_digests=[base_dataset["digest"]],
        resolved_configuration=resolved_configuration,
        seeds=seeds,
        libraries=libraries,
        receipt=baseline_receipt,
        artifacts=[baseline_artifact],
    )
    candidate_run = _run_manifest(
        run_id=_scoped_run_id(
            run_scope, EBM_FORCING_EXPERIMENT, EBM_FORCING_CANDIDATE_METHOD
        ),
        experiment_id=EBM_FORCING_EXPERIMENT,
        experiment_spec_digest=str(experiment["_runtime_spec_digest"]),
        revision=repository_revision,
        method_builds={
            EBM_BASELINE_METHOD: baseline_build,
            EBM_FORCING_CANDIDATE_METHOD: candidate_build,
        },
        execution_identity=candidate_identity,
        dataset_digests=[base_dataset["digest"], protocol_dataset["digest"]],
        resolved_configuration=resolved_configuration,
        seeds=seeds,
        libraries=libraries,
        receipt=candidate_receipt,
        artifacts=[candidate_artifact, metric_artifact],
    )
    return _finalize_two_run_outcome(
        output_dir=output_dir,
        experiment_id=EBM_FORCING_EXPERIMENT,
        baseline_run=baseline_run,
        candidate_run=candidate_run,
        artifacts=[baseline_artifact, candidate_artifact, metric_artifact],
        metrics=metrics,
    )


def _run_ebm_forced_ood_adapter(
    *,
    experiment: Mapping[str, Any],
    output_dir: Path,
    repository_revision: str,
    run_scope: str,
    methods: Mapping[str, Mapping[str, Any]],
    resolved_configuration: Mapping[str, Any],
    configuration_records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    baseline_descriptor, candidate_descriptor = _require_methods(
        experiment,
        methods,
        baseline_method=EBM_FORCING_CANDIDATE_METHOD,
        candidate_method=EBM_FORCED_OOD_CANDIDATE_METHOD,
    )
    datasets = _resolve_datasets(experiment)
    if len(datasets) != 3:
        raise ValueError("EBM forced-OOD adapter requires exactly three datasets")
    by_id = {dataset["id"]: (dataset, path) for dataset, path in datasets}
    try:
        base_dataset, ebm_fixture_path = by_id[
            "physics.two_layer_ebm.geoffroy_mean.v1"
        ]
        protocol_dataset, protocol_fixture_path = by_id[
            "physics.two_layer_ebm.forcing_protocols.v1"
        ]
        training_dataset, training_fixture_path = by_id[
            "physics.two_layer_ebm.forced_representation.v1"
        ]
    except KeyError as exc:
        raise ValueError(
            "EBM forced-OOD adapter datasets do not match registered identities"
        ) from exc

    baseline_build = _resolved_method_build(
        EBM_FORCING_CANDIDATE_METHOD, baseline_descriptor
    )
    candidate_build = _resolved_method_build(
        EBM_FORCED_OOD_CANDIDATE_METHOD, candidate_descriptor
    )
    baseline_receipt = _run_process(
        (
            sys.executable,
            str(ROOT / "reference" / "two_layer_forcing_protocols.py"),
            "--ebm-fixture",
            str(ebm_fixture_path),
            "--protocol-fixture",
            str(protocol_fixture_path),
            "--json",
        ),
        method_id=EBM_FORCING_CANDIDATE_METHOD,
        backend_id="scipy",
    )
    candidate_receipt = _run_process(
        (
            sys.executable,
            str(ROOT / "reference" / "two_layer_forced_representation.py"),
            "--ebm-fixture",
            str(ebm_fixture_path),
            "--protocol-fixture",
            str(protocol_fixture_path),
            "--training-fixture",
            str(training_fixture_path),
            "--json",
        ),
        method_id=EBM_FORCED_OOD_CANDIDATE_METHOD,
        backend_id="numpy.linalg",
    )
    baseline_payload = baseline_receipt["payload"]
    candidate_payload = candidate_receipt["payload"]
    if candidate_payload.get("ebm_fixture_id") != baseline_payload.get("ebm_fixture_id"):
        raise RuntimeError("forced-OOD candidate and baseline resolved different EBM fixtures")
    if (
        candidate_payload.get("forcing_protocol_fixture_id")
        != baseline_payload.get("fixture_id")
    ):
        raise RuntimeError(
            "forced-OOD candidate and baseline resolved different forcing protocols"
        )
    if float(candidate_payload["held_out_absolute_forcing_margin_w_m2"]) <= 0.0:
        raise RuntimeError("forced-OOD confirmation does not leave discovery forcing range")

    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_bytes = _write_json(
        output_dir / "baseline-forcing-protocols.json", baseline_payload
    )
    candidate_bytes = _write_json(
        output_dir / "candidate-forced-ood.json", candidate_payload
    )
    baseline_artifact = _artifact_ref(
        "multirepresentation.ebm_forced_ood.exact_forcing",
        "two_layer_forcing_response/v1",
        "baseline-forcing-protocols.json",
        baseline_bytes,
    )
    candidate_artifact = _artifact_ref(
        "multirepresentation.ebm_forced_ood.controlled_linear",
        "controlled_linear_representation_ood/v1",
        "candidate-forced-ood.json",
        candidate_bytes,
    )
    metrics = _derive_forced_ood_metrics(experiment, candidate_payload)
    metric_bytes = _write_json(
        output_dir / "metric-results.json",
        {"schema_version": 1, "metrics": metrics},
    )
    metric_artifact = _artifact_ref(
        "multirepresentation.ebm_forced_ood.metric_results",
        "metric_results/v1",
        "metric-results.json",
        metric_bytes,
    )

    libraries = {
        "numpy": importlib.metadata.version("numpy"),
        "scipy": importlib.metadata.version("scipy"),
    }
    seeds = _validated_seeds(experiment)
    baseline_identity = _execution_identity(
        EBM_FORCING_CANDIDATE_METHOD, baseline_build, "scipy"
    )
    candidate_identity = _execution_identity(
        EBM_FORCED_OOD_CANDIDATE_METHOD, candidate_build, "numpy.linalg"
    )
    baseline_run = _run_manifest(
        run_id=_scoped_run_id(
            run_scope, EBM_FORCED_OOD_EXPERIMENT, EBM_FORCING_CANDIDATE_METHOD
        ),
        experiment_id=EBM_FORCED_OOD_EXPERIMENT,
        experiment_spec_digest=str(experiment["_runtime_spec_digest"]),
        revision=repository_revision,
        method_builds={EBM_FORCING_CANDIDATE_METHOD: baseline_build},
        execution_identity=baseline_identity,
        dataset_digests=[base_dataset["digest"], protocol_dataset["digest"]],
        resolved_configuration=resolved_configuration,
        seeds=seeds,
        libraries=libraries,
        receipt=baseline_receipt,
        artifacts=[baseline_artifact],
    )
    candidate_run = _run_manifest(
        run_id=_scoped_run_id(
            run_scope, EBM_FORCED_OOD_EXPERIMENT, EBM_FORCED_OOD_CANDIDATE_METHOD
        ),
        experiment_id=EBM_FORCED_OOD_EXPERIMENT,
        experiment_spec_digest=str(experiment["_runtime_spec_digest"]),
        revision=repository_revision,
        method_builds={
            EBM_FORCING_CANDIDATE_METHOD: baseline_build,
            EBM_FORCED_OOD_CANDIDATE_METHOD: candidate_build,
        },
        execution_identity=candidate_identity,
        dataset_digests=[
            base_dataset["digest"],
            protocol_dataset["digest"],
            training_dataset["digest"],
        ],
        resolved_configuration=resolved_configuration,
        seeds=seeds,
        libraries=libraries,
        receipt=candidate_receipt,
        artifacts=[candidate_artifact, metric_artifact],
    )
    return _finalize_two_run_outcome(
        output_dir=output_dir,
        experiment_id=EBM_FORCED_OOD_EXPERIMENT,
        baseline_run=baseline_run,
        candidate_run=candidate_run,
        artifacts=[baseline_artifact, candidate_artifact, metric_artifact],
        metrics=metrics,
    )


def _run_ebm_observation_degradation_adapter(
    *,
    experiment: Mapping[str, Any],
    output_dir: Path,
    repository_revision: str,
    run_scope: str,
    methods: Mapping[str, Mapping[str, Any]],
    resolved_configuration: Mapping[str, Any],
    configuration_records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    baseline_descriptor, candidate_descriptor = _require_methods(
        experiment,
        methods,
        baseline_method=EBM_FORCING_CANDIDATE_METHOD,
        candidate_method=EBM_FORCED_OOD_CANDIDATE_METHOD,
    )
    datasets = _resolve_datasets(experiment)
    if len(datasets) != 4:
        raise ValueError(
            "EBM observation-degradation adapter requires exactly four datasets"
        )
    by_id = {dataset["id"]: (dataset, path) for dataset, path in datasets}
    try:
        base_dataset, ebm_fixture_path = by_id[
            "physics.two_layer_ebm.geoffroy_mean.v1"
        ]
        protocol_dataset, protocol_fixture_path = by_id[
            "physics.two_layer_ebm.forcing_protocols.v1"
        ]
        training_dataset, training_fixture_path = by_id[
            "physics.two_layer_ebm.forced_representation.v1"
        ]
        observation_dataset, observation_fixture_path = by_id[
            "physics.two_layer_ebm.observation_degradation.v1"
        ]
    except KeyError as exc:
        raise ValueError(
            "EBM observation-degradation datasets do not match registered identities"
        ) from exc

    baseline_build = _resolved_method_build(
        EBM_FORCING_CANDIDATE_METHOD, baseline_descriptor
    )
    candidate_build = _resolved_method_build(
        EBM_FORCED_OOD_CANDIDATE_METHOD, candidate_descriptor
    )
    baseline_receipt = _run_process(
        (
            sys.executable,
            str(ROOT / "reference" / "two_layer_forcing_protocols.py"),
            "--ebm-fixture",
            str(ebm_fixture_path),
            "--protocol-fixture",
            str(protocol_fixture_path),
            "--json",
        ),
        method_id=EBM_FORCING_CANDIDATE_METHOD,
        backend_id="scipy",
    )
    candidate_receipt = _run_process(
        (
            sys.executable,
            str(ROOT / "reference" / "two_layer_observation_degradation.py"),
            "--ebm-fixture",
            str(ebm_fixture_path),
            "--protocol-fixture",
            str(protocol_fixture_path),
            "--training-fixture",
            str(training_fixture_path),
            "--observation-fixture",
            str(observation_fixture_path),
            "--json",
        ),
        method_id=EBM_FORCED_OOD_CANDIDATE_METHOD,
        backend_id="numpy.linalg",
    )
    baseline_payload = baseline_receipt["payload"]
    candidate_payload = candidate_receipt["payload"]
    if candidate_payload.get("ebm_fixture_id") != baseline_payload.get("ebm_fixture_id"):
        raise RuntimeError(
            "observation-degradation candidate and baseline resolved different EBM fixtures"
        )
    if (
        candidate_payload.get("forcing_protocol_fixture_id")
        != baseline_payload.get("fixture_id")
    ):
        raise RuntimeError(
            "observation-degradation candidate and baseline resolved different forcing protocols"
        )
    expected_training_fixture_id = _load_json(training_fixture_path).get("fixture_id")
    expected_observation_fixture_id = _load_json(
        observation_fixture_path
    ).get("fixture_id")
    if candidate_payload.get("training_fixture_id") != expected_training_fixture_id:
        raise RuntimeError(
            "observation-degradation candidate resolved unexpected training fixture"
        )
    if candidate_payload.get("fixture_id") != expected_observation_fixture_id:
        raise RuntimeError(
            "observation-degradation candidate resolved unexpected observation fixture"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_bytes = _write_json(
        output_dir / "baseline-forcing-protocols.json", baseline_payload
    )
    candidate_bytes = _write_json(
        output_dir / "candidate-observation-degradation.json", candidate_payload
    )
    baseline_artifact = _artifact_ref(
        "multirepresentation.ebm_observation_degradation.exact_forcing",
        "two_layer_forcing_response/v1",
        "baseline-forcing-protocols.json",
        baseline_bytes,
    )
    candidate_artifact = _artifact_ref(
        "multirepresentation.ebm_observation_degradation.controlled_linear",
        "observation_degraded_representation_ood/v1",
        "candidate-observation-degradation.json",
        candidate_bytes,
    )
    metrics = _derive_observation_degradation_metrics(
        experiment, candidate_payload
    )
    metric_bytes = _write_json(
        output_dir / "metric-results.json",
        {"schema_version": 1, "metrics": metrics},
    )
    metric_artifact = _artifact_ref(
        "multirepresentation.ebm_observation_degradation.metric_results",
        "metric_results/v1",
        "metric-results.json",
        metric_bytes,
    )

    libraries = {
        "numpy": importlib.metadata.version("numpy"),
        "scipy": importlib.metadata.version("scipy"),
    }
    seeds = _validated_seeds(experiment)
    baseline_identity = _execution_identity(
        EBM_FORCING_CANDIDATE_METHOD, baseline_build, "scipy"
    )
    candidate_identity = _execution_identity(
        EBM_FORCED_OOD_CANDIDATE_METHOD, candidate_build, "numpy.linalg"
    )
    baseline_run = _run_manifest(
        run_id=_scoped_run_id(
            run_scope,
            EBM_OBSERVATION_DEGRADATION_EXPERIMENT,
            EBM_FORCING_CANDIDATE_METHOD,
        ),
        experiment_id=EBM_OBSERVATION_DEGRADATION_EXPERIMENT,
        experiment_spec_digest=str(experiment["_runtime_spec_digest"]),
        revision=repository_revision,
        method_builds={EBM_FORCING_CANDIDATE_METHOD: baseline_build},
        execution_identity=baseline_identity,
        dataset_digests=[base_dataset["digest"], protocol_dataset["digest"]],
        resolved_configuration=resolved_configuration,
        seeds=seeds,
        libraries=libraries,
        receipt=baseline_receipt,
        artifacts=[baseline_artifact],
    )
    candidate_run = _run_manifest(
        run_id=_scoped_run_id(
            run_scope,
            EBM_OBSERVATION_DEGRADATION_EXPERIMENT,
            EBM_FORCED_OOD_CANDIDATE_METHOD,
        ),
        experiment_id=EBM_OBSERVATION_DEGRADATION_EXPERIMENT,
        experiment_spec_digest=str(experiment["_runtime_spec_digest"]),
        revision=repository_revision,
        method_builds={
            EBM_FORCING_CANDIDATE_METHOD: baseline_build,
            EBM_FORCED_OOD_CANDIDATE_METHOD: candidate_build,
        },
        execution_identity=candidate_identity,
        dataset_digests=[
            base_dataset["digest"],
            protocol_dataset["digest"],
            training_dataset["digest"],
            observation_dataset["digest"],
        ],
        resolved_configuration=resolved_configuration,
        seeds=seeds,
        libraries=libraries,
        receipt=candidate_receipt,
        artifacts=[candidate_artifact, metric_artifact],
    )
    return _finalize_two_run_outcome(
        output_dir=output_dir,
        experiment_id=EBM_OBSERVATION_DEGRADATION_EXPERIMENT,
        baseline_run=baseline_run,
        candidate_run=candidate_run,
        artifacts=[baseline_artifact, candidate_artifact, metric_artifact],
        metrics=metrics,
    )


def _run_ebm_parameter_identifiability_adapter(
    *,
    experiment: Mapping[str, Any],
    output_dir: Path,
    repository_revision: str,
    run_scope: str,
    methods: Mapping[str, Mapping[str, Any]],
    resolved_configuration: Mapping[str, Any],
    configuration_records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    baseline_descriptor, candidate_descriptor = _require_methods(
        experiment,
        methods,
        baseline_method=EBM_FORCING_CANDIDATE_METHOD,
        candidate_method=EBM_PARAMETER_SENSITIVITY_METHOD,
    )
    datasets = _resolve_datasets(experiment)
    if len(datasets) != 3:
        raise ValueError(
            "EBM parameter-identifiability adapter requires exactly three datasets"
        )
    by_id = {dataset["id"]: (dataset, path) for dataset, path in datasets}
    try:
        base_dataset, ebm_fixture_path = by_id[
            "physics.two_layer_ebm.geoffroy_mean.v1"
        ]
        protocol_dataset, protocol_fixture_path = by_id[
            "physics.two_layer_ebm.forcing_protocols.v1"
        ]
        parameter_dataset, parameter_fixture_path = by_id[
            "physics.two_layer_ebm.parameter_identifiability.v1"
        ]
    except KeyError as exc:
        raise ValueError(
            "EBM parameter-identifiability datasets do not match registered identities"
        ) from exc

    baseline_build = _resolved_method_build(
        EBM_FORCING_CANDIDATE_METHOD, baseline_descriptor
    )
    candidate_build = _resolved_method_build(
        EBM_PARAMETER_SENSITIVITY_METHOD, candidate_descriptor
    )
    baseline_receipt = _run_process(
        (
            sys.executable,
            str(ROOT / "reference" / "two_layer_forcing_protocols.py"),
            "--ebm-fixture",
            str(ebm_fixture_path),
            "--protocol-fixture",
            str(protocol_fixture_path),
            "--json",
        ),
        method_id=EBM_FORCING_CANDIDATE_METHOD,
        backend_id="scipy",
    )
    candidate_receipt = _run_process(
        (
            sys.executable,
            str(ROOT / "reference" / "two_layer_parameter_identifiability.py"),
            "--ebm-fixture",
            str(ebm_fixture_path),
            "--protocol-fixture",
            str(protocol_fixture_path),
            "--parameter-fixture",
            str(parameter_fixture_path),
            "--json",
        ),
        method_id=EBM_PARAMETER_SENSITIVITY_METHOD,
        backend_id="numpy.linalg+scipy",
    )
    baseline_payload = baseline_receipt["payload"]
    candidate_payload = candidate_receipt["payload"]
    if candidate_payload.get("ebm_fixture_id") != baseline_payload.get("ebm_fixture_id"):
        raise RuntimeError(
            "parameter-identifiability candidate and baseline resolved different EBM fixtures"
        )
    if (
        candidate_payload.get("forcing_protocol_fixture_id")
        != baseline_payload.get("fixture_id")
    ):
        raise RuntimeError(
            "parameter-identifiability candidate and baseline resolved different forcing protocols"
        )
    expected_parameter_fixture_id = _load_json(parameter_fixture_path).get("fixture_id")
    if candidate_payload.get("fixture_id") != expected_parameter_fixture_id:
        raise RuntimeError(
            "parameter-identifiability candidate resolved unexpected sensitivity fixture"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_bytes = _write_json(
        output_dir / "baseline-forcing-protocols.json", baseline_payload
    )
    candidate_bytes = _write_json(
        output_dir / "candidate-parameter-identifiability.json", candidate_payload
    )
    baseline_artifact = _artifact_ref(
        "physics.two_layer_ebm.parameter_identifiability.exact_forcing",
        "two_layer_forcing_response/v1",
        "baseline-forcing-protocols.json",
        baseline_bytes,
    )
    candidate_artifact = _artifact_ref(
        "physics.two_layer_ebm.parameter_identifiability.sensitivity_rank",
        "parameter_sensitivity_rank/v1",
        "candidate-parameter-identifiability.json",
        candidate_bytes,
    )
    metrics = _derive_parameter_identifiability_metrics(
        experiment, candidate_payload
    )
    metric_bytes = _write_json(
        output_dir / "metric-results.json",
        {"schema_version": 1, "metrics": metrics},
    )
    metric_artifact = _artifact_ref(
        "physics.two_layer_ebm.parameter_identifiability.metric_results",
        "metric_results/v1",
        "metric-results.json",
        metric_bytes,
    )

    libraries = {
        "numpy": importlib.metadata.version("numpy"),
        "scipy": importlib.metadata.version("scipy"),
    }
    seeds = _validated_seeds(experiment)
    baseline_identity = _execution_identity(
        EBM_FORCING_CANDIDATE_METHOD, baseline_build, "scipy"
    )
    candidate_identity = _execution_identity(
        EBM_PARAMETER_SENSITIVITY_METHOD, candidate_build, "numpy.linalg+scipy"
    )
    baseline_run = _run_manifest(
        run_id=_scoped_run_id(
            run_scope,
            EBM_PARAMETER_IDENTIFIABILITY_EXPERIMENT,
            EBM_FORCING_CANDIDATE_METHOD,
        ),
        experiment_id=EBM_PARAMETER_IDENTIFIABILITY_EXPERIMENT,
        experiment_spec_digest=str(experiment["_runtime_spec_digest"]),
        revision=repository_revision,
        method_builds={EBM_FORCING_CANDIDATE_METHOD: baseline_build},
        execution_identity=baseline_identity,
        dataset_digests=[base_dataset["digest"], protocol_dataset["digest"]],
        resolved_configuration=resolved_configuration,
        seeds=seeds,
        libraries=libraries,
        receipt=baseline_receipt,
        artifacts=[baseline_artifact],
    )
    candidate_run = _run_manifest(
        run_id=_scoped_run_id(
            run_scope,
            EBM_PARAMETER_IDENTIFIABILITY_EXPERIMENT,
            EBM_PARAMETER_SENSITIVITY_METHOD,
        ),
        experiment_id=EBM_PARAMETER_IDENTIFIABILITY_EXPERIMENT,
        experiment_spec_digest=str(experiment["_runtime_spec_digest"]),
        revision=repository_revision,
        method_builds={
            EBM_FORCING_CANDIDATE_METHOD: baseline_build,
            EBM_PARAMETER_SENSITIVITY_METHOD: candidate_build,
        },
        execution_identity=candidate_identity,
        dataset_digests=[
            base_dataset["digest"],
            protocol_dataset["digest"],
            parameter_dataset["digest"],
        ],
        resolved_configuration=resolved_configuration,
        seeds=seeds,
        libraries=libraries,
        receipt=candidate_receipt,
        artifacts=[candidate_artifact, metric_artifact],
    )
    return _finalize_two_run_outcome(
        output_dir=output_dir,
        experiment_id=EBM_PARAMETER_IDENTIFIABILITY_EXPERIMENT,
        baseline_run=baseline_run,
        candidate_run=candidate_run,
        artifacts=[baseline_artifact, candidate_artifact, metric_artifact],
        metrics=metrics,
    )


def _run_ebm_stochastic_statistics_adapter(
    *,
    experiment: Mapping[str, Any],
    output_dir: Path,
    repository_revision: str,
    run_scope: str,
    methods: Mapping[str, Mapping[str, Any]],
    resolved_configuration: Mapping[str, Any],
    configuration_records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    baseline_descriptor, candidate_descriptor = _require_methods(
        experiment,
        methods,
        baseline_method=EBM_STOCHASTIC_BASELINE_METHOD,
        candidate_method=EBM_STOCHASTIC_STATISTICS_METHOD,
    )
    datasets = _resolve_datasets(experiment)
    if len(datasets) != 2:
        raise ValueError(
            "EBM stochastic-statistics adapter requires exactly two datasets"
        )
    by_id = {dataset["id"]: (dataset, path) for dataset, path in datasets}
    try:
        base_dataset, ebm_fixture_path = by_id[
            "physics.two_layer_ebm.geoffroy_mean.v1"
        ]
        stochastic_dataset, stochastic_fixture_path = by_id[
            "physics.two_layer_ebm.stochastic_internal_variability.v1"
        ]
    except KeyError as exc:
        raise ValueError(
            "EBM stochastic-statistics datasets do not match registered identities"
        ) from exc

    baseline_build = _resolved_method_build(
        EBM_STOCHASTIC_BASELINE_METHOD, baseline_descriptor
    )
    candidate_build = _resolved_method_build(
        EBM_STOCHASTIC_STATISTICS_METHOD, candidate_descriptor
    )
    baseline_receipt = _run_process(
        (
            sys.executable,
            str(ROOT / "reference" / "two_layer_stochastic_variability.py"),
            "--ebm-fixture",
            str(ebm_fixture_path),
            "--stochastic-fixture",
            str(stochastic_fixture_path),
            "--json",
        ),
        method_id=EBM_STOCHASTIC_BASELINE_METHOD,
        backend_id="numpy.random+scipy",
    )
    candidate_receipt = _run_process(
        (
            sys.executable,
            str(
                ROOT
                / "reference"
                / "two_layer_stochastic_representation_statistics.py"
            ),
            "--ebm-fixture",
            str(ebm_fixture_path),
            "--stochastic-fixture",
            str(stochastic_fixture_path),
            "--json",
        ),
        method_id=EBM_STOCHASTIC_STATISTICS_METHOD,
        backend_id="numpy.linalg+scipy",
    )
    baseline_payload = baseline_receipt["payload"]
    candidate_payload = candidate_receipt["payload"]
    if candidate_payload.get("ebm_fixture_id") != baseline_payload.get("ebm_fixture_id"):
        raise RuntimeError(
            "stochastic statistics candidate and baseline resolved different EBM fixtures"
        )
    if candidate_payload.get("fixture_id") != baseline_payload.get("fixture_id"):
        raise RuntimeError(
            "stochastic statistics candidate and baseline resolved different stochastic fixtures"
        )
    if candidate_payload.get("trajectory_digest") != baseline_payload.get(
        "trajectory_digest"
    ):
        raise RuntimeError(
            "stochastic statistics candidate and baseline resolved different trajectories"
        )
    fixture_seed = int(_load_json(stochastic_fixture_path)["seed"])
    seeds = _validated_seeds(experiment)
    if seeds != [fixture_seed]:
        raise ValueError(
            "stochastic statistics experiment seed must match its immutable fixture"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_bytes = _write_json(
        output_dir / "baseline-stochastic-variability.json", baseline_payload
    )
    candidate_bytes = _write_json(
        output_dir / "candidate-stochastic-statistics.json", candidate_payload
    )
    baseline_artifact = _artifact_ref(
        "multirepresentation.ebm_stochastic_statistics.physical_reference",
        "stochastic_two_layer_trajectory_summary/v1",
        "baseline-stochastic-variability.json",
        baseline_bytes,
    )
    candidate_artifact = _artifact_ref(
        "multirepresentation.ebm_stochastic_statistics.fixed_decode",
        "stochastic_representation_statistics/v1",
        "candidate-stochastic-statistics.json",
        candidate_bytes,
    )
    metrics = _derive_stochastic_statistics_metrics(
        experiment, candidate_payload
    )
    metric_bytes = _write_json(
        output_dir / "metric-results.json",
        {"schema_version": 1, "metrics": metrics},
    )
    metric_artifact = _artifact_ref(
        "multirepresentation.ebm_stochastic_statistics.metric_results",
        "metric_results/v1",
        "metric-results.json",
        metric_bytes,
    )

    libraries = {
        "numpy": importlib.metadata.version("numpy"),
        "scipy": importlib.metadata.version("scipy"),
    }
    baseline_identity = _execution_identity(
        EBM_STOCHASTIC_BASELINE_METHOD, baseline_build, "numpy.random+scipy"
    )
    candidate_identity = _execution_identity(
        EBM_STOCHASTIC_STATISTICS_METHOD, candidate_build, "numpy.linalg+scipy"
    )
    dataset_digests = [base_dataset["digest"], stochastic_dataset["digest"]]
    baseline_run = _run_manifest(
        run_id=_scoped_run_id(
            run_scope,
            EBM_STOCHASTIC_STATISTICS_EXPERIMENT,
            EBM_STOCHASTIC_BASELINE_METHOD,
        ),
        experiment_id=EBM_STOCHASTIC_STATISTICS_EXPERIMENT,
        experiment_spec_digest=str(experiment["_runtime_spec_digest"]),
        revision=repository_revision,
        method_builds={EBM_STOCHASTIC_BASELINE_METHOD: baseline_build},
        execution_identity=baseline_identity,
        dataset_digests=dataset_digests,
        resolved_configuration=resolved_configuration,
        seeds=seeds,
        libraries=libraries,
        receipt=baseline_receipt,
        artifacts=[baseline_artifact],
    )
    candidate_run = _run_manifest(
        run_id=_scoped_run_id(
            run_scope,
            EBM_STOCHASTIC_STATISTICS_EXPERIMENT,
            EBM_STOCHASTIC_STATISTICS_METHOD,
        ),
        experiment_id=EBM_STOCHASTIC_STATISTICS_EXPERIMENT,
        experiment_spec_digest=str(experiment["_runtime_spec_digest"]),
        revision=repository_revision,
        method_builds={
            EBM_STOCHASTIC_BASELINE_METHOD: baseline_build,
            EBM_STOCHASTIC_STATISTICS_METHOD: candidate_build,
        },
        execution_identity=candidate_identity,
        dataset_digests=dataset_digests,
        resolved_configuration=resolved_configuration,
        seeds=seeds,
        libraries=libraries,
        receipt=candidate_receipt,
        artifacts=[candidate_artifact, metric_artifact],
    )
    return _finalize_two_run_outcome(
        output_dir=output_dir,
        experiment_id=EBM_STOCHASTIC_STATISTICS_EXPERIMENT,
        baseline_run=baseline_run,
        candidate_run=candidate_run,
        artifacts=[baseline_artifact, candidate_artifact, metric_artifact],
        metrics=metrics,
    )


def _run_ebm_regime_feedback_adapter(
    *,
    experiment: Mapping[str, Any],
    output_dir: Path,
    repository_revision: str,
    run_scope: str,
    methods: Mapping[str, Mapping[str, Any]],
    resolved_configuration: Mapping[str, Any],
    configuration_records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    baseline_descriptor, candidate_descriptor = _require_methods(
        experiment,
        methods,
        baseline_method=EBM_REGIME_BASELINE_METHOD,
        candidate_method=EBM_REGIME_GATED_METHOD,
    )
    datasets = _resolve_datasets(experiment)
    if len(datasets) != 2:
        raise ValueError("EBM regime-feedback adapter requires exactly two datasets")
    by_id = {dataset["id"]: (dataset, path) for dataset, path in datasets}
    try:
        base_dataset, ebm_fixture_path = by_id[
            "physics.two_layer_ebm.geoffroy_mean.v1"
        ]
        regime_dataset, regime_fixture_path = by_id[
            "physics.two_layer_ebm.regime_feedback.v1"
        ]
    except KeyError as exc:
        raise ValueError(
            "EBM regime-feedback datasets do not match registered identities"
        ) from exc

    baseline_build = _resolved_method_build(
        EBM_REGIME_BASELINE_METHOD, baseline_descriptor
    )
    candidate_build = _resolved_method_build(
        EBM_REGIME_GATED_METHOD, candidate_descriptor
    )
    baseline_receipt = _run_process(
        (
            sys.executable,
            str(ROOT / "reference" / "two_layer_regime_feedback.py"),
            "--ebm-fixture",
            str(ebm_fixture_path),
            "--regime-fixture",
            str(regime_fixture_path),
            "--json",
        ),
        method_id=EBM_REGIME_BASELINE_METHOD,
        backend_id="numpy+scipy",
    )
    candidate_receipt = _run_process(
        (
            sys.executable,
            str(ROOT / "reference" / "two_layer_regime_representation.py"),
            "--ebm-fixture",
            str(ebm_fixture_path),
            "--regime-fixture",
            str(regime_fixture_path),
            "--json",
        ),
        method_id=EBM_REGIME_GATED_METHOD,
        backend_id="numpy.linalg+scipy",
    )
    baseline_payload = baseline_receipt["payload"]
    candidate_payload = candidate_receipt["payload"]
    if candidate_payload.get("ebm_fixture_id") != baseline_payload.get("ebm_fixture_id"):
        raise RuntimeError(
            "regime-feedback candidate and baseline resolved different EBM fixtures"
        )
    if candidate_payload.get("fixture_id") != baseline_payload.get("fixture_id"):
        raise RuntimeError(
            "regime-feedback candidate and baseline resolved different regime fixtures"
        )
    if candidate_payload.get("sample_digest") != baseline_payload.get("sample_digest"):
        raise RuntimeError(
            "regime-feedback candidate and baseline resolved different sample sets"
        )

    fixture = _load_json(regime_fixture_path)
    seeds = _validated_seeds(experiment)
    expected_seeds = [
        int(fixture["discovery_seeds"]["cold"]),
        int(fixture["discovery_seeds"]["warm"]),
        int(fixture["confirmation_seeds"]["cold"]),
        int(fixture["confirmation_seeds"]["warm"]),
    ]
    if seeds != expected_seeds:
        raise ValueError(
            "regime-feedback experiment seeds must match its immutable fixture"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_bytes = _write_json(
        output_dir / "baseline-regime-feedback.json", baseline_payload
    )
    candidate_bytes = _write_json(
        output_dir / "candidate-regime-representation.json", candidate_payload
    )
    baseline_artifact = _artifact_ref(
        "multirepresentation.ebm_regime_feedback.physical_reference",
        "two_layer_regime_feedback_samples/v1",
        "baseline-regime-feedback.json",
        baseline_bytes,
    )
    candidate_artifact = _artifact_ref(
        "multirepresentation.ebm_regime_feedback.regime_gated",
        "regime_representation_diagnostics/v1",
        "candidate-regime-representation.json",
        candidate_bytes,
    )
    metrics = _derive_regime_feedback_metrics(experiment, candidate_payload)
    metric_bytes = _write_json(
        output_dir / "metric-results.json",
        {"schema_version": 1, "metrics": metrics},
    )
    metric_artifact = _artifact_ref(
        "multirepresentation.ebm_regime_feedback.metric_results",
        "metric_results/v1",
        "metric-results.json",
        metric_bytes,
    )

    libraries = {
        "numpy": importlib.metadata.version("numpy"),
        "scipy": importlib.metadata.version("scipy"),
    }
    baseline_identity = _execution_identity(
        EBM_REGIME_BASELINE_METHOD, baseline_build, "numpy+scipy"
    )
    candidate_identity = _execution_identity(
        EBM_REGIME_GATED_METHOD, candidate_build, "numpy.linalg+scipy"
    )
    dataset_digests = [base_dataset["digest"], regime_dataset["digest"]]
    baseline_run = _run_manifest(
        run_id=_scoped_run_id(
            run_scope,
            EBM_REGIME_FEEDBACK_EXPERIMENT,
            EBM_REGIME_BASELINE_METHOD,
        ),
        experiment_id=EBM_REGIME_FEEDBACK_EXPERIMENT,
        experiment_spec_digest=str(experiment["_runtime_spec_digest"]),
        revision=repository_revision,
        method_builds={EBM_REGIME_BASELINE_METHOD: baseline_build},
        execution_identity=baseline_identity,
        dataset_digests=dataset_digests,
        resolved_configuration=resolved_configuration,
        seeds=seeds,
        libraries=libraries,
        receipt=baseline_receipt,
        artifacts=[baseline_artifact],
    )
    candidate_run = _run_manifest(
        run_id=_scoped_run_id(
            run_scope,
            EBM_REGIME_FEEDBACK_EXPERIMENT,
            EBM_REGIME_GATED_METHOD,
        ),
        experiment_id=EBM_REGIME_FEEDBACK_EXPERIMENT,
        experiment_spec_digest=str(experiment["_runtime_spec_digest"]),
        revision=repository_revision,
        method_builds={
            EBM_REGIME_BASELINE_METHOD: baseline_build,
            EBM_REGIME_GATED_METHOD: candidate_build,
        },
        execution_identity=candidate_identity,
        dataset_digests=dataset_digests,
        resolved_configuration=resolved_configuration,
        seeds=seeds,
        libraries=libraries,
        receipt=candidate_receipt,
        artifacts=[candidate_artifact, metric_artifact],
    )
    return _finalize_two_run_outcome(
        output_dir=output_dir,
        experiment_id=EBM_REGIME_FEEDBACK_EXPERIMENT,
        baseline_run=baseline_run,
        candidate_run=candidate_run,
        artifacts=[baseline_artifact, candidate_artifact, metric_artifact],
        metrics=metrics,
    )


def _available_runtime_libraries() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("numpy", "scipy", "pydmd"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


@dataclass(frozen=True)
class ResolvedExperimentContext:
    experiment: Mapping[str, Any]
    output_dir: Path
    repository_revision: str
    run_scope: str
    methods: Mapping[str, Mapping[str, Any]]
    resolved_configuration: Mapping[str, Any]
    configuration_records: Mapping[str, Mapping[str, Any]]


def _resolve_experiment_context(
    *,
    experiment_path: Path,
    output_dir: Path,
    repository_revision: str,
    run_scope: str,
) -> ResolvedExperimentContext:
    if not isinstance(repository_revision, str) or not repository_revision.strip():
        raise ValueError("repository_revision must be explicit")
    if not isinstance(run_scope, str) or not run_scope.strip():
        raise ValueError("run_scope must be explicit")

    experiment = _load_json(experiment_path)
    experiment["_runtime_spec_digest"] = _sha256_file(experiment_path)
    methods = _method_map()
    resolved_configuration, configuration_records = _resolve_configuration(experiment)
    return ResolvedExperimentContext(
        experiment=experiment,
        output_dir=output_dir,
        repository_revision=repository_revision,
        run_scope=run_scope,
        methods=methods,
        resolved_configuration=resolved_configuration,
        configuration_records=configuration_records,
    )


def _method_builds_for_failed_run(
    experiment: Mapping[str, Any],
    methods: Mapping[str, Mapping[str, Any]],
    failed_method_id: str,
) -> dict[str, str]:
    baseline = experiment.get("baseline_methods", [])
    candidate = experiment.get("candidate_methods", [])
    if not isinstance(baseline, list) or not isinstance(candidate, list):
        raise ValueError("experiment method lists must be arrays")
    if failed_method_id in baseline:
        ordered = list(baseline)
    elif failed_method_id in candidate:
        ordered = [*baseline, *candidate]
    else:
        raise ValueError(
            f"failed method {failed_method_id!r} is not declared by experiment"
        )

    builds: dict[str, str] = {}
    for method_id in ordered:
        descriptor = methods.get(method_id)
        if descriptor is None:
            raise ValueError(f"failed-run method does not resolve: {method_id}")
        builds[method_id] = _resolved_method_build(method_id, descriptor)
    return builds


def _dataset_digests_for_failed_command(
    experiment: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> list[str]:
    argv = receipt.get("argv", [])
    if not isinstance(argv, list):
        raise ValueError("failed process receipt lacks argv")
    tokens = {str(item) for item in argv}
    digests: list[str] = []
    for dataset, path in _resolve_datasets(experiment):
        if str(path) in tokens:
            digests.append(str(dataset["digest"]))
    return digests


def _persist_failed_method_run(
    *,
    failure: MethodProcessFailure,
    context: ResolvedExperimentContext,
) -> Path:
    experiment = context.experiment
    experiment_id = experiment.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id:
        raise ValueError("failed-run provenance requires experiment_id")

    method_builds = _method_builds_for_failed_run(
        experiment,
        context.methods,
        failure.method_id,
    )
    identity = _execution_identity(
        failure.method_id,
        method_builds[failure.method_id],
        failure.backend_id,
    )
    manifest = _run_manifest(
        run_id=_scoped_run_id(context.run_scope, experiment_id, failure.method_id),
        experiment_id=experiment_id,
        experiment_spec_digest=str(experiment["_runtime_spec_digest"]),
        revision=context.repository_revision,
        method_builds=method_builds,
        execution_identity=identity,
        dataset_digests=_dataset_digests_for_failed_command(
            experiment,
            failure.receipt,
        ),
        resolved_configuration=context.resolved_configuration,
        seeds=_validated_seeds(experiment),
        libraries=_available_runtime_libraries(),
        receipt=failure.receipt,
        artifacts=[],
    )

    baseline = experiment.get("baseline_methods", [])
    candidate = experiment.get("candidate_methods", [])
    if failure.method_id in baseline:
        filename = "run-baseline.json"
    elif failure.method_id in candidate:
        filename = "run-candidate.json"
    else:
        filename = "run-failed.json"
    context.output_dir.mkdir(parents=True, exist_ok=True)
    path = context.output_dir / filename
    _write_json(path, manifest)
    return path


ExperimentAdapter = Callable[..., dict[str, Any]]


def _experiment_adapters() -> dict[str, ExperimentAdapter]:
    """Return the explicit local adapter registry.

    The registry owns only experiment-id-to-family-evaluator routing. Scientific
    semantics remain inside each named adapter; adding a new family is therefore
    a small registration change rather than another branch in common execution.
    """
    return {
        EBM_DYNAMICS_EXPERIMENT: _run_ebm_dynamics_adapter,
        EBM_FORCING_EXPERIMENT: _run_ebm_forcing_adapter,
        EBM_FORCED_OOD_EXPERIMENT: _run_ebm_forced_ood_adapter,
        EBM_OBSERVATION_DEGRADATION_EXPERIMENT: _run_ebm_observation_degradation_adapter,
        EBM_PARAMETER_IDENTIFIABILITY_EXPERIMENT: _run_ebm_parameter_identifiability_adapter,
        EBM_STOCHASTIC_STATISTICS_EXPERIMENT: _run_ebm_stochastic_statistics_adapter,
        EBM_REGIME_FEEDBACK_EXPERIMENT: _run_ebm_regime_feedback_adapter,
    }


def _dispatch_experiment(
    context: ResolvedExperimentContext,
) -> dict[str, Any]:
    experiment_id = context.experiment.get("experiment_id")
    adapters = _experiment_adapters()
    if not isinstance(experiment_id, str) or experiment_id not in adapters:
        supported = ", ".join(sorted(adapters))
        raise ValueError(
            f"no local CPU adapter for experiment {experiment_id!r}; "
            f"supported: {supported}"
        )
    adapter = adapters[experiment_id]
    return adapter(
        experiment=context.experiment,
        output_dir=context.output_dir,
        repository_revision=context.repository_revision,
        run_scope=context.run_scope,
        methods=context.methods,
        resolved_configuration=context.resolved_configuration,
        configuration_records=context.configuration_records,
    )


def run_experiment(
    *,
    experiment_path: Path,
    output_dir: Path,
    repository_revision: str,
    run_scope: str,
) -> dict[str, Any]:
    context = _resolve_experiment_context(
        experiment_path=experiment_path,
        output_dir=output_dir,
        repository_revision=repository_revision,
        run_scope=run_scope,
    )
    try:
        return _dispatch_experiment(context)
    except MethodProcessFailure as failure:
        _persist_failed_method_run(
            failure=failure,
            context=context,
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="run a registered Climate CPU experiment")
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--repository-revision",
        default=os.environ.get("GITHUB_SHA") or os.environ.get("CLIMATE_REPOSITORY_REVISION"),
    )
    parser.add_argument(
        "--run-scope",
        default=os.environ.get("CLIMATE_RUN_SCOPE") or os.environ.get("GITHUB_RUN_ID"),
    )
    args = parser.parse_args()
    if not args.repository_revision:
        parser.error("--repository-revision is required outside a revision-aware execution environment")
    run_scope = args.run_scope or f"direct-{args.repository_revision[:12]}"
    run_experiment(
        experiment_path=args.experiment,
        output_dir=args.output_dir,
        repository_revision=args.repository_revision,
        run_scope=run_scope,
    )
    print(args.output_dir / "outcome.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
