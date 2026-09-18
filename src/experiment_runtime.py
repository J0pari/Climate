#!/usr/bin/env python3
"""Canonical local CPU experiment runtime for Climate's common evidence spine.

The first adapter executes the registered two-layer EBM representation-dynamics
experiment. The runtime owns identity resolution, immutable input checks,
subprocess receipts, content-addressed artifacts, typed metric results, and
contract-shaped run/outcome records. Scientific methods remain in their own
modules and are invoked without changing their semantics.
"""
from __future__ import annotations

import argparse
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
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = ROOT / "experiments" / "multirepresentation-ebm-dynamics.v1.json"
METHOD_REGISTRY = ROOT / "methods" / "registry.json"
CONTRACT = ROOT / "contracts" / "climate.cue"

SUPPORTED_EXPERIMENT = "multirepresentation.ebm_dynamics.v1"
BASELINE_METHOD = "physics.two_layer_ebm.exact_modes_v1"
CANDIDATE_METHOD = "dynamics.dmd.pydmd_v1"
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


def _resolve_dataset(experiment: Mapping[str, Any]) -> tuple[dict[str, Any], Path]:
    datasets = experiment.get("datasets")
    if not isinstance(datasets, list) or len(datasets) != 1 or not isinstance(datasets[0], dict):
        raise ValueError("the EBM runtime adapter requires exactly one dataset reference")
    dataset = dict(datasets[0])
    citation = dataset.get("citation")
    if not isinstance(citation, str) or not citation.startswith("fixtures/"):
        raise ValueError("EBM dataset citation must resolve to an immutable fixture path")
    path = (ROOT / citation).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError("dataset fixture escapes repository") from exc
    if not path.is_file():
        raise ValueError(f"dataset fixture does not exist: {citation}")
    if _sha256_file(path) != dataset.get("digest"):
        raise ValueError(f"dataset digest mismatch for {citation}")
    return dataset, path


def _run_process(command: Sequence[str]) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    process = subprocess.run(
        list(command), cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
    )
    ended = datetime.now(timezone.utc)
    if process.returncode != 0:
        raise RuntimeError(
            f"method process failed with exit code {process.returncode}: "
            + process.stderr.decode("utf-8", errors="replace")
        )
    try:
        payload = json.loads(process.stdout.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError("method process did not emit valid JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("method process JSON output must be an object")
    _canonical_json_bytes(payload)
    return {
        "payload": payload,
        "command": shlex.join(command),
        "started_at": started.isoformat(),
        "ended_at": ended.isoformat(),
        "stdout_digest": _sha256_bytes(process.stdout),
        "stderr_digest": _sha256_bytes(process.stderr),
    }


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


def _derive_metrics(experiment: Mapping[str, Any], candidate: Mapping[str, Any]) -> list[dict[str, Any]]:
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
    revision: str,
    method_builds: Mapping[str, str],
    execution_identity: Mapping[str, Any],
    dataset_digest: str,
    resolved_configuration: Mapping[str, Any],
    seeds: Sequence[int],
    libraries: Mapping[str, str],
    receipt: Mapping[str, Any],
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "experiment_id": experiment_id,
        "repository_revision": revision,
        "producer_build": revision,
        "contract_fingerprint": hashlib.sha256(CONTRACT.read_bytes()).hexdigest(),
        "method_builds": dict(method_builds),
        "execution": {
            "requested": dict(execution_identity),
            "status": "eligible",
            "resolved": dict(execution_identity),
        },
        "scientific_output_eligible": True,
        "resolved_dataset_digests": [dataset_digest],
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
        "exit_code": 0,
        "started_at": receipt["started_at"],
        "ended_at": receipt["ended_at"],
        "stdout_digest": receipt["stdout_digest"],
        "stderr_digest": receipt["stderr_digest"],
        "artifacts": [dict(item) for item in artifacts],
    }


def run_experiment(
    *,
    experiment_path: Path,
    output_dir: Path,
    repository_revision: str,
    run_scope: str,
) -> dict[str, Any]:
    experiment = _load_json(experiment_path)
    experiment_id = experiment.get("experiment_id")
    if experiment_id != SUPPORTED_EXPERIMENT:
        raise ValueError(f"runtime adapter supports {SUPPORTED_EXPERIMENT}, got {experiment_id!r}")
    if not repository_revision.strip() or not run_scope.strip():
        raise ValueError("repository_revision and run_scope must be explicit")

    methods = _method_map()
    for method_id in (BASELINE_METHOD, CANDIDATE_METHOD):
        descriptor = methods.get(method_id)
        if descriptor is None:
            raise ValueError(f"experiment method does not resolve: {method_id}")
        if descriptor.get("maturity") not in {
            "runnable", "verified", "validated", "replicated", "decision-eligible"
        }:
            raise ValueError(f"experiment method is not runnable: {method_id}")
    if experiment.get("baseline_methods") != [BASELINE_METHOD]:
        raise ValueError("EBM runtime requires the registered exact-mode baseline")
    if experiment.get("candidate_methods") != [CANDIDATE_METHOD]:
        raise ValueError("EBM runtime requires the registered PyDMD candidate")

    dataset, fixture_path = _resolve_dataset(experiment)
    resolved_configuration, records = _resolve_configuration(experiment)
    numerical_refs = resolved_configuration.get("numerical_policies", [])
    if len(numerical_refs) != 1:
        raise ValueError("EBM runtime requires exactly one numerical policy")
    settings = records[numerical_refs[0]["configuration_id"]]["settings"]
    dt_years = float(settings["transition_dt_years"])
    if not math.isfinite(dt_years) or dt_years <= 0.0:
        raise ValueError("transition_dt_years must be finite and positive")

    baseline_build = _resolved_method_build(BASELINE_METHOD, methods[BASELINE_METHOD])
    candidate_build = _resolved_method_build(CANDIDATE_METHOD, methods[CANDIDATE_METHOD])
    baseline_command = (
        sys.executable, str(ROOT / "reference" / "two_layer_energy_balance.py"),
        "--fixture", str(fixture_path), "--json",
    )
    candidate_command = (
        sys.executable, str(ROOT / "reference" / "two_layer_representation_dynamics.py"),
        "--fixture", str(fixture_path), "--dt-years", repr(dt_years), "--json",
    )
    baseline_receipt = _run_process(baseline_command)
    candidate_receipt = _run_process(candidate_command)
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
        raise RuntimeError("candidate evaluation and exact baseline disagree on control timescales")

    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_bytes = _write_json(output_dir / "baseline-exact-modes.json", baseline_payload)
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
    metrics = _derive_metrics(experiment, candidate_payload)
    metric_bytes = _write_json(
        output_dir / "metric-results.json", {"schema_version": 1, "metrics": metrics}
    )
    metric_artifact = _artifact_ref(
        "multirepresentation.ebm_dynamics.metric_results",
        "metric_results/v1",
        "metric-results.json",
        metric_bytes,
    )

    baseline_identity = _execution_identity(BASELINE_METHOD, baseline_build, "scipy")
    candidate_identity = _execution_identity(CANDIDATE_METHOD, candidate_build, "pydmd")
    libraries = {
        "numpy": importlib.metadata.version("numpy"),
        "scipy": importlib.metadata.version("scipy"),
        "pydmd": importlib.metadata.version("pydmd"),
    }
    seeds = experiment.get("seeds", [])
    if not isinstance(seeds, list) or not all(isinstance(seed, int) for seed in seeds):
        raise ValueError("experiment seeds must be an integer list")

    baseline_run = _run_manifest(
        run_id=f"{run_scope}.{BASELINE_METHOD}",
        experiment_id=experiment_id,
        revision=repository_revision,
        method_builds={BASELINE_METHOD: baseline_build},
        execution_identity=baseline_identity,
        dataset_digest=dataset["digest"],
        resolved_configuration=resolved_configuration,
        seeds=seeds,
        libraries={"numpy": libraries["numpy"], "scipy": libraries["scipy"]},
        receipt=baseline_receipt,
        artifacts=[baseline_artifact],
    )
    candidate_run = _run_manifest(
        run_id=f"{run_scope}.{CANDIDATE_METHOD}",
        experiment_id=experiment_id,
        revision=repository_revision,
        method_builds={BASELINE_METHOD: baseline_build, CANDIDATE_METHOD: candidate_build},
        execution_identity=candidate_identity,
        dataset_digest=dataset["digest"],
        resolved_configuration=resolved_configuration,
        seeds=seeds,
        libraries=libraries,
        receipt=candidate_receipt,
        artifacts=[candidate_artifact, metric_artifact],
    )
    _write_json(output_dir / "run-baseline.json", baseline_run)
    _write_json(output_dir / "run-candidate.json", candidate_run)

    outcome = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "runs": [baseline_run, candidate_run],
        "artifacts": [baseline_artifact, candidate_artifact, metric_artifact],
        "metrics": metrics,
        "evidence": [],
    }
    _write_json(output_dir / "outcome.json", outcome)
    return outcome


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
