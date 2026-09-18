#!/usr/bin/env python3
"""Preregistered claim-scoped evaluator for sheaf structural residual equivalence.

This evaluator does not decide that the sheaf-consistency claim is true. It
executes the repository-pinned structural ablation, verifies the exact checkout
and immutable inputs, applies the predeclared interpretation mapping, and emits
a candidate EvidenceRecord for later explicit registration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVALUATION = (
    ROOT / "evaluations" / "sheaf-consistency-structural-residual-equivalence.v1.json"
)
METHOD_REGISTRY = ROOT / "methods" / "registry.json"
CLAIM_REGISTRY = ROOT / "claims" / "registry.json"
CONTRACT = ROOT / "contracts" / "climate.cue"
METRIC_MARKER = "CLIMATE_SHEAF_CLAIM_METRICS="


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, payload: Any) -> bytes:
    data = _canonical_json_bytes(payload)
    path.write_bytes(data)
    return data


def _method_map() -> dict[str, dict[str, Any]]:
    methods = _load_json(METHOD_REGISTRY).get("methods")
    if not isinstance(methods, list):
        raise ValueError("method registry has no methods list")
    return {
        item["method_id"]: item
        for item in methods
        if isinstance(item, dict) and isinstance(item.get("method_id"), str)
    }


def _method_build(method_id: str, descriptor: Mapping[str, Any]) -> str:
    identity = descriptor.get("build_identity")
    if not isinstance(identity, dict) or identity.get("policy") != "source_digest_at_run":
        raise ValueError(f"{method_id} has no supported source-bound build identity")
    sources = identity.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError(f"{method_id} has no build sources")
    digest = hashlib.sha256()
    for relative in sources:
        if not isinstance(relative, str) or not relative:
            raise ValueError(f"{method_id} has malformed build source")
        path = ROOT / relative
        if not path.is_file():
            raise ValueError(f"{method_id} build source does not exist: {relative}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return "source-sha256:" + digest.hexdigest()


def _git_stdout(*args: str) -> str:
    process = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(
            "git provenance command failed: "
            + process.stderr.decode("utf-8", errors="replace")
        )
    return process.stdout.decode("utf-8").strip()


def _verify_checkout(repository_revision: str) -> dict[str, str]:
    head = _git_stdout("rev-parse", "HEAD")
    if head != repository_revision:
        raise RuntimeError(
            f"requested repository revision {repository_revision} != checkout {head}"
        )
    dirty = _git_stdout("status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise RuntimeError("claim evidence execution requires a clean tracked checkout")
    return {"git_head": head, "git_dirty": "false"}


def _tool_version(command: Sequence[str]) -> str:
    process = subprocess.run(
        list(command),
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode != 0:
        return "unavailable"
    text = process.stdout.decode("utf-8", errors="replace").strip()
    if not text:
        text = process.stderr.decode("utf-8", errors="replace").strip()
    return text.splitlines()[0] if text else "unknown"


def _artifact_ref(
    artifact_id: str,
    schema: str,
    filename: str,
    data: bytes,
) -> dict[str, Any]:
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
        for item in experiment.get(group, []):
            if not isinstance(item, dict) or not isinstance(item.get("metric_id"), str):
                raise ValueError(f"malformed metric definition in {group}")
            definitions[item["metric_id"]] = dict(item)
    return definitions


def _parse_metric_marker(stdout: bytes) -> dict[str, Any]:
    decoded = stdout.decode("utf-8")
    matches = [
        line.removeprefix(METRIC_MARKER)
        for line in decoded.splitlines()
        if line.startswith(METRIC_MARKER)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {METRIC_MARKER} line")
    payload = json.loads(matches[0])
    if not isinstance(payload, dict):
        raise ValueError("metric marker payload must be an object")
    return payload


def _dataset_digests(experiment: Mapping[str, Any]) -> list[str]:
    digests: list[str] = []
    datasets = experiment.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("experiment requires dataset references")
    for dataset in datasets:
        if not isinstance(dataset, dict):
            raise ValueError("dataset reference must be an object")
        citation = dataset.get("citation")
        digest = dataset.get("digest")
        if not isinstance(citation, str) or not isinstance(digest, str):
            raise ValueError("dataset reference lacks citation/digest")
        path = (ROOT / citation).resolve()
        path.relative_to(ROOT.resolve())
        if _sha256_file(path) != digest:
            raise ValueError(f"dataset digest mismatch: {citation}")
        digests.append(digest)
    return digests


def _validate_mapping(
    evaluation: Mapping[str, Any],
    experiment: Mapping[str, Any],
) -> None:
    if evaluation.get("experiment_id") != experiment.get("experiment_id"):
        raise ValueError("evaluation mapping experiment_id does not match experiment")
    claims = _load_json(CLAIM_REGISTRY).get("claims")
    if not isinstance(claims, list):
        raise ValueError("claim registry has no claims list")
    if evaluation.get("claim_id") not in {
        item.get("claim_id") for item in claims if isinstance(item, dict)
    }:
        raise ValueError("evaluation mapping claim_id does not resolve")
    if evaluation.get("claim_maturity_effect") != "unchanged":
        raise ValueError("this evaluator cannot promote claim maturity")


def run_evaluation(
    *,
    evaluation_path: Path,
    output_dir: Path,
    repository_revision: str,
    run_scope: str,
) -> dict[str, Any]:
    evaluation = _load_json(evaluation_path)
    experiment_path = ROOT / str(evaluation["experiment_path"])
    experiment = _load_json(experiment_path)
    _validate_mapping(evaluation, experiment)
    git_identity = _verify_checkout(repository_revision)

    methods = _method_map()
    method_ids = [
        *experiment.get("baseline_methods", []),
        *experiment.get("candidate_methods", []),
    ]
    if not method_ids or not all(isinstance(item, str) for item in method_ids):
        raise ValueError("experiment method identities are malformed")
    method_builds = {
        method_id: _method_build(method_id, methods[method_id])
        for method_id in method_ids
    }
    candidate_methods = experiment.get("candidate_methods", [])
    if candidate_methods != ["sheaf.real.spectral_v1"]:
        raise ValueError("claim evaluator requires sheaf.real.spectral_v1 candidate")

    dataset_digests = _dataset_digests(experiment)
    command = (
        "cargo",
        "test",
        "--quiet",
        "--test",
        "sheaf_claim_metrics",
        "--",
        "--nocapture",
    )
    started = datetime.now(timezone.utc)
    process = subprocess.run(
        list(command),
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    ended = datetime.now(timezone.utc)
    if process.returncode != 0:
        raise RuntimeError(
            f"preregistered sheaf claim command failed with {process.returncode}: "
            + process.stderr.decode("utf-8", errors="replace")
        )

    raw_metrics = _parse_metric_marker(process.stdout)
    definitions = _metric_definitions(experiment)
    metric_results: list[dict[str, Any]] = []
    for target in evaluation["required_metric_targets"]:
        metric_id = target["metric_id"]
        value = float(raw_metrics[metric_id])
        expected = float(target["target"])
        tolerance = float(target["absolute_tolerance"])
        if abs(value - expected) > tolerance:
            raise RuntimeError(
                f"preregistered metric target missed for {metric_id}: "
                f"value={value}, target={expected}, tolerance={tolerance}"
            )
        metric_results.append(
            {
                "metric": definitions[metric_id],
                "status": "finite",
                "value": value,
                "reference_population": (
                    "synthetic-linear-v1 deterministic fixture suite"
                ),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    metric_bytes = _write_json(
        output_dir / "metric-results.json",
        {"schema_version": 1, "metrics": metric_results},
    )
    metric_artifact = _artifact_ref(
        "climate.sheaf.consistency.structural_residual_equivalence.metrics",
        "metric_results/v1",
        "metric-results.json",
        metric_bytes,
    )

    candidate_method = candidate_methods[0]
    run_id = f"{run_scope}.{experiment['experiment_id']}.{candidate_method}"
    toolchains = {
        "python": platform.python_version(),
        "rustc": _tool_version(("rustc", "--version")),
        "cargo": _tool_version(("cargo", "--version")),
        **git_identity,
    }
    run_manifest = {
        "run_id": run_id,
        "experiment_id": experiment["experiment_id"],
        "experiment_spec_digest": _sha256_file(experiment_path),
        "repository_revision": repository_revision,
        "producer_build": repository_revision,
        "contract_fingerprint": hashlib.sha256(CONTRACT.read_bytes()).hexdigest(),
        "method_builds": method_builds,
        "execution": {
            "requested": {
                "method_id": candidate_method,
                "implementation_id": candidate_method,
                "implementation_build": method_builds[candidate_method],
                "backend_id": "rust-cargo-test",
                "precision": "FP64",
                "resource_class": "R2_toolchain_ci",
            },
            "status": "eligible",
            "resolved": {
                "method_id": candidate_method,
                "implementation_id": candidate_method,
                "implementation_build": method_builds[candidate_method],
                "backend_id": "rust-cargo-test",
                "precision": "FP64",
                "resource_class": "R2_toolchain_ci",
            },
        },
        "scientific_output_eligible": True,
        "resolved_dataset_digests": dataset_digests,
        "resolved_configuration": dict(experiment.get("configuration", {})),
        "seeds": list(experiment.get("seeds", [])),
        "environment": {
            "os": platform.platform(),
            "arch": platform.machine(),
            "toolchains": toolchains,
            "libraries": {},
        },
        "hardware": {"cpu": platform.processor() or platform.machine() or "unknown-cpu"},
        "commands": [shlex.join(command)],
        "exit_code": int(process.returncode),
        "started_at": started.isoformat(),
        "ended_at": ended.isoformat(),
        "stdout_digest": _sha256_bytes(process.stdout),
        "stderr_digest": _sha256_bytes(process.stderr),
        "artifacts": [metric_artifact],
    }
    _write_json(output_dir / "run-candidate.json", run_manifest)

    evidence = {
        "evidence_id": evaluation["evidence_id"],
        "claim_id": evaluation["claim_id"],
        "evidence_class": evaluation["evidence_class"],
        "verification": evaluation["verification_on_success"],
        "relation": evaluation["relation_on_success"],
        "run_id": run_id,
        "artifacts": [metric_artifact],
        "metrics": metric_results,
        "scope": evaluation["scope"],
        "threats_to_validity": list(evaluation["threats_to_validity"]),
        "notes": evaluation["interpretation_rule"],
    }
    _write_json(output_dir / "evidence-candidate.json", evidence)

    outcome = {
        "schema_version": 1,
        "experiment_id": experiment["experiment_id"],
        "runs": [run_manifest],
        "artifacts": [metric_artifact],
        "metrics": metric_results,
        "evidence": [evidence],
    }
    _write_json(output_dir / "outcome.json", outcome)
    return outcome


def main() -> int:
    parser = argparse.ArgumentParser(
        description="run preregistered sheaf claim evidence evaluation"
    )
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--repository-revision",
        default=os.environ.get("GITHUB_SHA"),
    )
    parser.add_argument(
        "--run-scope",
        default=os.environ.get("CLIMATE_RUN_SCOPE") or os.environ.get("GITHUB_RUN_ID"),
    )
    args = parser.parse_args()
    if not args.repository_revision:
        parser.error("--repository-revision is required")
    if not args.run_scope:
        parser.error("--run-scope is required")
    run_evaluation(
        evaluation_path=args.evaluation,
        output_dir=args.output_dir,
        repository_revision=args.repository_revision,
        run_scope=args.run_scope,
    )
    print(args.output_dir / "outcome.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
