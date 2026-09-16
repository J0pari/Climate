#!/usr/bin/env python3
"""Cross-record integrity checks for Climate experiment specifications.

CUE validates the shape of an ExperimentSpec. This module validates graph edges
that CUE deliberately does not resolve across repository files: method IDs,
local fixture digests, and resource compatibility.

The checker does not execute scientific methods.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = ROOT / "experiments"
METHODS_REGISTRY = ROOT / "methods" / "registry.json"


@dataclass(frozen=True)
class Finding:
    code: str
    path: str
    message: str

    def render(self) -> str:
        return f"{self.code}: {self.path}: {self.message}"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return f"sha256:{digest}"


def method_map(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for method in registry.get("methods", []):
        method_id = method.get("method_id")
        if isinstance(method_id, str) and method_id:
            result[method_id] = method
    return result


def _resource_number(resource: dict[str, Any] | None, key: str) -> float:
    if not resource:
        return 0.0
    value = resource.get(key, 0)
    return float(value) if isinstance(value, (int, float)) else 0.0


def check_experiment(
    root: Path,
    rel_path: str,
    experiment: dict[str, Any],
    methods: dict[str, dict[str, Any]],
) -> list[Finding]:
    findings: list[Finding] = []
    candidate_ids = experiment.get("candidate_methods", [])
    baseline_ids = experiment.get("baseline_methods", [])

    candidate_set = {value for value in candidate_ids if isinstance(value, str)}
    baseline_set = {value for value in baseline_ids if isinstance(value, str)}
    overlap = sorted(candidate_set & baseline_set)
    if overlap:
        findings.append(Finding(
            "experiments.method_role_overlap",
            rel_path,
            f"method(s) appear as both candidate and baseline: {', '.join(overlap)}",
        ))

    referenced_ids = list(candidate_ids) + list(baseline_ids)
    resolved: list[dict[str, Any]] = []
    for method_id in referenced_ids:
        if not isinstance(method_id, str) or method_id not in methods:
            findings.append(Finding(
                "experiments.method_missing",
                rel_path,
                f"referenced method does not exist in methods/registry.json: {method_id!r}",
            ))
            continue
        resolved.append(methods[method_id])

    experiment_resource = experiment.get("resource") or {}
    for key in ("cpu_cores", "memory_bytes", "gpu_count", "vram_bytes"):
        required = max((_resource_number(method.get("resource"), key) for method in resolved), default=0.0)
        provided = _resource_number(experiment_resource, key)
        if provided < required:
            findings.append(Finding(
                "experiments.resource_underprovisioned",
                rel_path,
                f"experiment {key}={provided:g} is below referenced-method requirement {required:g}",
            ))

    network_rank = {"none": 0, "restricted": 1, "required": 2}
    provided_network = experiment_resource.get("network", "none")
    provided_rank = network_rank.get(provided_network, -1)
    required_rank = max(
        (network_rank.get((method.get("resource") or {}).get("network", "none"), -1) for method in resolved),
        default=0,
    )
    if provided_rank < required_rank:
        findings.append(Finding(
            "experiments.network_underprovisioned",
            rel_path,
            "experiment network policy is weaker than at least one referenced method requires",
        ))

    metric_ids: list[str] = []
    for group in ("primary_metrics", "secondary_metrics"):
        for metric in experiment.get(group, []) or []:
            metric_id = metric.get("metric_id") if isinstance(metric, dict) else None
            if isinstance(metric_id, str):
                metric_ids.append(metric_id)
    duplicates = sorted({mid for mid in metric_ids if metric_ids.count(mid) > 1})
    if duplicates:
        findings.append(Finding(
            "experiments.metric_duplicate",
            rel_path,
            f"metric IDs are duplicated within experiment: {', '.join(duplicates)}",
        ))

    for dataset in experiment.get("datasets", []) or []:
        if not isinstance(dataset, dict):
            continue
        citation = dataset.get("citation")
        expected_digest = dataset.get("digest")
        if not isinstance(citation, str) or not citation:
            continue
        # Only repository-relative citations are checked here. URLs and formal
        # citations are metadata, not local artifact references.
        if "://" in citation or citation.startswith("doi:"):
            continue
        local_path = (root / citation).resolve()
        try:
            local_path.relative_to(root.resolve())
        except ValueError:
            findings.append(Finding(
                "experiments.dataset_path_escape",
                rel_path,
                f"dataset citation escapes repository root: {citation}",
            ))
            continue
        if not local_path.is_file():
            findings.append(Finding(
                "experiments.dataset_missing",
                rel_path,
                f"locally cited dataset/fixture does not exist: {citation}",
            ))
            continue
        actual_digest = sha256_file(local_path)
        if expected_digest != actual_digest:
            findings.append(Finding(
                "experiments.dataset_digest_mismatch",
                rel_path,
                f"digest for {citation} is {expected_digest!r}, actual {actual_digest!r}",
            ))

    return findings


def check(
    root: Path = ROOT,
    methods_registry: dict[str, Any] | None = None,
    experiments: dict[str, dict[str, Any]] | None = None,
) -> list[Finding]:
    if methods_registry is None:
        methods_registry = load_json(root / "methods" / "registry.json")
    methods = method_map(methods_registry)

    if experiments is None:
        experiments = {}
        experiments_dir = root / "experiments"
        if experiments_dir.is_dir():
            for path in sorted(experiments_dir.glob("*.json")):
                experiments[path.relative_to(root).as_posix()] = load_json(path)

    findings: list[Finding] = []
    ids: dict[str, str] = {}
    for rel_path, experiment in sorted(experiments.items()):
        experiment_id = experiment.get("experiment_id")
        if isinstance(experiment_id, str) and experiment_id:
            if experiment_id in ids:
                findings.append(Finding(
                    "experiments.id_duplicate",
                    rel_path,
                    f"experiment_id {experiment_id!r} already declared in {ids[experiment_id]}",
                ))
            else:
                ids[experiment_id] = rel_path
        findings.extend(check_experiment(root, rel_path, experiment, methods))

    return sorted(findings, key=lambda finding: (finding.path, finding.code, finding.message))


def main() -> int:
    findings = check()
    for finding in findings:
        print(finding.render())
    if findings:
        print(f"\n{len(findings)} experiment-integrity finding(s)")
        return 1
    print("experiment graph integrity: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
