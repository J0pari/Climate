#!/usr/bin/env python3
"""Cross-file integrity for preregistered external-artifact evaluations."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EVALUATIONS_DIR = ROOT / "evaluations"


@dataclass(frozen=True)
class Finding:
    code: str
    path: str
    message: str


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def check(root: Path = ROOT) -> list[Finding]:
    root = root.resolve()
    evaluations_dir = root / "evaluations"
    findings: list[Finding] = []
    ids: dict[str, str] = {}
    if not evaluations_dir.is_dir():
        return [Finding(
            "external_evaluations.directory_missing",
            "evaluations",
            "external evaluation specification directory is missing",
        )]
    for path in sorted(evaluations_dir.glob("*.json")):
        rel = path.relative_to(root).as_posix()
        try:
            spec = _load(path)
        except (OSError, json.JSONDecodeError) as error:
            findings.append(Finding(
                "external_evaluations.invalid_json", rel, str(error)))
            continue
        evaluation_id = spec.get("evaluation_id")
        if isinstance(evaluation_id, str):
            if evaluation_id in ids:
                findings.append(Finding(
                    "external_evaluations.duplicate_id", rel,
                    f"evaluation_id already declared in {ids[evaluation_id]}"))
            else:
                ids[evaluation_id] = rel
        is_external_artifact_evaluation = (
            "subject_contract" in spec or "task_set" in spec
        )
        if not is_external_artifact_evaluation:
            continue
        adapter = spec.get("adapter")
        if not isinstance(adapter, dict):
            findings.append(Finding(
                "external_evaluations.adapter_missing", rel,
                "external-artifact evaluation must declare its native adapter boundary"))
        else:
            status = adapter.get("status")
            if status not in {"unavailable", "native_output_import", "available"}:
                findings.append(Finding(
                    "external_evaluations.adapter_status_invalid", rel,
                    f"unknown external adapter status: {status!r}"))
            for field, code, message in (
                (
                    "native_capability",
                    "external_evaluations.native_capability_missing",
                    "external adapter must name the externally owned native capability",
                ),
                (
                    "climate_semantic_gap",
                    "external_evaluations.climate_semantic_gap_missing",
                    "external adapter must state the Climate-owned semantic gap",
                ),
            ):
                value = adapter.get(field)
                if not isinstance(value, str) or not value.strip():
                    findings.append(Finding(code, rel, message))
            if status in {"native_output_import", "available"} and adapter.get("receipt_contract") != "climate.external-model-runtime-receipt/v1":
                findings.append(Finding(
                    "external_evaluations.receipt_contract_missing", rel,
                    "available external adapter must bind climate.external-model-runtime-receipt/v1"))
        task_set = spec.get("task_set")
        if not isinstance(task_set, dict):
            findings.append(Finding(
                "external_evaluations.task_set_missing", rel,
                "external-artifact evaluation task_set must be an artifact reference"))
            continue
        uri = task_set.get("uri")
        if not isinstance(uri, str) or not uri:
            findings.append(Finding(
                "external_evaluations.task_set_uri_missing", rel,
                "task_set.uri must name the immutable local task artifact"))
            continue
        target = (root / uri).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            findings.append(Finding(
                "external_evaluations.task_set_path_escape", rel,
                f"task set escapes repository root: {uri}"))
            continue
        if not target.is_file():
            findings.append(Finding(
                "external_evaluations.task_set_missing", rel,
                f"task set does not exist: {uri}"))
            continue
        actual = _sha256(target)
        if task_set.get("digest") != actual:
            findings.append(Finding(
                "external_evaluations.task_set_digest_mismatch", rel,
                f"task set digest is {task_set.get('digest')!r}, actual {actual!r}"))
    return sorted(findings, key=lambda item: (item.path, item.code, item.message))


def main() -> int:
    findings = check()
    if findings:
        for finding in findings:
            print(f"{finding.code}: {finding.path}: {finding.message}")
        return 1
    print("external artifact evaluations: identity integrity ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
