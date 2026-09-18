#!/usr/bin/env python3
"""Integrity checks for executable method identities.

Method descriptors declare stable semantics and the source files that define
their local implementation. Actual build identity is derived at run time from
those sources and the resolved environment; hand-maintained commit-looking
build strings are deliberately not authority.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METHODS = ROOT / "methods" / "registry.json"

RUNNABLE_MATURITIES = {
    "runnable",
    "verified",
    "validated",
    "replicated",
    "decision-eligible",
}
BUILD_POLICIES = {"source_digest_at_run"}


@dataclass(frozen=True)
class Finding:
    code: str
    message: str
    method_id: str | None = None
    reference: str | None = None


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def check(root: Path, registry: dict[str, Any]) -> list[Finding]:
    findings: list[Finding] = []
    if registry.get("schema_version") != 2:
        findings.append(Finding(
            "methods.schema_version",
            "method registry must use schema_version 2 source-bound build identity",
        ))

    methods = registry.get("methods")
    if not isinstance(methods, list):
        return findings + [Finding("methods.registry_not_list", "methods must be a list")]

    records: dict[str, dict[str, Any]] = {}
    for index, method in enumerate(methods):
        if not isinstance(method, dict):
            findings.append(Finding("methods.entry_not_object", f"method at index {index} is not an object"))
            continue
        method_id = method.get("method_id")
        if not isinstance(method_id, str) or not method_id:
            findings.append(Finding("methods.id_missing", f"method at index {index} has no usable method_id"))
            continue
        if method_id in records:
            findings.append(Finding("methods.id_duplicate", "method_id appears more than once", method_id))
            continue
        records[method_id] = method

        if "implementation_build" in method:
            findings.append(Finding(
                "methods.stale_build_field",
                "implementation_build is a run receipt, not authored method authority",
                method_id,
            ))

        identity = method.get("build_identity")
        if method.get("maturity") in RUNNABLE_MATURITIES and not isinstance(identity, dict):
            findings.append(Finding(
                "methods.build_identity_missing",
                "runnable method must declare how actual build identity is derived",
                method_id,
            ))
            continue
        if identity is None:
            continue
        if not isinstance(identity, dict):
            findings.append(Finding("methods.build_identity_invalid", "build_identity must be an object", method_id))
            continue
        policy = identity.get("policy")
        if policy not in BUILD_POLICIES:
            findings.append(Finding(
                "methods.build_policy_unknown",
                f"unsupported build identity policy {policy!r}",
                method_id,
            ))
        sources = identity.get("sources")
        if not isinstance(sources, list) or not sources:
            findings.append(Finding(
                "methods.build_sources_missing",
                "source-bound build identity requires at least one source path",
                method_id,
            ))
            continue
        if len(set(sources)) != len(sources):
            findings.append(Finding("methods.build_sources_duplicate", "build source paths must be unique", method_id))
        for source in sources:
            if not isinstance(source, str) or not source:
                findings.append(Finding("methods.build_source_invalid", "build source path must be a non-empty string", method_id))
                continue
            candidate = (root / source).resolve()
            try:
                candidate.relative_to(root)
            except ValueError:
                findings.append(Finding("methods.build_source_escape", "build source path escapes repository", method_id, source))
                continue
            if not candidate.is_file():
                findings.append(Finding("methods.build_source_missing", "declared build source does not exist", method_id, source))

    known = set(records)
    for method_id, method in records.items():
        refs = method.get("reference_methods", [])
        if not isinstance(refs, list):
            findings.append(Finding("methods.reference_not_list", "reference_methods must be a list", method_id))
            continue
        for ref in refs:
            if ref == method_id:
                findings.append(Finding("methods.reference_self", "method cannot reference itself", method_id, str(ref)))
            elif ref not in known:
                findings.append(Finding("methods.reference_missing", "reference method does not resolve", method_id, str(ref)))

    return sorted(findings, key=lambda item: (item.code, item.method_id or "", item.reference or ""))


def main() -> int:
    parser = argparse.ArgumentParser(description="check Climate method identity integrity")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--methods", type=Path, default=DEFAULT_METHODS)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        registry = load_json(args.methods)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"methods.registry_load_error: {error}")
        return 1

    findings = check(args.root.resolve(), registry)
    if args.json:
        print(json.dumps({
            "ok": not findings,
            "finding_count": len(findings),
            "findings": [asdict(item) for item in findings],
        }, indent=2, sort_keys=True))
    elif findings:
        for finding in findings:
            suffix = f" [{finding.method_id}]" if finding.method_id else ""
            ref = f" -> {finding.reference}" if finding.reference else ""
            print(f"{finding.code}: {finding.message}{suffix}{ref}")
    else:
        print(f"methods: {len(registry.get('methods', []))} records; identity integrity ok")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
