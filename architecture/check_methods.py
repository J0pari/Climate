#!/usr/bin/env python3
"""Integrity checks for executable method identities.

Method descriptors declare stable semantics and the source files that define
their local implementation. Actual build identity is derived at run time from
those sources and the resolved environment; hand-maintained commit-looking
build strings are deliberately not authority.
"""
from __future__ import annotations

import argparse
import ast
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


def _module_file(root: Path, module: str) -> Path | None:
    if not module or module.startswith("."):
        return None
    relative = Path(*module.split("."))
    for candidate in (
        root / relative.with_suffix(".py"),
        root / relative / "__init__.py",
    ):
        if candidate.is_file():
            return candidate.resolve()
    return None


def _local_python_imports(root: Path, source: Path) -> set[Path]:
    try:
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    except (OSError, SyntaxError):
        return set()

    imports: set[Path] = set()
    for node in ast.walk(tree):
        module: str | None = None
        if isinstance(node, ast.Import):
            for alias in node.names:
                candidate = _module_file(root, alias.name)
                if candidate is not None:
                    imports.add(candidate)
            continue
        if isinstance(node, ast.ImportFrom):
            if node.level != 0:
                # Runnable repository methods currently use absolute local imports.
                # Relative imports are left to Python packaging rather than guessed.
                continue
            module = node.module
        if module:
            candidate = _module_file(root, module)
            if candidate is not None:
                imports.add(candidate)
    return imports


def _python_dependency_closure(root: Path, sources: list[str]) -> set[str]:
    root = root.resolve()
    queue = [
        (root / source).resolve()
        for source in sources
        if isinstance(source, str) and source.endswith(".py")
    ]
    seen: set[Path] = set()
    dependencies: set[str] = set()
    while queue:
        source = queue.pop()
        if source in seen or not source.is_file():
            continue
        seen.add(source)
        for dependency in _local_python_imports(root, source):
            try:
                relative = dependency.relative_to(root).as_posix()
            except ValueError:
                continue
            dependencies.add(relative)
            if dependency not in seen:
                queue.append(dependency)
    return dependencies


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

        declared_sources = {
            source for source in sources if isinstance(source, str) and source
        }
        for dependency in sorted(
            _python_dependency_closure(root, list(declared_sources))
            - declared_sources
        ):
            findings.append(Finding(
                "methods.build_source_dependency_missing",
                "source-bound build identity omits a transitive local Python dependency",
                method_id,
                dependency,
            ))

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
