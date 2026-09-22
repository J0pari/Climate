#!/usr/bin/env python3
"""Validate the worldwide no-fee station provider registry."""
from __future__ import annotations

import ast
import json
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "architecture" / "station_providers.json"
DATA_AUTHORITIES = ROOT / "architecture" / "data_authorities.json"


@dataclass(frozen=True)
class Finding:
    code: str
    path: str
    message: str


def _qualified_python_anchor_resolves(source: str, anchor: str) -> bool:
    parts = anchor.split(".")
    if len(parts) < 2:
        return False
    try:
        nodes = ast.parse(source).body
    except SyntaxError:
        return False
    for part in parts:
        match = next(
            (
                node
                for node in nodes
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == part
            ),
            None,
        )
        if match is None:
            return False
        nodes = match.body
    return True


def _boundary_resolves(candidate: Path, anchor: str) -> bool:
    if not candidate.is_file():
        return False
    source = candidate.read_text(encoding="utf-8")
    if candidate.suffix == ".py" and "." in anchor:
        return _qualified_python_anchor_resolves(source, anchor)
    return anchor in source


def check(root: Path = ROOT) -> list[Finding]:
    registry = json.loads((root / "architecture/station_providers.json").read_text())
    authorities = json.loads((root / "architecture/data_authorities.json").read_text())
    authority_by_id = {
        item.get("source_id"): item
        for item in authorities.get("sources", [])
        if isinstance(item, dict) and isinstance(item.get("source_id"), str)
    }
    source_ids = set(authority_by_id)
    findings: list[Finding] = []
    if registry.get("schema_version") != 1:
        findings.append(Finding("station_providers.schema", "architecture/station_providers.json", "schema_version must be 1"))
    target = registry.get("target_population")
    if not isinstance(target, str) or "without paid data access" not in target:
        findings.append(Finding("station_providers.target", "architecture/station_providers.json", "target population must retain the worldwide no-paid-access boundary"))
    providers = registry.get("providers")
    if not isinstance(providers, list) or not providers:
        return findings + [Finding("station_providers.empty", "architecture/station_providers.json", "providers must be non-empty")]
    seen = set()
    for index, provider in enumerate(providers):
        path = f"architecture/station_providers.json:providers[{index}]"
        provider_id = provider.get("provider_id")
        if not isinstance(provider_id, str) or not provider_id:
            findings.append(Finding("station_providers.id", path, "provider_id must be non-empty"))
        elif provider_id in seen:
            findings.append(Finding("station_providers.duplicate", path, f"duplicate provider_id {provider_id!r}"))
        else:
            seen.add(provider_id)
        if provider.get("source_id") not in source_ids:
            findings.append(Finding("station_providers.source", path, "source_id must resolve in data_authorities.json"))
        else:
            authority = authority_by_id[provider["source_id"]]
            for field in (
                "access",
                "license_or_access_constraints",
                "update_semantics",
            ):
                value = provider.get(field)
                if not isinstance(value, str) or not value.strip():
                    findings.append(Finding(
                        f"station_providers.{field}",
                        path,
                        f"{field} must be a non-empty string",
                    ))
                elif value != authority.get(field):
                    findings.append(Finding(
                        f"station_providers.{field}_drift",
                        path,
                        f"{field} must exactly match the resolved data authority",
                    ))
        revision_identity = provider.get("revision_identity")
        if not isinstance(revision_identity, str) or not revision_identity.strip():
            findings.append(Finding(
                "station_providers.revision_identity",
                path,
                "revision_identity must state how immutable provider revisions are bound",
            ))
        for field, message in (
            (
                "geographic_scope",
                "geographic_scope must state the provider population boundary",
            ),
            (
                "station_identity_namespace",
                "station_identity_namespace must name the provider-owned station identity",
            ),
            (
                "scale_path",
                "scale_path must state how complete provider artifacts enter bounded federation shards",
            ),
        ):
            value = provider.get(field)
            if not isinstance(value, str) or not value.strip():
                findings.append(Finding(
                    f"station_providers.{field}",
                    path,
                    message,
                ))
        if provider.get("access_cost") != "no_fee":
            findings.append(Finding("station_providers.cost", path, "station federation providers must require no paid data access"))
        if provider.get("status") not in {"current", "planned"}:
            findings.append(Finding("station_providers.status", path, "status must be current or planned"))
        if provider.get("discovery_mode") != "provider_catalog":
            findings.append(Finding("station_providers.discovery", path, "provider discovery must come from a provider catalog, not a hand-selected station list"))
        if "station_ids" in provider:
            findings.append(Finding("station_providers.fixed_station_list", path, "provider registry must not encode a fixed station list"))
        if provider.get("status") == "current":
            boundaries = provider.get("current_boundaries")
            if not isinstance(boundaries, list) or not boundaries:
                findings.append(Finding("station_providers.boundary", path, "current provider requires executable adapter boundaries"))
                continue
            for boundary in boundaries:
                if not isinstance(boundary, str) or "::" not in boundary:
                    findings.append(Finding("station_providers.boundary", path, "boundary must be path::anchor"))
                    continue
                rel, anchor = boundary.split("::", 1)
                candidate = root / rel
                if not _boundary_resolves(candidate, anchor):
                    findings.append(Finding("station_providers.boundary_missing", path, f"boundary does not resolve: {boundary}"))
    return findings


def main() -> int:
    findings = check()
    if findings:
        for finding in findings:
            print(f"{finding.code}: {finding.path}: {finding.message}")
        return 1
    print("station provider federation registry: integrity ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
