#!/usr/bin/env python3
"""Render one repository-local research-state view from machine authorities."""
from __future__ import annotations

import argparse
import difflib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from architecture.state_authorities import (
    load_manifest,
    projection_authority_paths,
)

DEFAULT_OUTPUT = ROOT / "docs" / "generated" / "STATE.md"


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _find(paths: list[Path], relative: str) -> Path:
    matches = [p for p in paths if p.relative_to(ROOT).as_posix() == relative]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one state authority {relative}")
    return matches[0]


def _matching(paths: list[Path], prefix: str, suffix: str = "") -> list[Path]:
    return sorted(
        p for p in paths
        if p.relative_to(ROOT).as_posix().startswith(prefix)
        and p.relative_to(ROOT).as_posix().endswith(suffix)
    )


def _evaluation_label(payload: dict[str, Any]) -> str:
    for key in (
        "evaluation_id",
        "experiment_id",
        "fixture_id",
        "campaign_id",
        "planning_node",
        "classification",
    ):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return "JSON record"


def render_state(root: Path = ROOT) -> str:
    manifest = load_manifest()
    authorities = projection_authority_paths(root, manifest, "repository_state")
    planning = _load(_find(authorities, "architecture/planning_graph.json"))
    claims = _load(_find(authorities, "claims/registry.json"))
    evidence = _load(_find(authorities, "evidence/registry.json"))
    methods = _load(_find(authorities, "methods/registry.json"))
    sheaf = _load(_find(authorities, "methods/sheaf-realization.v1.json"))
    data_authorities = _load(_find(authorities, "architecture/data_authorities.json"))
    stations = _load(_find(authorities, "architecture/station_providers.json"))
    hazards = _load(_find(authorities, "architecture/semantic_hazards.json"))
    commons = _load(_find(authorities, "architecture/commons_interface.json"))

    module_paths = _matching(authorities, "architecture/modules/", ".json")
    experiment_paths = _matching(authorities, "experiments/", ".json")
    evaluation_paths = _matching(authorities, "evaluations/")
    configuration_paths = _matching(authorities, "configurations/", ".json")

    modules: list[tuple[str, dict[str, Any]]] = []
    for path in module_paths:
        lifecycle = path.stem
        records = _load(path).get("modules")
        if not isinstance(records, list):
            raise ValueError(f"module registry missing modules array: {path}")
        for record in records:
            if not isinstance(record, dict):
                raise ValueError(f"module record must be object: {path}")
            modules.append((lifecycle, record))

    experiments = [path.relative_to(root).as_posix() for path in experiment_paths]
    evaluations = [
        path.relative_to(root).as_posix()
        for path in evaluation_paths
        if path.suffix == ".json"
    ]

    nodes = planning.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("planning graph missing nodes array")
    planning_counts = Counter(str(node["status"]) for node in nodes)
    frontier = sorted(
        (
            node for node in nodes
            if node["status"] in {"active", "ready", "blocked"}
        ),
        key=lambda node: (
            {"active": 0, "ready": 1, "blocked": 2}[node["status"]],
            {"P0": 0, "P1": 1, "P2": 2, "P3": 3}.get(node["priority"], 99),
            node["id"],
        ),
    )

    claim_records = claims.get("claims")
    evidence_records = evidence.get("evidence")
    method_records = methods.get("methods")
    obligations = sheaf.get("obligations")
    if not isinstance(claim_records, list):
        raise ValueError("claim registry missing claims array")
    if not isinstance(evidence_records, list):
        raise ValueError("evidence registry missing evidence array")
    if not isinstance(method_records, list):
        raise ValueError("method registry missing methods array")
    if not isinstance(obligations, list):
        raise ValueError("sheaf realization authority missing obligations array")

    claim_counts = Counter(str(item["maturity"]) for item in claim_records)
    module_counts = Counter(lifecycle for lifecycle, _ in modules)
    method_counts = Counter(str(item["maturity"]) for item in method_records)
    realization_counts = Counter(str(item["status"]) for item in obligations)
    hazard_records = hazards.get("hazards")
    source_records = data_authorities.get("sources")
    usage_records = data_authorities.get("usages")
    provider_records = stations.get("providers")
    if not isinstance(hazard_records, list):
        raise ValueError("semantic hazard authority missing hazards array")
    if not isinstance(source_records, list) or not isinstance(usage_records, list):
        raise ValueError("data authority registry is malformed")
    if not isinstance(provider_records, list):
        raise ValueError("station provider registry is malformed")

    lines = [
        "# Generated research state",
        "",
        "<!-- Generated by architecture/render_state.py. Do not hand-edit. -->",
        "",
        "This is the repository-local orientation view. Exact-head build, test, and workflow outcomes live in GitHub Actions for the commit being inspected.",
        "",
        "## Planning frontier",
        "",
        f"Active **{planning_counts['active']}** · Ready **{planning_counts['ready']}** · "
        f"Blocked **{planning_counts['blocked']}** · Done **{planning_counts['done']}** · "
        f"Dropped **{planning_counts['dropped']}**",
        "",
        "| Obligation | Status | Priority | Resource |",
        "| --- | --- | --- | --- |",
    ]
    for node in frontier:
        lines.append(
            f"| `{node['id']}` — {node['title']} | `{node['status']}` | "
            f"`{node['priority']}` | `{node['resource_class']}` |"
        )

    lines.extend([
        "",
        "## Registered implementation and experiment surface",
        "",
        f"Modules **{len(modules)}** · Methods **{len(method_records)}** · "
        f"ExperimentSpecs **{len(experiments)}** · Configurations **{len(configuration_paths)}**",
        "",
        "| Module lifecycle | Count |",
        "| --- | ---: |",
    ])
    for key, value in sorted(module_counts.items()):
        lines.append(f"| `{key}` | {value} |")

    lines.extend([
        "",
        "| Method maturity | Count |",
        "| --- | ---: |",
    ])
    for key, value in sorted(method_counts.items()):
        lines.append(f"| `{key}` | {value} |")

    lines.extend([
        "",
        "Registered experiments:",
        "",
    ])
    for relative in experiments:
        lines.append(f"- `{relative}`")

    lines.extend([
        "",
        "## Claims and evidence",
        "",
        f"Claims **{len(claim_records)}** · Evidence records **{len(evidence_records)}**",
        "",
        "| Claim maturity | Count |",
        "| --- | ---: |",
    ])
    for key, value in sorted(claim_counts.items()):
        lines.append(f"| `{key}` | {value} |")

    lines.extend([
        "",
        "| Claim | Maturity | Supporting evidence records |",
        "| --- | --- | ---: |",
    ])
    for claim in sorted(claim_records, key=lambda item: item["claim_id"]):
        supporting = claim.get("supporting_evidence")
        supporting_count = len(supporting) if isinstance(supporting, list) else 0
        lines.append(
            f"| `{claim['claim_id']}` | `{claim['maturity']}` | {supporting_count} |"
        )

    lines.extend([
        "",
        "## Committed evaluations",
        "",
        f"JSON evaluation records: **{len(evaluations)}**.",
        "",
    ])
    for relative in evaluations:
        lines.append(f"- `{relative}`")

    lines.extend([
        "",
        "## Supporting authority health",
        "",
        f"Semantic hazards **{len(hazard_records)}** · Data sources **{len(source_records)}** · "
        f"Data usages **{len(usage_records)}** · Station providers **{len(provider_records)}**",
        "",
        f"Commons interface: `{commons['interface_id']}` with control level "
        f"`{commons['supported_control_level']}`.",
        "",
        "| Sheaf realization status | Count |",
        "| --- | ---: |",
    ])
    for key, value in sorted(realization_counts.items()):
        lines.append(f"| `{key}` | {value} |")

    lines.extend([
        "",
        "## Authority",
        "",
        "Planning details and completion criteria remain in `architecture/planning_graph.json`. "
        "Claim promotion remains in `claims/registry.json` and `evidence/registry.json`. "
        "This generated view does not promote evidence or infer CI outcomes.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rendered = render_state(ROOT)
    if args.write:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"wrote {args.output.relative_to(ROOT)}")
        return 0

    if not args.output.exists():
        print(f"generated research state missing: {args.output.relative_to(ROOT)}")
        return 1
    observed = args.output.read_text(encoding="utf-8")
    if observed == rendered:
        print("generated research state matches repository-local authorities")
        return 0

    diff = difflib.unified_diff(
        observed.splitlines(),
        rendered.splitlines(),
        fromfile=str(args.output.relative_to(ROOT)),
        tofile="rendered research state",
        lineterm="",
    )
    print("\n".join(diff))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
