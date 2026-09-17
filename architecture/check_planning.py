#!/usr/bin/env python3
"""Validate Climate's sole-owner planning obligation graph.

The planning graph owns planned work, priority, dependency, blocker, and
completion concerns. Other registries own realized repository facts; durable
documentation owns current contracts and rationale. Keeping those authorities
separate prevents stale planning prose from becoming a second roadmap.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRAPH = ROOT / "architecture" / "planning_graph.json"

ID_RE = re.compile(r"^[a-z][a-z0-9_.-]*$")
STATUSES = {"ready", "active", "blocked", "done", "dropped"}
PRIORITIES = {"P0", "P1", "P2", "P3"}
RESOURCE_CLASSES = {
    "R0_static",
    "R1_portable_cpu",
    "R2_toolchain_ci",
    "R3_cuda_device",
    "R4_integrated_system",
    "R5_large_data",
}


@dataclass(frozen=True)
class Finding:
    code: str
    message: str
    node_id: str | None = None
    reference: str | None = None


def load_graph(path: Path = DEFAULT_GRAPH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("planning graph must contain a JSON object")
    return data


def _nonempty_strings(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(
        isinstance(item, str) and item.strip() for item in value
    )


def check(root: Path, graph: dict[str, Any]) -> list[Finding]:
    findings: list[Finding] = []
    if graph.get("schema_version") != 1:
        findings.append(Finding("planning.schema_version", "schema_version must be 1"))

    authority = graph.get("authority")
    if not isinstance(authority, str) or not authority.strip():
        findings.append(Finding("planning.authority_missing", "authority must be a non-empty string"))

    nodes = graph.get("nodes")
    if not isinstance(nodes, list):
        return findings + [Finding("planning.nodes_not_list", "nodes must be a list")]

    indexed: dict[str, dict[str, Any]] = {}
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            findings.append(Finding("planning.node_not_object", f"node {index} is not an object"))
            continue
        node_id = node.get("id")
        if not isinstance(node_id, str) or not ID_RE.fullmatch(node_id):
            findings.append(Finding("planning.id_invalid", "node id is missing or invalid", reference=str(node_id)))
            continue
        if node_id in indexed:
            findings.append(Finding("planning.id_duplicate", "node id appears more than once", node_id=node_id))
            continue
        indexed[node_id] = node

        for field in ("title", "summary"):
            value = node.get(field)
            if not isinstance(value, str) or not value.strip():
                findings.append(Finding(f"planning.{field}_missing", f"{field} must be non-empty", node_id=node_id))

        if node.get("status") not in STATUSES:
            findings.append(Finding("planning.status_invalid", f"unknown status {node.get('status')!r}", node_id=node_id))
        if node.get("priority") not in PRIORITIES:
            findings.append(Finding("planning.priority_invalid", f"unknown priority {node.get('priority')!r}", node_id=node_id))
        if node.get("resource_class") not in RESOURCE_CLASSES:
            findings.append(Finding("planning.resource_invalid", f"unknown resource class {node.get('resource_class')!r}", node_id=node_id))

        deps = node.get("depends_on")
        if not isinstance(deps, list) or not all(isinstance(dep, str) for dep in deps):
            findings.append(Finding("planning.dependencies_invalid", "depends_on must be a list of node ids", node_id=node_id))
        elif len(deps) != len(set(deps)):
            findings.append(Finding("planning.dependencies_duplicate", "depends_on contains duplicates", node_id=node_id))

        if not _nonempty_strings(node.get("completion")):
            findings.append(Finding("planning.completion_missing", "completion must contain explicit criteria", node_id=node_id))

        blockers = node.get("blockers")
        if blockers is not None and not _nonempty_strings(blockers):
            findings.append(Finding("planning.blockers_invalid", "blockers must be a non-empty string list when present", node_id=node_id))

        evidence = node.get("evidence")
        if evidence is not None:
            if not _nonempty_strings(evidence):
                findings.append(Finding("planning.evidence_invalid", "evidence must be a non-empty path list when present", node_id=node_id))
            else:
                for path in evidence:
                    if not (root / path).exists():
                        findings.append(Finding("planning.evidence_missing", "evidence path does not exist", node_id=node_id, reference=path))
        if node.get("status") == "done" and not _nonempty_strings(evidence):
            findings.append(Finding("planning.done_without_evidence", "done nodes require evidence paths", node_id=node_id))

    for node_id, node in indexed.items():
        deps = node.get("depends_on", [])
        if not isinstance(deps, list):
            continue
        for dep in deps:
            if dep == node_id:
                findings.append(Finding("planning.self_dependency", "node depends on itself", node_id=node_id))
            elif dep not in indexed:
                findings.append(Finding("planning.dependency_missing", "dependency does not resolve", node_id=node_id, reference=dep))

    for node_id, node in indexed.items():
        deps = node.get("depends_on", [])
        if not isinstance(deps, list):
            continue
        unresolved = [
            dep for dep in deps
            if dep in indexed and indexed[dep].get("status") != "done"
        ]
        status = node.get("status")
        blockers = node.get("blockers")
        has_external_blocker = _nonempty_strings(blockers)

        if status in {"ready", "active", "done"} and unresolved:
            findings.append(Finding(
                "planning.status_ignores_dependencies",
                f"status {status!r} requires all dependencies to be done",
                node_id=node_id,
                reference=", ".join(unresolved),
            ))
        if status == "blocked" and not unresolved and not has_external_blocker:
            findings.append(Finding(
                "planning.blocked_without_cause",
                "blocked node requires an unresolved dependency or explicit external blocker",
                node_id=node_id,
            ))

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str, trail: tuple[str, ...]) -> None:
        if node_id in visited:
            return
        if node_id in visiting:
            findings.append(Finding(
                "planning.dependency_cycle",
                "dependency graph contains a cycle",
                node_id=node_id,
                reference=" -> ".join((*trail, node_id)),
            ))
            return
        visiting.add(node_id)
        deps = indexed[node_id].get("depends_on", [])
        if isinstance(deps, list):
            for dep in deps:
                if dep in indexed:
                    visit(dep, (*trail, node_id))
        visiting.remove(node_id)
        visited.add(node_id)

    for node_id in indexed:
        visit(node_id, ())

    return sorted(findings, key=lambda item: (item.code, item.node_id or "", item.reference or ""))


def main() -> int:
    parser = argparse.ArgumentParser(description="check Climate planning graph integrity")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    try:
        graph = load_graph(args.graph)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        if args.json:
            print(json.dumps({"ok": False, "load_error": str(error)}, indent=2, sort_keys=True))
        else:
            print(f"planning.graph_load_error: {error}")
        return 1

    findings = check(args.root.resolve(), graph)
    if args.json:
        print(json.dumps({
            "ok": not findings,
            "node_count": len(graph.get("nodes", [])),
            "finding_count": len(findings),
            "findings": [asdict(item) for item in findings],
        }, indent=2, sort_keys=True))
    elif findings:
        for finding in findings:
            node = f" [{finding.node_id}]" if finding.node_id else ""
            ref = f" -> {finding.reference}" if finding.reference else ""
            print(f"{finding.code}: {finding.message}{node}{ref}")
    else:
        print(f"planning: {len(graph.get('nodes', []))} nodes; integrity ok")

    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
