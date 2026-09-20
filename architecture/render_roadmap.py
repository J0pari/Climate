#!/usr/bin/env python3
"""Render a concise human-readable view of the sole-owner planning graph."""
from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from architecture.state_authorities import (
    ROOT,
    fingerprint_paths,
    load_manifest,
    projection_authority_paths,
)

DEFAULT_GRAPH = ROOT / "architecture" / "planning_graph.json"
DEFAULT_OUTPUT = ROOT / "docs" / "ROADMAP.md"

STATUS_ORDER = {"active": 0, "ready": 1, "blocked": 2, "done": 3, "dropped": 4}
PRIORITY_ORDER = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}


def load_graph(path: Path = DEFAULT_GRAPH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("planning graph must contain a JSON object")
    return data


def render(
    graph: dict[str, Any],
    *,
    authority_fingerprint: str = "unbound-in-memory",
    authority_paths: list[str] | None = None,
) -> str:
    nodes = list(graph.get("nodes", []))
    nodes.sort(key=lambda node: (
        STATUS_ORDER.get(node.get("status"), 99),
        PRIORITY_ORDER.get(node.get("priority"), 99),
        node.get("id", ""),
    ))

    counts: dict[str, int] = {}
    for node in nodes:
        status = str(node.get("status"))
        counts[status] = counts.get(status, 0) + 1

    source_text = ", ".join(f"`{path}`" for path in (authority_paths or ["in-memory graph"]))
    lines = [
        "# Climate obligation roadmap",
        "",
        "> Generated planning projection. Do not hand-edit this file.",
        f"> Declared planning authority: {source_text}.",
        f"> Planning-authority fingerprint: `{authority_fingerprint}`.",
        "",
        "Freshness of this file means only that it matches the declared planning authority inputs above. It does **not** establish implementation/realization state, scientific evaluation outcomes, claim-evidence promotion, exact-head CI success, or commit-history state.",
        "",
        "The planning graph is the sole authority for planned work, priority, dependencies, blockers, resource class, completion criteria, and planning evidence paths.",
        "",
        "Repository-local research state, including planning, structural registration, claims/evidence, and committed evaluations, is generated in `docs/generated/STATE.md`.",
        "",
        "External implementation ownership does not remove integration correctness from scope: planning nodes should bind native external capabilities and add Climate-specific scientific semantics rather than create shadow cataloging, preprocessing, execution, training/inference, intercomparison, or provenance stacks.",
        "",
        "## Planning summary",
        "",
        f"- Active: {counts.get('active', 0)}",
        f"- Ready: {counts.get('ready', 0)}",
        f"- Blocked: {counts.get('blocked', 0)}",
        f"- Done: {counts.get('done', 0)}",
        f"- Dropped: {counts.get('dropped', 0)}",
        "",
        "## Graph projection",
        "",
        "| Obligation | Status | Priority | Resource | Dependencies |",
        "| --- | --- | --- | --- | --- |",
    ]

    for node in nodes:
        deps = node.get("depends_on", [])
        dependency_text = ", ".join(f"`{dep}`" for dep in deps) if deps else "—"
        lines.append(
            f"| `{node['id']}` — {node['title']} | `{node['status']}` | "
            f"`{node['priority']}` | `{node['resource_class']}` | {dependency_text} |"
        )

    lines.extend([
        "",
        "Node summaries, blockers, completion criteria, and evidence paths live only in `architecture/planning_graph.json` so this projection cannot become a second planning surface.",
        "External-integration obligations name both the native capability that remains externally owned and the Climate-specific semantic/evidence responsibility that remains in scope; completion must not be satisfied by a local shadow implementation with the same advertised identity.",
    ])
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="render Climate planning graph as Markdown")
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest()
    declared_paths = projection_authority_paths(ROOT, manifest, "planning")
    canonical_graph = declared_paths[0]
    if len(declared_paths) != 1:
        raise ValueError("planning projection must have exactly one declared authority file")

    graph_path = args.graph.resolve()
    if graph_path == canonical_graph.resolve():
        fingerprint = fingerprint_paths(ROOT, declared_paths)
        rendered_paths = [path.relative_to(ROOT).as_posix() for path in declared_paths]
    else:
        fingerprint = fingerprint_paths(ROOT, [graph_path])
        rendered_paths = [graph_path.relative_to(ROOT).as_posix()]

    expected = render(
        load_graph(args.graph),
        authority_fingerprint=fingerprint,
        authority_paths=rendered_paths,
    )
    if args.write:
        args.output.write_text(expected, encoding="utf-8")
        print(f"wrote {args.output}")
        return 0

    if args.check:
        try:
            observed = args.output.read_text(encoding="utf-8")
        except OSError as error:
            print(f"roadmap.read_error: {error}")
            return 1
        if observed == expected:
            print("roadmap: projection matches declared planning authority inputs")
            return 0
        diff = difflib.unified_diff(
            observed.splitlines(), expected.splitlines(),
            fromfile=str(args.output), tofile="generated planning projection",
            lineterm="",
        )
        print("\n".join(diff))
        return 1

    print(expected, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
