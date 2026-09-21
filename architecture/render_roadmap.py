#!/usr/bin/env python3
"""Render the sole-authority planning graph as Markdown."""
from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from architecture.state_authorities import (
    fingerprint_paths,
    load_manifest,
    projection_authority_paths,
    projection_fingerprint,
)

DEFAULT_GRAPH = ROOT / "architecture" / "planning_graph.json"
DEFAULT_OUTPUT = ROOT / "docs" / "ROADMAP.md"

STATUS_ORDER = {"active": 0, "ready": 1, "blocked": 2, "done": 3, "dropped": 4}
PRIORITY_ORDER = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}


def load_graph(path: Path = DEFAULT_GRAPH) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("planning graph must contain a JSON object")
    return data


def render(
    graph: dict[str, Any],
    *,
    authority_fingerprint: str = "unbound-in-memory",
    authority_paths: list[str] | None = None,
) -> str:
    nodes = list(graph["nodes"])
    nodes.sort(
        key=lambda node: (
            STATUS_ORDER[node["status"]],
            PRIORITY_ORDER[node["priority"]],
            node["id"],
        )
    )
    counts = {status: 0 for status in STATUS_ORDER}
    for node in nodes:
        counts[node["status"]] += 1

    source_text = ", ".join(
        f"`{path}`" for path in (authority_paths or ["in-memory graph"])
    )
    lines = [
        "# Climate obligation roadmap",
        "",
        "> Generated planning projection. Do not hand-edit.",
        f"> Declared planning authority: {source_text}.",
        f"> Planning-authority fingerprint: `{authority_fingerprint}`.",
        "",
        "Freshness means this projection matches its declared planning authority inputs in the checkout being inspected. Repository state is commit-scoped: movement of `main` invalidates cached conclusions until the current commit is re-oriented and the relevant verification is explicitly rerun.",
        "",
        "Repository-local research state is summarized in `docs/generated/STATE.md`.",
        "",
        "External-integration obligations bind native external capabilities while keeping Climate-specific scientific semantics local; completion cannot be satisfied by a local shadow implementation of the upstream capability.",
        "",
        "## Planning summary",
        "",
        f"- Active: {counts['active']}",
        f"- Ready: {counts['ready']}",
        f"- Blocked: {counts['blocked']}",
        f"- Done: {counts['done']}",
        f"- Dropped: {counts['dropped']}",
        "",
        "## Graph projection",
        "",
        "| Obligation | Status | Priority | Resource | Dependencies |",
        "| --- | --- | --- | --- | --- |",
    ]
    for node in nodes:
        deps = node["depends_on"]
        dependency_text = ", ".join(f"`{dep}`" for dep in deps) if deps else "—"
        lines.append(
            f"| `{node['id']}` — {node['title']} | `{node['status']}` | "
            f"`{node['priority']}` | `{node['resource_class']}` | {dependency_text} |"
        )
    lines.extend(
        [
            "",
            "Summaries, blockers, completion criteria, and evidence paths remain in `architecture/planning_graph.json`.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()

    manifest_path = ROOT / "architecture" / "state_authorities.json"
    manifest = load_manifest(manifest_path)
    declared_paths = projection_authority_paths(ROOT, manifest, "planning")
    canonical_graph = declared_paths[0]
    if len(declared_paths) != 1:
        raise ValueError("planning projection must have exactly one declared authority file")

    graph_path = args.graph.resolve()
    if graph_path == canonical_graph.resolve():
        fingerprint = projection_fingerprint(ROOT, manifest, "planning")
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
        args.output.parent.mkdir(parents=True, exist_ok=True)
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
            print("roadmap: generated projection matches planning authority")
            return 0
        print(
            "\n".join(
                difflib.unified_diff(
                    observed.splitlines(),
                    expected.splitlines(),
                    fromfile=str(args.output),
                    tofile="generated planning projection",
                    lineterm="",
                )
            )
        )
        return 1
    print(expected, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
