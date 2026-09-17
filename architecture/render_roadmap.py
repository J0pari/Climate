#!/usr/bin/env python3
"""Render a concise human-readable view of the sole-owner planning graph."""
from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
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


def render(graph: dict[str, Any]) -> str:
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

    lines = [
        "# Climate obligation roadmap",
        "",
        "> Generated from `architecture/planning_graph.json`. Do not hand-edit this file.",
        "> The graph is the sole authority for planned work, priority, dependencies, blockers, and completion criteria.",
        "",
        "Objective realized state is owned by the module, claim, experiment, hazard, and realization authorities and is rendered separately in `docs/generated/STATUS.md`.",
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

    expected = render(load_graph(args.graph))
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
            print("roadmap: generated projection is current")
            return 0
        diff = difflib.unified_diff(
            observed.splitlines(), expected.splitlines(),
            fromfile=str(args.output), tofile="generated planning graph",
            lineterm="",
        )
        print("\n".join(diff))
        return 1

    print(expected, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
