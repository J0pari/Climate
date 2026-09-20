#!/usr/bin/env python3
"""Render the repository-state/freshness contract from its machine authority."""
from __future__ import annotations

import argparse
import difflib
from pathlib import Path

from architecture.state_authorities import (
    DEFAULT_MANIFEST,
    ROOT,
    load_manifest,
    manifest_fingerprint,
)

DEFAULT_OUTPUT = ROOT / "docs" / "REPOSITORY-STATE.md"


def render(manifest: dict, *, fingerprint: str) -> str:
    lines = [
        "# Repository state and freshness contract",
        "",
        "> Generated from `architecture/state_authorities.json`. Do not hand-edit this file.",
        f"> State-authority manifest fingerprint: `{fingerprint}`.",
        "",
        "Repository state is **commit-scoped**. A statement about what is planned, realized, evaluated, evidenced, or passing is valid only for the exact resolved `main` commit on which its authorities were read. If `main` moves, cached present-state conclusions are stale until the reorientation sequence below is repeated.",
        "",
        "A generated projection being fresh means only that its checked-in bytes match its **declared authority inputs**. It does not mean that the projection is a complete description of repository state, and it does not establish exact-head CI success.",
        "",
        "## State surfaces",
        "",
        "| Surface | Kind | Authorities | Projection | Scope | Explicit exclusions |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for surface in manifest["surfaces"]:
        authorities = "<br>".join(f"`{item}`" for item in surface["authority_paths"])
        projection = (
            f"`{surface['projection']}`"
            if surface.get("projection")
            else "—"
        )
        excludes = "; ".join(surface.get("excludes", [])) or "—"
        lines.append(
            f"| `{surface['id']}` | `{surface['kind']}` | {authorities} | "
            f"{projection} | {surface['scope']} | {excludes} |"
        )

    lines.extend([
        "",
        "## Mandatory reorientation sequence",
        "",
        "Perform this sequence before making a present-state assertion, declaring an obligation done/ready/blocked, selecting the next repository action, or resuming substantial work from an earlier conversational state:",
        "",
    ])
    for index, step in enumerate(manifest["reorientation_sequence"], 1):
        lines.append(f"{index}. {step}")

    lines.extend([
        "",
        "## Generated-projection semantics",
        "",
        "`docs/ROADMAP.md` is a projection of the planning authority only. Its freshness says nothing about implementation, evaluations, evidence promotion, or CI.",
        "",
        "`docs/generated/STATUS.md` is a projection of the structural-realization authority set declared in the state-authority manifest. It is intentionally incomplete with respect to planning, committed evaluation records that have not been promoted through evidence authorities, exact-head GitHub Actions, commit history, and unregistered implementation facts.",
        "",
        "Therefore, neither projection may be cited alone as proof of the complete current repository state. Present-state claims require reconciliation across the relevant surfaces above at one exact commit.",
        "",
        "## Verification commands",
        "",
        "```text",
        "python architecture/check_repository_state.py",
        "python architecture/render_repository_state.py --check",
        "python architecture/render_roadmap.py --check",
        "python architecture/render_status.py --check",
        "```",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    expected = render(manifest, fingerprint=manifest_fingerprint(args.manifest))

    if args.write:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(expected, encoding="utf-8")
        print(f"wrote {args.output.relative_to(ROOT)}")
        return 0

    if not args.output.exists():
        print(f"repository-state projection missing: {args.output.relative_to(ROOT)}")
        return 1
    observed = args.output.read_text(encoding="utf-8")
    if observed == expected:
        print("repository-state projection matches declared state-authority manifest")
        return 0
    diff = difflib.unified_diff(
        observed.splitlines(),
        expected.splitlines(),
        fromfile=str(args.output.relative_to(ROOT)),
        tofile="rendered repository-state contract",
        lineterm="",
    )
    print("\n".join(diff))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
