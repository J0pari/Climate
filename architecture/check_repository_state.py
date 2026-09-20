#!/usr/bin/env python3
"""Validate the single generated repository-local research-state surface."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from architecture.render_state import render_state
from architecture.state_authorities import (
    expand_local_authority_paths,
    load_manifest,
    projection_authority_paths,
    surface_by_id,
)

REQUIRED_SURFACES = {"planning", "repository_state", "execution", "history"}
REQUIRED_REFERENCES = {
    "AGENTS.md": (
        "docs/generated/STATE.md",
        "architecture/planning_graph.json",
    ),
    "README.md": (
        "docs/generated/STATE.md",
        "docs/EXECUTION-RESOURCES.md",
    ),
    "docs/EXECUTION-RESOURCES.md": (
        "docs/generated/STATE.md",
        "docs/ROADMAP.md",
    ),
}

LEGACY_REFERENCES = (
    "docs/REPOSITORY-STATE.md",
    "docs/generated/STATUS.md",
    "docs/EXECUTION-TOPOLOGY.md",
    "architecture/render_repository_state.py",
    "architecture/render_status.py",
)

FORBIDDEN_LEGACY_PATHS = (
    "docs/REPOSITORY-STATE.md",
    "docs/generated/STATUS.md",
    "docs/EXECUTION-TOPOLOGY.md",
    "architecture/render_repository_state.py",
    "architecture/render_status.py",
)


@dataclass(frozen=True)
class Finding:
    code: str
    path: str
    message: str

    def render(self) -> str:
        return f"{self.code}: {self.path}: {self.message}"


def check(root: Path = ROOT) -> list[Finding]:
    findings: list[Finding] = []
    manifest_path = root / "architecture" / "state_authorities.json"
    try:
        manifest = load_manifest(manifest_path)
    except (OSError, ValueError) as exc:
        return [
            Finding(
                "repository_state.manifest_invalid",
                "architecture/state_authorities.json",
                str(exc),
            )
        ]

    surface_ids = {item["id"] for item in manifest["surfaces"]}
    missing = sorted(REQUIRED_SURFACES - surface_ids)
    extra = sorted(surface_ids - REQUIRED_SURFACES)
    if missing:
        findings.append(
            Finding(
                "repository_state.surface_missing",
                "architecture/state_authorities.json",
                "missing required surfaces: " + ", ".join(missing),
            )
        )
    if extra:
        findings.append(
            Finding(
                "repository_state.surface_parallel",
                "architecture/state_authorities.json",
                "unexpected parallel state surfaces: " + ", ".join(extra),
            )
        )

    for surface in manifest["surfaces"]:
        try:
            expand_local_authority_paths(root, surface)
        except ValueError as exc:
            findings.append(
                Finding(
                    "repository_state.surface_authority_invalid",
                    "architecture/state_authorities.json",
                    f"{surface['id']}: {exc}",
                )
            )

    for surface_id in ("planning", "repository_state"):
        if surface_id not in surface_ids:
            continue
        surface = surface_by_id(manifest, surface_id)
        projection = surface.get("projection")
        renderer = surface.get("renderer")
        if not isinstance(projection, str) or not projection:
            findings.append(
                Finding(
                    "repository_state.projection_missing",
                    "architecture/state_authorities.json",
                    f"{surface_id} requires a projection",
                )
            )
        elif not (root / projection).is_file():
            findings.append(
                Finding(
                    "repository_state.projection_file_missing",
                    projection,
                    f"{surface_id} projection is missing",
                )
            )
        if not isinstance(renderer, str) or not renderer:
            findings.append(
                Finding(
                    "repository_state.renderer_missing",
                    "architecture/state_authorities.json",
                    f"{surface_id} requires a renderer",
                )
            )
        elif not (root / renderer).is_file():
            findings.append(
                Finding(
                    "repository_state.renderer_file_missing",
                    renderer,
                    f"{surface_id} renderer is missing",
                )
            )
        try:
            projection_authority_paths(root, manifest, surface_id)
        except ValueError as exc:
            findings.append(
                Finding(
                    "repository_state.authority_inputs_invalid",
                    "architecture/state_authorities.json",
                    f"{surface_id}: {exc}",
                )
            )

    state_path = root / "docs" / "generated" / "STATE.md"
    if not state_path.is_file():
        findings.append(
            Finding(
                "repository_state.generated_state_missing",
                "docs/generated/STATE.md",
                "generated research state is missing",
            )
        )
    else:
        expected = render_state(root)
        if state_path.read_text(encoding="utf-8") != expected:
            findings.append(
                Finding(
                    "repository_state.generated_state_stale",
                    "docs/generated/STATE.md",
                    "generated research state does not match repository-local authorities",
                )
            )

    for relative, needles in REQUIRED_REFERENCES.items():
        path = root / relative
        if not path.is_file():
            findings.append(
                Finding(
                    "repository_state.orientation_doc_missing",
                    relative,
                    "required orientation document is missing",
                )
            )
            continue
        text = path.read_text(encoding="utf-8")
        for needle in needles:
            if needle not in text:
                findings.append(
                    Finding(
                        "repository_state.reference_missing",
                        relative,
                        f"must reference {needle}",
                    )
                )

    for relative in FORBIDDEN_LEGACY_PATHS:
        if (root / relative).exists():
            findings.append(
                Finding(
                    "repository_state.parallel_legacy_surface",
                    relative,
                    "obsolete parallel state/execution surface must be removed",
                )
            )

    reference_paths = [root / "README.md", root / "AGENTS.md", root / "architecture" / "planning_graph.json"]
    docs_root = root / "docs"
    if docs_root.is_dir():
        reference_paths.extend(
            path
            for path in docs_root.rglob("*.md")
            if "archive" not in path.relative_to(docs_root).parts
        )
    workflows = root / ".github" / "workflows"
    if workflows.is_dir():
        reference_paths.extend(workflows.glob("*.yml"))
        reference_paths.extend(workflows.glob("*.yaml"))

    for path in sorted(set(reference_paths)):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8", errors="replace")
        for legacy in LEGACY_REFERENCES:
            if legacy in text:
                findings.append(
                    Finding(
                        "repository_state.legacy_reference",
                        relative,
                        f"references obsolete path {legacy}",
                    )
                )

    return findings


def main() -> int:
    findings = check()
    for finding in findings:
        print(finding.render())
    if findings:
        print(f"\n{len(findings)} repository-state finding(s)", file=sys.stderr)
        return 1
    print("repository state authority integrity: clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
