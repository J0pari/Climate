#!/usr/bin/env python3
"""Validate repository-state authority and freshness semantics."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from architecture.render_repository_state import render as render_repository_state
from architecture.state_authorities import (
    DEFAULT_MANIFEST,
    expand_local_authority_paths,
    load_manifest,
    manifest_fingerprint,
    projection_authority_paths,
    surface_by_id,
)

REQUIRED_SURFACES = {
    "contract",
    "planning",
    "structural_realization",
    "scientific_evaluations",
    "execution",
    "history",
}

REQUIRED_REFERENCES = {
    "AGENTS.md": (
        "architecture/state_authorities.json",
        "docs/REPOSITORY-STATE.md",
    ),
    "README.md": (
        "docs/REPOSITORY-STATE.md",
        "docs/generated/STATUS.md",
        "docs/ROADMAP.md",
    ),
    "docs/EXECUTION-TOPOLOGY.md": (
        "docs/REPOSITORY-STATE.md",
        "docs/generated/STATUS.md",
        "docs/ROADMAP.md",
    ),
    "docs/META-EXPERIMENTATION.md": (
        "docs/REPOSITORY-STATE.md",
        "docs/ROADMAP.md",
    ),
}


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
    if missing:
        findings.append(
            Finding(
                "repository_state.surface_missing",
                "architecture/state_authorities.json",
                "missing required state surfaces: " + ", ".join(missing),
            )
        )

    freshness = manifest.get("freshness_semantics")
    if not isinstance(freshness, dict):
        findings.append(
            Finding(
                "repository_state.freshness_missing",
                "architecture/state_authorities.json",
                "freshness_semantics must be an object",
            )
        )
    else:
        if freshness.get("commit_scoped") is not True:
            findings.append(
                Finding(
                    "repository_state.not_commit_scoped",
                    "architecture/state_authorities.json",
                    "repository state must be explicitly commit-scoped",
                )
            )
        if freshness.get("head_movement_invalidates_cached_state") is not True:
            findings.append(
                Finding(
                    "repository_state.cache_not_invalidated",
                    "architecture/state_authorities.json",
                    "main movement must invalidate cached present-state conclusions",
                )
            )

    for surface_id in ("planning", "structural_realization"):
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
                    f"{surface_id} requires a generated projection",
                )
            )
        elif not (root / projection).is_file():
            findings.append(
                Finding(
                    "repository_state.projection_file_missing",
                    projection,
                    f"declared {surface_id} projection does not exist",
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
                    f"declared {surface_id} renderer does not exist",
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

    state_doc = root / "docs" / "REPOSITORY-STATE.md"
    expected_state_doc = render_repository_state(
        manifest,
        fingerprint=manifest_fingerprint(manifest_path),
    )
    if not state_doc.is_file():
        findings.append(
            Finding(
                "repository_state.contract_projection_missing",
                "docs/REPOSITORY-STATE.md",
                "generated repository-state/freshness contract is missing",
            )
        )
    elif state_doc.read_text(encoding="utf-8") != expected_state_doc:
        findings.append(
            Finding(
                "repository_state.contract_projection_stale",
                "docs/REPOSITORY-STATE.md",
                "generated repository-state/freshness contract does not match its manifest",
            )
        )

    for relative, needles in REQUIRED_REFERENCES.items():
        path = root / relative
        if not path.is_file():
            findings.append(
                Finding(
                    "repository_state.binding_doc_missing",
                    relative,
                    "required binding/orientation document is missing",
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

    for relative in ("docs/ROADMAP.md", "docs/generated/STATUS.md"):
        path = root / relative
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        if "docs/REPOSITORY-STATE.md" not in text:
            findings.append(
                Finding(
                    "repository_state.projection_reconciliation_missing",
                    relative,
                    "generated projection must point to commit-scoped state reconciliation",
                )
            )
        if "fingerprint" not in text.lower():
            findings.append(
                Finding(
                    "repository_state.projection_fingerprint_missing",
                    relative,
                    "generated projection must expose its authority fingerprint",
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
    print("repository state authority/freshness integrity: clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
