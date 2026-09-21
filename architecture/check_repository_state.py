#!/usr/bin/env python3
"""Validate repository-state authority, projection freshness, and local links."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from architecture import check_markdown_links, render_roadmap
from architecture.render_state import render_state
from architecture.state_authorities import (
    expand_local_authority_paths,
    load_manifest,
    orientation_fingerprint,
    projection_authority_paths,
    projection_fingerprint,
    surface_by_id,
)

REQUIRED_SURFACES = {
    "planning",
    "structural_realization",
    "experiment_definitions",
    "evaluation_records",
    "claim_evidence",
    "history",
}
REQUIRED_ORIENTATION_INCLUDES = {
    "planning",
    "structural_realization",
    "experiment_definitions",
    "evaluation_records",
    "claim_evidence",
}
REQUIRED_ORIENTATION_EXCLUDES = {"history"}
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

FORBIDDEN_LEGACY_PATHS = LEGACY_REFERENCES


@dataclass(frozen=True)
class Finding:
    code: str
    path: str
    message: str

    def render(self) -> str:
        return f"{self.code}: {self.path}: {self.message}"


def _check_projection_file(
    findings: list[Finding],
    *,
    root: Path,
    projection: str,
    expected: str,
    fingerprint: str,
    stale_code: str,
) -> None:
    path = root / projection
    if not path.is_file():
        findings.append(
            Finding(
                "repository_state.projection_file_missing",
                projection,
                "declared generated projection is missing",
            )
        )
        return
    observed = path.read_text(encoding="utf-8")
    if fingerprint not in observed:
        findings.append(
            Finding(
                "repository_state.projection_fingerprint_missing",
                projection,
                "generated projection must expose its current authority fingerprint",
            )
        )
    if observed != expected:
        findings.append(
            Finding(
                stale_code,
                projection,
                "generated projection does not match its declared authority inputs",
            )
        )


def check(root: Path = ROOT) -> list[Finding]:
    findings: list[Finding] = []
    forbidden_hosted_automation = root / ".github" / "workflows"
    if forbidden_hosted_automation.exists():
        findings.append(
            Finding(
                "repository_state.hosted_ci_forbidden",
                ".github/workflows",
                "hosted CI configuration is prohibited; run verification locally or in an explicitly declared external environment",
            )
        )

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

    freshness = manifest.get("freshness_semantics")
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
                "movement of main must invalidate cached present-state conclusions",
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

    planning = surface_by_id(manifest, "planning")
    if planning.get("projection") != "docs/ROADMAP.md":
        findings.append(
            Finding(
                "repository_state.planning_projection_invalid",
                "architecture/state_authorities.json",
                "planning projection must remain docs/ROADMAP.md",
            )
        )
    if planning.get("renderer") != "architecture/render_roadmap.py":
        findings.append(
            Finding(
                "repository_state.planning_renderer_invalid",
                "architecture/state_authorities.json",
                "planning renderer must remain architecture/render_roadmap.py",
            )
        )

    orientation = manifest["orientation_view"]
    if set(orientation["includes"]) != REQUIRED_ORIENTATION_INCLUDES:
        findings.append(
            Finding(
                "repository_state.orientation_includes_invalid",
                "architecture/state_authorities.json",
                "orientation view must compose the five typed repository-local research surfaces",
            )
        )
    if set(orientation["excludes"]) != REQUIRED_ORIENTATION_EXCLUDES:
        findings.append(
            Finding(
                "repository_state.orientation_excludes_invalid",
                "architecture/state_authorities.json",
                "orientation view must exclude git history",
            )
        )
    if orientation["projection"] != "docs/generated/STATE.md":
        findings.append(
            Finding(
                "repository_state.orientation_projection_invalid",
                "architecture/state_authorities.json",
                "orientation projection must remain docs/generated/STATE.md",
            )
        )
    if orientation["renderer"] != "architecture/render_state.py":
        findings.append(
            Finding(
                "repository_state.orientation_renderer_invalid",
                "architecture/state_authorities.json",
                "orientation renderer must remain architecture/render_state.py",
            )
        )

    declared_surface_projections = [
        (surface["id"], surface.get("projection"), surface.get("renderer"))
        for surface in manifest["surfaces"]
        if surface.get("projection") is not None or surface.get("renderer") is not None
    ]
    supported_surface_projections = {
        ("planning", "docs/ROADMAP.md", "architecture/render_roadmap.py")
    }
    for spec in declared_surface_projections:
        if spec not in supported_surface_projections:
            findings.append(
                Finding(
                    "repository_state.projection_checker_missing",
                    "architecture/state_authorities.json",
                    f"declared generated projection is not covered by freshness checking: {spec}",
                )
            )

    try:
        planning_paths = projection_authority_paths(root, manifest, "planning")
        planning_fp = projection_fingerprint(root, manifest, "planning")
        planning_expected = render_roadmap.render(
            render_roadmap.load_graph(planning_paths[0]),
            authority_fingerprint=planning_fp,
            authority_paths=[path.relative_to(root).as_posix() for path in planning_paths],
        )
        _check_projection_file(
            findings,
            root=root,
            projection="docs/ROADMAP.md",
            expected=planning_expected,
            fingerprint=planning_fp,
            stale_code="repository_state.generated_roadmap_stale",
        )
    except (OSError, ValueError, KeyError) as exc:
        findings.append(
            Finding(
                "repository_state.authority_inputs_invalid",
                "architecture/state_authorities.json",
                f"planning: {exc}",
            )
        )

    try:
        state_fp = orientation_fingerprint(
            root, manifest, manifest_path=manifest_path
        )
        state_expected = render_state(root)
        _check_projection_file(
            findings,
            root=root,
            projection="docs/generated/STATE.md",
            expected=state_expected,
            fingerprint=state_fp,
            stale_code="repository_state.generated_state_stale",
        )
    except (OSError, ValueError, KeyError) as exc:
        findings.append(
            Finding(
                "repository_state.orientation_inputs_invalid",
                "architecture/state_authorities.json",
                str(exc),
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

    reference_paths = [
        root / "README.md",
        root / "AGENTS.md",
        root / "architecture" / "planning_graph.json",
    ]
    docs_root = root / "docs"
    if docs_root.is_dir():
        reference_paths.extend(
            path
            for path in docs_root.rglob("*.md")
            if "archive" not in path.relative_to(docs_root).parts
        )
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

    for link_finding in check_markdown_links.check(root):
        findings.append(
            Finding(
                link_finding.code,
                f"{link_finding.path}:{link_finding.line}",
                f"{link_finding.message}: {link_finding.target}",
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
