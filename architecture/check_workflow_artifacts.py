#!/usr/bin/env python3
"""Reject ephemeral GitHub Actions artifact uploads as repository authority.

Workflow jobs may create temporary files for validation, but durable scientific,
architecture, and evidence identity must live in committed repository authorities
or explicitly external immutable stores with recorded digests. GitHub Actions
artifact uploads are ZIP-backed, retention-limited attachments and are therefore
not an acceptable persistence surface for this repository.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = ROOT / ".github" / "workflows"
UPLOAD_ARTIFACT = re.compile(
    r"^\s*(?:-\s*)?uses:\s*['\"]?actions/upload-artifact@",
    re.MULTILINE,
)


@dataclass(frozen=True)
class Finding:
    code: str
    path: str
    message: str


def check(root: Path = ROOT) -> list[Finding]:
    root = root.resolve()
    workflow_dir = root / ".github" / "workflows"
    findings: list[Finding] = []
    if not workflow_dir.is_dir():
        return findings

    for path in sorted(workflow_dir.iterdir()):
        if path.suffix not in {".yml", ".yaml"} or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        if UPLOAD_ARTIFACT.search(text):
            findings.append(
                Finding(
                    code="workflow.ephemeral_artifact_upload",
                    path=str(path.relative_to(root)),
                    message=(
                        "actions/upload-artifact is retention-limited ZIP persistence; "
                        "keep validation outputs ephemeral or persist authority through "
                        "committed identities/digests or an explicit immutable external store"
                    ),
                )
            )
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description="reject ephemeral workflow artifact persistence")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    findings = check(args.root)
    if args.json:
        print(json.dumps(
            {"ok": not findings, "findings": [asdict(item) for item in findings]},
            indent=2,
            sort_keys=True,
        ))
    elif findings:
        for finding in findings:
            print(f"{finding.code}: {finding.path}: {finding.message}")
    else:
        print("workflow artifacts: no ephemeral upload surface")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
