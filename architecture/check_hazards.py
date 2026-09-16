#!/usr/bin/env python3
"""Integrity checks for Climate's semantic-hazard ledger.

A hazard record is a review of a particular source state, not an eternal fact
about a filename.  The reviewed Git blob id therefore acts as a freshness pin:
if the source changes, the hazard must be revisited before this check passes.

`status=resolved` is intentionally stronger than an editable label.  A resolved
hazard must name at least one resolution witness; later work can strengthen the
witness graph without changing this basic authority boundary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HAZARDS = ROOT / "architecture" / "semantic_hazards.json"

HAZARD_CLASSES = {"S0", "S1", "S2", "S3", "S4"}
HAZARD_STATUSES = {"open", "resolved"}
GIT_BLOB_RE = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class Finding:
    code: str
    message: str
    hazard_id: str | None = None
    path: str | None = None


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def git_blob_sha(path: Path) -> str:
    """Return the SHA-1 object id Git uses for this file's blob bytes."""
    payload = path.read_bytes()
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def check(root: Path, hazard_registry: dict[str, Any]) -> list[Finding]:
    findings: list[Finding] = []
    hazards = hazard_registry.get("hazards", [])
    if not isinstance(hazards, list):
        return [Finding("hazards.registry_not_list", "hazards must be a list")]

    seen_ids: set[str] = set()

    for index, hazard in enumerate(hazards):
        if not isinstance(hazard, dict):
            findings.append(Finding(
                "hazards.entry_not_object",
                f"hazard at index {index} is not an object",
            ))
            continue

        hazard_id = hazard.get("hazard_id")
        if not isinstance(hazard_id, str) or not hazard_id:
            findings.append(Finding(
                "hazards.id_missing",
                f"hazard at index {index} has no usable hazard_id",
            ))
            hazard_id = None
        elif hazard_id in seen_ids:
            findings.append(Finding(
                "hazards.id_duplicate",
                "hazard_id appears more than once",
                hazard_id=hazard_id,
            ))
        else:
            seen_ids.add(hazard_id)

        path = hazard.get("path")
        if not isinstance(path, str) or not path:
            findings.append(Finding(
                "hazards.path_missing",
                "hazard has no usable source path",
                hazard_id=hazard_id,
            ))
            continue

        source_path = root / path
        if not source_path.is_file():
            findings.append(Finding(
                "hazards.source_missing",
                "hazard source path does not exist",
                hazard_id=hazard_id,
                path=path,
            ))
            continue

        reviewed_blob = hazard.get("reviewed_source_blob")
        if not isinstance(reviewed_blob, str) or not GIT_BLOB_RE.fullmatch(reviewed_blob):
            findings.append(Finding(
                "hazards.reviewed_blob_invalid",
                "reviewed_source_blob must be a lowercase 40-hex Git blob id",
                hazard_id=hazard_id,
                path=path,
            ))
        else:
            current_blob = git_blob_sha(source_path)
            if current_blob != reviewed_blob:
                findings.append(Finding(
                    "hazards.source_review_stale",
                    f"source changed since hazard review: reviewed={reviewed_blob} current={current_blob}",
                    hazard_id=hazard_id,
                    path=path,
                ))

        hazard_class = hazard.get("class")
        if hazard_class not in HAZARD_CLASSES:
            findings.append(Finding(
                "hazards.class_unknown",
                f"unknown hazard class {hazard_class!r}",
                hazard_id=hazard_id,
                path=path,
            ))

        status = hazard.get("status")
        if status not in HAZARD_STATUSES:
            findings.append(Finding(
                "hazards.status_unknown",
                f"unknown hazard status {status!r}",
                hazard_id=hazard_id,
                path=path,
            ))

        if not isinstance(hazard.get("blocks_evidence"), bool):
            findings.append(Finding(
                "hazards.blocks_evidence_missing",
                "blocks_evidence must be an explicit boolean",
                hazard_id=hazard_id,
                path=path,
            ))

        for field in ("disposition", "reason"):
            value = hazard.get(field)
            if not isinstance(value, str) or not value.strip():
                findings.append(Finding(
                    f"hazards.{field}_missing",
                    f"{field} must be a non-empty string",
                    hazard_id=hazard_id,
                    path=path,
                ))

        blueprint = hazard.get("blueprint")
        if blueprint is not None:
            if not isinstance(blueprint, str) or not blueprint:
                findings.append(Finding(
                    "hazards.blueprint_invalid",
                    "blueprint must be a non-empty path when present",
                    hazard_id=hazard_id,
                    path=path,
                ))
            elif not (root / blueprint).is_file():
                findings.append(Finding(
                    "hazards.blueprint_missing",
                    "referenced blueprint does not exist",
                    hazard_id=hazard_id,
                    path=blueprint,
                ))

        witnesses = hazard.get("resolution_witness_ids", [])
        if status == "resolved":
            if not isinstance(witnesses, list) or not witnesses or not all(
                isinstance(item, str) and item.strip() for item in witnesses
            ):
                findings.append(Finding(
                    "hazards.resolution_witness_missing",
                    "resolved hazard requires at least one named resolution witness",
                    hazard_id=hazard_id,
                    path=path,
                ))
        elif "resolution_witness_ids" in hazard and not isinstance(witnesses, list):
            findings.append(Finding(
                "hazards.resolution_witness_invalid",
                "resolution_witness_ids must be a list when present",
                hazard_id=hazard_id,
                path=path,
            ))

    return sorted(
        findings,
        key=lambda item: (item.code, item.hazard_id or "", item.path or ""),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="check Climate semantic hazard integrity")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--hazards", type=Path, default=DEFAULT_HAZARDS)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    registry = load_json(args.hazards)
    findings = check(args.root.resolve(), registry)

    if args.json:
        print(json.dumps({
            "ok": not findings,
            "finding_count": len(findings),
            "findings": [asdict(item) for item in findings],
        }, indent=2, sort_keys=True))
    elif findings:
        for finding in findings:
            hazard = f" [{finding.hazard_id}]" if finding.hazard_id else ""
            path = f" ({finding.path})" if finding.path else ""
            print(f"{finding.code}: {finding.message}{hazard}{path}")
    else:
        print(f"hazards: {len(registry.get('hazards', []))} records; integrity ok")

    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
