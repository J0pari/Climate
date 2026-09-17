#!/usr/bin/env python3
"""Cross-record integrity for focused configuration provenance.

CUE validates configuration record shape. This checker resolves nonlocal
configuration provenance against the repository's source-authority registry so
labels such as literature_fixed, calibrated, and learned cannot stand in for an
actual authority identity.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIGURATION_ROOT = ROOT / "configurations"
AUTHORITY_REGISTRY = ROOT / "architecture" / "data_authorities.json"
BOUND_PROVENANCE = frozenset({"literature_fixed", "calibrated", "learned"})


@dataclass(frozen=True)
class Finding:
    code: str
    path: str
    message: str


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
    return value


def _source_map(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for source in registry.get("sources", []):
        if not isinstance(source, dict):
            continue
        source_id = source.get("source_id")
        if isinstance(source_id, str) and source_id:
            result[source_id] = source
    return result


def _durable_identities(source: dict[str, Any]) -> set[str]:
    identities: set[str] = set()
    for key in ("doi", "official_url"):
        value = source.get(key)
        if isinstance(value, str) and value.strip():
            identities.add(value.strip())
    return identities


def check(root: Path = ROOT) -> list[Finding]:
    root = root.resolve()
    configuration_root = root / "configurations"
    authority_path = root / "architecture" / "data_authorities.json"
    findings: list[Finding] = []

    try:
        sources = _source_map(_load_json(authority_path))
    except (OSError, json.JSONDecodeError, ValueError) as error:
        return [Finding(
            "configuration.authority_registry_invalid",
            str(authority_path.relative_to(root)) if authority_path.is_relative_to(root) else str(authority_path),
            str(error),
        )]

    if not configuration_root.is_dir():
        return findings

    for path in sorted(configuration_root.glob("*/*.json")):
        relative = path.relative_to(root).as_posix()
        try:
            record = _load_json(path)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            findings.append(Finding(
                "configuration.record_invalid",
                relative,
                str(error),
            ))
            continue

        provenance = record.get("provenance")
        authority_refs = record.get("authority_refs")
        if provenance not in BOUND_PROVENANCE:
            continue

        if not isinstance(authority_refs, list) or not authority_refs:
            findings.append(Finding(
                "configuration.authority_missing",
                relative,
                f"provenance {provenance!r} requires at least one authority_ref",
            ))
            continue

        for index, reference in enumerate(authority_refs):
            if not isinstance(reference, dict):
                findings.append(Finding(
                    "configuration.authority_ref_invalid",
                    relative,
                    f"authority_refs[{index}] must be an object",
                ))
                continue

            source_id = reference.get("source_id")
            source = sources.get(source_id) if isinstance(source_id, str) else None
            if source is None:
                findings.append(Finding(
                    "configuration.authority_unresolved",
                    relative,
                    f"authority_refs[{index}] source_id {source_id!r} is not registered",
                ))
                continue

            identity = reference.get("identity")
            allowed = _durable_identities(source)
            if not isinstance(identity, str) or identity not in allowed:
                findings.append(Finding(
                    "configuration.authority_identity_mismatch",
                    relative,
                    (
                        f"authority_refs[{index}] identity {identity!r} does not match "
                        f"the registered DOI or official URL for {source_id!r}"
                    ),
                ))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description="check focused configuration provenance")
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
        print("configuration provenance: nonlocal authorities resolve")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
