#!/usr/bin/env python3
"""Validate Climate's local-versus-external data authority decisions.

This guard is intentionally semantic rather than a numeric-literal linter. The
registry reviews concrete climate-facing usage seams and records whether each
value belongs to a local mathematical/model/policy contract or to an external
standard, literature source, dataset, or API. Externalized usages must resolve
to declared data authorities with reproducible access/update semantics.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "architecture" / "data_authorities.json"

SOURCE_KINDS = {"standard", "literature", "dataset", "api_dataset"}
DATA_SOURCE_KINDS = {"dataset", "api_dataset"}
SCOPES = {"current", "planned"}
OWNERSHIP_CLASSES = {
    "reference_constant",
    "model_reference_constant",
    "published_parameterization",
    "literature_reference_fixture",
    "synthetic_control",
    "numerical_policy",
    "experiment_policy",
    "external_dataset",
    "external_api",
}
EXTERNAL_OWNERSHIP_CLASSES = {"external_dataset", "external_api"}
DISPOSITIONS = {"retain_local", "externalize"}
EXTERNAL_FALLBACK_PATTERNS = (
    ("data_authority.external_default_impl", re.compile(r"\bimpl\s+Default\s+for\b")),
    (
        "data_authority.external_fallback_constructor",
        re.compile(
            r"\b(?:pub\s+)?(?:const\s+)?fn\s+"
            r"(?:legacy_reference|default_reference|fallback)\s*\("
        ),
    ),
)


@dataclass(frozen=True)
class Finding:
    code: str
    path: str
    message: str


def _finding(code: str, path: str, message: str) -> Finding:
    return Finding(code=code, path=path, message=message)


def _is_https(value: object) -> bool:
    if not isinstance(value, str):
        return False
    parsed = urlparse(value)
    return parsed.scheme == "https" and bool(parsed.netloc)


def _nonempty_string(record: dict, key: str) -> bool:
    return isinstance(record.get(key), str) and bool(record[key].strip())


def _nonempty_string_list(record: dict, key: str) -> bool:
    value = record.get(key)
    return isinstance(value, list) and bool(value) and all(
        isinstance(item, str) and bool(item.strip()) for item in value
    )


def check(root: Path = ROOT, registry: dict | None = None) -> list[Finding]:
    root = root.resolve()
    if registry is None:
        registry = json.loads((root / "architecture" / "data_authorities.json").read_text(encoding="utf-8"))

    findings: list[Finding] = []
    if registry.get("schema_version") != 1:
        findings.append(_finding(
            "data_authority.schema_version",
            "architecture/data_authorities.json",
            "schema_version must be 1",
        ))

    sources = registry.get("sources")
    usages = registry.get("usages")
    if not isinstance(sources, list) or not sources:
        findings.append(_finding(
            "data_authority.sources_missing",
            "architecture/data_authorities.json",
            "sources must be a non-empty list",
        ))
        sources = []
    if not isinstance(usages, list) or not usages:
        findings.append(_finding(
            "data_authority.usages_missing",
            "architecture/data_authorities.json",
            "usages must be a non-empty list",
        ))
        usages = []

    source_by_id: dict[str, dict] = {}
    for index, source in enumerate(sources):
        path = f"architecture/data_authorities.json:sources[{index}]"
        if not isinstance(source, dict):
            findings.append(_finding("data_authority.source_shape", path, "source must be an object"))
            continue
        source_id = source.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            findings.append(_finding("data_authority.source_id", path, "source_id must be a non-empty string"))
            continue
        if source_id in source_by_id:
            findings.append(_finding(
                "data_authority.source_duplicate",
                path,
                f"duplicate source_id {source_id!r}",
            ))
        else:
            source_by_id[source_id] = source

        kind = source.get("kind")
        if kind not in SOURCE_KINDS:
            findings.append(_finding(
                "data_authority.source_kind",
                path,
                f"kind must be one of {sorted(SOURCE_KINDS)}, got {kind!r}",
            ))

        for key in ("provider", "product", "version", "access", "update_semantics"):
            if not _nonempty_string(source, key):
                findings.append(_finding(
                    "data_authority.source_metadata",
                    path,
                    f"{key} must be a non-empty string",
                ))
        if not _is_https(source.get("official_url")):
            findings.append(_finding(
                "data_authority.source_url",
                path,
                "official_url must be an absolute HTTPS URL",
            ))

        if kind in DATA_SOURCE_KINDS:
            if not _is_https(source.get("access_url")):
                findings.append(_finding(
                    "data_authority.access_url",
                    path,
                    "dataset/API sources require an absolute HTTPS access_url",
                ))
            for key in ("time_coverage", "license_or_access_constraints"):
                if not _nonempty_string(source, key):
                    findings.append(_finding(
                        "data_authority.dataset_metadata",
                        path,
                        f"dataset/API source requires non-empty {key}",
                    ))
            for key in ("variables", "units"):
                if not _nonempty_string_list(source, key):
                    findings.append(_finding(
                        "data_authority.dataset_metadata",
                        path,
                        f"dataset/API source requires a non-empty string list for {key}",
                    ))

    usage_ids: set[str] = set()
    for index, usage in enumerate(usages):
        path = f"architecture/data_authorities.json:usages[{index}]"
        if not isinstance(usage, dict):
            findings.append(_finding("data_authority.usage_shape", path, "usage must be an object"))
            continue
        usage_id = usage.get("usage_id")
        if not isinstance(usage_id, str) or not usage_id:
            findings.append(_finding("data_authority.usage_id", path, "usage_id must be a non-empty string"))
        elif usage_id in usage_ids:
            findings.append(_finding(
                "data_authority.usage_duplicate",
                path,
                f"duplicate usage_id {usage_id!r}",
            ))
        else:
            usage_ids.add(usage_id)

        scope = usage.get("scope")
        if scope not in SCOPES:
            findings.append(_finding(
                "data_authority.usage_scope",
                path,
                f"scope must be one of {sorted(SCOPES)}, got {scope!r}",
            ))
        ownership = usage.get("ownership_class")
        if ownership not in OWNERSHIP_CLASSES:
            findings.append(_finding(
                "data_authority.ownership_class",
                path,
                f"ownership_class must be one of {sorted(OWNERSHIP_CLASSES)}, got {ownership!r}",
            ))
        disposition = usage.get("disposition")
        if disposition not in DISPOSITIONS:
            findings.append(_finding(
                "data_authority.disposition",
                path,
                f"disposition must be one of {sorted(DISPOSITIONS)}, got {disposition!r}",
            ))
        if not _nonempty_string(usage, "rationale"):
            findings.append(_finding(
                "data_authority.rationale",
                path,
                "rationale must be a non-empty string",
            ))

        relative_path = usage.get("path")
        anchor = usage.get("anchor")
        candidate_text: str | None = None
        if not isinstance(relative_path, str) or not relative_path:
            findings.append(_finding("data_authority.usage_path", path, "path must be a non-empty repository-relative path"))
        else:
            candidate = (root / relative_path).resolve()
            try:
                candidate.relative_to(root)
            except ValueError:
                findings.append(_finding(
                    "data_authority.usage_path",
                    path,
                    f"path escapes repository root: {relative_path}",
                ))
            else:
                if not candidate.is_file():
                    findings.append(_finding(
                        "data_authority.usage_path_missing",
                        path,
                        f"usage path does not exist: {relative_path}",
                    ))
                elif not isinstance(anchor, str) or not anchor:
                    findings.append(_finding(
                        "data_authority.anchor",
                        path,
                        "anchor must be a non-empty string",
                    ))
                else:
                    candidate_text = candidate.read_text(encoding="utf-8")
                    if anchor not in candidate_text:
                        findings.append(_finding(
                            "data_authority.anchor_missing",
                            relative_path,
                            f"review anchor {anchor!r} is no longer present",
                        ))

        source_ids = usage.get("source_ids")
        if not isinstance(source_ids, list) or not all(isinstance(item, str) and item for item in source_ids):
            findings.append(_finding(
                "data_authority.source_refs",
                path,
                "source_ids must be a list of non-empty strings",
            ))
            source_ids = []

        resolved_sources: list[dict] = []
        for source_id in source_ids:
            source = source_by_id.get(source_id)
            if source is None:
                findings.append(_finding(
                    "data_authority.source_ref_missing",
                    path,
                    f"source_id {source_id!r} does not resolve",
                ))
            else:
                resolved_sources.append(source)

        is_external = ownership in EXTERNAL_OWNERSHIP_CLASSES
        if disposition == "externalize" and not is_external:
            findings.append(_finding(
                "data_authority.external_ownership",
                path,
                "externalize disposition requires external_dataset or external_api ownership_class",
            ))
        if is_external and disposition != "externalize":
            findings.append(_finding(
                "data_authority.external_disposition",
                path,
                "external dataset/API ownership must use externalize disposition",
            ))
        if disposition == "externalize":
            if not source_ids:
                findings.append(_finding(
                    "data_authority.external_source_missing",
                    path,
                    "externalized usage requires at least one source_id",
                ))
            non_data = [source.get("source_id") for source in resolved_sources if source.get("kind") not in DATA_SOURCE_KINDS]
            if non_data:
                findings.append(_finding(
                    "data_authority.external_source_kind",
                    path,
                    f"externalized usage resolves to non-data authorities: {non_data}",
                ))

            if scope == "current":
                authority_boundary = usage.get("authority_boundary")
                if not isinstance(authority_boundary, str) or not authority_boundary.strip():
                    findings.append(_finding(
                        "data_authority.current_external_boundary_missing",
                        path,
                        "current externalized usage requires a non-empty authority_boundary anchor",
                    ))
                elif candidate_text is not None and authority_boundary not in candidate_text:
                    findings.append(_finding(
                        "data_authority.current_external_boundary_missing",
                        relative_path if isinstance(relative_path, str) else path,
                        f"authority boundary {authority_boundary!r} is not present",
                    ))

                if candidate_text is not None:
                    for code, pattern in EXTERNAL_FALLBACK_PATTERNS:
                        if pattern.search(candidate_text):
                            findings.append(_finding(
                                code,
                                relative_path if isinstance(relative_path, str) else path,
                                "current externalized inputs must not share a canonical source file with an implicit or fallback constructor",
                            ))
        if ownership == "external_api" and resolved_sources and not any(
            source.get("kind") == "api_dataset" for source in resolved_sources
        ):
            findings.append(_finding(
                "data_authority.api_source_missing",
                path,
                "external_api usage requires at least one api_dataset source",
            ))

    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description="check Climate data authority registry")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    findings = check(args.root)
    if args.json:
        print(json.dumps({
            "ok": not findings,
            "finding_count": len(findings),
            "findings": [asdict(item) for item in findings],
        }, indent=2, sort_keys=True))
    elif findings:
        for finding in findings:
            print(f"{finding.code}: {finding.path}: {finding.message}")
    else:
        registry = json.loads((args.root / "architecture" / "data_authorities.json").read_text(encoding="utf-8"))
        external_count = sum(1 for item in registry["usages"] if item.get("disposition") == "externalize")
        print(
            f"data authorities: {len(registry['sources'])} sources; "
            f"{len(registry['usages'])} reviewed usages; {external_count} externalized; integrity ok"
        )
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
