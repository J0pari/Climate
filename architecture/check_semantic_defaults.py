#!/usr/bin/env python3
"""Reject implicit semantic input defaults in canonical scientific source.

Reference-value defaults are allowed only through focused *_reference_values
authorities. Boundary conditions, observations, forcing, calibration, policy,
configuration, and similar semantic choices must be explicit at the call site.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))

from architecture.source_surface import ROLE_SCIENTIFIC, iter_source_files

RUST_DEFAULT_IMPL = re.compile(r"\bimpl\s+Default\s+for\s+([A-Za-z_][A-Za-z0-9_]*)")
RUST_DERIVE_DEFAULT = re.compile(r"#\[derive\([^\]]*\bDefault\b[^\]]*\)\]")
RUST_FALLBACK_CONSTRUCTOR = re.compile(
    r"\bpub\s+(?:const\s+)?fn\s+"
    r"(legacy_reference|default_reference|fallback)\s*\("
)
RUST_ZERO_ARG_NEW = re.compile(r"\bpub\s+(?:const\s+)?fn\s+new\s*\(\s*\)")

FORTRAN_PUBLIC_TYPE = re.compile(
    r"^\s*type\s*,\s*public\s*::\s*([A-Za-z_][A-Za-z0-9_]*)",
    re.IGNORECASE,
)
FORTRAN_END_TYPE = re.compile(r"^\s*end\s+type\b", re.IGNORECASE)
FORTRAN_COMPONENT_DEFAULT = re.compile(
    r"::\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^!,\n]+)",
    re.IGNORECASE,
)
FORTRAN_REFERENCE_SYMBOL = re.compile(
    r"\bparameter\s*,\s*public\s*::\s*([A-Z][A-Z0-9_]*)\s*=",
    re.IGNORECASE,
)
SEMANTIC_INPUT_TYPE = re.compile(
    r"(?:boundary|parameters|configuration|config|policy|options|settings|"
    r"observation|forcing|calibration|input)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Finding:
    code: str
    path: str
    message: str


def _reference_symbols(root: Path) -> set[str]:
    symbols: set[str] = set()
    for path in sorted((root / "src" / "fortran").glob("*_reference_values.f90")):
        text = path.read_text(encoding="utf-8")
        symbols.update(match.group(1).upper() for match in FORTRAN_REFERENCE_SYMBOL.finditer(text))
    return symbols


def _check_rust(path: Path, relative_path: str, text: str) -> list[Finding]:
    findings: list[Finding] = []
    for match in RUST_DEFAULT_IMPL.finditer(text):
        findings.append(Finding(
            "semantic_default.rust_default_impl",
            relative_path,
            f"canonical scientific type {match.group(1)!r} implements Default; semantic construction must be explicit",
        ))
    if RUST_DERIVE_DEFAULT.search(text):
        findings.append(Finding(
            "semantic_default.rust_derive_default",
            relative_path,
            "canonical scientific types must not derive Default",
        ))
    for match in RUST_FALLBACK_CONSTRUCTOR.finditer(text):
        findings.append(Finding(
            "semantic_default.rust_fallback_constructor",
            relative_path,
            f"public fallback constructor {match.group(1)!r} is not an authority boundary",
        ))
    if RUST_ZERO_ARG_NEW.search(text):
        findings.append(Finding(
            "semantic_default.rust_zero_arg_constructor",
            relative_path,
            "zero-argument public new() can materialize omitted scientific semantics; require explicit inputs",
        ))
    return findings


def _check_fortran(
    relative_path: str,
    text: str,
    reference_symbols: set[str],
) -> list[Finding]:
    findings: list[Finding] = []
    active_type: str | None = None

    for line_number, line in enumerate(text.splitlines(), start=1):
        type_match = FORTRAN_PUBLIC_TYPE.match(line)
        if type_match:
            active_type = type_match.group(1)
            continue
        if FORTRAN_END_TYPE.match(line):
            active_type = None
            continue
        if active_type is None or not SEMANTIC_INPUT_TYPE.search(active_type):
            continue

        default_match = FORTRAN_COMPONENT_DEFAULT.search(line)
        if default_match is None:
            continue

        field = default_match.group(1)
        rhs = default_match.group(2).strip()
        if "parameters" in active_type.lower() and rhs.upper() in reference_symbols:
            continue

        findings.append(Finding(
            "semantic_default.fortran_input_component",
            relative_path,
            (
                f"line {line_number}: public semantic input type {active_type!r} "
                f"defaults {field!r} to {rhs!r}; require an explicit caller choice "
                "or a named *_reference_values authority"
            ),
        ))
    return findings


def check(root: Path = ROOT) -> list[Finding]:
    root = root.resolve()
    reference_symbols = _reference_symbols(root)
    findings: list[Finding] = []

    for source in iter_source_files(root, roles={ROLE_SCIENTIFIC}):
        text = source.path.read_text(encoding="utf-8")
        if source.language == "rust":
            findings.extend(_check_rust(source.path, source.relative_path, text))
        elif source.language == "fortran":
            findings.extend(_check_fortran(source.relative_path, text, reference_symbols))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description="reject implicit semantic defaults")
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
        print("semantic defaults: canonical scientific inputs are explicit")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
