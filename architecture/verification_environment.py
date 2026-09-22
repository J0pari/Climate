#!/usr/bin/env python3
"""Report whether repository verification inputs have reproducible identities."""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SEMVER = re.compile(r"^v?\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$")
DIGEST_IMAGE = re.compile(r"@sha256:[0-9a-f]{64}$")
PINNED_REQUIREMENT = re.compile(
    r"^([A-Za-z0-9_.-]+)(?:\[[A-Za-z0-9_,.-]+\])?==([^=<>!~\s;]+)(?:\s*;(.*))?$"
)


def _parsed_python_requirement(line: str) -> tuple[str, str, str | None] | None:
    match = PINNED_REQUIREMENT.fullmatch(line)
    if not match or "*" in match.group(2):
        return None
    name = re.sub(r"[-_.]+", "-", match.group(1)).lower()
    marker = match.group(3).strip() if match.group(3) else None
    return name, match.group(2), marker


def _exact_python_requirement(line: str) -> bool:
    return _parsed_python_requirement(line) is not None


def _exact_apt_version(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and not any(token in value for token in ("*", "?", "[", "]", "$", "`"))
    )


@dataclass(frozen=True)
class Check:
    check_id: str
    satisfied: bool
    observed: Any
    requirement: str


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _devcontainer_checks(root: Path) -> list[Check]:
    path = root / ".devcontainer" / "devcontainer.json"
    if not path.is_file():
        return [
            Check(
                "devcontainer.present",
                False,
                None,
                "repository verification environment has a declared container/workspace definition",
            )
        ]
    payload = _read_json(path)
    image = payload.get("image")
    features = payload.get("features", {})
    checks = [
        Check(
            "devcontainer.image_immutable",
            isinstance(image, str) and bool(DIGEST_IMAGE.search(image)),
            image,
            "base image is bound by an immutable sha256 digest",
        )
    ]
    if not isinstance(features, dict):
        features = {}
    for feature_id, config in sorted(features.items()):
        version = config.get("version") if isinstance(config, dict) else None
        checks.append(
            Check(
                f"devcontainer.feature_version.{feature_id}",
                isinstance(version, str)
                and version.lower() != "latest"
                and bool(SEMVER.fullmatch(version)),
                version,
                "devcontainer feature runtime version is exact rather than latest/major/minor-only",
            )
        )
    return checks


def _cargo_checks(root: Path) -> list[Check]:
    lock = root / "Cargo.lock"
    return [
        Check(
            "rust.cargo_lock",
            lock.is_file(),
            "Cargo.lock" if lock.is_file() else None,
            "Rust verification uses a committed Cargo.lock",
        )
    ]


def _lean_checks(root: Path) -> list[Check]:
    path = root / "formal" / "lean-toolchain"
    observed = path.read_text(encoding="utf-8").strip() if path.is_file() else None
    satisfied = (
        isinstance(observed, str)
        and ":" in observed
        and observed.rsplit(":", 1)[-1].startswith("v")
        and bool(SEMVER.fullmatch(observed.rsplit(":", 1)[-1]))
    )
    return [
        Check(
            "lean.toolchain_exact",
            satisfied,
            observed,
            "Lean toolchain resolves to an explicit version",
        )
    ]


def _requirements_checks(root: Path) -> list[Check]:
    req_dir = root / "requirements"
    files = sorted(req_dir.glob("*.txt")) if req_dir.is_dir() else []
    if not files:
        return [
            Check(
                "python.requirements_present",
                False,
                [],
                "focused Python dependency slices are declared",
            )
        ]
    unpinned: dict[str, list[str]] = {}
    unconditional_pins: dict[str, dict[str, set[str]]] = {}
    for path in files:
        relative = path.relative_to(root).as_posix()
        bad = []
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parsed = _parsed_python_requirement(line)
            if parsed is None:
                bad.append(line)
                continue
            name, version, marker = parsed
            if marker is None:
                unconditional_pins.setdefault(name, {}).setdefault(version, set()).add(
                    relative
                )
        if bad:
            unpinned[relative] = bad

    conflicts = {
        name: {
            version: sorted(paths)
            for version, paths in sorted(versions.items())
        }
        for name, versions in sorted(unconditional_pins.items())
        if len(versions) > 1
    }
    return [
        Check(
            "python.focused_requirements_exact",
            not unpinned,
            {
                "files": [path.relative_to(root).as_posix() for path in files],
                "unpinned": unpinned,
                "cross_slice_conflicts": conflicts,
            },
            "every declared focused Python requirement uses an exact == pin; cross-slice differences remain diagnostic because focused experiment environments are independent",
        ),
        Check(
            "python.repository_lock",
            any(
                (root / name).is_file()
                for name in (
                    "requirements.lock",
                    "requirements/verification.lock",
                    "uv.lock",
                    "poetry.lock",
                    "Pipfile.lock",
                )
            ),
            None,
            "complete Python/architecture verification environment has one resolved repository-wide lock or immutable container identity",
        ),
    ]


def _bootstrap_checks(root: Path) -> list[Check]:
    path = root / ".devcontainer" / "bootstrap.sh"
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    cue = re.search(r"cuelang\.org/go/cmd/cue@(v[^\s]+)", text)
    cue_version = cue.group(1) if cue else None
    apt_packages = {}
    for package in ("gfortran", "libblas-dev", "liblapack-dev"):
        match = re.search(
            rf"(?<![A-Za-z0-9_.-]){re.escape(package)}(?:=([^\s\\]+))?(?=[\s\\]|$)",
            text,
        )
        apt_packages[package] = None if match is None else match.group(1)
    return [
        Check(
            "cue.version_exact",
            isinstance(cue_version, str) and bool(SEMVER.fullmatch(cue_version)),
            cue_version,
            "CUE CLI install is version-pinned",
        ),
        Check(
            "fortran.compiler_package_exact",
            _exact_apt_version(apt_packages.get("gfortran")),
            apt_packages.get("gfortran"),
            "gfortran package identity is version-pinned for the verification environment",
        ),
        Check(
            "blas.package_exact",
            _exact_apt_version(apt_packages.get("libblas-dev")),
            apt_packages.get("libblas-dev"),
            "BLAS provider package identity is version-pinned for the verification environment",
        ),
        Check(
            "lapack.package_exact",
            _exact_apt_version(apt_packages.get("liblapack-dev")),
            apt_packages.get("liblapack-dev"),
            "LAPACK provider package identity is version-pinned for the verification environment",
        ),
    ]


def _source_identity_checks(root: Path) -> list[Check]:
    generator = root / "architecture" / "snapshot" / "generate.py"
    manifest = root / "architecture" / "snapshot" / "manifest.py"
    archive = root / "architecture" / "snapshot" / "archive.py"
    cli = root / "architecture" / "check_snapshot.py"
    generator_text = generator.read_text(encoding="utf-8") if generator.is_file() else ""
    manifest_text = manifest.read_text(encoding="utf-8") if manifest.is_file() else ""
    archive_text = archive.read_text(encoding="utf-8") if archive.is_file() else ""
    cli_text = cli.read_text(encoding="utf-8") if cli.is_file() else ""
    return [
        Check(
            "source_snapshot.identity_tooling",
            generator.is_file()
            and "build_manifest" in generator_text
            and '"cat-file", "commit"' in generator_text
            and manifest.is_file()
            and "source_commit_object_base64" in manifest_text
            and 'git_object_id("commit", source_commit_object)' in manifest_text
            and archive.is_file()
            and "create_archive" in archive_text
            and '"archive"' in cli_text
            and '"identity"' in cli_text
            and "--expected-commit" in cli_text,
            {
                "generator": generator.relative_to(root).as_posix() if generator.is_file() else None,
                "manifest": manifest.relative_to(root).as_posix() if manifest.is_file() else None,
                "archive": archive.relative_to(root).as_posix() if archive.is_file() else None,
                "cli": cli.relative_to(root).as_posix() if cli.is_file() else None,
            },
            "self-contained archive/no-VCS handoffs bind exact tree identity to an embedded Git commit object and an externally resolved expected commit",
        )
    ]


def inspect(root: Path = ROOT) -> dict[str, Any]:
    root = root.resolve()
    checks = [
        *_devcontainer_checks(root),
        *_cargo_checks(root),
        *_lean_checks(root),
        *_requirements_checks(root),
        *_bootstrap_checks(root),
        *_source_identity_checks(root),
    ]
    unresolved = [item.check_id for item in checks if not item.satisfied]
    return {
        "schema_version": 1,
        "status": "resolved" if not unresolved else "unresolved",
        "unresolved": unresolved,
        "checks": [asdict(item) for item in checks],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--require-resolved", action="store_true")
    args = parser.parse_args()

    report = inspect(args.root)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_resolved and report["status"] != "resolved":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
