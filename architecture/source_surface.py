"""Shared classification of source files across Climate architecture tooling.

The classification is intentionally conservative: a source file outside a known
control/test/contract/excluded tree is scientific by default. New package
layouts therefore cannot escape module registration merely by moving code under
an unfamiliar directory.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

SOURCE_LANGUAGES = {
    ".rs": "rust",
    ".py": "python",
    ".cu": "cuda",
    ".cuh": "cuda-header",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".h": "c-header",
    ".hpp": "cpp-header",
    ".f90": "fortran",
    ".F90": "fortran",
    ".jl": "julia",
    ".hs": "haskell",
    ".cue": "cue",
}

ROLE_SCIENTIFIC = "scientific"
ROLE_REFERENCE = "reference"
ROLE_PACKAGE = "package"
ROLE_EXPERIMENT = "experiment"
ROLE_ARCHITECTURE = "architecture"
ROLE_TEST = "test"
ROLE_CONTRACT = "contract"
ROLE_FIXTURE = "fixture"
ROLE_EXCLUDED = "excluded"

MODULE_TRACKED_ROLES = frozenset({ROLE_SCIENTIFIC, ROLE_REFERENCE})
AUDITED_ROLES = frozenset({ROLE_SCIENTIFIC, ROLE_REFERENCE, ROLE_PACKAGE, ROLE_EXPERIMENT})

_EXCLUDED_PARTS = frozenset({
    ".git", ".github", "build", "target", ".venv", "venv", "__pycache__",
})


@dataclass(frozen=True)
class SourceFile:
    path: Path
    relative_path: str
    language: str
    role: str


def classify_relative_path(relative: Path) -> str | None:
    """Return the architectural role for a repository-relative source path.

    Unknown source-bearing directories deliberately fall through to
    ``scientific``. Adding a new package directory must not create an escape
    hatch from maturity/registration checks. Python package marker files remain
    visible to source audits but are not scientific components by themselves.
    """
    if relative.suffix not in SOURCE_LANGUAGES:
        return None

    parts = relative.parts
    if not parts:
        return None
    if any(part in _EXCLUDED_PARTS or part.startswith(".") for part in parts[:-1]):
        return ROLE_EXCLUDED

    root = parts[0] if len(parts) > 1 else None
    if root == "architecture":
        return ROLE_ARCHITECTURE
    if root == "tests":
        return ROLE_TEST
    if root == "contracts":
        return ROLE_CONTRACT
    if root == "experiments":
        return ROLE_EXPERIMENT
    if root == "fixtures":
        return ROLE_FIXTURE
    if root in {"docs", "blueprints", "evidence"}:
        return ROLE_EXCLUDED

    if relative.name == "__init__.py":
        return ROLE_PACKAGE
    if root == "reference":
        return ROLE_REFERENCE

    return ROLE_SCIENTIFIC


def iter_source_files(root: Path, *, roles: frozenset[str] | set[str] | None = None) -> Iterable[SourceFile]:
    root = root.resolve()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        role = classify_relative_path(relative)
        if role is None or role == ROLE_EXCLUDED:
            continue
        if roles is not None and role not in roles:
            continue
        yield SourceFile(
            path=path,
            relative_path=relative.as_posix(),
            language=SOURCE_LANGUAGES[path.suffix],
            role=role,
        )


def module_tracked_paths(root: Path) -> set[str]:
    return {
        item.relative_path
        for item in iter_source_files(root, roles=MODULE_TRACKED_ROLES)
    }
