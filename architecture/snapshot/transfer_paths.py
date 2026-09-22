"""Canonical persisted-partial paths for snapshot transport."""
from __future__ import annotations

from pathlib import Path

from .git_objects import validate_relative_path

PART_SUFFIXES = (".b64part", ".textpart")


def _part_path(work_dir: Path, relative: str, suffix: str) -> Path:
    if suffix not in PART_SUFFIXES:
        raise ValueError(f"unsupported snapshot partial suffix: {suffix!r}")
    path = work_dir / "parts" / Path(*validate_relative_path(relative).parts)
    return path.with_name(path.name + suffix)


def part_path(work_dir: Path, relative: str, suffix: str) -> Path:
    path = _part_path(work_dir, relative, suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def discard_partials(work_dir: Path, relative: str) -> None:
    for suffix in PART_SUFFIXES:
        part = _part_path(work_dir, relative, suffix)
        if part.exists():
            part.unlink()
