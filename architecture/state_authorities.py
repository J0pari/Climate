#!/usr/bin/env python3
"""Helpers for generated repository-state and planning projections."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "architecture" / "state_authorities.json"


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported repository-state authority schema")
    surfaces = payload.get("surfaces")
    if not isinstance(surfaces, list) or not surfaces:
        raise ValueError("repository-state authority manifest requires surfaces")
    ids: list[str] = []
    for surface in surfaces:
        if not isinstance(surface, dict):
            raise ValueError("repository-state surfaces must be objects")
        surface_id = surface.get("id")
        if not isinstance(surface_id, str) or not surface_id:
            raise ValueError("every repository-state surface requires a non-empty id")
        ids.append(surface_id)
    if len(ids) != len(set(ids)):
        raise ValueError("repository-state surface ids must be unique")
    orientation = payload.get("worker_orientation")
    if not isinstance(orientation, list) or not orientation or not all(
        isinstance(item, str) and item for item in orientation
    ):
        raise ValueError("repository-state authority manifest requires worker_orientation")
    return payload


def surface_by_id(manifest: dict[str, Any], surface_id: str) -> dict[str, Any]:
    for surface in manifest["surfaces"]:
        if surface["id"] == surface_id:
            return surface
    raise KeyError(f"unknown repository-state surface: {surface_id}")


def _is_local_pattern(value: str) -> bool:
    return not (
        value.startswith("GitHub ")
        or value.startswith("Git ")
        or "://" in value
    )


def expand_local_authority_paths(root: Path, surface: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    patterns = surface.get("authority_paths")
    if not isinstance(patterns, list) or not patterns:
        raise ValueError(f"surface {surface['id']} requires authority_paths")
    for raw in patterns:
        if not isinstance(raw, str) or not raw:
            raise ValueError(f"surface {surface['id']} has invalid authority path")
        if not _is_local_pattern(raw):
            continue
        if any(token in raw for token in ("*", "?", "[")):
            matches = sorted(path for path in root.glob(raw) if path.is_file())
            if not matches:
                raise ValueError(
                    f"surface {surface['id']} authority glob matched no files: {raw}"
                )
            paths.extend(matches)
        else:
            path = root / raw
            if not path.is_file():
                raise ValueError(
                    f"surface {surface['id']} authority path is missing: {raw}"
                )
            paths.append(path)
    return sorted({path.resolve() for path in paths})


def projection_authority_paths(
    root: Path,
    manifest: dict[str, Any],
    surface_id: str,
) -> list[Path]:
    surface = surface_by_id(manifest, surface_id)
    if not surface.get("projection"):
        raise ValueError(f"surface {surface_id} has no generated projection")
    paths = expand_local_authority_paths(root, surface)
    if not paths:
        raise ValueError(f"projected surface {surface_id} has no local authority inputs")
    return paths


def fingerprint_paths(root: Path, paths: list[Path]) -> str:
    digest = hashlib.sha256()
    root = root.resolve()
    for path in sorted(paths):
        resolved = path.resolve()
        relative = resolved.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(resolved.read_bytes()).hexdigest().encode("ascii"))
        digest.update(b"\n")
    return "sha256:" + digest.hexdigest()
