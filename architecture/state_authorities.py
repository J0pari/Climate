#!/usr/bin/env python3
"""Helpers for typed repository-state authorities and generated projections."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "architecture" / "state_authorities.json"


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2:
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
        paths = surface.get("authority_paths")
        if not isinstance(paths, list) or not paths or not all(
            isinstance(item, str) and item for item in paths
        ):
            raise ValueError(f"surface {surface_id} requires authority_paths")
        for field in ("owns", "does_not_own"):
            values = surface.get(field)
            if not isinstance(values, list) or not values or not all(
                isinstance(item, str) and item for item in values
            ):
                raise ValueError(f"surface {surface_id} requires {field}")
        ids.append(surface_id)
    if len(ids) != len(set(ids)):
        raise ValueError("repository-state surface ids must be unique")

    orientation = payload.get("orientation_view")
    if not isinstance(orientation, dict):
        raise ValueError("repository-state authority manifest requires orientation_view")
    projection = orientation.get("projection")
    renderer = orientation.get("renderer")
    includes = orientation.get("includes")
    excludes = orientation.get("excludes")
    rule = orientation.get("composition_rule")
    if not isinstance(projection, str) or not projection:
        raise ValueError("orientation_view requires projection")
    if not isinstance(renderer, str) or not renderer:
        raise ValueError("orientation_view requires renderer")
    if not isinstance(includes, list) or not includes:
        raise ValueError("orientation_view requires included surfaces")
    if not isinstance(excludes, list):
        raise ValueError("orientation_view requires excluded surfaces")
    if not isinstance(rule, str) or not rule:
        raise ValueError("orientation_view requires composition_rule")
    unknown = (set(includes) | set(excludes)) - set(ids)
    if unknown:
        raise ValueError(
            "orientation_view references unknown surfaces: " + ", ".join(sorted(unknown))
        )
    if set(includes) & set(excludes):
        raise ValueError("orientation_view cannot both include and exclude a surface")

    worker_orientation = payload.get("worker_orientation")
    if not isinstance(worker_orientation, list) or not worker_orientation or not all(
        isinstance(item, str) and item for item in worker_orientation
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
    for raw in surface["authority_paths"]:
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


def orientation_authority_paths(
    root: Path,
    manifest: dict[str, Any],
) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    for surface_id in manifest["orientation_view"]["includes"]:
        surface = surface_by_id(manifest, surface_id)
        paths = expand_local_authority_paths(root, surface)
        if not paths:
            raise ValueError(
                f"orientation surface {surface_id} has no local authority inputs"
            )
        result[surface_id] = paths
    return result
