"""Schema and parsing for authoritative repository snapshot manifests."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .git_objects import ALLOWED_FILE_MODES, validate_relative_path

GIT_SHA1 = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class SnapshotFile:
    path: str
    git_blob_sha1: str
    mode: str
    size: int


@dataclass(frozen=True)
class SnapshotManifest:
    files: tuple[SnapshotFile, ...]
    complete_tree: bool = False
    tree_sha1: str | None = None
    source_commit: str | None = None


def parse_manifest(payload: object) -> SnapshotManifest:
    if not isinstance(payload, dict):
        raise ValueError("snapshot manifest must be a JSON object")
    schema_version = payload.get("schema_version")
    if type(schema_version) is not int or schema_version != 1:
        raise ValueError("snapshot manifest schema_version must be integer 1")
    raw_files = payload.get("files")
    if not isinstance(raw_files, list):
        raise ValueError("snapshot manifest files must be a list")

    seen: set[str] = set()
    files: list[SnapshotFile] = []
    for row in raw_files:
        if not isinstance(row, dict):
            raise ValueError("snapshot manifest file rows must be objects")
        missing = {"path", "git_blob_sha1", "mode", "size"} - row.keys()
        if missing:
            raise ValueError(f"snapshot manifest row missing {sorted(missing)}")
        path = str(row["path"])
        validate_relative_path(path)
        if path in seen:
            raise ValueError(f"duplicate snapshot manifest path: {path}")
        seen.add(path)
        mode = str(row["mode"])
        if mode not in ALLOWED_FILE_MODES:
            raise ValueError(f"unsupported file mode for {path}: {mode!r}")
        blob = row["git_blob_sha1"]
        if not isinstance(blob, str) or not GIT_SHA1.fullmatch(blob):
            raise ValueError(f"invalid Git blob SHA-1 for {path}")
        size = row["size"]
        if type(size) is not int or size < 0:
            raise ValueError(f"invalid byte size for {path}")
        files.append(SnapshotFile(path=path, git_blob_sha1=blob, mode=mode, size=size))

    complete = payload.get("complete_tree", False)
    if not isinstance(complete, bool):
        raise ValueError("snapshot manifest complete_tree must be boolean")

    tree_sha1 = payload.get("tree_sha1")
    if tree_sha1 is not None and (
        not isinstance(tree_sha1, str) or not GIT_SHA1.fullmatch(tree_sha1)
    ):
        raise ValueError("invalid root tree SHA-1")
    if complete and tree_sha1 is None:
        raise ValueError("complete snapshot manifest requires tree_sha1")

    source_commit = payload.get("source_commit")
    if source_commit is not None and (
        not isinstance(source_commit, str) or not GIT_SHA1.fullmatch(source_commit)
    ):
        raise ValueError("invalid source commit SHA-1")
    if complete and source_commit is None:
        raise ValueError("complete snapshot manifest requires source_commit")

    return SnapshotManifest(
        files=tuple(files),
        complete_tree=complete,
        tree_sha1=tree_sha1,
        source_commit=source_commit,
    )


def load_manifest(path: Path) -> SnapshotManifest:
    return parse_manifest(json.loads(path.read_text(encoding="utf-8")))
