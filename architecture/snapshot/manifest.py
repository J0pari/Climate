"""Schema and parsing for authoritative repository snapshot manifests."""
from __future__ import annotations

import base64
import binascii
import json
from dataclasses import dataclass
from pathlib import Path

from .git_objects import ALLOWED_FILE_MODES, git_object_id, is_git_sha1, validate_relative_path


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
    source_commit_object: bytes | None = None


def _commit_tree_sha1(payload: bytes) -> str:
    first_line = payload.split(b"\n", 1)[0]
    if not first_line.startswith(b"tree "):
        raise ValueError("embedded source commit object is missing its tree header")
    raw_tree = first_line[5:]
    try:
        tree = raw_tree.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("embedded source commit tree is not ASCII") from exc
    if not is_git_sha1(tree):
        raise ValueError("embedded source commit has invalid tree SHA-1")
    return tree


def parse_manifest(payload: object) -> SnapshotManifest:
    if not isinstance(payload, dict):
        raise ValueError("snapshot manifest must be a JSON object")
    schema_version = payload.get("schema_version")
    if type(schema_version) is not int or schema_version not in {1, 2}:
        raise ValueError("snapshot manifest schema_version must be integer 1 or 2")
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
        path = row["path"]
        if not isinstance(path, str):
            raise ValueError("snapshot manifest path must be a string")
        validate_relative_path(path)
        if path in seen:
            raise ValueError(f"duplicate snapshot manifest path: {path}")
        seen.add(path)
        mode = row["mode"]
        if not isinstance(mode, str) or mode not in ALLOWED_FILE_MODES:
            raise ValueError(f"unsupported file mode for {path}: {mode!r}")
        blob = row["git_blob_sha1"]
        if not isinstance(blob, str) or not is_git_sha1(blob):
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
        not is_git_sha1(tree_sha1)
    ):
        raise ValueError("invalid root tree SHA-1")
    if complete and tree_sha1 is None:
        raise ValueError("complete snapshot manifest requires tree_sha1")

    source_commit = payload.get("source_commit")
    if source_commit is not None and (
        not is_git_sha1(source_commit)
    ):
        raise ValueError("invalid source commit SHA-1")
    if complete and source_commit is None:
        raise ValueError("complete snapshot manifest requires source_commit")

    source_commit_object: bytes | None = None
    encoded_commit = payload.get("source_commit_object_base64")
    if encoded_commit is not None:
        if not isinstance(encoded_commit, str):
            raise ValueError("source_commit_object_base64 must be a string")
        try:
            source_commit_object = base64.b64decode(encoded_commit, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("source_commit_object_base64 is not valid base64") from exc
        if source_commit is None:
            raise ValueError("embedded source commit object requires source_commit")
        observed_commit = git_object_id("commit", source_commit_object)
        if observed_commit != source_commit:
            raise ValueError(
                f"embedded source commit object hashes to {observed_commit}, expected {source_commit}"
            )
        observed_tree = _commit_tree_sha1(source_commit_object)
        if tree_sha1 is None or observed_tree != tree_sha1:
            raise ValueError(
                f"embedded source commit points to tree {observed_tree}, expected {tree_sha1}"
            )
    elif schema_version >= 2 and complete:
        raise ValueError("schema v2 complete snapshot requires source_commit_object_base64")

    return SnapshotManifest(
        files=tuple(files),
        complete_tree=complete,
        tree_sha1=tree_sha1,
        source_commit=source_commit,
        source_commit_object=source_commit_object,
    )


def load_manifest(path: Path) -> SnapshotManifest:
    return parse_manifest(json.loads(path.read_text(encoding="utf-8")))
