"""Small Git object-identity primitives used by snapshot verification."""
from __future__ import annotations

import hashlib
import re
import stat
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Iterable

ALLOWED_FILE_MODES = {"100644", "100755"}
GIT_SHA1 = re.compile(r"^[0-9a-f]{40}$")


def is_git_sha1(value: object) -> bool:
    return isinstance(value, str) and bool(GIT_SHA1.fullmatch(value))


def git_object_id(kind: str, payload: bytes) -> str:
    header = f"{kind} {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def git_blob_id(payload: bytes) -> str:
    return git_object_id("blob", payload)


def sha256_digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def validate_relative_path(raw: str) -> PurePosixPath:
    """Validate one canonical, cross-platform repository-relative path.

    Snapshot manifests use POSIX separators regardless of host OS. Validate the
    raw spelling before pathlib normalization so aliases such as ``a//b`` and
    ``a/./b`` cannot collapse onto another manifest row. Backslashes and
    Windows drive-qualified spellings are rejected because later materialization
    converts validated POSIX parts into a host-native ``Path``.
    """
    if not isinstance(raw, str) or not raw or "\0" in raw or "\\" in raw:
        raise ValueError(f"invalid repository-relative path: {raw!r}")
    parts = raw.split("/")
    if raw.startswith("/") or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"invalid repository-relative path: {raw!r}")
    windows = PureWindowsPath(raw)
    if windows.drive or windows.root:
        raise ValueError(f"invalid repository-relative path: {raw!r}")
    path = PurePosixPath(raw)
    if path.as_posix() != raw:
        raise ValueError(f"invalid repository-relative path: {raw!r}")
    return path


def local_file_mode(path: Path) -> str:
    executable = bool(path.stat().st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH))
    return "100755" if executable else "100644"


def _tree_id(entries: Iterable[tuple[str, str, str]]) -> str:
    encoded: list[tuple[bytes, bytes]] = []
    for name, mode, object_id in entries:
        if not name or "/" in name or "\0" in name:
            raise ValueError(f"invalid Git tree entry name: {name!r}")
        if not is_git_sha1(object_id):
            raise ValueError(f"invalid Git object SHA-1 for {name!r}")
        is_tree = mode == "40000"
        sort_key = name.encode("utf-8") + (b"/" if is_tree else b"")
        body = mode.encode("ascii") + b" " + name.encode("utf-8") + b"\0" + bytes.fromhex(object_id)
        encoded.append((sort_key, body))
    encoded.sort(key=lambda item: item[0])
    return git_object_id("tree", b"".join(body for _, body in encoded))


def tree_id_from_files(files: Iterable[tuple[str, str, str]]) -> str:
    """Compute a root Git tree ID from (path, mode, blob_id) rows."""
    root: dict[str, object] = {}
    for raw_path, mode, blob_id in files:
        path = validate_relative_path(raw_path)
        if mode not in ALLOWED_FILE_MODES:
            raise ValueError(f"unsupported file mode {mode!r} for {raw_path}")
        if not is_git_sha1(blob_id):
            raise ValueError(f"invalid Git blob SHA-1 for {raw_path}")

        node = root
        for part in path.parts[:-1]:
            current = node.get(part)
            if current is None:
                child: dict[str, object] = {}
                node[part] = child
                node = child
            elif isinstance(current, dict):
                node = current
            else:
                raise ValueError(f"path collision at {raw_path}")
        leaf = path.parts[-1]
        if leaf in node:
            raise ValueError(f"duplicate or colliding path: {raw_path}")
        node[leaf] = (mode, blob_id)

    def emit(node: dict[str, object]) -> str:
        entries: list[tuple[str, str, str]] = []
        for name, value in node.items():
            if isinstance(value, dict):
                entries.append((name, "40000", emit(value)))
            else:
                mode, blob_id = value
                entries.append((name, str(mode), str(blob_id)))
        return _tree_id(entries)

    return emit(root)
