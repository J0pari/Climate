"""Persistent state and request planning for bounded snapshot transport."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .git_objects import (
    ALLOWED_FILE_MODES,
    git_blob_id,
    is_git_sha1,
    local_file_mode,
    validate_relative_path,
)
from .manifest import SnapshotManifest

PROTOCOL = "bounded-github-snapshot-transport/v1"
DEFAULT_MAX_RESPONSE_CHARS = 12_000
DEFAULT_BASE64_CHUNK_CHARS = 10_000
DEFAULT_LINE_WINDOW = 120
DEFAULT_LINE_OVERLAP = 1
METHODS = {"base64", "text-lines"}
STATUSES = {"pending", "receiving", "verified"}


def _encoded_length(size: int) -> int:
    return 4 * ((size + 2) // 3)


def _reset_progress(row: dict[str, Any]) -> None:
    row["status"] = "pending"
    row["base64_received_chars"] = 0
    row["text_next_start_line"] = 1


def validate_state(state: object) -> None:
    if not isinstance(state, dict):
        raise ValueError("snapshot transport state must be a JSON object")
    schema_version = state.get("schema_version")
    if type(schema_version) is not int or schema_version != 1:
        raise ValueError("unsupported snapshot transport state schema")
    if state.get("protocol") != PROTOCOL:
        raise ValueError("unsupported snapshot transport protocol")

    limits = state.get("limits")
    if not isinstance(limits, dict):
        raise ValueError("transport state limits must be an object")
    for key in (
        "max_response_chars",
        "base64_chunk_chars",
        "line_window",
        "line_overlap",
    ):
        if type(limits.get(key)) is not int:
            raise ValueError(f"transport limit {key} must be an integer")
    max_response_chars = limits["max_response_chars"]
    base64_chunk_chars = limits["base64_chunk_chars"]
    line_window = limits["line_window"]
    line_overlap = limits["line_overlap"]
    if max_response_chars <= 0:
        raise ValueError("max response chars must be positive")
    if not (0 < base64_chunk_chars < max_response_chars):
        raise ValueError("base64 chunk size must be positive and below response cap")
    if line_window <= 1 or not (0 <= line_overlap < line_window):
        raise ValueError("invalid line window/overlap")

    files = state.get("files")
    if not isinstance(files, list):
        raise ValueError("transport state files must be a list")
    seen: set[str] = set()
    required = {
        "path",
        "git_blob_sha1",
        "mode",
        "size",
        "method",
        "status",
        "base64_expected_chars",
        "base64_received_chars",
        "text_next_start_line",
    }
    for row in files:
        if not isinstance(row, dict):
            raise ValueError("transport state rows must be objects")
        missing = required - row.keys()
        if missing:
            raise ValueError(f"transport state row missing {sorted(missing)}")

        path = row["path"]
        if not isinstance(path, str):
            raise ValueError("transport state path must be a string")
        validate_relative_path(path)
        if path in seen:
            raise ValueError(f"duplicate transport state path: {path}")
        seen.add(path)

        if not is_git_sha1(row["git_blob_sha1"]):
            raise ValueError(f"{path}: invalid Git blob SHA-1")
        mode = row["mode"]
        if not isinstance(mode, str) or mode not in ALLOWED_FILE_MODES:
            raise ValueError(f"{path}: unsupported file mode {mode!r}")
        size = row["size"]
        if type(size) is not int or size < 0:
            raise ValueError(f"{path}: invalid byte size")
        method = row["method"]
        if not isinstance(method, str) or method not in METHODS:
            raise ValueError(f"{path}: unsupported transfer method {method!r}")
        status = row["status"]
        if not isinstance(status, str) or status not in STATUSES:
            raise ValueError(f"{path}: unsupported transfer status {status!r}")

        expected = row["base64_expected_chars"]
        actual_expected = _encoded_length(size)
        if type(expected) is not int or expected != actual_expected:
            raise ValueError(
                f"{path}: base64 expected chars must be {actual_expected}"
            )
        received = row["base64_received_chars"]
        if type(received) is not int or not (0 <= received <= expected):
            raise ValueError(f"{path}: invalid base64 received offset")
        next_line = row["text_next_start_line"]
        if type(next_line) is not int or next_line < 1:
            raise ValueError(f"{path}: invalid next text line")

        if method == "base64":
            if next_line != 1:
                raise ValueError(f"{path}: base64 state cannot carry text-line progress")
            if status == "pending" and received != 0:
                raise ValueError(f"{path}: pending base64 state cannot have progress")
            if status == "receiving" and not (0 < received < expected):
                raise ValueError(f"{path}: receiving base64 state must have partial progress")
            if status == "verified" and received != expected:
                raise ValueError(f"{path}: verified base64 state must be complete")
        else:
            if received != 0:
                raise ValueError(f"{path}: text-line state cannot carry base64 progress")
            if status == "pending" and next_line != 1:
                raise ValueError(f"{path}: pending text state cannot have line progress")
            if status == "receiving" and next_line <= 1:
                raise ValueError(f"{path}: receiving text state must have line progress")


def init_state(
    manifest: SnapshotManifest,
    *,
    method: str = "base64",
    max_response_chars: int = DEFAULT_MAX_RESPONSE_CHARS,
    base64_chunk_chars: int = DEFAULT_BASE64_CHUNK_CHARS,
    line_window: int = DEFAULT_LINE_WINDOW,
    line_overlap: int = DEFAULT_LINE_OVERLAP,
) -> dict[str, Any]:
    if method not in METHODS:
        raise ValueError("method must be base64 or text-lines")
    for name, value in (
        ("max_response_chars", max_response_chars),
        ("base64_chunk_chars", base64_chunk_chars),
        ("line_window", line_window),
        ("line_overlap", line_overlap),
    ):
        if type(value) is not int:
            raise ValueError(f"{name} must be an integer")
    state = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "limits": {
            "max_response_chars": max_response_chars,
            "base64_chunk_chars": base64_chunk_chars,
            "line_window": line_window,
            "line_overlap": line_overlap,
        },
        "files": [
            {
                "path": row.path,
                "git_blob_sha1": row.git_blob_sha1,
                "mode": row.mode,
                "size": row.size,
                "method": method,
                "status": "pending",
                "base64_expected_chars": _encoded_length(row.size),
                "base64_received_chars": 0,
                "text_next_start_line": 1,
            }
            for row in manifest.files
        ],
    }
    validate_state(state)
    return state


def load_state(path: Path) -> dict[str, Any]:
    state = json.loads(path.read_text(encoding="utf-8"))
    validate_state(state)
    return state


def save_state(path: Path, state: dict[str, Any]) -> None:
    validate_state(state)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def row_for(state: dict[str, Any], relative: str) -> dict[str, Any]:
    validate_relative_path(relative)
    rows = [row for row in state["files"] if row["path"] == relative]
    if len(rows) != 1:
        raise ValueError(f"expected one transport row for {relative}")
    return rows[0]


def next_request(state: dict[str, Any]) -> dict[str, Any] | None:
    validate_state(state)
    limits = state["limits"]
    for row in state["files"]:
        if row["status"] == "verified":
            continue
        if row["method"] == "base64":
            start = row["base64_received_chars"]
            end = min(
                row["base64_expected_chars"],
                start + limits["base64_chunk_chars"],
            )
            return {
                "path": row["path"],
                "method": "base64",
                "slice_start": start,
                "slice_end": end,
                "max_response_chars": limits["max_response_chars"],
            }
        next_line = row["text_next_start_line"]
        overlap = min(limits["line_overlap"], next_line - 1)
        start = next_line - overlap
        return {
            "path": row["path"],
            "method": "text-lines",
            "start_line": start,
            "end_line": start + limits["line_window"] - 1,
            "overlap_lines": overlap,
            "max_response_chars": limits["max_response_chars"],
        }
    return None


def reconcile_existing(
    state: dict[str, Any],
    *,
    root: Path,
) -> list[dict[str, str]]:
    validate_state(state)
    issues: list[dict[str, str]] = []
    for row in state["files"]:
        path = root / Path(*validate_relative_path(row["path"]).parts)
        if not path.exists():
            if row["status"] == "verified":
                issues.append({
                    "path": row["path"],
                    "problem": "verified file is absent",
                })
                _reset_progress(row)
            continue

        problems: list[str] = []
        if path.is_symlink() or not path.is_file():
            problems.append("not an ordinary file")
        else:
            data = path.read_bytes()
            if len(data) != row["size"]:
                problems.append("size mismatch")
            if git_blob_id(data) != row["git_blob_sha1"]:
                problems.append("blob mismatch")
            if local_file_mode(path) != row["mode"]:
                problems.append("mode mismatch")
        if problems:
            issues.append({"path": row["path"], "problem": "; ".join(problems)})
            if row["status"] == "verified":
                _reset_progress(row)
        else:
            row["status"] = "verified"
            row["base64_received_chars"] = (
                row["base64_expected_chars"] if row["method"] == "base64" else 0
            )
            row["text_next_start_line"] = 1

    validate_state(state)
    return issues
