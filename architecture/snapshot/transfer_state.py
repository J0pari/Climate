"""Persistent state and request planning for bounded snapshot transport."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .git_objects import git_blob_id, local_file_mode, validate_relative_path
from .manifest import SnapshotManifest

PROTOCOL = "bounded-github-snapshot-transport/v1"
DEFAULT_MAX_RESPONSE_CHARS = 12_000
DEFAULT_BASE64_CHUNK_CHARS = 10_000
DEFAULT_LINE_WINDOW = 120
DEFAULT_LINE_OVERLAP = 1


def _encoded_length(size: int) -> int:
    return 4 * ((size + 2) // 3)


def init_state(
    manifest: SnapshotManifest,
    *,
    method: str = "base64",
    max_response_chars: int = DEFAULT_MAX_RESPONSE_CHARS,
    base64_chunk_chars: int = DEFAULT_BASE64_CHUNK_CHARS,
    line_window: int = DEFAULT_LINE_WINDOW,
    line_overlap: int = DEFAULT_LINE_OVERLAP,
) -> dict[str, Any]:
    if method not in {"base64", "text-lines"}:
        raise ValueError("method must be base64 or text-lines")
    if not (0 < base64_chunk_chars < max_response_chars):
        raise ValueError("base64 chunk size must be positive and below response cap")
    if line_window <= 1 or not (0 <= line_overlap < line_window):
        raise ValueError("invalid line window/overlap")
    return {
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


def load_state(path: Path) -> dict[str, Any]:
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("schema_version") != 1 or state.get("protocol") != PROTOCOL:
        raise ValueError("unsupported snapshot transport state")
    files = state.get("files")
    if not isinstance(files, list):
        raise ValueError("transport state files must be a list")
    for row in files:
        if not isinstance(row, dict):
            raise ValueError("transport state rows must be objects")
        validate_relative_path(str(row.get("path", "")))
    return state


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def row_for(state: dict[str, Any], relative: str) -> dict[str, Any]:
    rows = [row for row in state["files"] if row["path"] == relative]
    if len(rows) != 1:
        raise ValueError(f"expected one transport row for {relative}")
    return rows[0]


def next_request(state: dict[str, Any]) -> dict[str, Any] | None:
    limits = state["limits"]
    for row in state["files"]:
        if row["status"] == "verified":
            continue
        if row["method"] == "base64":
            start = int(row["base64_received_chars"])
            end = min(int(row["base64_expected_chars"]), start + int(limits["base64_chunk_chars"]))
            return {
                "path": row["path"],
                "method": "base64",
                "slice_start": start,
                "slice_end": end,
                "max_response_chars": limits["max_response_chars"],
            }
        next_line = int(row["text_next_start_line"])
        overlap = int(limits["line_overlap"]) if next_line > 1 else 0
        start = max(1, next_line - overlap)
        return {
            "path": row["path"],
            "method": "text-lines",
            "start_line": start,
            "end_line": start + int(limits["line_window"]) - 1,
            "overlap_lines": overlap,
            "max_response_chars": limits["max_response_chars"],
        }
    return None


def reconcile_existing(state: dict[str, Any], *, root: Path) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    for row in state["files"]:
        if row["status"] == "verified":
            continue
        path = root / Path(*validate_relative_path(row["path"]).parts)
        if not path.exists():
            continue
        problems: list[str] = []
        if path.is_symlink() or not path.is_file():
            problems.append("not an ordinary file")
        else:
            data = path.read_bytes()
            if len(data) != int(row["size"]):
                problems.append("size mismatch")
            if git_blob_id(data) != row["git_blob_sha1"]:
                problems.append("blob mismatch")
            if local_file_mode(path) != row["mode"]:
                problems.append("mode mismatch")
        if problems:
            issues.append({"path": row["path"], "problem": "; ".join(problems)})
        else:
            row["status"] = "verified"
            row["base64_received_chars"] = row["base64_expected_chars"]
    return issues
