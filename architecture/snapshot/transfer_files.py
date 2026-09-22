"""Filesystem operations shared by snapshot transport implementations."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .git_objects import git_blob_id, validate_relative_path
from .transfer_paths import discard_partials, part_path
from .transfer_state import row_for


def finish_file(row: dict[str, Any], data: bytes, root: Path) -> None:
    if len(data) != int(row["size"]):
        raise ValueError(f"{row['path']}: size mismatch")
    blob = git_blob_id(data)
    if blob != row["git_blob_sha1"]:
        raise ValueError(f"{row['path']}: Git blob {blob} != expected {row['git_blob_sha1']}")
    target = root / Path(*validate_relative_path(row["path"]).parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    target.chmod(0o755 if row["mode"] == "100755" else 0o644)
    row["status"] = "verified"


def reset_file(state: dict[str, Any], *, relative: str, work_dir: Path) -> None:
    row = row_for(state, relative)
    discard_partials(work_dir, relative)
    row.update({"status": "pending", "base64_received_chars": 0, "text_next_start_line": 1})
