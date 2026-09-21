"""Filesystem operations shared by snapshot transport implementations."""
from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any

from .git_objects import git_blob_id
from .transfer_state import row_for


def part_path(work_dir: Path, relative: str, suffix: str) -> Path:
    path = work_dir / "parts" / Path(*PurePosixPath(relative).parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_name(path.name + suffix)


def finish_file(row: dict[str, Any], data: bytes, root: Path) -> None:
    if len(data) != int(row["size"]):
        raise ValueError(f"{row['path']}: size mismatch")
    blob = git_blob_id(data)
    if blob != row["git_blob_sha1"]:
        raise ValueError(f"{row['path']}: Git blob {blob} != expected {row['git_blob_sha1']}")
    target = root / Path(*PurePosixPath(row["path"]).parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    target.chmod(0o755 if row["mode"] == "100755" else 0o644)
    row["status"] = "verified"


def reset_file(state: dict[str, Any], *, relative: str, work_dir: Path) -> None:
    row = row_for(state, relative)
    for suffix in (".b64part", ".textpart"):
        part = part_path(work_dir, relative, suffix)
        if part.exists():
            part.unlink()
    row.update({"status": "pending", "base64_received_chars": 0, "text_next_start_line": 1})
