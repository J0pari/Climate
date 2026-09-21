"""Line-window fallback acceptance for bounded snapshot transport."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .transfer_files import finish_file, part_path
from .transfer_state import row_for


def _existing_part(part: Path, *, relative: str, next_line: int) -> str:
    if not part.exists():
        if next_line != 1:
            raise ValueError(
                f"{relative}: text state expects prior lines but persisted partial is missing; reset required"
            )
        return ""
    value = part.read_text(encoding="utf-8")
    if value and not value.endswith("\n"):
        raise ValueError(
            f"{relative}: persisted text partial ended mid-line; reset required"
        )
    lines = value.splitlines(keepends=True)
    if len(lines) != next_line - 1:
        raise ValueError(
            f"{relative}: persisted text partial has {len(lines)} lines, "
            f"state expects {next_line - 1}; reset required"
        )
    return value


def accept_window(
    state: dict[str, Any],
    *,
    relative: str,
    start_line: int,
    text: str,
    eof: bool,
    root: Path,
    work_dir: Path,
) -> None:
    row = row_for(state, relative)
    if row["method"] != "text-lines":
        raise ValueError(
            f"{relative}: state method is {row['method']!r}, not text-lines"
        )
    if row["status"] == "verified":
        raise ValueError(f"{relative}: file is already verified")
    if len(text) > int(state["limits"]["max_response_chars"]):
        raise ValueError("text transport window exceeds configured response cap")
    if not eof and not text:
        raise ValueError("non-EOF text window must make progress")
    if not eof and not text.endswith("\n"):
        raise ValueError("non-EOF text window ended mid-line")

    lines = text.splitlines(keepends=True)
    if len(lines) > int(state["limits"]["line_window"]):
        raise ValueError("text transport window exceeds configured line bound")

    next_line = int(row["text_next_start_line"])
    overlap = min(
        int(state["limits"]["line_overlap"]),
        next_line - 1,
    )
    expected_start = next_line - overlap
    if start_line != expected_start:
        raise ValueError(
            f"{relative}: expected line window at {expected_start}, got {start_line}"
        )

    part = part_path(work_dir, relative, ".textpart")
    existing = _existing_part(part, relative=relative, next_line=next_line)
    previous = existing.splitlines(keepends=True)
    if overlap:
        if lines[:overlap] != previous[-overlap:]:
            raise ValueError(f"{relative}: line overlap mismatch")
        new_lines = lines[overlap:]
    else:
        new_lines = lines

    if not eof and not new_lines:
        raise ValueError("non-EOF text window must add at least one new line")

    candidate = existing + "".join(new_lines)
    if eof:
        # Validate and materialize the complete candidate before mutating the
        # persisted partial or advancing line state.
        finish_file(row, candidate.encode("utf-8"), root)
        row["text_next_start_line"] = next_line + len(new_lines)
        if part.exists():
            part.unlink()
        return

    tmp = part.with_suffix(part.suffix + ".tmp")
    tmp.write_text(candidate, encoding="utf-8")
    os.replace(tmp, part)
    row["text_next_start_line"] = next_line + len(new_lines)
    row["status"] = "receiving"
