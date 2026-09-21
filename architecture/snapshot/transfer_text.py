"""Line-window fallback acceptance for bounded snapshot transport."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .transfer_files import finish_file, part_path
from .transfer_state import row_for


def accept_window(
    state: dict[str, Any], *, relative: str, start_line: int, text: str, eof: bool, root: Path, work_dir: Path
) -> None:
    row = row_for(state, relative)
    if not eof and text and not text.endswith("\n"):
        raise ValueError("non-EOF text window ended mid-line")

    lines = text.splitlines(keepends=True)
    next_line = int(row["text_next_start_line"])
    overlap = int(state["limits"]["line_overlap"]) if next_line > 1 else 0
    expected_start = max(1, next_line - overlap)
    if start_line != expected_start:
        raise ValueError(f"{relative}: expected line window at {expected_start}, got {start_line}")

    part = part_path(work_dir, relative, ".textpart")
    if overlap:
        existing = part.read_text(encoding="utf-8") if part.exists() else ""
        previous = existing.splitlines(keepends=True)
        if lines[:overlap] != previous[-overlap:]:
            raise ValueError(f"{relative}: line overlap mismatch")
        lines = lines[overlap:]

    with part.open("a", encoding="utf-8") as handle:
        handle.write("".join(lines))
    row["text_next_start_line"] = next_line + len(lines)
    row["status"] = "receiving"
    if eof:
        finish_file(row, part.read_bytes(), root)
        part.unlink()
