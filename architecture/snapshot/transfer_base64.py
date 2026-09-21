"""Binary-safe base64 chunk acceptance for bounded snapshot transport."""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from .transfer_files import finish_file, part_path
from .transfer_state import row_for


def accept_chunk(
    state: dict[str, Any], *, relative: str, offset: int, chunk: str, root: Path, work_dir: Path
) -> None:
    row = row_for(state, relative)
    expected = int(row["base64_received_chars"])
    if offset != expected:
        raise ValueError(f"{relative}: expected base64 offset {expected}, got {offset}")
    if any(ch.isspace() for ch in chunk):
        raise ValueError("base64 transport chunks must not contain whitespace")
    if len(chunk) > int(state["limits"]["base64_chunk_chars"]):
        raise ValueError("base64 transport chunk exceeds configured bound")

    part = part_path(work_dir, relative, ".b64part")
    with part.open("a", encoding="ascii") as handle:
        handle.write(chunk)
    received = expected + len(chunk)
    row["base64_received_chars"] = received
    row["status"] = "receiving"

    total = int(row["base64_expected_chars"])
    if received > total:
        raise ValueError(f"{relative}: received too much base64 data")
    if received == total:
        finish_file(row, base64.b64decode(part.read_text(encoding="ascii"), validate=True), root)
        part.unlink()
