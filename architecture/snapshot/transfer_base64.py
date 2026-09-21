"""Binary-safe base64 chunk acceptance for bounded snapshot transport."""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from .transfer_files import finish_file, part_path
from .transfer_state import row_for


def _existing_part(part: Path) -> str:
    if not part.exists():
        return ""
    value = part.read_text(encoding="ascii")
    if any(ch.isspace() for ch in value):
        raise ValueError(f"{part}: persisted base64 partial contains whitespace")
    return value


def accept_chunk(
    state: dict[str, Any], *, relative: str, offset: int, chunk: str, root: Path, work_dir: Path
) -> None:
    """Accept one contiguous base64 slice without mutating state before validation.

    The persisted partial length is itself an invariant: state may never advance beyond
    bytes that actually exist in the partial file.  The final chunk is validated as a
    complete Git blob before either the partial file or the state is advanced.
    """
    row = row_for(state, relative)
    expected = int(row["base64_received_chars"])
    if offset != expected:
        raise ValueError(f"{relative}: expected base64 offset {expected}, got {offset}")
    if any(ch.isspace() for ch in chunk):
        raise ValueError("base64 transport chunks must not contain whitespace")
    if len(chunk) > int(state["limits"]["base64_chunk_chars"]):
        raise ValueError("base64 transport chunk exceeds configured bound")

    part = part_path(work_dir, relative, ".b64part")
    existing = _existing_part(part)
    if len(existing) != expected:
        raise ValueError(
            f"{relative}: persisted base64 partial has {len(existing)} chars, state expects {expected}; reset required"
        )

    received = expected + len(chunk)
    total = int(row["base64_expected_chars"])
    if received > total:
        raise ValueError(f"{relative}: received too much base64 data")

    if received == total:
        # Validate the complete candidate before mutating either the persisted partial
        # or the state. finish_file itself validates byte size and Git blob identity.
        data = base64.b64decode(existing + chunk, validate=True)
        finish_file(row, data, root)
        row["base64_received_chars"] = received
        if part.exists():
            part.unlink()
        return

    with part.open("a", encoding="ascii") as handle:
        handle.write(chunk)
    row["base64_received_chars"] = received
    row["status"] = "receiving"
