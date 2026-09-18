"""Fail-closed Climate view of the Commons-owned gpu-scheduler/v1.

This module verifies ownership and ABI compatibility and exposes read-only
control-plane status/inspection. It intentionally does not submit Climate's
current CPU experiment runtime through a GPU-only scheduler contract.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
PIN = ROOT / "contracts" / "gpu-scheduler-pin.json"
ABI_KEYS = ("schema", "contractVersion", "compatibility", "public",
            "types", "endpoints", "gpu_lock")


class CommonsControlError(RuntimeError):
    pass


def load_pin() -> dict[str, Any]:
    return json.loads(PIN.read_text(encoding="utf-8"))


def abi_fingerprint(contract: Mapping[str, Any]) -> str:
    abi = {key: contract.get(key) for key in ABI_KEYS if key in contract}
    canonical = json.dumps(abi, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def commons_root(env: Mapping[str, str] | None = None) -> Path:
    values = os.environ if env is None else env
    configured = values.get("COMMONS_ROOT")
    if configured:
        path = Path(configured).expanduser().resolve()
        if path.is_dir():
            return path
        raise CommonsControlError(
            f"COMMONS_ROOT does not name a directory: {path}")
    for sibling in (ROOT.parent / "Commons", ROOT.parent / "commons"):
        if sibling.is_dir():
            return sibling.resolve()
    raise CommonsControlError(
        "Commons checkout unavailable; set COMMONS_ROOT explicitly")


def scheduler_path(env: Mapping[str, str] | None = None) -> Path:
    path = commons_root(env) / "control" / "gpu_scheduler.py"
    if not path.is_file():
        raise CommonsControlError(
            f"Commons scheduler entrypoint is unavailable: {path}")
    return path


def _invoke(args: list[str], env: Mapping[str, str] | None = None) -> dict[str, Any]:
    process = subprocess.run(
        [sys.executable, str(scheduler_path(env)), *args],
        cwd=str(commons_root(env)),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if process.returncode != 0:
        detail = (process.stderr or process.stdout).strip()
        raise CommonsControlError(
            f"Commons scheduler refused with exit {process.returncode}: {detail}")
    try:
        return json.loads(process.stdout)
    except json.JSONDecodeError as exc:
        raise CommonsControlError(
            "Commons scheduler returned non-JSON output") from exc


def verify_scheduler_contract(
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    contract = _invoke(["contract", "--json"], env)
    pin = load_pin()
    if contract.get("schema") != pin["schema"]:
        raise CommonsControlError(
            f"scheduler schema {contract.get('schema')!r} != pinned {pin['schema']!r}")
    if contract.get("owner") != pin["owner"]:
        raise CommonsControlError(
            f"scheduler owner {contract.get('owner')!r} != pinned {pin['owner']!r}")
    actual = abi_fingerprint(contract)
    if actual != pin["fingerprint"]:
        raise CommonsControlError(
            f"scheduler fingerprint drift: {actual} != {pin['fingerprint']}")
    return contract


def scheduler_status(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    verify_scheduler_contract(env)
    return _invoke(["status"], env)


def inspect_job(
    job_id: str,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    verify_scheduler_contract(env)
    return _invoke(["inspect", "--job", job_id], env)
