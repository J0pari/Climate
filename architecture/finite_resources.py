#!/usr/bin/env python3
"""Hard limits and granular self-accounting for finite execution resources.

This authority tracks Climate-initiated resource use independently of provider
billing. Provider account usage may be recorded later for reconciliation, but it
is never required to know whether Climate itself has budget left: every
Codespaces campaign must be reserved in this ledger before the environment is
allowed to bootstrap, and the full reservation is charged conservatively.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LEDGER_PATH = ROOT / "architecture" / "finite_resource_ledger.json"
CLAIM_ROOT = ROOT / "run-artifacts" / "resource-claims"
ACTIONS_ID = "github_actions"
CODESPACES_ID = "github_codespaces"


class FiniteResourceError(RuntimeError):
    pass


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FiniteResourceError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result <= 0):
        adjective = "positive" if positive else "non-negative"
        raise FiniteResourceError(f"{name} must be finite and {adjective}")
    return result


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise FiniteResourceError(f"{name} must be a positive integer")
    return value


def _timestamp(value: Any, name: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise FiniteResourceError(f"{name} must be an ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise FiniteResourceError(f"{name} must be an ISO timestamp") from exc
    if parsed.tzinfo is None:
        raise FiniteResourceError(f"{name} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _safe_id(value: str, name: str) -> str:
    if not value or any(
        ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
        for ch in value
    ):
        raise FiniteResourceError(f"{name} contains unsafe characters")
    return value


def load_ledger(path: Path = LEDGER_PATH) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FiniteResourceError(f"cannot load finite-resource ledger: {exc}") from exc
    validate_ledger(payload)
    return payload


def codespaces_accounting(payload: dict[str, Any]) -> dict[str, float]:
    resource = payload["resources"][CODESPACES_ID]
    completed_core = 0.0
    completed_storage = 0.0
    reserved_core = 0.0
    reserved_storage = 0.0
    for item in resource["authorizations"]:
        status = item["status"]
        if status == "completed":
            completed_core += float(item["charged_core_minutes"])
            completed_storage += float(item["charged_storage_gib_hours"])
        elif status == "authorized":
            reserved_core += float(item["reserved_core_minutes"])
            reserved_storage += float(item["reserved_storage_gib_hours"])
    limits = resource["self_hard_limits"]
    return {
        "completed_core_minutes": completed_core,
        "completed_storage_gib_hours": completed_storage,
        "reserved_core_minutes": reserved_core,
        "reserved_storage_gib_hours": reserved_storage,
        "remaining_core_minutes": float(limits["core_minutes"])
        - completed_core
        - reserved_core,
        "remaining_storage_gib_hours": float(limits["storage_gib_hours"])
        - completed_storage
        - reserved_storage,
    }


def validate_ledger(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != 1:
        raise FiniteResourceError("finite-resource ledger schema_version must be 1")
    resources = payload.get("resources")
    if not isinstance(resources, dict):
        raise FiniteResourceError("finite-resource ledger requires resources")

    actions = resources.get(ACTIONS_ID)
    if not isinstance(actions, dict):
        raise FiniteResourceError("github_actions resource is required")
    if actions.get("enabled") is not False:
        raise FiniteResourceError("GitHub Actions must remain disabled")
    if _number(
        actions.get("self_hard_limit"), "github_actions.self_hard_limit"
    ) != 0:
        raise FiniteResourceError("GitHub Actions self_hard_limit must remain zero")

    codespaces = resources.get(CODESPACES_ID)
    if not isinstance(codespaces, dict):
        raise FiniteResourceError("github_codespaces resource is required")
    if not isinstance(codespaces.get("enabled"), bool):
        raise FiniteResourceError("github_codespaces.enabled must be boolean")
    limits = codespaces.get("self_hard_limits")
    if not isinstance(limits, dict):
        raise FiniteResourceError("github_codespaces.self_hard_limits is required")
    core_limit = _number(
        limits.get("core_minutes"), "Codespaces core-minute hard limit"
    )
    storage_limit = _number(
        limits.get("storage_gib_hours"), "Codespaces storage hard limit"
    )
    authorizations = codespaces.get("authorizations")
    if not isinstance(authorizations, list):
        raise FiniteResourceError("github_codespaces.authorizations must be a list")

    if codespaces["enabled"]:
        period = codespaces.get("budget_period")
        if not isinstance(period, dict):
            raise FiniteResourceError("enabled Codespaces requires a budget_period")
        period_id = period.get("period_id")
        if not isinstance(period_id, str) or not period_id:
            raise FiniteResourceError("Codespaces budget_period.period_id is required")
        start = _timestamp(period.get("starts_at"), "Codespaces budget start")
        end = _timestamp(period.get("ends_at"), "Codespaces budget end")
        if start >= end:
            raise FiniteResourceError(
                "Codespaces budget period requires starts_at < ends_at"
            )
        if core_limit <= 0 or storage_limit <= 0:
            raise FiniteResourceError(
                "enabled Codespaces requires positive self hard limits"
            )
    elif authorizations:
        raise FiniteResourceError("disabled Codespaces cannot contain authorizations")

    seen: set[str] = set()
    outstanding = 0
    for index, item in enumerate(authorizations):
        if not isinstance(item, dict):
            raise FiniteResourceError(f"authorization[{index}] must be an object")
        auth_id = item.get("authorization_id")
        campaign_id = item.get("campaign_id")
        if not isinstance(auth_id, str):
            raise FiniteResourceError(
                f"authorization[{index}].authorization_id is required"
            )
        _safe_id(auth_id, "authorization_id")
        if auth_id in seen:
            raise FiniteResourceError(f"duplicate authorization_id {auth_id!r}")
        seen.add(auth_id)
        if not isinstance(campaign_id, str):
            raise FiniteResourceError(
                f"authorization[{index}].campaign_id is required"
            )
        _safe_id(campaign_id, "campaign_id")
        status = item.get("status")
        if status not in {"authorized", "completed"}:
            raise FiniteResourceError(
                f"authorization {auth_id!r} has invalid status"
            )
        cores = _positive_int(
            item.get("billable_cores"), f"{auth_id}.billable_cores"
        )
        max_wall = _number(
            item.get("max_wall_minutes"),
            f"{auth_id}.max_wall_minutes",
            positive=True,
        )
        reserved_core = _number(
            item.get("reserved_core_minutes"),
            f"{auth_id}.reserved_core_minutes",
            positive=True,
        )
        minimum_core = math.ceil(max_wall) * cores
        if reserved_core < minimum_core:
            raise FiniteResourceError(
                f"authorization {auth_id!r} reserves {reserved_core:g} core-minutes "
                f"but its wall/core limit requires at least {minimum_core:g}"
            )
        _number(
            item.get("reserved_storage_gib_hours"),
            f"{auth_id}.reserved_storage_gib_hours",
        )
        if codespaces["enabled"]:
            period = codespaces["budget_period"]
            created = _timestamp(
                item.get("authorized_at"), f"{auth_id}.authorized_at"
            )
            start = _timestamp(period["starts_at"], "Codespaces budget start")
            end = _timestamp(period["ends_at"], "Codespaces budget end")
            if not (start <= created < end):
                raise FiniteResourceError(
                    f"authorization {auth_id!r} is outside its budget period"
                )
        if status == "authorized":
            outstanding += 1
            for forbidden in (
                "actual_wall_seconds",
                "actual_core_minutes_ceiling",
                "charged_core_minutes",
                "charged_storage_gib_hours",
                "resource_receipt_path",
            ):
                if forbidden in item:
                    raise FiniteResourceError(
                        f"authorized entry {auth_id!r} cannot predeclare {forbidden}"
                    )
        else:
            receipt = item.get("resource_receipt_path")
            if not isinstance(receipt, str) or not receipt:
                raise FiniteResourceError(
                    f"completed authorization {auth_id!r} requires "
                    "resource_receipt_path"
                )
            _number(
                item.get("actual_wall_seconds"), f"{auth_id}.actual_wall_seconds"
            )
            actual_core = _number(
                item.get("actual_core_minutes_ceiling"),
                f"{auth_id}.actual_core_minutes_ceiling",
            )
            charged_core = _number(
                item.get("charged_core_minutes"),
                f"{auth_id}.charged_core_minutes",
            )
            charged_storage = _number(
                item.get("charged_storage_gib_hours"),
                f"{auth_id}.charged_storage_gib_hours",
            )
            if actual_core > reserved_core:
                raise FiniteResourceError(
                    f"completed authorization {auth_id!r} exceeded its reservation"
                )
            if charged_core != reserved_core:
                raise FiniteResourceError(
                    f"completed authorization {auth_id!r} must charge its full "
                    "core reservation"
                )
            if charged_storage != float(item["reserved_storage_gib_hours"]):
                raise FiniteResourceError(
                    f"completed authorization {auth_id!r} must charge its full "
                    "storage reservation"
                )

    if outstanding > 1:
        raise FiniteResourceError(
            "only one unreconciled Codespaces authorization may exist at a time"
        )

    if codespaces["enabled"]:
        accounting = codespaces_accounting(payload)
        if accounting["remaining_core_minutes"] < 0:
            raise FiniteResourceError(
                "Codespaces self core-minute budget is overcommitted"
            )
        if accounting["remaining_storage_gib_hours"] < 0:
            raise FiniteResourceError(
                "Codespaces self storage budget is overcommitted"
            )


def codespaces_authorization(
    authorization_id: str,
    *,
    campaign_id: str | None = None,
    path: Path = LEDGER_PATH,
    now: datetime | None = None,
) -> dict[str, Any]:
    payload = load_ledger(path)
    resource = payload["resources"][CODESPACES_ID]
    if not resource["enabled"]:
        raise FiniteResourceError(
            "Codespaces self-budget is disabled; no Codespaces execution is authorized"
        )
    for item in resource["authorizations"]:
        if item["authorization_id"] != authorization_id:
            continue
        if item["status"] != "authorized":
            raise FiniteResourceError(
                f"Codespaces authorization {authorization_id!r} is not outstanding"
            )
        if campaign_id is not None and item["campaign_id"] != campaign_id:
            raise FiniteResourceError(
                "campaign_id does not match Codespaces authorization"
            )
        current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        period = resource["budget_period"]
        if not (
            _timestamp(period["starts_at"], "Codespaces budget start")
            <= current
            < _timestamp(period["ends_at"], "Codespaces budget end")
        ):
            raise FiniteResourceError(
                "Codespaces self-budget period is not currently active"
            )
        return dict(item)
    raise FiniteResourceError(
        f"unknown Codespaces authorization {authorization_id!r}"
    )


def _claim_path(authorization_id: str) -> Path:
    return CLAIM_ROOT / f"{_safe_id(authorization_id, 'authorization_id')}.json"


def claim_codespaces(
    authorization_id: str,
    *,
    codespace_name: str,
    detected_logical_cpus: int,
    now: datetime | None = None,
) -> dict[str, Any]:
    if not codespace_name:
        raise FiniteResourceError("CODESPACE_NAME is required")
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    item = codespaces_authorization(authorization_id, now=now)
    detected_logical_cpus = _positive_int(
        detected_logical_cpus, "detected_logical_cpus"
    )
    if detected_logical_cpus > int(item["billable_cores"]):
        raise FiniteResourceError(
            "Codespace exposes more logical CPUs than the pre-authorized "
            "billable core count"
        )
    path = _claim_path(authorization_id)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("codespace_name") != codespace_name:
            raise FiniteResourceError(
                "authorization is already claimed by another Codespace"
            )
        return existing
    CLAIM_ROOT.mkdir(parents=True, exist_ok=True)
    claim = {
        "schema_version": 1,
        "authorization_id": authorization_id,
        "campaign_id": item["campaign_id"],
        "codespace_name": codespace_name,
        "claimed_at": now.isoformat(),
        "billable_cores": item["billable_cores"],
        "detected_logical_cpus": detected_logical_cpus,
        "max_wall_minutes": item["max_wall_minutes"],
        "reserved_core_minutes": item["reserved_core_minutes"],
        "reserved_storage_gib_hours": item["reserved_storage_gib_hours"],
        "filesystem_used_gib_at_claim": (
            shutil.disk_usage(ROOT).used / float(1024 ** 3)
        ),
    }
    path.write_text(
        json.dumps(claim, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return claim


def require_codespaces_session(
    authorization_id: str,
    campaign_id: str,
    *,
    codespace_name: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    item = codespaces_authorization(
        authorization_id, campaign_id=campaign_id, now=now
    )
    path = _claim_path(authorization_id)
    if not path.is_file():
        raise FiniteResourceError(
            "Codespaces authorization has not been claimed by bootstrap"
        )
    claim = json.loads(path.read_text(encoding="utf-8"))
    if claim.get("codespace_name") != codespace_name:
        raise FiniteResourceError("Codespaces claim belongs to another Codespace")
    claimed = _timestamp(claim.get("claimed_at"), "Codespaces claim time")
    deadline = claimed + timedelta(minutes=float(item["max_wall_minutes"]))
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if current >= deadline:
        raise FiniteResourceError(
            "Codespaces authorization wall-time is exhausted"
        )
    return {**claim, "deadline": deadline.isoformat()}


def remaining_seconds(
    session: dict[str, Any], *, now: datetime | None = None
) -> float:
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    deadline = _timestamp(session.get("deadline"), "Codespaces deadline")
    remaining = (deadline - current).total_seconds()
    if remaining <= 0:
        raise FiniteResourceError(
            "Codespaces authorization wall-time is exhausted"
        )
    return remaining


def finish_codespaces_receipt(
    session: dict[str, Any],
    *,
    receipt_path: Path,
    campaign_status: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    claimed = _timestamp(session.get("claimed_at"), "Codespaces claim time")
    elapsed = max(0.0, (current - claimed).total_seconds())
    cores = int(session["billable_cores"])
    actual_core = math.ceil(elapsed / 60.0) * cores
    reserved_core = float(session["reserved_core_minutes"])
    if actual_core > reserved_core:
        raise FiniteResourceError(
            "actual core-minute ceiling exceeded the precommitted reservation"
        )
    payload = {
        "schema_version": 1,
        "resource_id": CODESPACES_ID,
        "authorization_id": session["authorization_id"],
        "campaign_id": session["campaign_id"],
        "codespace_name": session["codespace_name"],
        "claimed_at": session["claimed_at"],
        "finished_at": current.isoformat(),
        "campaign_status": campaign_status,
        "billable_cores": cores,
        "detected_logical_cpus": session["detected_logical_cpus"],
        "max_wall_minutes": session["max_wall_minutes"],
        "reserved_core_minutes": reserved_core,
        "reserved_storage_gib_hours": session["reserved_storage_gib_hours"],
        "actual_wall_seconds": elapsed,
        "actual_core_minutes_ceiling": actual_core,
        "self_budget_charged_core_minutes": reserved_core,
        "self_budget_charged_storage_gib_hours": (
            session["reserved_storage_gib_hours"]
        ),
        "filesystem_used_gib_at_claim": session[
            "filesystem_used_gib_at_claim"
        ],
        "filesystem_used_gib_at_finish": (
            shutil.disk_usage(ROOT).used / float(1024 ** 3)
        ),
        "accounting_note": (
            "The self-budget charges the full reservation. Actual measurements "
            "are audit data and do not refund budget automatically."
        ),
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="validate finite-resource self-accounting"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check")
    claim = sub.add_parser("claim-codespaces")
    claim.add_argument("--authorization-id", required=True)
    args = parser.parse_args()
    try:
        if args.command == "check":
            payload = load_ledger()
            accounting = codespaces_accounting(payload)
            state = (
                "enabled"
                if payload["resources"][CODESPACES_ID]["enabled"]
                else "disabled"
            )
            print(
                "finite resource ledger: integrity ok; "
                f"Codespaces={state}; remaining_core_minutes="
                f"{accounting['remaining_core_minutes']:g}; "
                "remaining_storage_gib_hours="
                f"{accounting['remaining_storage_gib_hours']:g}"
            )
            return 0
        if args.command == "claim-codespaces":
            if os.environ.get("CODESPACES", "").lower() != "true":
                raise FiniteResourceError(
                    "claim-codespaces is valid only in GitHub Codespaces"
                )
            claim_codespaces(
                args.authorization_id,
                codespace_name=os.environ.get("CODESPACE_NAME", ""),
                detected_logical_cpus=os.cpu_count() or 1,
            )
            print(f"Codespaces authorization {args.authorization_id} claimed")
            return 0
    except FiniteResourceError as exc:
        print(f"finite_resources: {exc}", file=os.sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
