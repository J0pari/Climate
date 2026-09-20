from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from architecture import finite_resources


NOW = datetime(2026, 9, 20, 12, 0, tzinfo=timezone.utc)


def ledger(
    *,
    enabled=True,
    core_limit=120.0,
    storage_limit=20.0,
    authorizations=None,
):
    return {
        "schema_version": 1,
        "resources": {
            "github_actions": {
                "enabled": False,
                "accounting_unit": "runner_minutes",
                "self_hard_limit": 0,
                "note": "disabled",
            },
            "github_codespaces": {
                "enabled": enabled,
                "accounting_scope": "Climate-initiated/self-use only",
                "accounting_policy": "full reservation charged",
                "budget_period": (
                    {
                        "period_id": "self-budget-2026-09",
                        "starts_at": (NOW - timedelta(days=1)).isoformat(),
                        "ends_at": (NOW + timedelta(days=1)).isoformat(),
                    }
                    if enabled
                    else None
                ),
                "self_hard_limits": {
                    "core_minutes": core_limit if enabled else 0,
                    "storage_gib_hours": storage_limit if enabled else 0,
                },
                "authorizations": authorizations or [],
            },
        },
    }


def authorized(auth_id="auth-1", campaign_id="campaign-1"):
    return {
        "authorization_id": auth_id,
        "campaign_id": campaign_id,
        "authorized_at": NOW.isoformat(),
        "billable_cores": 2,
        "max_wall_minutes": 20,
        "reserved_core_minutes": 40,
        "reserved_storage_gib_hours": 2,
        "status": "authorized",
    }


class FiniteResourceLedgerTests(unittest.TestCase):
    def write(self, root: Path, payload: dict) -> Path:
        path = root / "finite_resource_ledger.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_disabled_default_has_zero_self_budget(self):
        payload = ledger(enabled=False)
        finite_resources.validate_ledger(payload)
        accounting = finite_resources.codespaces_accounting(payload)
        self.assertEqual(accounting["remaining_core_minutes"], 0)
        self.assertEqual(accounting["remaining_storage_gib_hours"], 0)

    def test_actions_cannot_be_reenabled_or_given_budget(self):
        payload = ledger(enabled=False)
        payload["resources"]["github_actions"]["enabled"] = True
        with self.assertRaisesRegex(
            finite_resources.FiniteResourceError, "must remain disabled"
        ):
            finite_resources.validate_ledger(payload)

    def test_codespaces_reservation_is_charged_before_execution(self):
        payload = ledger(authorizations=[authorized()])
        finite_resources.validate_ledger(payload)
        accounting = finite_resources.codespaces_accounting(payload)
        self.assertEqual(accounting["reserved_core_minutes"], 40)
        self.assertEqual(accounting["remaining_core_minutes"], 80)

    def test_codespaces_overcommit_fails_closed(self):
        payload = ledger(core_limit=30, authorizations=[authorized()])
        with self.assertRaisesRegex(
            finite_resources.FiniteResourceError, "overcommitted"
        ):
            finite_resources.validate_ledger(payload)

    def test_only_one_outstanding_authorization_is_allowed(self):
        payload = ledger(
            authorizations=[
                authorized("auth-1", "campaign-1"),
                authorized("auth-2", "campaign-2"),
            ]
        )
        with self.assertRaisesRegex(
            finite_resources.FiniteResourceError, "only one"
        ):
            finite_resources.validate_ledger(payload)

    def test_completed_campaign_charges_full_reservation_not_actual(self):
        item = authorized()
        item.update(
            {
                "status": "completed",
                "actual_wall_seconds": 301.0,
                "actual_core_minutes_ceiling": 12,
                "charged_core_minutes": 40,
                "charged_storage_gib_hours": 2,
                "resource_receipt_path": (
                    "evaluations/resource-receipts/campaign-1.json"
                ),
            }
        )
        payload = ledger(authorizations=[item])
        finite_resources.validate_ledger(payload)
        accounting = finite_resources.codespaces_accounting(payload)
        self.assertEqual(accounting["completed_core_minutes"], 40)
        self.assertEqual(accounting["remaining_core_minutes"], 80)

    def test_claim_and_session_are_bound_to_one_codespace(self):
        payload = ledger(authorizations=[authorized()])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger_path = self.write(root, payload)
            claim_root = root / "claims"
            with (
                patch.object(finite_resources, "LEDGER_PATH", ledger_path),
                patch.object(finite_resources, "CLAIM_ROOT", claim_root),
                patch.object(
                    finite_resources.shutil, "disk_usage"
                ) as disk_usage,
            ):
                disk_usage.return_value = type(
                    "DU", (), {"used": 3 * 1024**3}
                )()
                claim = finite_resources.claim_codespaces(
                    "auth-1",
                    codespace_name="space-a",
                    detected_logical_cpus=2,
                    now=NOW,
                )
                self.assertEqual(claim["campaign_id"], "campaign-1")
                session = finite_resources.require_codespaces_session(
                    "auth-1",
                    "campaign-1",
                    codespace_name="space-a",
                    now=NOW + timedelta(minutes=1),
                )
                self.assertGreater(
                    finite_resources.remaining_seconds(
                        session, now=NOW + timedelta(minutes=1)
                    ),
                    0,
                )
                with self.assertRaisesRegex(
                    finite_resources.FiniteResourceError,
                    "another Codespace",
                ):
                    finite_resources.require_codespaces_session(
                        "auth-1",
                        "campaign-1",
                        codespace_name="space-b",
                        now=NOW + timedelta(minutes=1),
                    )


if __name__ == "__main__":
    unittest.main()
