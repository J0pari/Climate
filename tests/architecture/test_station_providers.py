from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from architecture import check_station_providers


def authority(source_id: str = "x") -> dict:
    return {
        "source_id": source_id,
        "access": "HTTPS provider archive",
        "license_or_access_constraints": "public provider data",
        "update_semantics": "mutable provider feed; capture and digest artifacts",
    }


def provider(**overrides) -> dict:
    value = {
        "provider_id": "x",
        "source_id": "x",
        "status": "planned",
        "access_cost": "no_fee",
        "access": "HTTPS provider archive",
        "license_or_access_constraints": "public provider data",
        "update_semantics": "mutable provider feed; capture and digest artifacts",
        "revision_identity": "provider revision plus captured artifact digest",
        "discovery_mode": "provider_catalog",
    }
    value.update(overrides)
    return value


class StationProviderRegistryTests(unittest.TestCase):
    def test_current_registry_is_clean(self):
        self.assertEqual(check_station_providers.check(), [])

    def test_paid_provider_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "architecture").mkdir()
            (root / "architecture/data_authorities.json").write_text(
                json.dumps({"sources": [authority()]}), encoding="utf-8")
            (root / "adapter.py").write_text("def boundary(): pass\n", encoding="utf-8")
            (root / "architecture/station_providers.json").write_text(json.dumps({
                "schema_version": 1,
                "target_population": "every station worldwide without paid data access",
                "providers": [provider(
                    status="current",
                    access_cost="paid",
                    current_boundaries=["adapter.py::boundary"],
                )]
            }), encoding="utf-8")
            codes = {item.code for item in check_station_providers.check(root)}
            self.assertIn("station_providers.cost", codes)

    def test_provider_authority_semantics_are_required_and_cannot_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "architecture").mkdir()
            (root / "architecture/data_authorities.json").write_text(
                json.dumps({"sources": [authority()]}), encoding="utf-8"
            )
            (root / "architecture/station_providers.json").write_text(
                json.dumps({
                    "schema_version": 1,
                    "target_population": "every station worldwide without paid data access",
                    "providers": [provider(
                        license_or_access_constraints="different terms",
                        revision_identity="",
                    )],
                }),
                encoding="utf-8",
            )
            codes = {item.code for item in check_station_providers.check(root)}
            self.assertIn(
                "station_providers.license_or_access_constraints_drift", codes
            )
            self.assertIn("station_providers.revision_identity", codes)

    def test_provider_authority_semantics_match_resolved_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "architecture").mkdir()
            (root / "architecture/data_authorities.json").write_text(
                json.dumps({"sources": [authority()]}), encoding="utf-8"
            )
            (root / "architecture/station_providers.json").write_text(
                json.dumps({
                    "schema_version": 1,
                    "target_population": "every station worldwide without paid data access",
                    "providers": [provider()],
                }),
                encoding="utf-8",
            )
            self.assertEqual(check_station_providers.check(root), [])

    def test_fixed_station_list_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "architecture").mkdir()
            (root / "architecture/data_authorities.json").write_text(
                json.dumps({"sources": [authority()]}), encoding="utf-8")
            (root / "architecture/station_providers.json").write_text(json.dumps({
                "schema_version": 1,
                "target_population": "every station worldwide without paid data access",
                "providers": [provider(station_ids=["toy"])]
            }), encoding="utf-8")
            codes = {item.code for item in check_station_providers.check(root)}
            self.assertIn("station_providers.fixed_station_list", codes)


if __name__ == "__main__":
    unittest.main()
