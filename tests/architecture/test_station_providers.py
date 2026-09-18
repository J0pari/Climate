from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from architecture import check_station_providers


class StationProviderRegistryTests(unittest.TestCase):
    def test_current_registry_is_clean(self):
        self.assertEqual(check_station_providers.check(), [])

    def test_paid_provider_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "architecture").mkdir()
            (root / "architecture/data_authorities.json").write_text(
                json.dumps({"sources": [{"source_id": "x"}]}), encoding="utf-8")
            (root / "adapter.py").write_text("def boundary(): pass\n", encoding="utf-8")
            (root / "architecture/station_providers.json").write_text(json.dumps({
                "schema_version": 1,
                "target_population": "every station worldwide without paid data access",
                "providers": [{
                    "provider_id": "x",
                    "source_id": "x",
                    "status": "current",
                    "access_cost": "paid",
                    "discovery_mode": "provider_catalog",
                    "current_boundaries": ["adapter.py::boundary"]
                }]
            }), encoding="utf-8")
            codes = {item.code for item in check_station_providers.check(root)}
            self.assertIn("station_providers.cost", codes)

    def test_fixed_station_list_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "architecture").mkdir()
            (root / "architecture/data_authorities.json").write_text(
                json.dumps({"sources": [{"source_id": "x"}]}), encoding="utf-8")
            (root / "architecture/station_providers.json").write_text(json.dumps({
                "schema_version": 1,
                "target_population": "every station worldwide without paid data access",
                "providers": [{
                    "provider_id": "x",
                    "source_id": "x",
                    "status": "planned",
                    "access_cost": "no_fee",
                    "discovery_mode": "provider_catalog",
                    "station_ids": ["toy"]
                }]
            }), encoding="utf-8")
            codes = {item.code for item in check_station_providers.check(root)}
            self.assertIn("station_providers.fixed_station_list", codes)


if __name__ == "__main__":
    unittest.main()
