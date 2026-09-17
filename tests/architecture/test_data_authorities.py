from __future__ import annotations

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from architecture import check_data_authorities


class DataAuthorityTests(unittest.TestCase):
    def _root_and_registry(self):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        (root / "src").mkdir()
        (root / "src" / "example.txt").write_text(
            "observed_temperature\nlocal_policy\n",
            encoding="utf-8",
        )
        registry = {
            "schema_version": 1,
            "authority": "test authority",
            "sources": [
                {
                    "source_id": "provider.dataset.v1",
                    "kind": "api_dataset",
                    "provider": "Provider",
                    "product": "Dataset",
                    "version": "v1",
                    "official_url": "https://example.test/product",
                    "access": "REST API",
                    "access_url": "https://example.test/api",
                    "time_coverage": "2000-present",
                    "variables": ["temperature"],
                    "units": ["K"],
                    "license_or_access_constraints": "test terms",
                    "update_semantics": "responses used as evidence are captured and digested",
                },
                {
                    "source_id": "paper.reference.v1",
                    "kind": "literature",
                    "provider": "Journal",
                    "product": "Published reference",
                    "version": "v1",
                    "official_url": "https://example.test/paper",
                    "access": "published article",
                    "update_semantics": "fixed publication",
                },
            ],
            "usages": [
                {
                    "usage_id": "observation.temperature",
                    "scope": "current",
                    "path": "src/example.txt",
                    "anchor": "observed_temperature",
                    "ownership_class": "external_api",
                    "disposition": "externalize",
                    "source_ids": ["provider.dataset.v1"],
                    "rationale": "measured temperature belongs to the provider",
                },
                {
                    "usage_id": "policy.local",
                    "scope": "current",
                    "path": "src/example.txt",
                    "anchor": "local_policy",
                    "ownership_class": "experiment_policy",
                    "disposition": "retain_local",
                    "source_ids": [],
                    "rationale": "policy is experiment-owned",
                },
            ],
        }
        return temporary, root, registry

    def test_valid_registry_passes(self):
        temporary, root, registry = self._root_and_registry()
        with temporary:
            self.assertEqual(check_data_authorities.check(root, registry), [])

    def test_duplicate_source_and_usage_ids_are_rejected(self):
        temporary, root, registry = self._root_and_registry()
        with temporary:
            registry["sources"].append(deepcopy(registry["sources"][0]))
            registry["usages"].append(deepcopy(registry["usages"][0]))
            codes = {item.code for item in check_data_authorities.check(root, registry)}
            self.assertIn("data_authority.source_duplicate", codes)
            self.assertIn("data_authority.usage_duplicate", codes)

    def test_missing_usage_anchor_is_rejected(self):
        temporary, root, registry = self._root_and_registry()
        with temporary:
            registry["usages"][0]["anchor"] = "invented_observation"
            codes = {item.code for item in check_data_authorities.check(root, registry)}
            self.assertIn("data_authority.anchor_missing", codes)

    def test_unresolved_external_source_is_rejected(self):
        temporary, root, registry = self._root_and_registry()
        with temporary:
            registry["usages"][0]["source_ids"] = ["missing.dataset"]
            codes = {item.code for item in check_data_authorities.check(root, registry)}
            self.assertIn("data_authority.source_ref_missing", codes)

    def test_literature_cannot_satisfy_external_dataset_authority(self):
        temporary, root, registry = self._root_and_registry()
        with temporary:
            registry["usages"][0]["ownership_class"] = "external_dataset"
            registry["usages"][0]["source_ids"] = ["paper.reference.v1"]
            codes = {item.code for item in check_data_authorities.check(root, registry)}
            self.assertIn("data_authority.external_source_kind", codes)

    def test_external_api_requires_api_capable_source(self):
        temporary, root, registry = self._root_and_registry()
        with temporary:
            registry["sources"][0]["kind"] = "dataset"
            codes = {item.code for item in check_data_authorities.check(root, registry)}
            self.assertIn("data_authority.api_source_missing", codes)

    def test_dataset_requires_access_and_update_metadata(self):
        temporary, root, registry = self._root_and_registry()
        with temporary:
            source = registry["sources"][0]
            del source["access_url"]
            source["update_semantics"] = ""
            codes = {item.code for item in check_data_authorities.check(root, registry)}
            self.assertIn("data_authority.access_url", codes)
            self.assertIn("data_authority.source_metadata", codes)

    def test_external_ownership_cannot_be_retained_as_local_truth(self):
        temporary, root, registry = self._root_and_registry()
        with temporary:
            registry["usages"][0]["disposition"] = "retain_local"
            codes = {item.code for item in check_data_authorities.check(root, registry)}
            self.assertIn("data_authority.external_disposition", codes)

    def test_real_registry_is_current(self):
        findings = check_data_authorities.check()
        self.assertEqual(findings, [], "\n".join(str(item) for item in findings))


if __name__ == "__main__":
    unittest.main()
