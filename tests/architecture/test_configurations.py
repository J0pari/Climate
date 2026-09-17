from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from architecture import check_configurations


class ConfigurationAuthorityTests(unittest.TestCase):
    def _root(self):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        (root / "architecture").mkdir()
        (root / "configurations" / "kernel").mkdir(parents=True)
        (root / "architecture" / "data_authorities.json").write_text(
            json.dumps({
                "schema_version": 1,
                "sources": [{
                    "source_id": "paper.reference.v1",
                    "doi": "10.0000/example",
                    "official_url": "https://example.test/paper",
                }],
                "usages": [],
            }),
            encoding="utf-8",
        )
        return temporary, root

    def _write_configuration(self, root: Path, value: dict):
        (root / "configurations" / "kernel" / "test.v1.json").write_text(
            json.dumps(value),
            encoding="utf-8",
        )

    def test_nonlocal_provenance_requires_authority_reference(self):
        temporary, root = self._root()
        with temporary:
            self._write_configuration(root, {
                "provenance": "literature_fixed",
            })
            codes = {item.code for item in check_configurations.check(root)}
            self.assertIn("configuration.authority_missing", codes)

    def test_authority_source_must_resolve(self):
        temporary, root = self._root()
        with temporary:
            self._write_configuration(root, {
                "provenance": "literature_fixed",
                "authority_refs": [{
                    "source_id": "missing.source",
                    "identity": "10.0000/example",
                }],
            })
            codes = {item.code for item in check_configurations.check(root)}
            self.assertIn("configuration.authority_unresolved", codes)

    def test_authority_identity_must_match_registry(self):
        temporary, root = self._root()
        with temporary:
            self._write_configuration(root, {
                "provenance": "literature_fixed",
                "authority_refs": [{
                    "source_id": "paper.reference.v1",
                    "identity": "not-the-registered-paper",
                }],
            })
            codes = {item.code for item in check_configurations.check(root)}
            self.assertIn("configuration.authority_identity_mismatch", codes)

    def test_registered_doi_satisfies_nonlocal_provenance(self):
        temporary, root = self._root()
        with temporary:
            self._write_configuration(root, {
                "provenance": "literature_fixed",
                "authority_refs": [{
                    "source_id": "paper.reference.v1",
                    "identity": "10.0000/example",
                }],
            })
            self.assertEqual(check_configurations.check(root), [])

    def test_explicit_local_policy_does_not_require_external_authority(self):
        temporary, root = self._root()
        with temporary:
            self._write_configuration(root, {"provenance": "explicit"})
            self.assertEqual(check_configurations.check(root), [])

    def test_repository_configuration_authorities_resolve(self):
        self.assertEqual(check_configurations.check(), [])


if __name__ == "__main__":
    unittest.main()
