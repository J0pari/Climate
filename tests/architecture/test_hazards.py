from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from architecture.check_hazards import check, git_blob_sha


class HazardIntegrityTests(unittest.TestCase):
    def _source(self, root: Path, text: str = "fn prototype() {}\n") -> Path:
        path = root / "prototype.rs"
        path.write_text(text, encoding="utf-8")
        return path

    def _hazard(self, source: Path, **overrides):
        record = {
            "hazard_id": "prototype.fake_output",
            "path": source.name,
            "reviewed_source_blob": git_blob_sha(source),
            "class": "S4",
            "status": "open",
            "disposition": "fail_closed",
            "blocks_evidence": True,
            "reason": "planted fake-output witness",
        }
        record.update(overrides)
        return record

    def test_current_open_hazard_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self._source(root)
            findings = check(root, {"hazards": [self._hazard(source)]})
            self.assertEqual(findings, [])

    def test_source_change_makes_review_stale(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self._source(root)
            hazard = self._hazard(source)
            source.write_text("fn prototype() { fake_value(); }\n", encoding="utf-8")
            findings = check(root, {"hazards": [hazard]})
            self.assertIn("hazards.source_review_stale", {item.code for item in findings})

    def test_resolved_hazard_requires_resolution_witness(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self._source(root)
            hazard = self._hazard(source, status="resolved")
            findings = check(root, {"hazards": [hazard]})
            self.assertIn("hazards.resolution_witness_missing", {item.code for item in findings})

    def test_resolved_hazard_with_witness_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self._source(root)
            hazard = self._hazard(
                source,
                status="resolved",
                resolution_witness_ids=["test:prototype_rejects_fake_output"],
            )
            findings = check(root, {"hazards": [hazard]})
            self.assertEqual(findings, [])

    def test_duplicate_hazard_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self._source(root)
            hazard = self._hazard(source)
            findings = check(root, {"hazards": [hazard, dict(hazard)]})
            self.assertIn("hazards.id_duplicate", {item.code for item in findings})

    def test_missing_source_is_distinct_from_stale_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self._source(root)
            hazard = self._hazard(source)
            source.unlink()
            findings = check(root, {"hazards": [hazard]})
            codes = {item.code for item in findings}
            self.assertIn("hazards.source_missing", codes)
            self.assertNotIn("hazards.source_review_stale", codes)

    def test_invalid_blob_pin_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self._source(root)
            hazard = self._hazard(source, reviewed_source_blob="not-a-git-blob")
            findings = check(root, {"hazards": [hazard]})
            self.assertIn("hazards.reviewed_blob_invalid", {item.code for item in findings})

    def test_missing_blueprint_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self._source(root)
            hazard = self._hazard(source, blueprint="blueprints/missing.md")
            findings = check(root, {"hazards": [hazard]})
            self.assertIn("hazards.blueprint_missing", {item.code for item in findings})

    def test_real_registry_is_current(self):
        root = Path(__file__).resolve().parents[2]
        import json

        with (root / "architecture" / "semantic_hazards.json").open(encoding="utf-8") as fh:
            registry = json.load(fh)
        findings = check(root, registry)
        self.assertEqual(findings, [], "\n".join(f"{f.code}: {f.message}" for f in findings))


if __name__ == "__main__":
    unittest.main()
