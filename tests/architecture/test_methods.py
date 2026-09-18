"""Negative witnesses for method identity integrity."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from architecture import check_methods


def method(method_id: str = "fixture.method", **overrides):
    record = {
        "method_id": method_id,
        "maturity": "runnable",
        "build_identity": {
            "policy": "source_digest_at_run",
            "sources": ["implementation.py"],
        },
        "reference_methods": [],
    }
    record.update(overrides)
    return record


class MethodIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "implementation.py").write_text("pass\n", encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def check(self, *records):
        return check_methods.check(
            self.root,
            {"schema_version": 2, "methods": list(records)},
        )

    def test_runnable_method_requires_build_identity(self):
        record = method()
        del record["build_identity"]
        findings = self.check(record)
        self.assertIn("methods.build_identity_missing", {f.code for f in findings})

    def test_stale_implementation_build_is_rejected(self):
        findings = self.check(method(implementation_build="deadbeef"))
        self.assertIn("methods.stale_build_field", {f.code for f in findings})

    def test_missing_source_is_rejected(self):
        findings = self.check(method(build_identity={
            "policy": "source_digest_at_run",
            "sources": ["missing.py"],
        }))
        self.assertIn("methods.build_source_missing", {f.code for f in findings})

    def test_reference_must_resolve(self):
        findings = self.check(method(reference_methods=["missing.method"]))
        self.assertIn("methods.reference_missing", {f.code for f in findings})

    def test_concept_method_may_be_unimplemented(self):
        findings = self.check(method(maturity="concept", build_identity=None))
        self.assertEqual(findings, [])

    def test_valid_source_bound_method_passes(self):
        self.assertEqual(self.check(method()), [])


if __name__ == "__main__":
    unittest.main()
