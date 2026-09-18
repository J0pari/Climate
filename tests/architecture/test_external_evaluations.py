"""Negative witnesses for external-artifact evaluation identity."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from architecture import check_external_evaluations


def _write_spec(root: Path, *, evaluation_id: str = "eval.one.v1", digest: str | None = None) -> None:
    fixture = root / "fixtures" / "evaluation" / "tasks.json"
    fixture.parent.mkdir(parents=True, exist_ok=True)
    fixture.write_text('{"tasks": []}\n', encoding="utf-8")
    actual = "sha256:" + hashlib.sha256(fixture.read_bytes()).hexdigest()
    spec = {
        "evaluation_id": evaluation_id,
        "task_set": {
            "uri": "fixtures/evaluation/tasks.json",
            "digest": actual if digest is None else digest,
        },
    }
    evaluations = root / "evaluations"
    evaluations.mkdir(parents=True, exist_ok=True)
    (evaluations / (evaluation_id + ".json")).write_text(
        json.dumps(spec), encoding="utf-8")


class ExternalEvaluationIntegrityTests(unittest.TestCase):
    def test_matching_task_digest_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_spec(root)
            self.assertEqual(check_external_evaluations.check(root), [])

    def test_task_digest_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_spec(root, digest="sha256:" + "0" * 64)
            codes = {f.code for f in check_external_evaluations.check(root)}
            self.assertIn(
                "external_evaluations.task_set_digest_mismatch", codes)

    def test_duplicate_evaluation_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_spec(root, evaluation_id="eval.same.v1")
            first = root / "evaluations" / "eval.same.v1.json"
            second = root / "evaluations" / "copy.json"
            second.write_text(first.read_text(encoding="utf-8"), encoding="utf-8")
            codes = {f.code for f in check_external_evaluations.check(root)}
            self.assertIn("external_evaluations.duplicate_id", codes)


if __name__ == "__main__":
    unittest.main()
