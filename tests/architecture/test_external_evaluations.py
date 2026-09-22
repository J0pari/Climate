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
        "adapter": {
            "interface": "climate.external-model-runtime/v1",
            "status": "unavailable",
            "native_capability": "fixture producer-owned inference",
            "climate_semantic_gap": "fixture Climate-owned evaluation semantics",
            "required_capabilities": ["bind_subject_digest"],
        },
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

    def test_non_external_claim_evaluation_does_not_require_task_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evaluations = root / "evaluations"
            evaluations.mkdir(parents=True)
            (evaluations / "claim.json").write_text(
                json.dumps({
                    "evaluation_id": "claim.eval.v1",
                    "claim_id": "claim.one",
                    "experiment_id": "experiment.one",
                }),
                encoding="utf-8",
            )
            self.assertEqual(check_external_evaluations.check(root), [])

    def test_duplicate_id_is_global_across_evaluation_families(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_spec(root, evaluation_id="shared.eval.v1")
            (root / "evaluations" / "claim.json").write_text(
                json.dumps({
                    "evaluation_id": "shared.eval.v1",
                    "claim_id": "claim.one",
                    "experiment_id": "experiment.one",
                }),
                encoding="utf-8",
            )
            codes = {f.code for f in check_external_evaluations.check(root)}
            self.assertIn("external_evaluations.duplicate_id", codes)

    def test_task_digest_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_spec(root, digest="sha256:" + "0" * 64)
            codes = {f.code for f in check_external_evaluations.check(root)}
            self.assertIn(
                "external_evaluations.task_set_digest_mismatch", codes)



    def test_external_adapter_requires_native_capability_and_climate_gap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_spec(root)
            path = root / "evaluations" / "eval.one.v1.json"
            spec = json.loads(path.read_text(encoding="utf-8"))
            spec["adapter"].pop("native_capability")
            spec["adapter"]["climate_semantic_gap"] = ""
            path.write_text(json.dumps(spec), encoding="utf-8")
            codes = {f.code for f in check_external_evaluations.check(root)}
            self.assertIn("external_evaluations.native_capability_missing", codes)
            self.assertIn("external_evaluations.climate_semantic_gap_missing", codes)

    def test_external_artifact_evaluation_requires_adapter_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_spec(root)
            path = root / "evaluations" / "eval.one.v1.json"
            spec = json.loads(path.read_text(encoding="utf-8"))
            spec.pop("adapter")
            path.write_text(json.dumps(spec), encoding="utf-8")
            codes = {f.code for f in check_external_evaluations.check(root)}
            self.assertIn("external_evaluations.adapter_missing", codes)

    def test_native_output_import_requires_receipt_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_spec(root)
            path = root / "evaluations" / "eval.one.v1.json"
            spec = json.loads(path.read_text(encoding="utf-8"))
            spec["adapter"] = {
                "interface": "climate.external-model-runtime/v1",
                "status": "native_output_import",
                "native_capability": "fixture producer-owned inference",
                "climate_semantic_gap": "fixture Climate-owned evaluation semantics",
                "required_capabilities": ["bind_subject_digest"],
            }
            path.write_text(json.dumps(spec), encoding="utf-8")
            codes = {f.code for f in check_external_evaluations.check(root)}
            self.assertIn("external_evaluations.receipt_contract_missing", codes)
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
