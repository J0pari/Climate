"""Climate's fail-closed Commons control boundary."""
from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from architecture import commons_control


class _FakeControlClient:
    def __init__(self):
        self.calls = []
        self.job = {"jobId": "0123456789abcdef", "status": "queued",
                    "idempotent": False}

    def status(self, etag=None):
        self.calls.append(("status", {"etag": etag}))
        return {"readinessGate": "open"}

    def list_jobs(self, repo=None, status=None):
        self.calls.append(("jobs", {"repo": repo, "status": status}))
        return {"jobs": [], "count": 0}

    def inspect(self, job_id=None):
        self.calls.append(("inspect", {"jobId": job_id}))
        return {"jobId": job_id, "status": "done"}

    def cancel(self, job_id=None):
        self.calls.append(("cancel", {"jobId": job_id}))
        return {"cancelled": True, "jobId": job_id}

    def submit(self, **kwargs):
        self.calls.append(("submit", kwargs))
        return dict(self.job)

    def send(self, **kwargs):
        self.calls.append(("send", kwargs))
        return {"messageId": "0" * 16, "status": "pending",
                "idempotent": False}

    def inbox(self, **kwargs):
        self.calls.append(("inbox", kwargs))
        return {"messages": [], "cursor": None}

    def ack(self, **kwargs):
        self.calls.append(("ack", kwargs))
        return {"messageId": kwargs["message_id"], "status": "acked",
                "idempotent": False}


class CommonsControlTests(unittest.TestCase):
    def test_pin_and_interface_adopt_resource_generic_read_execution(self):
        root = Path(__file__).resolve().parents[2]
        pin = json.loads((root / "contracts/work-scheduler-pin.json").read_text())
        control_pin = json.loads(
            (root / "contracts/control-api-pin.json").read_text())
        interface = json.loads((root / "architecture/commons_interface.json").read_text())
        self.assertEqual(pin["owner"], "commons")
        self.assertEqual(pin["schema"], "work-scheduler/v1")
        self.assertEqual(control_pin["owner"], "commons")
        self.assertEqual(control_pin["schema"], "control-api/v1")
        self.assertEqual(interface["supported_control_level"], "read")
        self.assertEqual(interface["scheduler_contract"]["pin_path"],
                         "contracts/work-scheduler-pin.json")
        self.assertEqual(interface["control_api"]["pin_path"],
                         "contracts/control-api-pin.json")
        self.assertEqual(interface["control_api"]["client"],
                         "control/client.py")

    def test_abi_fingerprint_ignores_administration_but_not_resource_semantics(self):
        base = {
            "schema": "work-scheduler/v1",
            "contractVersion": "1",
            "compatibility": {"additive": "x"},
            "public": {"commands": []},
            "types": {"A": {"required": ["x"], "optional": [], "docs": "x"}},
            "endpoints": {},
            "resource_semantics": {"cpu": "zero gpu"},
            "owner": "commons",
            "consumers": {"climate": {}},
        }
        first = commons_control.abi_fingerprint(base)
        admin = copy.deepcopy(base)
        admin["consumers"]["other"] = {}
        self.assertEqual(first, commons_control.abi_fingerprint(admin))
        changed = copy.deepcopy(base)
        changed["resource_semantics"]["cpu"] = "pretend gpu"
        self.assertNotEqual(first, commons_control.abi_fingerprint(changed))

    def test_missing_commons_checkout_refuses_instead_of_substituting(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(commons_control.CommonsControlError):
                commons_control.commons_root({"COMMONS_ROOT": str(Path(td) / "missing")})

    def test_cpu_experiment_submission_preserves_climate_runtime_identity(self):
        root = Path(__file__).resolve().parents[2]
        experiment = root / "experiments" / "multirepresentation-ebm-dynamics.v1.json"
        client = _FakeControlClient()
        with patch.object(commons_control, "verify_contracts", return_value={}), \
             patch.object(commons_control, "control_client", return_value=client):
            out = commons_control.submit_cpu_experiment(
                experiment_path=experiment,
                repository_revision="a" * 40,
                run_scope="commons-fixture",
                ram_mib=1024,
            )
        kind, submit = client.calls[-1]
        self.assertEqual(kind, "submit")
        self.assertEqual(submit["repo"], "climate")
        self.assertEqual(submit["resourceClass"], "cpu")
        self.assertNotIn("vramMib", submit)
        self.assertEqual(submit["ramMib"], 1024)
        self.assertEqual(submit["cwd"], str(root))
        self.assertIn(str(root / "src" / "experiment_runtime.py"), submit["command"])
        self.assertEqual(submit["idempotency_key"], "climate-commons-fixture")
        self.assertEqual(out["outputDir"], "run-artifacts/commons/commons-fixture")
        self.assertEqual(out["idempotencyKey"], "climate-commons-fixture")

    def test_explicit_idempotency_key_is_preserved(self):
        root = Path(__file__).resolve().parents[2]
        experiment = root / "experiments" / "multirepresentation-ebm-dynamics.v1.json"
        client = _FakeControlClient()
        with patch.object(commons_control, "verify_contracts", return_value={}), \
             patch.object(commons_control, "control_client", return_value=client):
            out = commons_control.submit_cpu_experiment(
                experiment_path=experiment,
                repository_revision="a" * 40,
                run_scope="scope-a",
                ram_mib=2048,
                idempotency_key="stable-declaration-key",
            )
        self.assertEqual(client.calls[-1][1]["idempotency_key"],
                         "stable-declaration-key")
        self.assertEqual(out["idempotencyKey"], "stable-declaration-key")

    def test_control_api_operations_delegate_to_the_pinned_client(self):
        client = _FakeControlClient()
        with patch.object(commons_control, "verify_contracts", return_value={}), \
             patch.object(commons_control, "control_client", return_value=client):
            self.assertEqual(
                commons_control.scheduler_status(etag='"etag"'),
                {"readinessGate": "open"})
            commons_control.list_jobs(repo="climate", status="done")
            commons_control.inspect_job("0" * 16)
            commons_control.cancel_job("0" * 16)
            commons_control.send_message(repo="commons", payload={"ok": True},
                                         name="result", idempotency_key="key")
            commons_control.inbox("climate", wait_seconds=1.0, limit=5,
                                  after="cursor")
            commons_control.ack_message("0" * 16)
        self.assertEqual(
            [call[0] for call in client.calls],
            ["status", "jobs", "inspect", "cancel", "send", "inbox", "ack"])
        self.assertEqual(client.calls[0][1], {"etag": '"etag"'})
        self.assertEqual(client.calls[1][1],
                         {"repo": "climate", "status": "done"})
        self.assertEqual(client.calls[4][1]["idempotency_key"], "key")
        self.assertEqual(client.calls[5][1],
                         {"repo": "climate", "wait_seconds": 1.0,
                          "limit": 5, "after": "cursor"})

    def test_experiment_outside_registered_directory_refuses(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(commons_control.CommonsControlError):
                commons_control.submit_cpu_experiment(
                    experiment_path=Path(td) / "x.json",
                    repository_revision="a" * 40,
                    run_scope="outside",
                    ram_mib=1024,
                )

    def test_contract_verification_refuses_missing_or_drifted_declarations(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with patch.object(commons_control, "commons_root", return_value=root):
                with self.assertRaisesRegex(commons_control.CommonsControlError,
                                            "unavailable"):
                    commons_control.verify_contracts()
            contracts = root / "contracts"
            contracts.mkdir()
            (contracts / "control-api-v1.json").write_text(json.dumps({
                "schema": "control-api/v1", "owner": "commons",
                "contractVersion": "1", "compatibility": {}, "public": {},
                "semantics": {}, "types": {}, "endpoints": {},
            }), encoding="utf-8")
            (contracts / "work-scheduler-v1.json").write_text(json.dumps({
                "schema": "work-scheduler/v1", "owner": "commons",
                "contractVersion": "1", "compatibility": {}, "public": {},
                "types": {}, "endpoints": {}, "resource_semantics": {},
            }), encoding="utf-8")
            with patch.object(commons_control, "commons_root", return_value=root):
                with self.assertRaisesRegex(commons_control.CommonsControlError,
                                            "fingerprint drift"):
                    commons_control.verify_contracts()

    def test_control_api_fingerprint_covers_semantics_not_administration(self):
        base = {
            "schema": "control-api/v1",
            "contractVersion": "1",
            "compatibility": {"additive": "x"},
            "public": {"resources": []},
            "semantics": {"jobs": "the durable queue"},
            "types": {"A": {"required": ["x"], "optional": [], "docs": "x"}},
            "endpoints": {},
            "consumers": {"climate": {}},
        }
        first = commons_control.control_api_fingerprint(base)
        admin = copy.deepcopy(base)
        admin["consumers"]["other"] = {}
        self.assertEqual(first, commons_control.control_api_fingerprint(admin))
        changed = copy.deepcopy(base)
        changed["semantics"]["jobs"] = "a different delivery semantic"
        self.assertNotEqual(first, commons_control.control_api_fingerprint(changed))

    def test_adapter_avoids_cli_spawns_ports_and_state_parsing(self):
        source = (
            Path(__file__).resolve().parents[2]
            / "architecture" / "commons_control.py"
        ).read_text(encoding="utf-8")
        body = source.split('"""', 2)[-1]
        self.assertIn("importlib.import_module", body)
        self.assertNotIn("subprocess", body)
        self.assertNotIn("work_scheduler.py", body)
        self.assertNotIn("scheduler-state", body)
        self.assertNotIn("127.0.0.1", body)


    def test_external_artifact_ref_requires_full_digest(self):
        good = {
            "producer_repository": "J0pari/Training",
            "local_artifact_id": "0123456789abcdef",
            "digest": "a" * 64,
            "artifact_contract": "training.model-artifact/v1",
        }
        self.assertEqual(
            commons_control.validate_external_artifact_ref(good)["digest"],
            "a" * 64,
        )
        bad = dict(good)
        bad["digest"] = "a" * 16
        with self.assertRaises(commons_control.CommonsControlError):
            commons_control.validate_external_artifact_ref(bad)

    def test_registered_external_evaluator_exposes_native_import_path(self):
        subject = {
            "producer_repository": "J0pari/Training",
            "local_artifact_id": "0123456789abcdef",
            "digest": "b" * 64,
            "artifact_contract": "training.model-artifact/v1",
        }
        evaluation_id = "external.training_artifact.climate_contract_reasoning.v1"
        spec = commons_control.load_external_evaluation_spec(evaluation_id)
        self.assertEqual(spec["subject_contract"], "training.model-artifact/v1")
        self.assertEqual(spec["adapter"]["status"], "native_output_import")
        self.assertEqual(
            commons_control.require_external_evaluator(subject, evaluation_id)["evaluation_id"],
            evaluation_id,
        )


    def _write_external_import_fixture(self, root: Path):
        evaluation_id = "external.training_artifact.climate_contract_reasoning.v1"
        subject = {
            "producer_repository": "J0pari/Training",
            "local_artifact_id": "0123456789abcdef",
            "digest": "a" * 64,
            "artifact_contract": "training.model-artifact/v1",
        }
        predictions = {
            "schema": "climate-contract-reasoning-predictions/v1",
            "evaluation_id": evaluation_id,
            "subject_digest": subject["digest"],
            "runtime": {
                "interface": "climate.external-model-runtime/v1",
                "implementation": "fixture-native-runtime",
                "implementation_version": "1.2.3",
                "configuration_digest": "sha256:" + "b" * 64,
                "subject_digest": subject["digest"],
            },
            "responses": [],
        }
        predictions_path = root / "predictions.json"
        predictions_path.write_text(json.dumps(predictions, sort_keys=True) + "\n", encoding="utf-8")
        prediction_digest = "sha256:" + hashlib.sha256(predictions_path.read_bytes()).hexdigest()
        receipt = {
            "schema": "climate.external-model-runtime-receipt/v1",
            "interface": "climate.external-model-runtime/v1",
            "execution_mode": "native_output_import",
            "producer_system": "fixture-upstream",
            "implementation": "fixture-native-runtime",
            "implementation_version": "1.2.3",
            "subject": {
                "producer_repository": subject["producer_repository"],
                "artifact_contract": subject["artifact_contract"],
                "digest": subject["digest"],
            },
            "configuration_digest": "sha256:" + "b" * 64,
            "transformations": [],
            "status": "succeeded",
            "prediction_digest": prediction_digest,
            "exit_code": 0,
        }
        receipt_path = root / "receipt.json"
        receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
        return evaluation_id, subject, predictions_path, receipt_path

    def test_native_output_import_binds_subject_runtime_and_prediction_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            evaluation_id, subject, predictions_path, receipt_path = self._write_external_import_fixture(Path(td))
            result = commons_control.validate_external_output_import(
                subject=subject,
                evaluation_id=evaluation_id,
                predictions_path=predictions_path,
                receipt_path=receipt_path,
            )
        self.assertEqual(result["subject"]["digest"], "a" * 64)
        self.assertEqual(result["runtime"]["implementation_version"], "1.2.3")
        self.assertTrue(result["prediction_digest"].startswith("sha256:"))
        self.assertTrue(result["receipt_digest"].startswith("sha256:"))
        self.assertEqual(len(result["artifacts"]), 2)
        prediction_artifact, receipt_artifact = result["artifacts"]
        self.assertEqual(prediction_artifact["artifact_id"], evaluation_id + ".predictions")
        self.assertEqual(prediction_artifact["digest"], result["prediction_digest"])
        self.assertEqual(prediction_artifact["schema"], "climate-contract-reasoning-predictions/v1")
        self.assertEqual(prediction_artifact["media_type"], "application/json")
        self.assertGreater(prediction_artifact["bytes"], 0)
        self.assertEqual(receipt_artifact["artifact_id"], evaluation_id + ".runtime_receipt")
        self.assertEqual(receipt_artifact["digest"], result["receipt_digest"])
        self.assertEqual(receipt_artifact["schema"], "climate.external-model-runtime-receipt/v1")
        self.assertEqual(receipt_artifact["media_type"], "application/json")
        self.assertGreater(receipt_artifact["bytes"], 0)

    def test_native_output_import_rejects_subject_substitution(self):
        with tempfile.TemporaryDirectory() as td:
            evaluation_id, subject, predictions_path, receipt_path = self._write_external_import_fixture(Path(td))
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["subject"]["digest"] = "c" * 64
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(commons_control.CommonsControlError, "subject digest mismatch"):
                commons_control.validate_external_output_import(
                    subject=subject, evaluation_id=evaluation_id,
                    predictions_path=predictions_path, receipt_path=receipt_path)

    def test_native_output_import_rejects_runtime_substitution(self):
        with tempfile.TemporaryDirectory() as td:
            evaluation_id, subject, predictions_path, receipt_path = self._write_external_import_fixture(Path(td))
            predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
            predictions["runtime"]["implementation"] = "replacement-runtime"
            predictions_path.write_text(json.dumps(predictions), encoding="utf-8")
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["prediction_digest"] = commons_control._sha256_file(predictions_path)
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(commons_control.CommonsControlError, "runtime implementation mismatch"):
                commons_control.validate_external_output_import(
                    subject=subject, evaluation_id=evaluation_id,
                    predictions_path=predictions_path, receipt_path=receipt_path)

    def test_native_output_import_rejects_prediction_digest_drift(self):
        with tempfile.TemporaryDirectory() as td:
            evaluation_id, subject, predictions_path, receipt_path = self._write_external_import_fixture(Path(td))
            predictions_path.write_text(predictions_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
            with self.assertRaisesRegex(commons_control.CommonsControlError, "prediction_digest mismatch"):
                commons_control.validate_external_output_import(
                    subject=subject, evaluation_id=evaluation_id,
                    predictions_path=predictions_path, receipt_path=receipt_path)

    def test_unavailable_external_adapter_still_refuses(self):
        subject = {
            "producer_repository": "J0pari/Training",
            "local_artifact_id": "0123456789abcdef",
            "digest": "d" * 64,
            "artifact_contract": "training.model-artifact/v1",
        }
        spec = commons_control.load_external_evaluation_spec(
            "external.training_artifact.climate_contract_reasoning.v1")
        spec = copy.deepcopy(spec)
        spec["adapter"]["status"] = "unavailable"
        with patch.object(commons_control, "load_external_evaluation_spec", return_value=spec):
            with self.assertRaises(commons_control.ExternalEvaluationUnavailable):
                commons_control.require_external_evaluator(
                    subject, "external.training_artifact.climate_contract_reasoning.v1")

    def test_unknown_external_evaluator_refuses(self):
        with self.assertRaises(commons_control.ExternalEvaluationUnavailable):
            commons_control.load_external_evaluation_spec("missing.evaluator.v1")

    def test_cli_status_entrypoint_dispatches(self):
        with patch.object(
            commons_control, "scheduler_status", return_value={"ok": True}
        ):
            self.assertEqual(commons_control.main(["status"]), 0)


if __name__ == "__main__":
    unittest.main()
