"""Negative witnesses for Climate experiment graph integrity."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from architecture import check_experiments


def method(method_id: str, *, gpu_count: int = 0, vram_bytes: int = 0, network: str = "none"):
    return {
        "method_id": method_id,
        "resource": {
            "cpu_cores": 1,
            "memory_bytes": 128,
            "gpu_count": gpu_count,
            "vram_bytes": vram_bytes,
            "network": network,
        },
    }


def experiment(*, candidate="candidate", baseline="baseline", digest=None, citation=None, resource=None):
    dataset = {
        "id": "fixture.v1",
        "digest": digest or ("sha256:" + "0" * 64),
        "source_family": "fixture",
        "variables": ["x"],
    }
    if citation is not None:
        dataset["citation"] = citation
    return {
        "experiment_id": "fixture.experiment.v1",
        "candidate_methods": [candidate],
        "baseline_methods": [baseline],
        "datasets": [dataset],
        "primary_metrics": [{"metric_id": "metric.one"}],
        "secondary_metrics": [],
        "resource": resource or {
            "cpu_cores": 2,
            "memory_bytes": 1024,
            "gpu_count": 1,
            "vram_bytes": 1024,
            "network": "none",
        },
    }


def write_numerical_configuration(root: Path) -> tuple[dict, str]:
    record = {
        "configuration_id": "fixture.numerical.v1",
        "semantic_version": "1.0.0",
        "kind": "numerical_policy",
        "owner": "fixture",
        "provenance": "experiment_policy",
        "settings": {"missing_result": "fail"},
    }
    path = root / "configurations" / "numerical" / "fixture.v1.json"
    path.parent.mkdir(parents=True)
    content = (json.dumps(record, indent=2) + "\n").encode()
    path.write_bytes(content)
    digest = "sha256:" + hashlib.sha256(content).hexdigest()
    reference = {
        **{key: record[key] for key in (
            "configuration_id",
            "semantic_version",
            "kind",
            "owner",
            "provenance",
        )},
        "record_path": "configurations/numerical/fixture.v1.json",
        "digest": digest,
    }
    return reference, digest


class ExperimentIntegrityTests(unittest.TestCase):
    def registry(self, *methods):
        return {"methods": list(methods)}

    def test_missing_method_is_rejected(self):
        findings = check_experiments.check(
            Path("."),
            self.registry(method("baseline")),
            {"experiments/x.json": experiment(candidate="missing")},
        )
        self.assertIn("experiments.method_missing", {f.code for f in findings})

    def test_candidate_baseline_overlap_is_rejected(self):
        findings = check_experiments.check(
            Path("."),
            self.registry(method("same")),
            {"experiments/x.json": experiment(candidate="same", baseline="same")},
        )
        self.assertIn("experiments.method_role_overlap", {f.code for f in findings})

    def test_underprovisioned_gpu_resource_is_rejected(self):
        findings = check_experiments.check(
            Path("."),
            self.registry(method("candidate", gpu_count=1, vram_bytes=2048), method("baseline")),
            {"experiments/x.json": experiment(resource={
                "cpu_cores": 2,
                "memory_bytes": 1024,
                "gpu_count": 0,
                "vram_bytes": 1024,
                "network": "none",
            })},
        )
        codes = {f.code for f in findings}
        self.assertIn("experiments.resource_underprovisioned", codes)

    def test_network_requirement_is_enforced(self):
        findings = check_experiments.check(
            Path("."),
            self.registry(method("candidate", network="required"), method("baseline")),
            {"experiments/x.json": experiment()},
        )
        self.assertIn("experiments.network_underprovisioned", {f.code for f in findings})

    def test_duplicate_metric_id_is_rejected(self):
        spec = experiment()
        spec["secondary_metrics"] = [{"metric_id": "metric.one"}]
        findings = check_experiments.check(
            Path("."),
            self.registry(method("candidate"), method("baseline")),
            {"experiments/x.json": spec},
        )
        self.assertIn("experiments.metric_duplicate", {f.code for f in findings})

    def test_local_fixture_digest_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture_path = root / "fixtures" / "x.json"
            fixture_path.parent.mkdir(parents=True)
            fixture_path.write_text("{}\n", encoding="utf-8")
            spec = experiment(citation="fixtures/x.json")
            findings = check_experiments.check(
                root,
                self.registry(method("candidate"), method("baseline")),
                {"experiments/x.json": spec},
            )
            self.assertIn("experiments.dataset_digest_mismatch", {f.code for f in findings})

    def test_matching_local_fixture_digest_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture_path = root / "fixtures" / "x.json"
            fixture_path.parent.mkdir(parents=True)
            content = b"{}\n"
            fixture_path.write_bytes(content)
            digest = "sha256:" + hashlib.sha256(content).hexdigest()
            spec = experiment(citation="fixtures/x.json", digest=digest)
            findings = check_experiments.check(
                root,
                self.registry(method("candidate"), method("baseline")),
                {"experiments/x.json": spec},
            )
            self.assertEqual(findings, [])

    def test_matching_configuration_reference_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference, _ = write_numerical_configuration(root)
            spec = experiment()
            spec["configuration"] = {"numerical_policies": [reference]}
            findings = check_experiments.check(
                root,
                self.registry(method("candidate"), method("baseline")),
                {"experiments/x.json": spec},
            )
            self.assertEqual(findings, [])

    def test_configuration_digest_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference, _ = write_numerical_configuration(root)
            reference["digest"] = "sha256:" + "0" * 64
            spec = experiment()
            spec["configuration"] = {"numerical_policies": [reference]}
            findings = check_experiments.check(
                root,
                self.registry(method("candidate"), method("baseline")),
                {"experiments/x.json": spec},
            )
            self.assertIn(
                "experiments.configuration_digest_mismatch",
                {finding.code for finding in findings},
            )

    def test_configuration_identity_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference, _ = write_numerical_configuration(root)
            reference["owner"] = "other.owner"
            spec = experiment()
            spec["configuration"] = {"numerical_policies": [reference]}
            findings = check_experiments.check(
                root,
                self.registry(method("candidate"), method("baseline")),
                {"experiments/x.json": spec},
            )
            self.assertIn(
                "experiments.configuration_identity_mismatch",
                {finding.code for finding in findings},
            )


if __name__ == "__main__":
    unittest.main()
