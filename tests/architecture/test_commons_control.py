"""Climate's fail-closed Commons control boundary."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from architecture import commons_control


class CommonsControlTests(unittest.TestCase):
    def test_pin_and_interface_adopt_resource_generic_read_execution(self):
        root = Path(__file__).resolve().parents[2]
        pin = json.loads((root / "contracts/work-scheduler-pin.json").read_text())
        interface = json.loads((root / "architecture/commons_interface.json").read_text())
        self.assertEqual(pin["owner"], "commons")
        self.assertEqual(pin["schema"], "work-scheduler/v1")
        self.assertEqual(interface["supported_control_level"], "read")
        self.assertEqual(interface["scheduler_contract"]["pin_path"],
                         "contracts/work-scheduler-pin.json")

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
        calls = []
        def fake_invoke(args, env=None):
            calls.append(args)
            if args[:2] == ["contract", "--json"]:
                pin = commons_control.load_pin()
                contract = {
                    "schema": pin["schema"],
                    "owner": "commons",
                    "contractVersion": "1",
                    "compatibility": {},
                    "public": {},
                    "types": {},
                    "endpoints": {},
                    "resource_semantics": {},
                }
                # This test targets command construction, not the stored pin hash.
                with patch.object(commons_control, "abi_fingerprint",
                                  return_value=pin["fingerprint"]):
                    return contract
            return {"jobId": "0123456789abcdef", "status": "queued"}
        with patch.object(commons_control, "_invoke", side_effect=fake_invoke), \
             patch.object(commons_control, "abi_fingerprint",
                          return_value=commons_control.load_pin()["fingerprint"]):
            out = commons_control.submit_cpu_experiment(
                experiment_path=experiment,
                repository_revision="a" * 40,
                run_scope="commons-fixture",
                ram_mib=1024,
            )
        submit = calls[-1]
        self.assertIn("--resource-class", submit)
        self.assertEqual(submit[submit.index("--resource-class") + 1], "cpu")
        self.assertNotIn("--gpu-mib", submit)
        self.assertIn(str(root / "src" / "experiment_runtime.py"), submit)
        self.assertEqual(out["outputDir"], "run-artifacts/commons/commons-fixture")

    def test_experiment_outside_registered_directory_refuses(self):
        with tempfile.TemporaryDirectory() as td, \
             patch.object(commons_control, "verify_scheduler_contract",
                          return_value={}):
            with self.assertRaises(commons_control.CommonsControlError):
                commons_control.submit_cpu_experiment(
                    experiment_path=Path(td) / "x.json",
                    repository_revision="a" * 40,
                    run_scope="outside",
                    ram_mib=1024,
                )


if __name__ == "__main__":
    unittest.main()
