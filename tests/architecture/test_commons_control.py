"""Climate's fail-closed Commons control boundary."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from architecture import commons_control


class CommonsControlTests(unittest.TestCase):
    def test_pin_and_interface_preserve_observe_only_authority(self):
        root = Path(__file__).resolve().parents[2]
        pin = json.loads((root / "contracts/gpu-scheduler-pin.json").read_text())
        interface = json.loads((root / "architecture/commons_interface.json").read_text())
        self.assertEqual(pin["owner"], "commons")
        self.assertEqual(pin["schema"], "gpu-scheduler/v1")
        self.assertEqual(interface["supported_control_level"], "observe")
        self.assertEqual(interface["scheduler_contract"]["pin_path"],
                         "contracts/gpu-scheduler-pin.json")

    def test_abi_fingerprint_ignores_administration_but_not_types(self):
        base = {
            "schema": "gpu-scheduler/v1",
            "contractVersion": "1",
            "compatibility": {"additive": "x"},
            "public": {"commands": []},
            "types": {"A": {"required": ["x"], "optional": [], "docs": "x"}},
            "endpoints": {},
            "gpu_lock": {"version": 1},
            "owner": "commons",
            "consumers": {"climate": {}},
        }
        first = commons_control.abi_fingerprint(base)
        admin = copy.deepcopy(base)
        admin["consumers"]["other"] = {}
        self.assertEqual(first, commons_control.abi_fingerprint(admin))
        changed = copy.deepcopy(base)
        changed["types"]["A"]["required"].append("y")
        self.assertNotEqual(first, commons_control.abi_fingerprint(changed))

    def test_missing_commons_checkout_refuses_instead_of_substituting(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(commons_control.CommonsControlError):
                commons_control.commons_root({"COMMONS_ROOT": str(Path(td) / "missing")})


if __name__ == "__main__":
    unittest.main()
