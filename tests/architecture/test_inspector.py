"""Negative witnesses for the non-executing Climate repository inspector."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from architecture import inspect_repository as inspector


class RepositoryInspectorTests(unittest.TestCase):
    def test_missing_cargo_workspace_member_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Cargo.toml").write_text(
                '[workspace]\nmembers = ["CORE"]\n', encoding="utf-8"
            )
            findings = inspector.inspect_cargo(root)
            self.assertEqual(len(findings), 1)
            self.assertEqual(findings[0].code, "cargo.workspace_member_missing")
            self.assertEqual(findings[0].path, "CORE")

    def test_existing_cargo_workspace_member_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "CORE").mkdir()
            (root / "Cargo.toml").write_text(
                '[workspace]\nmembers = ["CORE"]\n', encoding="utf-8"
            )
            self.assertEqual(inspector.inspect_cargo(root), [])

    def test_missing_literal_cmake_reference_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "CMakeLists.txt").write_text(
                'set(SOURCES CORE/climate_physics_core.f90)\nadd_subdirectory(tests)\n',
                encoding="utf-8",
            )
            findings = inspector.inspect_cmake(root)
            codes = {(f.code, f.path) for f in findings}
            self.assertIn(("cmake.referenced_path_missing", "CORE/climate_physics_core.f90"), codes)
            self.assertIn(("cmake.referenced_path_missing", "tests"), codes)

    def test_source_audit_is_exposed_as_warning_not_structural_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "bad.rs").write_text(
                'fn demo() { let mut rng = rand::thread_rng(); }\n', encoding="utf-8"
            )
            findings = inspector.inspect_source_risks(root)
            self.assertEqual(len(findings), 1)
            self.assertEqual(findings[0].code, "source_audit.ambient_rng")
            self.assertEqual(findings[0].severity, "warning")

    def test_full_report_is_json_serializable_and_refuses_execution_readiness(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Create only the architecture files so the intended structural
            # failure comes from the Cargo member rather than the contract spine.
            for rel in (
                "AGENTS.md",
                "docs/ARCHITECTURE.md",
                "docs/GPU-ENGINEERING.md",
                "docs/VALIDATION-AND-EVIDENCE.md",
                "docs/META-EXPERIMENTATION.md",
                "docs/COMMONS-INTEGRATION.md",
                "docs/ROADMAP.md",
                "contracts/climate.cue",
            ):
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture\n", encoding="utf-8")
            (root / "Cargo.toml").write_text(
                '[workspace]\nmembers = ["CORE"]\n', encoding="utf-8"
            )

            report = inspector.inspect(root)
            json.dumps(report)
            self.assertFalse(report["ready_for_execution"])
            self.assertTrue(any(
                f["code"] == "cargo.workspace_member_missing"
                for f in report["findings"]
            ))

    def test_documented_direct_script_entrypoint_works_without_pythonpath(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = os.environ.copy()
            env.pop("PYTHONPATH", None)
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            script = inspector.ROOT / "architecture" / "inspect_repository.py"
            result = subprocess.run(
                [sys.executable, str(script), "--root", tmp, "--json"],
                cwd=inspector.ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report["repository"], "J0pari/Climate")
            self.assertEqual(report["mode"], "static-observe")


if __name__ == "__main__":
    unittest.main()
