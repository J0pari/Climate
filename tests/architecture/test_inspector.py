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
from architecture import source_surface


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

    def test_root_cargo_package_without_target_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Cargo.toml").write_text(
                '[package]\nname = "fixture"\nversion = "0.1.0"\n', encoding="utf-8"
            )
            findings = inspector.inspect_cargo(root)
            self.assertEqual([f.code for f in findings], ["cargo.package_target_missing"])

    def test_root_cargo_package_with_conventional_target_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "src/lib.rs").write_text("pub fn x() {}\n", encoding="utf-8")
            (root / "Cargo.toml").write_text(
                '[package]\nname = "fixture"\nversion = "0.1.0"\n', encoding="utf-8"
            )
            self.assertEqual(inspector.inspect_cargo(root), [])

    def test_missing_literal_cmake_reference_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "CMakeLists.txt").write_text(
                'set(SOURCES CORE/climate_physics_core.f90)\n',
                encoding="utf-8",
            )
            findings = inspector.inspect_cmake(root)
            codes = {(f.code, f.path) for f in findings}
            self.assertIn(("cmake.referenced_path_missing", "CORE/climate_physics_core.f90"), codes)

    def test_unconditional_cuda_project_language_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "CMakeLists.txt").write_text(
                'project(Fixture LANGUAGES CXX CUDA Fortran)\n'
                'option(ENABLE_CUDA "Enable CUDA" OFF)\n',
                encoding="utf-8",
            )
            findings = inspector.inspect_cmake(root)
            self.assertIn("cmake.cuda_language_unconditional", {f.code for f in findings})

    def test_default_off_missing_subdirectory_is_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "CMakeLists.txt").write_text(
                'option(ENABLE_BENCHMARKS "Enable benchmarks" OFF)\n'
                'if(ENABLE_BENCHMARKS)\n'
                '  add_subdirectory(benchmarks)\n'
                'endif()\n',
                encoding="utf-8",
            )
            findings = inspector.inspect_cmake(root)
            self.assertEqual(len(findings), 1)
            self.assertEqual(findings[0].code, "cmake.optional_subdirectory_missing")
            self.assertEqual(findings[0].severity, "warning")

    def test_enabled_subdirectory_without_manifest_is_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tests").mkdir()
            (root / "CMakeLists.txt").write_text(
                'option(ENABLE_TESTING "Enable tests" ON)\n'
                'if(ENABLE_TESTING)\n'
                '  add_subdirectory(tests)\n'
                'endif()\n',
                encoding="utf-8",
            )
            findings = inspector.inspect_cmake(root)
            self.assertEqual(len(findings), 1)
            self.assertEqual(findings[0].code, "cmake.subdirectory_manifest_missing")
            self.assertEqual(findings[0].severity, "error")

    def test_source_inventory_uses_shared_roles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = {
                "future/solver.rs": "fn x() {}\n",
                "reference/oracle.py": "pass\n",
                "architecture/gate.py": "pass\n",
                "tests/test_gate.py": "pass\n",
                "contracts/model.cue": "package fixture\n",
                "build/generated.rs": "fn generated() {}\n",
            }
            for rel, body in files.items():
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(body, encoding="utf-8")

            entries = inspector.source_inventory(root)
            by_path = {entry.path: entry.role for entry in entries}
            self.assertEqual(by_path["future/solver.rs"], source_surface.ROLE_SCIENTIFIC)
            self.assertEqual(by_path["reference/oracle.py"], source_surface.ROLE_REFERENCE)
            self.assertEqual(by_path["architecture/gate.py"], source_surface.ROLE_ARCHITECTURE)
            self.assertEqual(by_path["tests/test_gate.py"], source_surface.ROLE_TEST)
            self.assertEqual(by_path["contracts/model.cue"], source_surface.ROLE_CONTRACT)
            self.assertNotIn("build/generated.rs", by_path)

            summary = inspector.summarize_inventory(entries)
            self.assertEqual(summary["source_files"], 5)
            self.assertEqual(summary["by_role"][source_surface.ROLE_SCIENTIFIC]["files"], 1)
            self.assertEqual(summary["by_role"][source_surface.ROLE_REFERENCE]["files"], 1)

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
            self.assertIn("by_role", report["inventory"])

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