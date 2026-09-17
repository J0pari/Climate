from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from architecture import check_root_layout


ROOT = Path(__file__).resolve().parents[2]


class RootLayoutTests(unittest.TestCase):
    def test_manifests_are_allowed_but_root_scientific_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "README.md").write_text("description\n", encoding="utf-8")
            (root / "Cargo.toml").write_text("[package]\n", encoding="utf-8")
            (root / "solver.rs").write_text("fn solve() {}\n", encoding="utf-8")
            findings = check_root_layout.check(root)
            self.assertEqual(
                {(item.code, item.path) for item in findings},
                {("root_layout.scientific_source", "solver.rs")},
            )

    def test_climate_prefixed_root_config_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "climate_config.toml").write_text("[global]\n", encoding="utf-8")
            findings = check_root_layout.check(root)
            self.assertEqual(
                {(item.code, item.path) for item in findings},
                {("root_layout.climate_prefixed_file", "climate_config.toml")},
            )

    def test_owned_source_tree_is_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "src" / "solver.rs").write_text("fn solve() {}\n", encoding="utf-8")
            self.assertEqual(check_root_layout.check(root), [])

    def test_documented_direct_script_entrypoint_works(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "architecture" / "check_root_layout.py")],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
        self.assertIn("root layout: clean", result.stdout)


if __name__ == "__main__":
    unittest.main()
