"""Witnesses for the shared Climate source-surface classifier."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from architecture import source_surface


class SourceSurfaceTests(unittest.TestCase):
    def test_root_source_defaults_to_scientific(self):
        self.assertEqual(
            source_surface.classify_relative_path(Path("climate_state.rs")),
            source_surface.ROLE_SCIENTIFIC,
        )

    def test_unknown_nested_package_cannot_escape_scientific_role(self):
        self.assertEqual(
            source_surface.classify_relative_path(Path("future_package/solver/kernel.rs")),
            source_surface.ROLE_SCIENTIFIC,
        )

    def test_package_scaffolding_is_audited_but_not_module_tracked(self):
        for path in ("reference/__init__.py", "src/lib.rs"):
            with self.subTest(path=path):
                self.assertEqual(
                    source_surface.classify_relative_path(Path(path)),
                    source_surface.ROLE_PACKAGE,
                )
        self.assertIn(source_surface.ROLE_PACKAGE, source_surface.AUDITED_ROLES)
        self.assertNotIn(source_surface.ROLE_PACKAGE, source_surface.MODULE_TRACKED_ROLES)

    def test_non_root_src_file_remains_scientific(self):
        self.assertEqual(
            source_surface.classify_relative_path(Path("src/solver.rs")),
            source_surface.ROLE_SCIENTIFIC,
        )

    def test_known_roles_are_distinct(self):
        cases = {
            "reference/geometry.py": source_surface.ROLE_REFERENCE,
            "experiments/run.py": source_surface.ROLE_EXPERIMENT,
            "architecture/check.py": source_surface.ROLE_ARCHITECTURE,
            "tests/test_check.py": source_surface.ROLE_TEST,
            "contracts/model.cue": source_surface.ROLE_CONTRACT,
            "fixtures/generator.py": source_surface.ROLE_FIXTURE,
        }
        for path, expected in cases.items():
            with self.subTest(path=path):
                self.assertEqual(
                    source_surface.classify_relative_path(Path(path)), expected
                )

    def test_build_and_hidden_trees_are_excluded(self):
        for path in ("build/generated.rs", "target/generated.rs", ".github/tool.py"):
            with self.subTest(path=path):
                self.assertEqual(
                    source_surface.classify_relative_path(Path(path)),
                    source_surface.ROLE_EXCLUDED,
                )

    def test_unknown_extension_is_not_source(self):
        self.assertIsNone(source_surface.classify_relative_path(Path("physics/notes.txt")))

    def test_module_surface_is_recursive_and_non_vacuous(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "physics").mkdir()
            (root / "physics" / "nested.rs").write_text("fn x() {}\n", encoding="utf-8")
            (root / "reference").mkdir()
            (root / "reference" / "oracle.py").write_text("pass\n", encoding="utf-8")
            (root / "reference" / "__init__.py").write_text("\n", encoding="utf-8")
            (root / "architecture").mkdir()
            (root / "architecture" / "gate.py").write_text("pass\n", encoding="utf-8")
            (root / "src").mkdir()
            (root / "src" / "lib.rs").write_text("\n", encoding="utf-8")
            (root / "src" / "solver.rs").write_text("fn y() {}\n", encoding="utf-8")

            paths = source_surface.module_tracked_paths(root)
            self.assertEqual(paths, {"physics/nested.rs", "reference/oracle.py", "src/solver.rs"})
            self.assertNotIn("reference/__init__.py", paths)
            self.assertNotIn("src/lib.rs", paths)
            self.assertNotIn("architecture/gate.py", paths)


if __name__ == "__main__":
    unittest.main()
