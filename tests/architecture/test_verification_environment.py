from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from architecture import verification_environment


class VerificationEnvironmentTests(unittest.TestCase):
    def _root(self) -> tuple[tempfile.TemporaryDirectory[str], Path]:
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        (root / ".devcontainer").mkdir()
        (root / "requirements").mkdir()
        (root / "formal").mkdir()
        (root / "architecture" / "snapshot").mkdir(parents=True)
        (root / ".devcontainer" / "devcontainer.json").write_text(
            json.dumps(
                {
                    "image": "example.invalid/dev:ubuntu-24.04",
                    "features": {
                        "python": {"version": "3.12"},
                        "rust": {"version": "latest"},
                    },
                }
            ),
            encoding="utf-8",
        )
        (root / ".devcontainer" / "bootstrap.sh").write_text(
            """#!/bin/sh
apt-get install -y gfortran \
  libblas-dev \
  liblapack-dev
go install cuelang.org/go/cmd/cue@v0.17.1
""",
            encoding="utf-8",
        )
        (root / "requirements" / "focused.txt").write_text(
            "numpy==2.5.3\nscipy==1.18.1\n", encoding="utf-8"
        )
        (root / "formal" / "lean-toolchain").write_text(
            "leanprover/lean4:v4.34.0\n", encoding="utf-8"
        )
        (root / "architecture" / "snapshot" / "generate.py").write_text(
            "def build_manifest(): pass\n", encoding="utf-8"
        )
        (root / "architecture" / "check_snapshot.py").write_text(
            'command = "identity"\n', encoding="utf-8"
        )
        return temporary, root

    def test_current_like_environment_reports_specific_unresolved_identities(self) -> None:
        temporary, root = self._root()
        self.addCleanup(temporary.cleanup)
        report = verification_environment.inspect(root)
        self.assertEqual(report["status"], "unresolved")
        unresolved = set(report["unresolved"])
        self.assertIn("devcontainer.image_immutable", unresolved)
        self.assertIn("devcontainer.feature_version.python", unresolved)
        self.assertIn("devcontainer.feature_version.rust", unresolved)
        self.assertIn("rust.cargo_lock", unresolved)
        self.assertIn("python.repository_lock", unresolved)
        self.assertIn("fortran.compiler_package_exact", unresolved)
        self.assertIn("blas.package_exact", unresolved)
        self.assertIn("lapack.package_exact", unresolved)
        self.assertNotIn("cue.version_exact", unresolved)
        self.assertNotIn("lean.toolchain_exact", unresolved)
        self.assertNotIn("python.focused_requirements_exact", unresolved)
        self.assertNotIn("source_snapshot.identity_tooling", unresolved)

    def test_resolved_fixture_passes_strict_readiness(self) -> None:
        temporary, root = self._root()
        self.addCleanup(temporary.cleanup)
        (root / ".devcontainer" / "devcontainer.json").write_text(
            json.dumps(
                {
                    "image": "example.invalid/dev@sha256:" + "a" * 64,
                    "features": {
                        "python": {"version": "3.12.9"},
                        "rust": {"version": "1.91.0"},
                    },
                }
            ),
            encoding="utf-8",
        )
        (root / ".devcontainer" / "bootstrap.sh").write_text(
            """#!/bin/sh
apt-get install -y gfortran=14.2.0 \
  libblas-dev=3.12.0 \
  liblapack-dev=3.12.0
go install cuelang.org/go/cmd/cue@v0.17.1
""",
            encoding="utf-8",
        )
        (root / "Cargo.lock").write_text("# fixture\n", encoding="utf-8")
        (root / "requirements.lock").write_text("numpy==2.5.3\n", encoding="utf-8")

        report = verification_environment.inspect(root)
        self.assertEqual(report["status"], "resolved")
        self.assertEqual(report["unresolved"], [])

    def test_unpinned_focused_python_requirement_is_visible(self) -> None:
        temporary, root = self._root()
        self.addCleanup(temporary.cleanup)
        (root / "requirements" / "focused.txt").write_text("numpy>=2\n", encoding="utf-8")
        report = verification_environment.inspect(root)
        self.assertIn("python.focused_requirements_exact", report["unresolved"])

    def test_wildcard_versions_are_not_treated_as_exact_pins(self) -> None:
        temporary, root = self._root()
        self.addCleanup(temporary.cleanup)
        (root / "requirements" / "focused.txt").write_text(
            "numpy==2.*\n", encoding="utf-8"
        )
        (root / ".devcontainer" / "bootstrap.sh").write_text(
            """#!/bin/sh
apt-get install -y gfortran=14.* \
  libblas-dev=3.12.0 \
  liblapack-dev=3.12.0
go install cuelang.org/go/cmd/cue@v0.17.1
""",
            encoding="utf-8",
        )
        report = verification_environment.inspect(root)
        unresolved = set(report["unresolved"])
        self.assertIn("python.focused_requirements_exact", unresolved)
        self.assertIn("fortran.compiler_package_exact", unresolved)
        self.assertNotIn("blas.package_exact", unresolved)
        self.assertNotIn("lapack.package_exact", unresolved)


if __name__ == "__main__":
    unittest.main()
