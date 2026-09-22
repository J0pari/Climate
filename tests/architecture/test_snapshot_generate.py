from __future__ import annotations

import base64
import json
import os
import subprocess
import tempfile
import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

from architecture.snapshot import (
    SnapshotGenerationError,
    build_manifest,
    git_object_id,
    load_manifest,
    verify_snapshot,
    write_manifest,
)


class SnapshotGenerateTests(unittest.TestCase):
    def _repo(self) -> tuple[tempfile.TemporaryDirectory[str], Path]:
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.email", "snapshot@example.invalid"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.name", "Snapshot Test"], cwd=root, check=True)
        (root / "alpha.txt").write_text("alpha\n", encoding="utf-8")
        (root / "nested").mkdir()
        script = root / "nested" / "run.sh"
        script.write_text("#!/bin/sh\necho ok\n", encoding="utf-8")
        script.chmod(0o755)
        subprocess.run(["git", "add", "-A"], cwd=root, check=True)
        subprocess.run(["git", "commit", "-qm", "fixture"], cwd=root, check=True)
        return temporary, root

    def test_build_manifest_reconstructs_exact_head_tree(self) -> None:
        temporary, root = self._repo()
        self.addCleanup(temporary.cleanup)

        manifest = build_manifest(root)
        expected_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        expected_tree = subprocess.run(
            ["git", "rev-parse", "HEAD^{tree}"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()

        self.assertTrue(manifest.complete_tree)
        self.assertEqual(manifest.source_commit, expected_commit)
        self.assertEqual(manifest.tree_sha1, expected_tree)
        self.assertIsNotNone(manifest.source_commit_object)
        self.assertEqual(
            git_object_id("commit", manifest.source_commit_object or b""),
            expected_commit,
        )
        self.assertTrue(
            (manifest.source_commit_object or b"").startswith(
                f"tree {expected_tree}\n".encode("ascii")
            )
        )
        self.assertEqual([row.path for row in manifest.files], ["alpha.txt", "nested/run.sh"])
        self.assertEqual([row.mode for row in manifest.files], ["100644", "100755"])
        self.assertEqual(verify_snapshot(root, manifest), [])

    def test_dirty_or_untracked_checkout_is_rejected(self) -> None:
        temporary, root = self._repo()
        self.addCleanup(temporary.cleanup)

        (root / "alpha.txt").write_text("changed\n", encoding="utf-8")
        with self.assertRaisesRegex(SnapshotGenerationError, "not clean"):
            build_manifest(root)

        subprocess.run(["git", "restore", "alpha.txt"], cwd=root, check=True)
        (root / "untracked.txt").write_text("extra\n", encoding="utf-8")
        with self.assertRaisesRegex(SnapshotGenerationError, "not clean"):
            build_manifest(root)

    def test_source_ref_must_match_checked_out_head(self) -> None:
        temporary, root = self._repo()
        self.addCleanup(temporary.cleanup)
        first = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        (root / "beta.txt").write_text("beta\n", encoding="utf-8")
        subprocess.run(["git", "add", "beta.txt"], cwd=root, check=True)
        subprocess.run(["git", "commit", "-qm", "second"], cwd=root, check=True)

        with self.assertRaisesRegex(SnapshotGenerationError, "checkout HEAD"):
            build_manifest(root, source_ref=first)

    def test_manifest_round_trip_outside_tree(self) -> None:
        temporary, root = self._repo()
        self.addCleanup(temporary.cleanup)
        manifest = build_manifest(root)

        with tempfile.TemporaryDirectory() as outdir:
            path = Path(outdir) / "snapshot.json"
            write_manifest(manifest, path, root=root)
            observed = load_manifest(path)
            self.assertEqual(observed, manifest)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], 2)
            self.assertEqual(payload["source_commit"], manifest.source_commit)
            self.assertEqual(payload["tree_sha1"], manifest.tree_sha1)
            self.assertEqual(
                base64.b64decode(payload["source_commit_object_base64"], validate=True),
                manifest.source_commit_object,
            )

    def test_manifest_output_inside_source_tree_is_rejected(self) -> None:
        temporary, root = self._repo()
        self.addCleanup(temporary.cleanup)
        manifest = build_manifest(root)
        with self.assertRaisesRegex(SnapshotGenerationError, "outside the source tree"):
            write_manifest(manifest, root / "snapshot.json", root=root)

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks unavailable")
    def test_symlink_tree_is_rejected_until_transport_support_is_explicit(self) -> None:
        temporary, root = self._repo()
        self.addCleanup(temporary.cleanup)
        os.symlink("alpha.txt", root / "alias.txt")
        subprocess.run(["git", "add", "alias.txt"], cwd=root, check=True)
        subprocess.run(["git", "commit", "-qm", "symlink"], cwd=root, check=True)
        with self.assertRaisesRegex(SnapshotGenerationError, "unsupported file mode"):
            build_manifest(root)


class SnapshotCliTests(unittest.TestCase):
    def _repo(self) -> tuple[tempfile.TemporaryDirectory[str], Path]:
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.email", "snapshot@example.invalid"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.name", "Snapshot Test"], cwd=root, check=True)
        (root / "alpha.txt").write_text("alpha\\n", encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=root, check=True)
        subprocess.run(["git", "commit", "-qm", "fixture"], cwd=root, check=True)
        return temporary, root

    def test_manifest_then_identity_cli(self) -> None:
        temporary, checkout = self._repo()
        self.addCleanup(temporary.cleanup)
        with tempfile.TemporaryDirectory() as outdir:
            manifest_path = Path(outdir) / "snapshot.json"
            generated = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "architecture" / "check_snapshot.py"),
                    "manifest",
                    str(manifest_path),
                    "--root",
                    str(checkout),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(generated.returncode, 0, generated.stderr)
            payload = json.loads(generated.stdout)
            self.assertEqual(payload["file_count"], 1)

            identity = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "architecture" / "check_snapshot.py"),
                    "identity",
                    str(manifest_path),
                    "--root",
                    str(checkout),
                    "--expected-commit",
                    payload["source_commit"],
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(identity.returncode, 0, identity.stderr)
            observed = json.loads(identity.stdout)
            self.assertEqual(observed["source_commit"], payload["source_commit"])
            self.assertEqual(observed["tree_sha1"], payload["tree_sha1"])
            self.assertTrue(observed["source_commit_object_verified"])
            self.assertTrue(observed["expected_commit_matched"])
            self.assertTrue(observed["complete_tree"])


if __name__ == "__main__":
    unittest.main()
