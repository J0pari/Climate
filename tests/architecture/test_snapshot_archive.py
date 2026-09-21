from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from architecture.snapshot import (
    SnapshotManifest,
    create_archive,
    load_manifest,
    restore_modes,
    verify_snapshot,
)
from architecture.snapshot.archive import MANIFEST_MEMBER, REPOSITORY_PREFIX, SnapshotArchiveError

ROOT = Path(__file__).resolve().parents[2]


class SnapshotArchiveTests(unittest.TestCase):
    def _repo(self) -> tuple[tempfile.TemporaryDirectory[str], Path]:
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name) / "repo"
        root.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.email", "snapshot@example.invalid"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.name", "Snapshot Test"], cwd=root, check=True)
        (root / "alpha.txt").write_text("alpha\n", encoding="utf-8")
        script = root / "run.sh"
        script.write_text("#!/bin/sh\necho ok\n", encoding="utf-8")
        script.chmod(0o755)
        subprocess.run(["git", "add", "-A"], cwd=root, check=True)
        subprocess.run(["git", "commit", "-qm", "fixture"], cwd=root, check=True)
        return temporary, root

    def test_archive_round_trip_restores_only_lost_modes(self) -> None:
        temporary, root = self._repo()
        self.addCleanup(temporary.cleanup)
        output = Path(temporary.name) / "snapshot.zip"
        manifest = create_archive(root, output)

        with zipfile.ZipFile(output) as archive:
            self.assertEqual(
                set(archive.namelist()),
                {
                    MANIFEST_MEMBER,
                    REPOSITORY_PREFIX + "alpha.txt",
                    REPOSITORY_PREFIX + "run.sh",
                },
            )
            extracted = Path(temporary.name) / "extracted"
            archive.extractall(extracted)

        repository = extracted / "repository"
        (repository / "run.sh").chmod(0o644)
        loaded = load_manifest(extracted / MANIFEST_MEMBER)
        self.assertEqual(loaded.source_commit, manifest.source_commit)
        self.assertEqual(loaded.tree_sha1, manifest.tree_sha1)
        self.assertIn(
            "snapshot.mode",
            {finding.code for finding in verify_snapshot(repository, loaded)},
        )
        self.assertEqual(restore_modes(repository, loaded), [])
        self.assertEqual(verify_snapshot(repository, loaded), [])

    def test_mode_restoration_refuses_content_or_file_set_mismatch(self) -> None:
        temporary, root = self._repo()
        self.addCleanup(temporary.cleanup)
        output = Path(temporary.name) / "snapshot.zip"
        create_archive(root, output)
        with zipfile.ZipFile(output) as archive:
            extracted = Path(temporary.name) / "extracted"
            archive.extractall(extracted)

        repository = extracted / "repository"
        loaded = load_manifest(extracted / MANIFEST_MEMBER)
        (repository / "unexpected.txt").write_text("extra\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "refusing mode restoration"):
            restore_modes(repository, loaded)
        (repository / "unexpected.txt").unlink()
        (repository / "alpha.txt").write_text("changed\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "refusing mode restoration"):
            restore_modes(repository, loaded)

    def test_mode_restoration_requires_complete_tree_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "complete-tree snapshot manifest"):
                restore_modes(root, SnapshotManifest(files=()))

    def test_archive_must_be_outside_source_tree_and_not_overwrite(self) -> None:
        temporary, root = self._repo()
        self.addCleanup(temporary.cleanup)
        with self.assertRaisesRegex(SnapshotArchiveError, "outside the source tree"):
            create_archive(root, root / "snapshot.zip")
        output = Path(temporary.name) / "snapshot.zip"
        create_archive(root, output)
        with self.assertRaisesRegex(SnapshotArchiveError, "refusing to overwrite"):
            create_archive(root, output)

    def test_cli_archive_then_identity_with_mode_restoration(self) -> None:
        temporary, root = self._repo()
        self.addCleanup(temporary.cleanup)
        output = Path(temporary.name) / "snapshot.zip"
        created = subprocess.run(
            [
                sys.executable,
                str(ROOT / "architecture" / "check_snapshot.py"),
                "archive",
                str(output),
                "--root",
                str(root),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(created.returncode, 0, created.stderr)
        metadata = json.loads(created.stdout)
        self.assertEqual(metadata["manifest_member"], MANIFEST_MEMBER)
        self.assertEqual(metadata["repository_prefix"], REPOSITORY_PREFIX)

        with zipfile.ZipFile(output) as archive:
            extracted = Path(temporary.name) / "extracted"
            archive.extractall(extracted)
        repository = extracted / "repository"
        (repository / "run.sh").chmod(0o644)

        identity = subprocess.run(
            [
                sys.executable,
                str(ROOT / "architecture" / "check_snapshot.py"),
                "identity",
                str(extracted / MANIFEST_MEMBER),
                "--root",
                str(repository),
                "--restore-modes",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(identity.returncode, 0, identity.stderr)
        observed = json.loads(identity.stdout)
        self.assertEqual(observed["source_commit"], metadata["source_commit"])
        self.assertEqual(observed["tree_sha1"], metadata["tree_sha1"])
        self.assertTrue(observed["complete_tree"])


if __name__ == "__main__":
    unittest.main()
