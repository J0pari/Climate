from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from architecture.snapshot import git_blob_id, tree_id_from_files


class SnapshotGitObjectsTests(unittest.TestCase):
    def test_blob_identity_matches_git(self):
        payload = b"alpha\nbeta\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.txt"
            path.write_bytes(payload)
            expected = subprocess.run(
                ["git", "hash-object", str(path)], capture_output=True, text=True, check=True
            ).stdout.strip()
        self.assertEqual(git_blob_id(payload), expected)

    def test_tree_identity_matches_git_with_executable_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "alpha.txt").write_text("alpha\n", encoding="utf-8", newline="\n")
            (root / "nested").mkdir()
            script = root / "nested" / "run.sh"
            script.write_text("#!/bin/sh\necho ok\n", encoding="utf-8", newline="\n")
            script.chmod(0o755)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "config", "core.autocrlf", "false"], cwd=root, check=True)
            subprocess.run(["git", "add", "-A"], cwd=root, check=True)
            subprocess.run(["git", "update-index", "--chmod=+x", "nested/run.sh"], cwd=root, check=True)
            expected = subprocess.run(
                ["git", "write-tree"], cwd=root, capture_output=True, text=True, check=True
            ).stdout.strip()
            rows = [
                ("alpha.txt", "100644", git_blob_id((root / "alpha.txt").read_bytes())),
                ("nested/run.sh", "100755", git_blob_id((root / "nested" / "run.sh").read_bytes())),
            ]
            self.assertEqual(tree_id_from_files(rows), expected)


if __name__ == "__main__":
    unittest.main()
