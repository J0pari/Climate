from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from architecture.snapshot import git_blob_id, verify_snapshot
from architecture.snapshot.manifest import parse_manifest


class SnapshotManifestVerifyTests(unittest.TestCase):
    def _manifest(self, payload: bytes):
        return parse_manifest({
            "schema_version": 1,
            "complete_tree": False,
            "files": [{"path": "x.txt", "git_blob_sha1": git_blob_id(payload), "mode": "100644", "size": len(payload)}],
        })

    def test_manifest_rejects_parent_traversal(self):
        with self.assertRaises(ValueError):
            parse_manifest({
                "schema_version": 1,
                "files": [{"path": "../escape", "git_blob_sha1": "0" * 40, "mode": "100644", "size": 0}],
            })

    def test_verifier_accepts_exact_file_and_rejects_corruption(self):
        payload = b"right\n"
        manifest = self._manifest(payload)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "x.txt"
            path.write_bytes(payload)
            path.chmod(0o644)
            self.assertEqual(verify_snapshot(root, manifest), [])
            path.write_bytes(b"wrong\n")
            codes = {finding.code for finding in verify_snapshot(root, manifest)}
            self.assertIn("snapshot.blob", codes)


if __name__ == "__main__":
    unittest.main()
