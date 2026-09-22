from __future__ import annotations

import base64
import tempfile
import unittest
from pathlib import Path

from architecture.snapshot import git_blob_id, git_object_id, verify_snapshot
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

    def test_manifest_rejects_noncanonical_identity_fields(self):
        valid_row = {
            "path": "x.txt",
            "git_blob_sha1": "a" * 40,
            "mode": "100644",
            "size": 0,
        }
        with self.assertRaisesRegex(ValueError, "Git blob SHA-1"):
            parse_manifest({
                "schema_version": 1,
                "files": [{**valid_row, "git_blob_sha1": "z" * 40}],
            })
        with self.assertRaisesRegex(ValueError, "root tree SHA-1"):
            parse_manifest({
                "schema_version": 1,
                "tree_sha1": "A" * 40,
                "files": [valid_row],
            })
        with self.assertRaisesRegex(ValueError, "source commit SHA-1"):
            parse_manifest({
                "schema_version": 1,
                "source_commit": "not-a-commit",
                "files": [valid_row],
            })

    def test_complete_manifest_requires_exact_commit_and_tree_identity(self):
        payload = {
            "schema_version": 1,
            "complete_tree": True,
            "tree_sha1": "a" * 40,
            "files": [],
        }
        with self.assertRaisesRegex(ValueError, "requires source_commit"):
            parse_manifest(payload)
        payload["source_commit"] = "b" * 40
        observed = parse_manifest(payload)
        self.assertTrue(observed.complete_tree)
        self.assertEqual(observed.tree_sha1, "a" * 40)
        self.assertEqual(observed.source_commit, "b" * 40)

    def test_manifest_boolean_and_integer_fields_do_not_accept_json_type_aliases(self):
        with self.assertRaisesRegex(ValueError, "schema_version"):
            parse_manifest({"schema_version": True, "files": []})
        with self.assertRaisesRegex(ValueError, "complete_tree"):
            parse_manifest({
                "schema_version": 1,
                "complete_tree": "false",
                "files": [],
            })
        with self.assertRaisesRegex(ValueError, "byte size"):
            parse_manifest({
                "schema_version": 1,
                "files": [{
                    "path": "x.txt",
                    "git_blob_sha1": "a" * 40,
                    "mode": "100644",
                    "size": True,
                }],
            })
        with self.assertRaisesRegex(ValueError, "path must be a string"):
            parse_manifest({
                "schema_version": 1,
                "files": [{
                    "path": 7,
                    "git_blob_sha1": "a" * 40,
                    "mode": "100644",
                    "size": 0,
                }],
            })
        with self.assertRaisesRegex(ValueError, "unsupported file mode"):
            parse_manifest({
                "schema_version": 1,
                "files": [{
                    "path": "x.txt",
                    "git_blob_sha1": "a" * 40,
                    "mode": 100644,
                    "size": 0,
                }],
            })

    def test_manifest_rejects_noncanonical_and_cross_platform_escape_paths(self):
        valid = {
            "schema_version": 1,
            "files": [{
                "path": "x.txt",
                "git_blob_sha1": "a" * 40,
                "mode": "100644",
                "size": 0,
            }],
        }
        for path in (
            "a//b",
            "a/./b",
            "a/../b",
            "..\\escape",
            "nested\\escape",
            "C:/escape",
            "C:escape",
            "/absolute",
        ):
            with self.subTest(path=path):
                payload = {
                    **valid,
                    "files": [{**valid["files"][0], "path": path}],
                }
                with self.assertRaisesRegex(ValueError, "repository-relative path"):
                    parse_manifest(payload)

    def test_schema_v2_binds_source_commit_object_to_declared_tree(self):
        tree = "a" * 40
        commit_object = (
            f"tree {tree}\n"
            "author Snapshot <snapshot@example.invalid> 0 +0000\n"
            "committer Snapshot <snapshot@example.invalid> 0 +0000\n"
            "\nfixture\n"
        ).encode("ascii")
        source_commit = git_object_id("commit", commit_object)
        payload = {
            "schema_version": 2,
            "complete_tree": True,
            "tree_sha1": tree,
            "source_commit": source_commit,
            "source_commit_object_base64": base64.b64encode(commit_object).decode("ascii"),
            "files": [],
        }
        observed = parse_manifest(payload)
        self.assertEqual(observed.source_commit_object, commit_object)

        tampered = dict(payload)
        tampered["source_commit_object_base64"] = base64.b64encode(
            commit_object + b"x"
        ).decode("ascii")
        with self.assertRaisesRegex(ValueError, "hashes to"):
            parse_manifest(tampered)

        wrong_tree = dict(payload)
        wrong_tree["tree_sha1"] = "b" * 40
        with self.assertRaisesRegex(ValueError, "points to tree"):
            parse_manifest(wrong_tree)

        missing_object = dict(payload)
        del missing_object["source_commit_object_base64"]
        with self.assertRaisesRegex(ValueError, "requires source_commit_object_base64"):
            parse_manifest(missing_object)


if __name__ == "__main__":
    unittest.main()
