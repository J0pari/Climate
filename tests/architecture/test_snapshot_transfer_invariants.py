from __future__ import annotations

import base64
import tempfile
import unittest
from pathlib import Path

from architecture.snapshot import git_blob_id
from architecture.snapshot.manifest import parse_manifest
from architecture.snapshot import transfer


class SnapshotTransferInvariantTests(unittest.TestCase):
    @staticmethod
    def _manifest(payload: bytes):
        return parse_manifest({
            "schema_version": 1,
            "files": [{
                "path": "x",
                "git_blob_sha1": git_blob_id(payload),
                "mode": "100644",
                "size": len(payload),
            }],
        })

    def test_overshoot_does_not_mutate_partial_or_state(self):
        payload = b"payload\n"
        state = transfer.init_state(self._manifest(payload), base64_chunk_chars=1000)
        encoded = base64.b64encode(payload).decode("ascii")
        with tempfile.TemporaryDirectory() as tmp:
            root, work = Path(tmp) / "root", Path(tmp) / "work"
            root.mkdir()
            with self.assertRaises(ValueError):
                transfer.accept_base64_chunk(
                    state, relative="x", offset=0, chunk=encoded + "AAAA", root=root, work_dir=work
                )
            self.assertEqual(state["files"][0]["base64_received_chars"], 0)
            self.assertFalse((work / "parts" / "x.b64part").exists())

    def test_partial_state_divergence_fails_before_append(self):
        payload = b"abcdefghij\n" * 100
        state = transfer.init_state(self._manifest(payload), base64_chunk_chars=1000)
        encoded = base64.b64encode(payload).decode("ascii")
        with tempfile.TemporaryDirectory() as tmp:
            root, work = Path(tmp) / "root", Path(tmp) / "work"
            root.mkdir()
            transfer.accept_base64_chunk(
                state, relative="x", offset=0, chunk=encoded[:100], root=root, work_dir=work
            )
            state["files"][0]["base64_received_chars"] = 101
            with self.assertRaisesRegex(ValueError, "reset required"):
                transfer.accept_base64_chunk(
                    state, relative="x", offset=101, chunk=encoded[101:150], root=root, work_dir=work
                )
            self.assertEqual((work / "parts" / "x.b64part").read_text(), encoded[:100])

    def test_bad_final_chunk_does_not_advance_state_or_partial(self):
        payload = b"payload\n"
        state = transfer.init_state(self._manifest(payload), base64_chunk_chars=1000)
        encoded = base64.b64encode(payload).decode("ascii")
        bad = encoded[:-4] + base64.b64encode(b"wrong").decode("ascii")[:4]
        with tempfile.TemporaryDirectory() as tmp:
            root, work = Path(tmp) / "root", Path(tmp) / "work"
            root.mkdir()
            with self.assertRaises(ValueError):
                transfer.accept_base64_chunk(
                    state, relative="x", offset=0, chunk=bad, root=root, work_dir=work
                )
            self.assertEqual(state["files"][0]["base64_received_chars"], 0)
            self.assertFalse((work / "parts" / "x.b64part").exists())


if __name__ == "__main__":
    unittest.main()
