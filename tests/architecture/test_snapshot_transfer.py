from __future__ import annotations

import base64
import tempfile
import unittest
from pathlib import Path

from architecture.snapshot import git_blob_id
from architecture.snapshot.manifest import parse_manifest
from architecture.snapshot import transfer


class SnapshotTransferTests(unittest.TestCase):
    def _manifest(self, path: str, payload: bytes):
        return parse_manifest({
            "schema_version": 1,
            "files": [{"path": path, "git_blob_sha1": git_blob_id(payload), "mode": "100644", "size": len(payload)}],
        })

    def test_base64_requests_are_bounded_resumable_and_verified(self):
        payload = b"abcdef\n" * 400
        state = transfer.init_state(self._manifest("data.txt", payload), max_response_chars=1200, base64_chunk_chars=1000)
        encoded = base64.b64encode(payload).decode("ascii")
        with tempfile.TemporaryDirectory() as tmp:
            root, work = Path(tmp) / "root", Path(tmp) / "work"
            root.mkdir()
            while (request := transfer.next_request(state)) is not None:
                self.assertLessEqual(request["slice_end"] - request["slice_start"], 1000)
                start, end = request["slice_start"], request["slice_end"]
                transfer.accept_base64_chunk(state, relative="data.txt", offset=start, chunk=encoded[start:end], root=root, work_dir=work)
            self.assertEqual((root / "data.txt").read_bytes(), payload)

    def test_base64_rejects_gap_and_corrupt_final_payload(self):
        payload = b"payload\n"
        state = transfer.init_state(self._manifest("x", payload))
        encoded = base64.b64encode(payload).decode("ascii")
        with tempfile.TemporaryDirectory() as tmp:
            root, work = Path(tmp) / "root", Path(tmp) / "work"
            root.mkdir()
            with self.assertRaises(ValueError):
                transfer.accept_base64_chunk(state, relative="x", offset=4, chunk=encoded, root=root, work_dir=work)
            bad = encoded[:-4] + base64.b64encode(b"wrong").decode("ascii")[:4]
            with self.assertRaises(ValueError):
                transfer.accept_base64_chunk(state, relative="x", offset=0, chunk=bad, root=root, work_dir=work)
            self.assertFalse((root / "x").exists())

    def test_text_fallback_requires_overlap_and_complete_lines(self):
        payload = b"one\ntwo\nthree\n"
        state = transfer.init_state(self._manifest("x.txt", payload), method="text-lines", line_window=3, line_overlap=1)
        with tempfile.TemporaryDirectory() as tmp:
            root, work = Path(tmp) / "root", Path(tmp) / "work"
            root.mkdir()
            transfer.accept_text_window(state, relative="x.txt", start_line=1, text="one\ntwo\n", eof=False, root=root, work_dir=work)
            self.assertEqual(transfer.next_request(state)["start_line"], 2)
            with self.assertRaises(ValueError):
                transfer.accept_text_window(state, relative="x.txt", start_line=2, text="two\nthr", eof=False, root=root, work_dir=work)
            transfer.accept_text_window(state, relative="x.txt", start_line=2, text="two\nthree\n", eof=True, root=root, work_dir=work)
            self.assertEqual((root / "x.txt").read_bytes(), payload)

    def test_reconcile_adopts_only_exact_existing_files(self):
        payload = b"exact\n"
        state = transfer.init_state(self._manifest("x.txt", payload))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "x.txt"
            path.write_bytes(payload)
            path.chmod(0o644)
            self.assertEqual(transfer.reconcile_existing(state, root=root), [])
            self.assertEqual(state["files"][0]["status"], "verified")

    def test_reconcile_reopens_verified_file_that_disappeared(self):
        payload = b"exact\n"
        state = transfer.init_state(self._manifest("x.txt", payload))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "x.txt"
            path.write_bytes(payload)
            path.chmod(0o644)
            self.assertEqual(transfer.reconcile_existing(state, root=root), [])
            self.assertEqual(state["files"][0]["status"], "verified")
            path.unlink()
            issues = transfer.reconcile_existing(state, root=root)
            self.assertEqual(issues[0]["problem"], "verified file is absent")
            self.assertEqual(state["files"][0]["status"], "pending")
            self.assertIsNotNone(transfer.next_request(state))


if __name__ == "__main__":
    unittest.main()
