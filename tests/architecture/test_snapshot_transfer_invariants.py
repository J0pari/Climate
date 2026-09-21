from __future__ import annotations

import base64
import copy
import json
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

    def test_persisted_state_rejects_corrupt_cross_field_invariants(self):
        payload = b"payload\n"
        baseline = transfer.init_state(self._manifest(payload))
        mutations = [
            lambda state: state["files"][0].update(status="bogus"),
            lambda state: state["files"][0].update(base64_expected_chars=999),
            lambda state: state["files"][0].update(base64_received_chars=1),
            lambda state: state["limits"].update(line_overlap=state["limits"]["line_window"]),
            lambda state: state["files"].append(copy.deepcopy(state["files"][0])),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            for mutate in mutations:
                state = copy.deepcopy(baseline)
                mutate(state)
                path.write_text(json.dumps(state), encoding="utf-8")
                with self.assertRaises(ValueError):
                    transfer.load_state(path)

    def test_bad_final_text_window_does_not_advance_state_or_partial(self):
        payload = b"one\ntwo\nthree\n"
        state = transfer.init_state(
            self._manifest(payload),
            method="text-lines",
            line_window=3,
            line_overlap=1,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root, work = Path(tmp) / "root", Path(tmp) / "work"
            root.mkdir()
            transfer.accept_text_window(
                state,
                relative="x",
                start_line=1,
                text="one\ntwo\n",
                eof=False,
                root=root,
                work_dir=work,
            )
            row_before = copy.deepcopy(state["files"][0])
            part = work / "parts" / "x.textpart"
            partial_before = part.read_text(encoding="utf-8")
            with self.assertRaises(ValueError):
                transfer.accept_text_window(
                    state,
                    relative="x",
                    start_line=2,
                    text="two\nWRONG\n",
                    eof=True,
                    root=root,
                    work_dir=work,
                )
            self.assertEqual(state["files"][0], row_before)
            self.assertEqual(part.read_text(encoding="utf-8"), partial_before)
            self.assertFalse((root / "x").exists())

    def test_text_transport_enforces_caps_methods_and_available_overlap(self):
        payload = b"one\ntwo\nthree\n"
        state = transfer.init_state(
            self._manifest(payload),
            method="text-lines",
            max_response_chars=20,
            base64_chunk_chars=10,
            line_window=3,
            line_overlap=2,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root, work = Path(tmp) / "root", Path(tmp) / "work"
            root.mkdir()
            with self.assertRaisesRegex(ValueError, "not base64"):
                transfer.accept_base64_chunk(
                    state,
                    relative="x",
                    offset=0,
                    chunk="AAAA",
                    root=root,
                    work_dir=work,
                )
            with self.assertRaisesRegex(ValueError, "response cap"):
                transfer.accept_text_window(
                    state,
                    relative="x",
                    start_line=1,
                    text="x" * 21,
                    eof=True,
                    root=root,
                    work_dir=work,
                )
            transfer.accept_text_window(
                state,
                relative="x",
                start_line=1,
                text="one\n",
                eof=False,
                root=root,
                work_dir=work,
            )
            request = transfer.next_request(state)
            self.assertEqual(request["overlap_lines"], 1)
            self.assertEqual(request["start_line"], 1)
            transfer.accept_text_window(
                state,
                relative="x",
                start_line=1,
                text="one\ntwo\nthree\n",
                eof=True,
                root=root,
                work_dir=work,
            )
            self.assertEqual((root / "x").read_bytes(), payload)


if __name__ == "__main__":
    unittest.main()
