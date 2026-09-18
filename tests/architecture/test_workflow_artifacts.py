from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from architecture import check_workflow_artifacts


class WorkflowArtifactTests(unittest.TestCase):
    def test_upload_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workflows = root / ".github" / "workflows"
            workflows.mkdir(parents=True)
            (workflows / "bad.yml").write_text(
                "jobs:\n"
                "  test:\n"
                "    steps:\n"
                "      - uses: actions/upload-artifact@v4\n",
                encoding="utf-8",
            )
            findings = check_workflow_artifacts.check(root)
            self.assertEqual(
                [item.code for item in findings],
                ["workflow.ephemeral_artifact_upload"],
            )

    def test_named_upload_artifact_step_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workflows = root / ".github" / "workflows"
            workflows.mkdir(parents=True)
            (workflows / "bad.yml").write_text(
                "jobs:\n"
                "  test:\n"
                "    steps:\n"
                "      - name: Persist experiment receipts\n"
                "        uses: actions/upload-artifact@v4\n"
                "        with:\n"
                "          path: run-artifacts/\n",
                encoding="utf-8",
            )
            findings = check_workflow_artifacts.check(root)
            self.assertEqual(
                [item.code for item in findings],
                ["workflow.ephemeral_artifact_upload"],
            )

    def test_temporary_generation_without_upload_is_allowed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workflows = root / ".github" / "workflows"
            workflows.mkdir(parents=True)
            (workflows / "good.yml").write_text(
                "jobs:\n"
                "  test:\n"
                "    steps:\n"
                "      - run: python make_result.py > result.json\n",
                encoding="utf-8",
            )
            self.assertEqual(check_workflow_artifacts.check(root), [])

    def test_repository_workflows_have_no_ephemeral_uploads(self):
        self.assertEqual(check_workflow_artifacts.check(), [])


if __name__ == "__main__":
    unittest.main()
