from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from architecture import check_markdown_links


class MarkdownLinkTests(unittest.TestCase):
    def test_missing_relative_inline_target_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "docs").mkdir()
            (root / "docs" / "page.md").write_text(
                "[missing](generated/STATUS.md)\n", encoding="utf-8"
            )
            findings = check_markdown_links.check(root)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].code, "markdown_link.missing_target")
        self.assertEqual(findings[0].target, "generated/STATUS.md")

    def test_external_anchor_and_inline_code_targets_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "README.md").write_text(
                "[web](https://example.com) [mail](mailto:x@example.com) "
                "[anchor](#local) ` [code](missing.md) `\n",
                encoding="utf-8",
            )
            self.assertEqual(check_markdown_links.check(root), [])

    def test_reference_definition_and_percent_decoded_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "docs").mkdir()
            (root / "docs" / "target file.md").write_text("ok\n", encoding="utf-8")
            (root / "docs" / "page.md").write_text(
                "[ok]: target%20file.md#section\n[bad]: missing.md\n",
                encoding="utf-8",
            )
            findings = check_markdown_links.check(root)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].target, "missing.md")

    def test_target_outside_repository_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "README.md").write_text("[escape](../outside.md)\n", encoding="utf-8")
            findings = check_markdown_links.check(root)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].code, "markdown_link.outside_repository")


if __name__ == "__main__":
    unittest.main()
