#!/usr/bin/env python3
"""CLI for exact snapshot identity, verification, and resumable transport state."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from architecture.snapshot.archive import MANIFEST_MEMBER, REPOSITORY_PREFIX
from architecture.snapshot import (
    SnapshotArchiveError,
    SnapshotGenerationError,
    build_manifest,
    create_archive,
    load_manifest,
    restore_modes,
    verify_snapshot,
    write_manifest,
)


def _print_findings(findings) -> None:
    for finding in findings:
        print(f"{finding.code}: {finding.path}: {finding.message}")


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    manifest = sub.add_parser(
        "manifest",
        help="bind a clean Git checkout to a complete-tree snapshot manifest",
    )
    manifest.add_argument("output", type=Path)
    manifest.add_argument("--root", type=Path, default=Path.cwd())
    manifest.add_argument("--source-ref", default="HEAD")

    archive = sub.add_parser(
        "archive",
        help="write one self-contained ZIP bound to an exact clean Git checkout",
    )
    archive.add_argument("output", type=Path)
    archive.add_argument("--root", type=Path, default=Path.cwd())
    archive.add_argument("--source-ref", default="HEAD")

    verify = sub.add_parser("verify")
    verify.add_argument("manifest", type=Path)
    verify.add_argument("--root", type=Path, default=Path.cwd())

    identity = sub.add_parser(
        "identity",
        help="verify an extracted snapshot and print its bound commit/tree identity",
    )
    identity.add_argument("manifest", type=Path)
    identity.add_argument("--root", type=Path, default=Path.cwd())
    identity.add_argument(
        "--restore-modes",
        action="store_true",
        help="repair only 100644/100755 mode differences after all other snapshot invariants match",
    )

    nxt = sub.add_parser("next")
    nxt.add_argument("state", type=Path)

    reconcile = sub.add_parser("reconcile")
    reconcile.add_argument("state", type=Path)
    reconcile.add_argument("--root", type=Path, required=True)

    args = parser.parse_args()

    if args.command == "archive":
        try:
            generated = create_archive(
                args.root,
                args.output,
                source_ref=args.source_ref,
            )
        except (OSError, ValueError, SnapshotArchiveError, SnapshotGenerationError) as exc:
            print(f"snapshot.archive: {exc}", file=sys.stderr)
            return 1
        print(
            json.dumps(
                {
                    "archive": str(args.output.resolve()),
                    "manifest_member": MANIFEST_MEMBER,
                    "repository_prefix": REPOSITORY_PREFIX,
                    "source_commit": generated.source_commit,
                    "tree_sha1": generated.tree_sha1,
                    "file_count": len(generated.files),
                },
                indent=2,
            )
        )
        return 0

    if args.command == "manifest":
        try:
            generated = build_manifest(args.root, source_ref=args.source_ref)
            write_manifest(generated, args.output, root=args.root)
        except (OSError, ValueError, SnapshotGenerationError) as exc:
            print(f"snapshot.manifest: {exc}", file=sys.stderr)
            return 1
        print(
            json.dumps(
                {
                    "manifest": str(args.output.resolve()),
                    "source_commit": generated.source_commit,
                    "tree_sha1": generated.tree_sha1,
                    "file_count": len(generated.files),
                },
                indent=2,
            )
        )
        return 0

    if args.command in {"verify", "identity"}:
        try:
            loaded = load_manifest(args.manifest)
            if args.command == "identity" and args.restore_modes:
                findings = restore_modes(args.root, loaded)
            else:
                findings = verify_snapshot(args.root, loaded)
        except (OSError, ValueError) as exc:
            print(f"snapshot.verify: {exc}", file=sys.stderr)
            return 1
        _print_findings(findings)
        if findings:
            return 1
        if args.command == "verify":
            print("snapshot integrity: clean")
        else:
            print(
                json.dumps(
                    {
                        "source_commit": loaded.source_commit,
                        "tree_sha1": loaded.tree_sha1,
                        "file_count": len(loaded.files),
                        "complete_tree": loaded.complete_tree,
                    },
                    indent=2,
                )
            )
        return 0

    from architecture.snapshot import transfer

    try:
        state = transfer.load_state(args.state)
        if args.command == "next":
            print(json.dumps(transfer.next_request(state), indent=2))
            return 0

        issues = transfer.reconcile_existing(state, root=args.root)
        transfer.save_state(args.state, state)
        print(
            json.dumps(
                {
                    "issues": issues,
                    "next_request": transfer.next_request(state),
                },
                indent=2,
            )
        )
        return 1 if issues else 0
    except (OSError, ValueError) as exc:
        print(f"snapshot.transport: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
