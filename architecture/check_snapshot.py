#!/usr/bin/env python3
"""CLI for exact snapshot identity, verification, and resumable transport state."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from architecture.snapshot import (
    SnapshotGenerationError,
    build_manifest,
    load_manifest,
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

    verify = sub.add_parser("verify")
    verify.add_argument("manifest", type=Path)
    verify.add_argument("--root", type=Path, default=Path.cwd())

    identity = sub.add_parser(
        "identity",
        help="verify an extracted snapshot and print its bound commit/tree identity",
    )
    identity.add_argument("manifest", type=Path)
    identity.add_argument("--root", type=Path, default=Path.cwd())

    nxt = sub.add_parser("next")
    nxt.add_argument("state", type=Path)

    reconcile = sub.add_parser("reconcile")
    reconcile.add_argument("state", type=Path)
    reconcile.add_argument("--root", type=Path, required=True)

    args = parser.parse_args()

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

    state = transfer.load_state(args.state)
    if args.command == "next":
        print(json.dumps(transfer.next_request(state), indent=2))
        return 0

    issues = transfer.reconcile_existing(state, root=args.root)
    transfer.save_state(args.state, state)
    print(json.dumps({"issues": issues, "next_request": transfer.next_request(state)}, indent=2))
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
