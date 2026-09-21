#!/usr/bin/env python3
"""CLI for snapshot verification and resumable transport state."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__ in {None, ""}:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from architecture.snapshot import load_manifest, verify_snapshot
from architecture.snapshot import transfer


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    verify = sub.add_parser("verify")
    verify.add_argument("manifest", type=Path)
    verify.add_argument("--root", type=Path, default=Path.cwd())

    nxt = sub.add_parser("next")
    nxt.add_argument("state", type=Path)

    reconcile = sub.add_parser("reconcile")
    reconcile.add_argument("state", type=Path)
    reconcile.add_argument("--root", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "verify":
        findings = verify_snapshot(args.root, load_manifest(args.manifest))
        for finding in findings:
            print(f"{finding.code}: {finding.path}: {finding.message}")
        if not findings:
            print("snapshot integrity: clean")
        return 1 if findings else 0

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
