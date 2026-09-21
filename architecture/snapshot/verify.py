"""Verify local files against an authoritative snapshot manifest."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .git_objects import git_blob_id, local_file_mode, sha256_digest, tree_id_from_files
from .manifest import SnapshotManifest


@dataclass(frozen=True)
class Finding:
    code: str
    path: str
    message: str


def _filesystem_files(root: Path) -> set[str]:
    files: set[str] = set()
    for path in root.rglob("*"):
        if ".git" in path.relative_to(root).parts:
            continue
        if path.is_symlink() or path.is_file():
            files.add(path.relative_to(root).as_posix())
    return files


def verify_snapshot(root: Path, manifest: SnapshotManifest) -> list[Finding]:
    root = root.resolve()
    findings: list[Finding] = []
    verified: list[tuple[str, str, str]] = []
    expected_paths = {row.path for row in manifest.files}

    for row in manifest.files:
        path = root / Path(*PurePosixPath(row.path).parts)
        if not path.exists():
            findings.append(Finding("snapshot.missing", row.path, "expected file is absent"))
            continue
        if path.is_symlink() or not path.is_file():
            findings.append(Finding("snapshot.file_type", row.path, "expected an ordinary file"))
            continue

        payload = path.read_bytes()
        if len(payload) != row.size:
            findings.append(Finding("snapshot.size", row.path, f"expected {row.size} bytes, found {len(payload)}"))
        blob = git_blob_id(payload)
        if blob != row.git_blob_sha1:
            findings.append(Finding(
                "snapshot.blob",
                row.path,
                f"expected Git blob {row.git_blob_sha1}, found {blob}; sha256={sha256_digest(payload)}",
            ))
        mode = local_file_mode(path)
        if mode != row.mode:
            findings.append(Finding("snapshot.mode", row.path, f"expected mode {row.mode}, found {mode}"))
        verified.append((row.path, mode, blob))

    if manifest.complete_tree:
        for unexpected in sorted(_filesystem_files(root) - expected_paths):
            findings.append(Finding("snapshot.unexpected", unexpected, "file is not declared by complete manifest"))
        blocking = {"snapshot.missing", "snapshot.file_type", "snapshot.blob", "snapshot.mode"}
        if len(verified) == len(manifest.files) and not any(f.code in blocking for f in findings):
            actual_tree = tree_id_from_files(verified)
            if actual_tree != manifest.tree_sha1:
                findings.append(Finding("snapshot.tree", ".", f"expected root Git tree {manifest.tree_sha1}, found {actual_tree}"))

    return findings


def restore_modes(root: Path, manifest: SnapshotManifest) -> list[Finding]:
    """Restore only executable-mode metadata after all non-mode invariants match."""
    if (
        not manifest.complete_tree
        or manifest.tree_sha1 is None
        or manifest.source_commit is None
    ):
        raise ValueError(
            "mode restoration requires a complete-tree snapshot manifest with commit/tree identity"
        )
    findings = verify_snapshot(root, manifest)
    blockers = [finding for finding in findings if finding.code != "snapshot.mode"]
    if blockers:
        detail = "; ".join(
            f"{finding.code}:{finding.path}" for finding in blockers[:5]
        )
        raise ValueError(
            "refusing mode restoration because snapshot content is not otherwise exact: "
            + detail
        )

    root = root.resolve()
    for row in manifest.files:
        path = root / Path(*PurePosixPath(row.path).parts)
        path.chmod(0o755 if row.mode == "100755" else 0o644)
    return verify_snapshot(root, manifest)
