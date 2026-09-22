"""Generate authoritative manifests from an exact clean Git checkout."""
from __future__ import annotations

import base64
import json
import subprocess
from pathlib import Path

from .git_objects import ALLOWED_FILE_MODES, git_blob_id, git_object_id, tree_id_from_files
from .manifest import SnapshotFile, SnapshotManifest


class SnapshotGenerationError(RuntimeError):
    pass


def _git(root: Path, *args: str, text: bool = True) -> str | bytes:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=text,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = ""
        if isinstance(exc, subprocess.CalledProcessError):
            stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            detail = f": {stderr.strip()}" if stderr.strip() else ""
        raise SnapshotGenerationError(f"git {' '.join(args)} failed{detail}") from exc
    return result.stdout


def _repo_root(root: Path) -> Path:
    requested = root.resolve()
    actual = Path(str(_git(requested, "rev-parse", "--show-toplevel")).strip()).resolve()
    if requested != actual:
        raise SnapshotGenerationError(
            f"snapshot root must be the Git worktree root: requested {requested}, actual {actual}"
        )
    return actual


def _source_identity(root: Path, source_ref: str) -> tuple[str, str, bytes]:
    source_commit = str(_git(root, "rev-parse", f"{source_ref}^{{commit}}")).strip()
    head_commit = str(_git(root, "rev-parse", "HEAD^{commit}")).strip()
    if source_commit != head_commit:
        raise SnapshotGenerationError(
            f"source ref {source_ref!r} resolves to {source_commit}, but checkout HEAD is {head_commit}"
        )
    tree_sha1 = str(_git(root, "rev-parse", f"{source_commit}^{{tree}}")).strip()
    commit_object = _git(root, "cat-file", "commit", source_commit, text=False)
    assert isinstance(commit_object, bytes)
    observed_commit = git_object_id("commit", commit_object)
    if observed_commit != source_commit:
        raise SnapshotGenerationError(
            f"source commit object hashes to {observed_commit}, expected {source_commit}"
        )
    if not commit_object.startswith(f"tree {tree_sha1}\n".encode("ascii")):
        raise SnapshotGenerationError(
            f"source commit object does not point to root tree {tree_sha1}"
        )
    return source_commit, tree_sha1, commit_object


def _require_clean(root: Path) -> None:
    status = _git(
        root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        text=False,
    )
    if status:
        raise SnapshotGenerationError(
            "checkout is not clean; tracked modifications or untracked files would make "
            "the exported snapshot differ from the bound source commit"
        )


def _tree_rows(root: Path, source_commit: str) -> list[SnapshotFile]:
    raw = _git(root, "ls-tree", "-r", "-z", "--full-tree", source_commit, text=False)
    assert isinstance(raw, bytes)
    files: list[SnapshotFile] = []
    for entry in raw.split(b"\0"):
        if not entry:
            continue
        meta, raw_path = entry.split(b"\t", 1)
        mode_b, kind_b, object_b = meta.split(b" ", 2)
        mode = mode_b.decode("ascii")
        kind = kind_b.decode("ascii")
        object_id = object_b.decode("ascii")
        path = raw_path.decode("utf-8", errors="strict")
        if kind != "blob":
            raise SnapshotGenerationError(
                f"snapshot tree contains unsupported Git object type {kind!r} at {path}"
            )
        if mode not in ALLOWED_FILE_MODES:
            raise SnapshotGenerationError(
                f"snapshot tree contains unsupported file mode {mode!r} at {path}; "
                "symlinks and other non-regular entries require an explicit transport contract"
            )
        local = root / path
        if not local.is_file() or local.is_symlink():
            raise SnapshotGenerationError(f"tracked path is not an ordinary local file: {path}")
        payload = local.read_bytes()
        actual_blob = git_blob_id(payload)
        if actual_blob != object_id:
            raise SnapshotGenerationError(
                f"working-tree bytes for {path} do not match source commit: "
                f"expected {object_id}, found {actual_blob}"
            )
        files.append(
            SnapshotFile(
                path=path,
                git_blob_sha1=object_id,
                mode=mode,
                size=len(payload),
            )
        )
    return sorted(files, key=lambda row: row.path.encode("utf-8"))


def build_manifest(root: Path, *, source_ref: str = "HEAD") -> SnapshotManifest:
    root = _repo_root(root)
    source_commit, tree_sha1, commit_object = _source_identity(root, source_ref)
    _require_clean(root)
    files = _tree_rows(root, source_commit)
    rebuilt = tree_id_from_files(
        (row.path, row.mode, row.git_blob_sha1) for row in files
    )
    if rebuilt != tree_sha1:
        raise SnapshotGenerationError(
            f"manifest rows reconstruct Git tree {rebuilt}, expected {tree_sha1}"
        )
    return SnapshotManifest(
        files=tuple(files),
        complete_tree=True,
        tree_sha1=tree_sha1,
        source_commit=source_commit,
        source_commit_object=commit_object,
    )


def manifest_payload(manifest: SnapshotManifest) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 2 if manifest.source_commit_object is not None else 1,
        "complete_tree": manifest.complete_tree,
        "tree_sha1": manifest.tree_sha1,
        "source_commit": manifest.source_commit,
    }
    if manifest.source_commit_object is not None:
        payload["source_commit_object_base64"] = base64.b64encode(
            manifest.source_commit_object
        ).decode("ascii")
    payload["files"] = [
        {
            "path": row.path,
            "git_blob_sha1": row.git_blob_sha1,
            "mode": row.mode,
            "size": row.size,
        }
        for row in manifest.files
    ]
    return payload


def write_manifest(
    manifest: SnapshotManifest,
    output: Path,
    *,
    root: Path | None = None,
) -> None:
    output = output.resolve()
    if root is not None:
        resolved_root = root.resolve()
        try:
            output.relative_to(resolved_root)
        except ValueError:
            pass
        else:
            raise SnapshotGenerationError(
                "authoritative snapshot manifests must be written outside the source tree; "
                "otherwise the manifest would invalidate complete-tree verification"
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest_payload(manifest), indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
