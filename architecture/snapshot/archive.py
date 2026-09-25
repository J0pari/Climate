"""Create self-contained ZIP handoffs bound to exact Git snapshot identity."""
from __future__ import annotations

import json
import os
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from .generate import build_manifest, manifest_payload
from .git_objects import filesystem_tracks_executable_bit, git_blob_id, local_file_mode
from .manifest import SnapshotManifest

MANIFEST_MEMBER = "climate-snapshot.json"
REPOSITORY_PREFIX = "repository/"
_FIXED_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


class SnapshotArchiveError(RuntimeError):
    pass


def _zip_info(name: str, permissions: int) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=_FIXED_TIMESTAMP)
    info.create_system = 3
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = (stat.S_IFREG | permissions) << 16
    return info


def _repository_path(root: Path, relative: str) -> Path:
    return root / Path(*PurePosixPath(relative).parts)


def create_archive(
    root: Path,
    output: Path,
    *,
    source_ref: str = "HEAD",
) -> SnapshotManifest:
    """Write one ZIP containing an embedded manifest and the exact source tree."""
    root = root.resolve()
    output = output.resolve()
    try:
        output.relative_to(root)
    except ValueError:
        pass
    else:
        raise SnapshotArchiveError(
            "snapshot archive must be written outside the source tree"
        )
    if output.exists():
        raise SnapshotArchiveError(
            f"refusing to overwrite existing snapshot archive: {output}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(root, source_ref=source_ref)
    fd, raw_tmp = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
    )
    os.close(fd)
    tmp = Path(raw_tmp)
    try:
        with zipfile.ZipFile(
            tmp,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive_file:
            metadata = (
                json.dumps(manifest_payload(manifest), indent=2, sort_keys=False)
                + "\n"
            )
            archive_file.writestr(
                _zip_info(MANIFEST_MEMBER, 0o644),
                metadata.encode("utf-8"),
            )
            for row in manifest.files:
                path = _repository_path(root, row.path)
                payload = path.read_bytes()
                if len(payload) != row.size or git_blob_id(payload) != row.git_blob_sha1:
                    raise SnapshotArchiveError(
                        f"source bytes changed after manifest generation: {row.path}"
                    )
                if (
                    filesystem_tracks_executable_bit()
                    and local_file_mode(path) != row.mode
                ):
                    raise SnapshotArchiveError(
                        f"source mode changed after manifest generation: {row.path}"
                    )
                permissions = 0o755 if row.mode == "100755" else 0o644
                archive_file.writestr(
                    _zip_info(REPOSITORY_PREFIX + row.path, permissions),
                    payload,
                )

        observed = build_manifest(
            root,
            source_ref=manifest.source_commit or source_ref,
        )
        if observed != manifest:
            raise SnapshotArchiveError(
                "source checkout changed while snapshot archive was being written"
            )
        os.replace(tmp, output)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
    return manifest
