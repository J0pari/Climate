"""Repository snapshot transport and verification helpers.

These helpers support controlled repository recovery/materialization. They are
repository-governance infrastructure, not scientific-evidence surfaces.
"""

from .archive import SnapshotArchiveError, create_archive
from .generate import SnapshotGenerationError, build_manifest, manifest_payload, write_manifest
from .git_objects import git_blob_id, git_object_id, sha256_digest, tree_id_from_files
from .manifest import SnapshotFile, SnapshotManifest, load_manifest
from .verify import Finding, restore_modes, verify_snapshot

__all__ = [
    "Finding",
    "SnapshotArchiveError",
    "SnapshotFile",
    "SnapshotGenerationError",
    "SnapshotManifest",
    "build_manifest",
    "create_archive",
    "git_blob_id",
    "git_object_id",
    "load_manifest",
    "manifest_payload",
    "restore_modes",
    "sha256_digest",
    "tree_id_from_files",
    "verify_snapshot",
    "write_manifest",
]
