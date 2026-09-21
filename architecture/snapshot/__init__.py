"""Repository snapshot transport and verification helpers.

These helpers support controlled repository recovery/materialization. They are
repository-governance infrastructure, not scientific-evidence surfaces.
"""

from .generate import SnapshotGenerationError, build_manifest, manifest_payload, write_manifest
from .git_objects import git_blob_id, git_object_id, sha256_digest, tree_id_from_files
from .manifest import SnapshotFile, SnapshotManifest, load_manifest
from .verify import Finding, verify_snapshot

__all__ = [
    "Finding",
    "SnapshotFile",
    "SnapshotGenerationError",
    "SnapshotManifest",
    "build_manifest",
    "git_blob_id",
    "git_object_id",
    "load_manifest",
    "manifest_payload",
    "sha256_digest",
    "tree_id_from_files",
    "verify_snapshot",
    "write_manifest",
]
