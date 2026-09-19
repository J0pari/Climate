"""Immutable content-addressed publication for raw GHCNh annual archives."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import tarfile
from typing import Mapping, Sequence

from data.ncei_ghcnh_bulk import (
    SOURCE_ID,
    GHCNhAliasShardLookup,
    GHCNhObservationPartitionKey,
    GHCNhObservationSink,
    RoutedGHCNhObservation,
    stream_station_year_psv,
)
from src.station_federation import ObservationPartitionRef


SCHEMA = "climate-ghcnh-raw-psv-partition/v1"
MEDIA_TYPE = "application/vnd.climate.ghcnh-raw-psv-partition"
VARIABLE_ID = "GHCNH_RAW_PSV"
_MEMBER = re.compile(r"^GHCNh_([A-Za-z0-9]{11})_([0-9]{4})\.psv$")


def _canonical_json(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return "sha256:" + hasher.hexdigest()


@dataclass
class _PartitionState:
    key: GHCNhObservationPartitionKey
    directory: Path
    records_path: Path
    hasher: object = field(default_factory=hashlib.sha256)
    row_count: int = 0
    byte_count: int = 0
    time_start: str | None = None
    time_end: str | None = None
    provider_fields: set[str] = field(default_factory=set)


class GHCNhRawPartitionSink(GHCNhObservationSink):
    """Disk-backed bounded-memory raw-row sink for one captured source revision."""

    def __init__(
        self,
        object_root: Path,
        *,
        source_revision: str,
        supersedes: Mapping[GHCNhObservationPartitionKey, Sequence[str]] | None = None,
    ) -> None:
        if not source_revision.startswith("sha256:") or len(source_revision) != 71:
            raise ValueError("source_revision must be a SHA-256 identity")
        self.object_root = Path(object_root)
        self.source_revision = source_revision
        self.supersedes = {
            key: tuple(values)
            for key, values in (supersedes or {}).items()
        }
        token = source_revision.removeprefix("sha256:")[:20]
        self.staging = self.object_root / ".staging" / f"ghcnh-{token}"
        if self.staging.exists():
            shutil.rmtree(self.staging)
        self.staging.mkdir(parents=True, exist_ok=False)
        self._states: dict[GHCNhObservationPartitionKey, _PartitionState] = {}
        self._closed = False

    def _state(self, key: GHCNhObservationPartitionKey) -> _PartitionState:
        if key.source_id != SOURCE_ID:
            raise ValueError("GHCNh sink received another provider source_id")
        state = self._states.get(key)
        if state is not None:
            return state
        token = hashlib.sha256(
            _canonical_json(
                {
                    "source_id": key.source_id,
                    "spatial_partition": key.spatial_partition,
                    "year": key.year,
                }
            )
        ).hexdigest()
        directory = self.staging / token
        directory.mkdir(parents=True)
        state = _PartitionState(
            key=key,
            directory=directory,
            records_path=directory / "records.ndjson",
        )
        self._states[key] = state
        return state

    def write(
        self,
        key: GHCNhObservationPartitionKey,
        record: RoutedGHCNhObservation,
    ) -> None:
        if self._closed:
            raise ValueError("GHCNh partition sink is closed")
        state = self._state(key)
        date = record.observation_datetime[:10]
        encoded = _canonical_json(
            {
                "canonical_station_id": record.canonical_station_id,
                "observation_datetime": record.observation_datetime,
                "provider_fields": [list(item) for item in record.provider_fields],
            }
        )
        with state.records_path.open("ab") as handle:
            handle.write(encoded)
        state.hasher.update(encoded)
        state.row_count += 1
        state.byte_count += len(encoded)
        state.time_start = date if state.time_start is None else min(state.time_start, date)
        state.time_end = date if state.time_end is None else max(state.time_end, date)
        state.provider_fields.update(name for name, _ in record.provider_fields)

    def finalize(self) -> tuple[ObservationPartitionRef, ...]:
        if self._closed:
            raise ValueError("GHCNh partition sink is closed")
        self._closed = True
        refs: list[ObservationPartitionRef] = []
        objects = self.object_root / "objects"
        objects.mkdir(parents=True, exist_ok=True)

        for key in sorted(self._states):
            state = self._states[key]
            if state.row_count <= 0 or state.time_start is None or state.time_end is None:
                raise ValueError("GHCNh partition cannot finalize empty state")
            records_digest = "sha256:" + state.hasher.hexdigest()
            manifest = {
                "schema": SCHEMA,
                "source_id": SOURCE_ID,
                "source_revision": self.source_revision,
                "spatial_partition": key.spatial_partition,
                "year": key.year,
                "time_start": state.time_start,
                "time_end": state.time_end,
                "variable_ids": [VARIABLE_ID],
                "provider_field_names": sorted(state.provider_fields),
                "row_count": state.row_count,
                "records_digest": records_digest,
                "records_bytes": state.byte_count,
            }
            manifest_bytes = _canonical_json(manifest)
            digest = _sha256_bytes(manifest_bytes)
            target = objects / digest.removeprefix("sha256:")
            if target.exists():
                observed_manifest = (target / "manifest.json").read_bytes()
                if observed_manifest != manifest_bytes:
                    raise ValueError(
                        "content-addressed GHCNh object has conflicting manifest"
                    )
                if _sha256_file(target / "records.ndjson") != records_digest:
                    raise ValueError(
                        "content-addressed GHCNh object has conflicting records"
                    )
                shutil.rmtree(state.directory)
            else:
                (state.directory / "manifest.json").write_bytes(manifest_bytes)
                os.replace(state.directory, target)

            refs.append(
                ObservationPartitionRef(
                    source_id=SOURCE_ID,
                    spatial_partition=key.spatial_partition,
                    time_start=state.time_start,
                    time_end=state.time_end,
                    variable_ids=(VARIABLE_ID,),
                    digest=digest,
                    source_revision=self.source_revision,
                    media_type=MEDIA_TYPE,
                    row_count=state.row_count,
                    byte_count=state.byte_count + len(manifest_bytes),
                    supersedes=self.supersedes.get(key, ()),
                )
            )

        shutil.rmtree(self.staging, ignore_errors=True)
        return tuple(refs)

    def abort(self) -> None:
        self._closed = True
        shutil.rmtree(self.staging, ignore_errors=True)


@dataclass(frozen=True)
class GHCNhArchivePublication:
    source_revision: str
    archive_member_count: int
    partitions: tuple[ObservationPartitionRef, ...]


def publish_year_archive(
    archive_path: Path,
    *,
    expected_year: int,
    lookup: GHCNhAliasShardLookup,
    object_root: Path,
    supersedes: Mapping[GHCNhObservationPartitionKey, Sequence[str]] | None = None,
) -> GHCNhArchivePublication:
    """Stream a captured annual GHCNh tar.gz into immutable raw-row partitions."""
    archive_path = Path(archive_path)
    source_revision = _sha256_file(archive_path)
    sink = GHCNhRawPartitionSink(
        object_root,
        source_revision=source_revision,
        supersedes=supersedes,
    )
    member_count = 0
    try:
        with archive_path.open("rb") as raw:
            with tarfile.open(fileobj=raw, mode="r|gz") as archive:
                for member in archive:
                    if not member.isfile():
                        continue
                    basename = Path(member.name).name
                    if not basename.endswith(".psv"):
                        continue
                    matched = _MEMBER.fullmatch(basename)
                    if matched is None:
                        raise ValueError(
                            f"GHCNh archive contains unexpected PSV member {basename!r}"
                        )
                    station_id, year_text = matched.groups()
                    if int(year_text) != expected_year:
                        raise ValueError(
                            f"GHCNh archive member {basename!r} does not match "
                            f"expected year {expected_year}"
                        )
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        raise ValueError(
                            f"GHCNh archive member {basename!r} could not be read"
                        )
                    with io.TextIOWrapper(
                        extracted,
                        encoding="utf-8",
                        newline="",
                    ) as text:
                        stream_station_year_psv(
                            text,
                            expected_year=expected_year,
                            expected_station_id=station_id,
                            lookup=lookup,
                            sink=sink,
                        )
                    member_count += 1
        if member_count == 0:
            raise ValueError("GHCNh annual archive contains no PSV station members")
        partitions = sink.finalize()
    except Exception:
        sink.abort()
        raise
    return GHCNhArchivePublication(
        source_revision=source_revision,
        archive_member_count=member_count,
        partitions=partitions,
    )
