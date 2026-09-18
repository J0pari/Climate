"""Durable Parquet publication for routed NCEI GHCN-Daily observations.

Provider parsing and station identity remain in data.ncei_ghcnd_bulk. This
module owns only the provider-specific storage adapter that turns routed raw
GHCN observations into immutable, content-addressed Parquet partition objects
and provider-neutral ObservationPartitionRef records.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from data.ncei_ghcnd_bulk import (
    GHCNObservationPartitionKey,
    GHCNRoutingSummary,
    RoutedGHCNObservation,
    SOURCE_ID,
    StationShardLookup,
    stream_by_year_partitions,
)
from src.station_federation import ObservationPartitionRef


MEDIA_TYPE = "application/vnd.climate.station-parquet-partition+json"
_SCHEMA_ID = "ncei-ghcnd-parquet-partition/v1"
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> tuple[str, int]:
    hasher = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
            byte_count += len(chunk)
    return "sha256:" + hasher.hexdigest(), byte_count


def _require_digest(value: str, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be sha256:<64 lowercase hex>")
    return value


def _key_token(key: GHCNObservationPartitionKey) -> str:
    payload = _canonical_json(
        {
            "source_id": key.source_id,
            "spatial_partition": key.spatial_partition,
            "year": key.year,
            "element": key.element,
        }
    )
    return hashlib.sha256(payload).hexdigest()


def _record_bytes(record: RoutedGHCNObservation) -> bytes:
    return _canonical_json(
        [
            record.canonical_station_id,
            record.observation_date,
            record.element,
            record.value,
            record.measurement_flag,
            record.quality_flag,
            record.source_flag,
            record.observation_time,
        ]
    )


@dataclass
class _PartitionState:
    key: GHCNObservationPartitionKey
    row_count: int = 0
    time_start: str | None = None
    time_end: str | None = None
    logical_hasher: object = field(default_factory=hashlib.sha256)
    fragments: list[dict[str, object]] = field(default_factory=list)


@dataclass(frozen=True)
class GHCNParquetPublication:
    source_revision: str
    source_byte_count: int
    routing: GHCNRoutingSummary
    partitions: tuple[ObservationPartitionRef, ...]


class GHCNParquetPartitionPublisher:
    """Bounded-memory sink that publishes immutable Parquet partition objects."""

    def __init__(
        self,
        root: Path,
        *,
        source_revision: str,
        batch_rows: int = 50_000,
        compression: str = "zstd",
    ) -> None:
        if batch_rows <= 0:
            raise ValueError("batch_rows must be positive")
        if not compression or not compression.strip():
            raise ValueError("compression must be non-empty")
        self.root = Path(root)
        self.source_revision = _require_digest(source_revision, "source_revision")
        self.batch_rows = batch_rows
        self.compression = compression.strip()
        self._buffer: list[
            tuple[GHCNObservationPartitionKey, RoutedGHCNObservation]
        ] = []
        self._states: dict[GHCNObservationPartitionKey, _PartitionState] = {}
        self._final_refs: tuple[ObservationPartitionRef, ...] | None = None

        staging_root = self.root / ".staging"
        staging_root.mkdir(parents=True, exist_ok=True)
        self._staging = Path(
            tempfile.mkdtemp(prefix="ghcnd-parquet-", dir=staging_root)
        )
        (self.root / "objects").mkdir(parents=True, exist_ok=True)

    def _state(self, key: GHCNObservationPartitionKey) -> _PartitionState:
        state = self._states.get(key)
        if state is None:
            state = _PartitionState(key)
            self._states[key] = state
        return state

    def write(
        self,
        key: GHCNObservationPartitionKey,
        record: RoutedGHCNObservation,
    ) -> None:
        if self._final_refs is not None:
            raise RuntimeError("publisher is already finalized")
        if key.source_id != SOURCE_ID:
            raise ValueError(
                f"GHCN Parquet publisher requires source_id {SOURCE_ID!r}"
            )
        if record.element != key.element:
            raise ValueError("record element must match partition key element")
        try:
            observed = date.fromisoformat(record.observation_date)
        except (TypeError, ValueError) as exc:
            raise ValueError("record observation_date must be an ISO date") from exc
        if observed.year != key.year:
            raise ValueError("record year must match partition key year")

        state = self._state(key)
        state.row_count += 1
        if state.time_start is None or record.observation_date < state.time_start:
            state.time_start = record.observation_date
        if state.time_end is None or record.observation_date > state.time_end:
            state.time_end = record.observation_date
        state.logical_hasher.update(_record_bytes(record))
        self._buffer.append((key, record))
        if len(self._buffer) >= self.batch_rows:
            self._flush_batch()

    def _fragment_table(
        self,
        records: Sequence[RoutedGHCNObservation],
    ) -> pa.Table:
        schema = pa.schema(
            [
                ("canonical_station_id", pa.string()),
                ("observation_date", pa.date32()),
                ("provider_element", pa.string()),
                ("provider_value", pa.int64()),
                ("measurement_flag", pa.string()),
                ("quality_flag", pa.string()),
                ("source_flag", pa.string()),
                ("observation_time", pa.string()),
            ]
        )
        return pa.Table.from_pydict(
            {
                "canonical_station_id": [
                    item.canonical_station_id for item in records
                ],
                "observation_date": [
                    date.fromisoformat(item.observation_date) for item in records
                ],
                "provider_element": [item.element for item in records],
                "provider_value": [item.value for item in records],
                "measurement_flag": [item.measurement_flag for item in records],
                "quality_flag": [item.quality_flag for item in records],
                "source_flag": [item.source_flag for item in records],
                "observation_time": [item.observation_time for item in records],
            },
            schema=schema,
        )

    def _flush_batch(self) -> None:
        if not self._buffer:
            return
        grouped: dict[
            GHCNObservationPartitionKey, list[RoutedGHCNObservation]
        ] = {}
        for key, record in self._buffer:
            grouped.setdefault(key, []).append(record)

        for key in sorted(grouped):
            records = grouped[key]
            state = self._states[key]
            partition_dir = self._staging / _key_token(key)
            partition_dir.mkdir(parents=True, exist_ok=True)
            filename = f"part-{len(state.fragments):08d}.parquet"
            path = partition_dir / filename
            table = self._fragment_table(records)
            pq.write_table(
                table,
                path,
                compression=self.compression,
                use_dictionary=True,
                write_statistics=True,
            )
            digest, byte_count = _sha256_file(path)
            state.fragments.append(
                {
                    "filename": filename,
                    "sha256": digest,
                    "byte_count": byte_count,
                    "row_count": len(records),
                }
            )
        self._buffer.clear()

    def _manifest_bytes(self, state: _PartitionState) -> bytes:
        assert state.time_start is not None and state.time_end is not None
        return _canonical_json(
            {
                "schema": _SCHEMA_ID,
                "source_id": state.key.source_id,
                "source_revision": self.source_revision,
                "spatial_partition": state.key.spatial_partition,
                "year": state.key.year,
                "element": state.key.element,
                "time_start": state.time_start,
                "time_end": state.time_end,
                "row_count": state.row_count,
                "logical_content_digest": (
                    "sha256:" + state.logical_hasher.hexdigest()
                ),
                "value_semantics": (
                    "provider-native GHCN integer; null denotes the provider "
                    "missing sentinel; climate unit normalization is external"
                ),
                "storage": {
                    "format": "parquet",
                    "compression": self.compression,
                    "pyarrow_version": pa.__version__,
                    "batch_rows": self.batch_rows,
                    "fragments": state.fragments,
                },
            }
        )

    @staticmethod
    def _verify_existing_object(
        object_dir: Path,
        manifest_bytes: bytes,
        fragments: Sequence[dict[str, object]],
    ) -> None:
        manifest_path = object_dir / "manifest.json"
        if not manifest_path.is_file() or manifest_path.read_bytes() != manifest_bytes:
            raise ValueError(
                "content-addressed partition path contains conflicting manifest"
            )
        for fragment in fragments:
            path = object_dir / str(fragment["filename"])
            if not path.is_file():
                raise ValueError(
                    "content-addressed partition is missing a Parquet fragment"
                )
            digest, byte_count = _sha256_file(path)
            if digest != fragment["sha256"] or byte_count != fragment["byte_count"]:
                raise ValueError(
                    "content-addressed partition contains a corrupt Parquet fragment"
                )

    def finalize(
        self,
        *,
        supersedes_by_key: Mapping[
            GHCNObservationPartitionKey, Sequence[str]
        ] | None = None,
    ) -> tuple[ObservationPartitionRef, ...]:
        if self._final_refs is not None:
            return self._final_refs
        self._flush_batch()
        supersedes = dict(supersedes_by_key or {})
        unknown_keys = set(supersedes) - set(self._states)
        if unknown_keys:
            raise ValueError("supersession metadata names an unpublished partition")
        for values in supersedes.values():
            for digest in values:
                _require_digest(digest, "supersedes digest")

        refs: list[ObservationPartitionRef] = []
        for key in sorted(self._states):
            state = self._states[key]
            manifest_bytes = self._manifest_bytes(state)
            digest = _sha256_bytes(manifest_bytes)
            object_dir = self.root / "objects" / digest.removeprefix("sha256:")
            partition_dir = self._staging / _key_token(key)
            manifest_path = partition_dir / "manifest.json"
            manifest_path.write_bytes(manifest_bytes)

            if object_dir.exists():
                self._verify_existing_object(
                    object_dir, manifest_bytes, state.fragments
                )
                shutil.rmtree(partition_dir)
            else:
                os.replace(partition_dir, object_dir)

            byte_count = len(manifest_bytes) + sum(
                int(item["byte_count"]) for item in state.fragments
            )
            refs.append(
                ObservationPartitionRef(
                    source_id=key.source_id,
                    spatial_partition=key.spatial_partition,
                    time_start=state.time_start or "",
                    time_end=state.time_end or "",
                    variable_ids=(key.element,),
                    digest=digest,
                    source_revision=self.source_revision,
                    media_type=MEDIA_TYPE,
                    row_count=state.row_count,
                    byte_count=byte_count,
                    supersedes=tuple(supersedes.get(key, ())),
                )
            )

        self._final_refs = tuple(refs)
        shutil.rmtree(self._staging, ignore_errors=True)
        return self._final_refs

    def abort(self) -> None:
        if self._final_refs is None:
            shutil.rmtree(self._staging, ignore_errors=True)


def publish_gzip_by_year(
    path: Path,
    *,
    expected_year: int,
    lookup: StationShardLookup,
    root: Path,
    batch_rows: int = 50_000,
    compression: str = "zstd",
    supersedes_by_key: Mapping[
        GHCNObservationPartitionKey, Sequence[str]
    ] | None = None,
) -> GHCNParquetPublication:
    """Publish one captured provider by-year artifact without full-year buffering."""
    source_revision, source_byte_count = _sha256_file(Path(path))
    publisher = GHCNParquetPartitionPublisher(
        root,
        source_revision=source_revision,
        batch_rows=batch_rows,
        compression=compression,
    )
    try:
        with gzip.open(path, mode="rt", encoding="ascii", newline="") as handle:
            routing = stream_by_year_partitions(
                handle,
                expected_year=expected_year,
                lookup=lookup,
                sink=publisher,
            )
        partitions = publisher.finalize(
            supersedes_by_key=supersedes_by_key
        )
    except Exception:
        publisher.abort()
        raise
    return GHCNParquetPublication(
        source_revision=source_revision,
        source_byte_count=source_byte_count,
        routing=routing,
        partitions=partitions,
    )
