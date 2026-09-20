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
from typing import Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from data.ncei_ghcnd_bulk import (
    GHCNObservationPartitionKey,
    GHCNRoutingSummary,
    RoutedGHCNObservation,
    SOURCE_ID,
    StationRoutingLookup,
    stream_by_year_partitions,
)
from src.station_federation import ObservationPartitionRef


MEDIA_TYPE = "application/vnd.climate.station-parquet-partition+json"
_SCHEMA_ID = "ncei-ghcnd-parquet-partition/v1"
_CHECKPOINT_SCHEMA = "ncei-ghcnd-parquet-resume/v1"
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


def _input_record_bytes(
    key: GHCNObservationPartitionKey,
    record: RoutedGHCNObservation,
) -> bytes:
    return _canonical_json(
        [
            key.source_id,
            key.spatial_partition,
            key.year,
            key.element,
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


def _write_json_atomic(path: Path, payload: object) -> None:
    encoded = _canonical_json(payload)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


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
        max_partitions: int,
        batch_rows: int = 50_000,
        compression: str = "zstd",
    ) -> None:
        if max_partitions <= 0:
            raise ValueError("max_partitions must be positive")
        if batch_rows <= 0:
            raise ValueError("batch_rows must be positive")
        if not compression or not compression.strip():
            raise ValueError("compression must be non-empty")
        self.root = Path(root)
        self.source_revision = _require_digest(source_revision, "source_revision")
        self.max_partitions = int(max_partitions)
        self.batch_rows = batch_rows
        self.compression = compression.strip()
        self._buffer: list[
            tuple[GHCNObservationPartitionKey, RoutedGHCNObservation]
        ] = []
        self._states: dict[GHCNObservationPartitionKey, _PartitionState] = {}
        self._final_refs: tuple[ObservationPartitionRef, ...] | None = None

        staging_root = self.root / ".staging"
        staging_root.mkdir(parents=True, exist_ok=True)
        resume_token = hashlib.sha256(
            _canonical_json(
                {
                    "schema": _CHECKPOINT_SCHEMA,
                    "source_revision": self.source_revision,
                    "max_partitions": self.max_partitions,
                    "batch_rows": self.batch_rows,
                    "compression": self.compression,
                    "pyarrow_version": pa.__version__,
                }
            )
        ).hexdigest()
        self._staging = staging_root / f"ghcnd-parquet-{resume_token}"
        self._checkpoint_path = self._staging / "checkpoint.json"
        self._prefix_path = self._staging / "input-prefix.ndjson"
        self._skip_remaining = 0
        self._resume_reader = None

        if self._staging.exists():
            if self._checkpoint_path.is_file():
                self._restore_checkpoint()
            else:
                shutil.rmtree(self._staging)
                self._staging.mkdir(parents=True)
        else:
            self._staging.mkdir(parents=True)
        (self.root / "objects").mkdir(parents=True, exist_ok=True)

    def _partition_dir(self, key: GHCNObservationPartitionKey) -> Path:
        return self._staging / _key_token(key)

    def _logical_path(self, key: GHCNObservationPartitionKey) -> Path:
        return self._partition_dir(key) / "logical.ndjson"

    def _checkpoint_payload(self) -> dict[str, object]:
        states = []
        for key in sorted(self._states):
            state = self._states[key]
            logical_path = self._logical_path(key)
            states.append(
                {
                    "key": {
                        "source_id": key.source_id,
                        "spatial_partition": key.spatial_partition,
                        "year": key.year,
                        "element": key.element,
                    },
                    "row_count": state.row_count,
                    "time_start": state.time_start,
                    "time_end": state.time_end,
                    "logical_bytes": logical_path.stat().st_size,
                    "logical_content_digest": (
                        "sha256:" + state.logical_hasher.hexdigest()
                    ),
                    "fragments": list(state.fragments),
                }
            )
        return {
            "schema": _CHECKPOINT_SCHEMA,
            "source_revision": self.source_revision,
            "max_partitions": self.max_partitions,
            "batch_rows": self.batch_rows,
            "compression": self.compression,
            "pyarrow_version": pa.__version__,
            "committed_rows": sum(
                state.row_count for state in self._states.values()
            ),
            "prefix_bytes": self._prefix_path.stat().st_size,
            "states": states,
        }

    def _write_checkpoint(self) -> None:
        _write_json_atomic(self._checkpoint_path, self._checkpoint_payload())

    def _restore_checkpoint(self) -> None:
        try:
            payload = json.loads(self._checkpoint_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("GHCN Parquet resume checkpoint is invalid") from exc
        if not isinstance(payload, dict) or payload.get("schema") != _CHECKPOINT_SCHEMA:
            raise ValueError("GHCN Parquet resume checkpoint schema is invalid")
        expected = {
            "source_revision": self.source_revision,
            "max_partitions": self.max_partitions,
            "batch_rows": self.batch_rows,
            "compression": self.compression,
            "pyarrow_version": pa.__version__,
        }
        for name, value in expected.items():
            if payload.get(name) != value:
                raise ValueError(
                    f"GHCN Parquet resume checkpoint changed at {name}"
                )

        prefix_bytes = payload.get("prefix_bytes")
        committed_rows = payload.get("committed_rows")
        states = payload.get("states")
        if (
            not isinstance(prefix_bytes, int)
            or isinstance(prefix_bytes, bool)
            or prefix_bytes < 0
            or not isinstance(committed_rows, int)
            or isinstance(committed_rows, bool)
            or committed_rows < 0
            or not isinstance(states, list)
        ):
            raise ValueError("GHCN Parquet resume checkpoint counters are invalid")
        if not self._prefix_path.is_file():
            raise ValueError("GHCN Parquet resume prefix is missing")
        if self._prefix_path.stat().st_size < prefix_bytes:
            raise ValueError("GHCN Parquet resume prefix is truncated")
        with self._prefix_path.open("r+b") as handle:
            handle.truncate(prefix_bytes)

        restored_rows = 0
        for item in states:
            if not isinstance(item, dict) or not isinstance(item.get("key"), dict):
                raise ValueError("GHCN Parquet resume partition state is invalid")
            raw_key = item["key"]
            try:
                key = GHCNObservationPartitionKey(
                    str(raw_key["source_id"]),
                    str(raw_key["spatial_partition"]),
                    int(raw_key["year"]),
                    str(raw_key["element"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("GHCN Parquet resume partition key is invalid") from exc
            if key.source_id != SOURCE_ID:
                raise ValueError("GHCN Parquet resume partition source changed")
            row_count = item.get("row_count")
            logical_bytes = item.get("logical_bytes")
            fragments = item.get("fragments")
            if (
                not isinstance(row_count, int)
                or isinstance(row_count, bool)
                or row_count < 0
                or not isinstance(logical_bytes, int)
                or isinstance(logical_bytes, bool)
                or logical_bytes < 0
                or not isinstance(fragments, list)
            ):
                raise ValueError("GHCN Parquet resume partition counters are invalid")

            partition_dir = self._partition_dir(key)
            logical_path = self._logical_path(key)
            if not logical_path.is_file() or logical_path.stat().st_size < logical_bytes:
                raise ValueError("GHCN Parquet resume logical sidecar is missing/truncated")
            with logical_path.open("r+b") as handle:
                handle.truncate(logical_bytes)

            expected_fragment_names = {
                str(fragment.get("filename"))
                for fragment in fragments
                if isinstance(fragment, dict)
            }
            for extra in partition_dir.glob("part-*.parquet"):
                if extra.name not in expected_fragment_names:
                    extra.unlink()

            hasher = hashlib.sha256()
            logical_rows = 0
            with logical_path.open("rb") as handle:
                for line in handle:
                    hasher.update(line)
                    logical_rows += 1
            if logical_rows != row_count:
                raise ValueError("GHCN Parquet resume logical row count changed")
            observed_logical_digest = "sha256:" + hasher.hexdigest()
            if observed_logical_digest != item.get("logical_content_digest"):
                raise ValueError("GHCN Parquet resume logical digest changed")

            fragment_rows = 0
            normalized_fragments: list[dict[str, object]] = []
            for fragment in fragments:
                if not isinstance(fragment, dict):
                    raise ValueError("GHCN Parquet resume fragment metadata is invalid")
                path = partition_dir / str(fragment.get("filename", ""))
                if not path.is_file():
                    raise ValueError("GHCN Parquet resume fragment is missing")
                digest, byte_count = _sha256_file(path)
                if (
                    digest != fragment.get("sha256")
                    or byte_count != fragment.get("byte_count")
                ):
                    raise ValueError("GHCN Parquet resume fragment digest changed")
                fragment_row_count = fragment.get("row_count")
                if (
                    not isinstance(fragment_row_count, int)
                    or isinstance(fragment_row_count, bool)
                    or fragment_row_count < 0
                ):
                    raise ValueError("GHCN Parquet resume fragment row count is invalid")
                fragment_rows += fragment_row_count
                normalized_fragments.append(dict(fragment))
            if fragment_rows != row_count:
                raise ValueError("GHCN Parquet resume fragment rows changed")

            state = _PartitionState(
                key=key,
                row_count=row_count,
                time_start=item.get("time_start"),
                time_end=item.get("time_end"),
                logical_hasher=hasher,
                fragments=normalized_fragments,
            )
            self._states[key] = state
            restored_rows += row_count

        if restored_rows != committed_rows:
            raise ValueError("GHCN Parquet resume committed row count changed")
        self._skip_remaining = committed_rows
        if committed_rows:
            self._resume_reader = self._prefix_path.open("rb")

    def _resume_prefix_record(
        self,
        key: GHCNObservationPartitionKey,
        record: RoutedGHCNObservation,
    ) -> bool:
        if self._skip_remaining <= 0:
            return False
        assert self._resume_reader is not None
        expected = self._resume_reader.readline()
        observed = _input_record_bytes(key, record)
        if expected != observed:
            raise ValueError(
                "GHCN Parquet resume replay does not match the committed input prefix"
            )
        self._skip_remaining -= 1
        if self._skip_remaining == 0:
            if self._resume_reader.read(1) != b"":
                raise ValueError("GHCN Parquet resume prefix has uncommitted records")
            self._resume_reader.close()
            self._resume_reader = None
        return True

    def _state(self, key: GHCNObservationPartitionKey) -> _PartitionState:
        state = self._states.get(key)
        if state is None:
            if len(self._states) >= self.max_partitions:
                raise ValueError(
                    "GHCN Parquet publication exceeded max_partitions"
                )
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
        if self._resume_prefix_record(key, record):
            return

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
        if self._skip_remaining:
            raise ValueError(
                "GHCN Parquet resume replay is incomplete before new rows"
            )
        grouped: dict[
            GHCNObservationPartitionKey, list[RoutedGHCNObservation]
        ] = {}
        for key, record in self._buffer:
            grouped.setdefault(key, []).append(record)

        pending_fragments: dict[
            GHCNObservationPartitionKey, dict[str, object]
        ] = {}
        for key in sorted(grouped):
            records = grouped[key]
            state = self._states[key]
            partition_dir = self._partition_dir(key)
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
            pending_fragments[key] = {
                "filename": filename,
                "sha256": digest,
                "byte_count": byte_count,
                "row_count": len(records),
            }

        for key in sorted(grouped):
            logical_path = self._logical_path(key)
            with logical_path.open("ab") as handle:
                for record in grouped[key]:
                    handle.write(_record_bytes(record))
                handle.flush()
                os.fsync(handle.fileno())

        with self._prefix_path.open("ab") as handle:
            for key, record in self._buffer:
                handle.write(_input_record_bytes(key, record))
            handle.flush()
            os.fsync(handle.fileno())

        for key, fragment in pending_fragments.items():
            self._states[key].fragments.append(fragment)
        self._buffer.clear()
        self._write_checkpoint()

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
        if self._skip_remaining:
            raise ValueError(
                "GHCN Parquet resume replay ended before committed prefix was verified"
            )
        self._flush_batch()
        supersedes = dict(supersedes_by_key or {})
        unknown_keys = set(supersedes) - set(self._states)
        if unknown_keys:
            raise ValueError("supersession metadata names an unpublished partition")
        for values in supersedes.values():
            for digest in values:
                _require_digest(digest, "supersedes digest")

        # The source stream is fully consumed. Resume-only state is no longer
        # needed; a crash during short final object publication safely restarts
        # deterministic publication from the captured source artifact.
        self._checkpoint_path.unlink(missing_ok=True)
        self._prefix_path.unlink(missing_ok=True)

        refs: list[ObservationPartitionRef] = []
        for key in sorted(self._states):
            state = self._states[key]
            self._logical_path(key).unlink(missing_ok=True)
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

    def abort(self, *, preserve_checkpoint: bool = True) -> None:
        if self._resume_reader is not None:
            self._resume_reader.close()
            self._resume_reader = None
        self._buffer.clear()
        if self._final_refs is not None:
            return
        if preserve_checkpoint and self._checkpoint_path.is_file():
            return
        shutil.rmtree(self._staging, ignore_errors=True)


def publish_gzip_by_year(
    path: Path,
    *,
    expected_year: int,
    lookup: StationRoutingLookup,
    root: Path,
    max_partitions: int,
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
        max_partitions=max_partitions,
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
                max_partition_keys=max_partitions,
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
