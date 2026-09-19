"""Bulk NCEI GHCN-Daily catalog/inventory adapter for station federation.

This is the scale path for provider discovery. It parses the provider's complete
station and inventory artifacts and converts them into bounded federation shards.
The subset REST adapter remains useful for captured fixtures but is not the
worldwide ingestion loop.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
import gzip
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Mapping, Protocol, Sequence

from src.station_federation import (
    AliasBinding,
    CatalogShardRef,
    FederatedStation,
    ProviderAlias,
    StationCatalogShard,
    StationLocationEpoch,
    canonical_station_id,
)

SOURCE_ID = "ncei.ghcnd.v3"
GHCN_BASE_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily"
GHCN_STATION_CATALOG_URL = f"{GHCN_BASE_URL}/ghcnd-stations.txt"
GHCN_INVENTORY_URL = f"{GHCN_BASE_URL}/ghcnd-inventory.txt"
GHCN_VERSION_URL = f"{GHCN_BASE_URL}/ghcnd-version.txt"
GHCN_BY_YEAR_URL = f"{GHCN_BASE_URL}/by_year"


_HTTP_RESUME_SCHEMA = "climate-http-resume/v1"
_HTTP_RECEIPT_SCHEMA = "climate-http-artifact/v1"


@dataclass(frozen=True)
class HTTPArtifactIdentity:
    url: str
    content_length: int
    etag: str | None
    last_modified: str | None
    accept_ranges: bool

    def __post_init__(self) -> None:
        if not isinstance(self.url, str) or not self.url.strip():
            raise ValueError("artifact url must be non-empty")
        if self.content_length < 0:
            raise ValueError("artifact content_length must be non-negative")


@dataclass(frozen=True)
class DownloadedHTTPArtifact:
    url: str
    path: Path
    sha256: str
    byte_count: int
    validator_kind: str
    validator_value: str


def _resume_validator(identity: HTTPArtifactIdentity) -> tuple[str, str]:
    etag = identity.etag.strip() if isinstance(identity.etag, str) else ""
    if etag and not etag.startswith("W/"):
        return "etag", etag
    modified = (
        identity.last_modified.strip()
        if isinstance(identity.last_modified, str)
        else ""
    )
    if modified:
        return "last_modified", modified
    raise ValueError(
        "resumable acquisition requires a strong ETag or Last-Modified validator"
    )


def _header_identity(
    url: str,
    headers: Mapping[str, str],
) -> HTTPArtifactIdentity:
    lowered = {key.lower(): value for key, value in headers.items()}
    raw_length = lowered.get("content-length")
    if raw_length is None:
        raise ValueError("remote artifact does not declare Content-Length")
    try:
        content_length = int(raw_length)
    except ValueError as exc:
        raise ValueError("remote Content-Length is not an integer") from exc
    return HTTPArtifactIdentity(
        url=url,
        content_length=content_length,
        etag=lowered.get("etag"),
        last_modified=lowered.get("last-modified"),
        accept_ranges=lowered.get("accept-ranges", "").lower() == "bytes",
    )


def _curl_executable() -> str:
    executable = shutil.which("curl")
    if executable is None:
        raise RuntimeError("curl is required for native HTTP artifact acquisition")
    return executable


def _parse_curl_head_output(url: str, output: str) -> HTTPArtifactIdentity:
    blocks = re.split(r"\r?\n\r?\n", output.strip())
    for raw in reversed(blocks):
        lines = [line for line in raw.splitlines() if line.strip()]
        if not lines or not lines[0].startswith("HTTP/"):
            continue
        parts = lines[0].split()
        if len(parts) < 2:
            continue
        try:
            status = int(parts[1])
        except ValueError:
            continue
        if not 200 <= status < 300:
            continue
        headers: dict[str, str] = {}
        for line in lines[1:]:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            headers[key.strip()] = value.strip()
        return _header_identity(url, headers)
    raise ValueError("curl HEAD response did not contain a successful final response")


def _inspect_with_curl(url: str, *, timeout_seconds: float) -> HTTPArtifactIdentity:
    result = subprocess.run(
        [
            _curl_executable(),
            "--fail",
            "--silent",
            "--show-error",
            "--location",
            "--head",
            "--max-time",
            str(timeout_seconds),
            url,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return _parse_curl_head_output(url, result.stdout)


def _download_with_curl(
    url: str,
    partial: Path,
    *,
    start: int,
    identity: HTTPArtifactIdentity,
    timeout_seconds: float,
) -> None:
    validator_kind, validator_value = _resume_validator(identity)
    command = [
        _curl_executable(),
        "--fail",
        "--silent",
        "--show-error",
        "--location",
        "--max-time",
        str(timeout_seconds),
        "--output",
        str(partial),
        "--write-out",
        "%{http_code}",
    ]
    if start:
        command.extend([
            "--continue-at",
            str(start),
            "--header",
            f"If-Range: {validator_value}",
        ])
    command.append(url)
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or f"curl exit {result.returncode}"
        raise RuntimeError(f"curl artifact acquisition failed: {detail}")
    status_text = result.stdout.strip()
    if not status_text.isdigit():
        raise RuntimeError("curl did not emit an HTTP status code")
    status = int(status_text)
    expected = 206 if start else 200
    if status != expected:
        raise ValueError(
            f"curl returned HTTP {status}; expected {expected} for "
            f"{validator_kind}-bound acquisition"
        )


def _sha256_path(path: Path) -> tuple[str, int]:
    hasher = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
            byte_count += len(chunk)
    return "sha256:" + hasher.hexdigest(), byte_count


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_json_object(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid resumable artifact state at {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"resumable artifact state at {path} must be an object")
    return payload


def _identity_payload(identity: HTTPArtifactIdentity) -> dict[str, object]:
    validator_kind, validator_value = _resume_validator(identity)
    return {
        "url": identity.url,
        "content_length": identity.content_length,
        "validator_kind": validator_kind,
        "validator_value": validator_value,
    }


def _validate_identity_payload(
    payload: Mapping[str, object],
    identity: HTTPArtifactIdentity,
    *,
    label: str,
) -> None:
    expected = _identity_payload(identity)
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"{label} remote artifact identity changed at field {key}"
            )


def download_resumable_http_artifact(
    url: str,
    destination: Path,
    *,
    timeout_seconds: float = 120.0,
) -> DownloadedHTTPArtifact:
    """Capture one immutable remote artifact using curl-owned HTTP transport.

    Climate owns the immutable identity/checkpoint/digest contract only. curl
    owns HTTP/TLS/redirect/range mechanics. A missing curl executable, changed
    remote validator, unsupported byte range, or curl failure aborts acquisition;
    no alternate client or unchecked restart path is selected.
    """
    if timeout_seconds <= 0.0:
        raise ValueError("timeout_seconds must be positive")

    destination = Path(destination)
    partial = destination.with_name(destination.name + ".partial")
    state_path = destination.with_name(destination.name + ".resume.json")
    receipt_path = destination.with_name(destination.name + ".artifact.json")
    identity = _inspect_with_curl(url, timeout_seconds=timeout_seconds)
    if not identity.accept_ranges:
        raise ValueError("remote artifact does not advertise byte-range resume support")
    validator_kind, validator_value = _resume_validator(identity)

    if receipt_path.exists():
        receipt = _load_json_object(receipt_path)
        if receipt.get("schema") != _HTTP_RECEIPT_SCHEMA:
            raise ValueError("existing artifact receipt has an unsupported schema")
        _validate_identity_payload(receipt, identity, label="completed")
        if not destination.is_file():
            raise ValueError("artifact receipt exists but captured artifact is missing")
        digest, byte_count = _sha256_path(destination)
        if (
            receipt.get("sha256") != digest
            or receipt.get("byte_count") != byte_count
            or byte_count != identity.content_length
        ):
            raise ValueError("captured artifact does not match its durable receipt")
        return DownloadedHTTPArtifact(
            url=url, path=destination, sha256=digest, byte_count=byte_count,
            validator_kind=validator_kind, validator_value=validator_value,
        )

    if destination.exists() and not state_path.exists():
        raise ValueError(
            "destination exists without a durable artifact receipt; refuse overwrite"
        )

    empty_digest = "sha256:" + hashlib.sha256(b"").hexdigest()
    if state_path.exists():
        state = _load_json_object(state_path)
        if state.get("schema") != _HTTP_RESUME_SCHEMA:
            raise ValueError("resume state has an unsupported schema")
        _validate_identity_payload(state, identity, label="resume")
        committed = state.get("committed_bytes")
        prefix_digest = state.get("prefix_sha256")
        if (
            not isinstance(committed, int)
            or isinstance(committed, bool)
            or committed < 0
            or committed > identity.content_length
        ):
            raise ValueError("resume state committed_bytes is invalid")
        if not isinstance(prefix_digest, str):
            raise ValueError("resume state prefix_sha256 is missing")
    else:
        if partial.exists():
            raise ValueError("partial artifact exists without durable resume state")
        committed = 0
        prefix_digest = empty_digest
        _write_json_atomic(
            state_path,
            {
                "schema": _HTTP_RESUME_SCHEMA,
                **_identity_payload(identity),
                "committed_bytes": committed,
                "prefix_sha256": prefix_digest,
            },
        )

    if not partial.exists():
        if committed:
            raise ValueError("resume state references a missing partial artifact")
        partial.parent.mkdir(parents=True, exist_ok=True)
        partial.touch()
    partial_size = partial.stat().st_size
    if partial_size < committed:
        raise ValueError("partial artifact is shorter than committed resume state")
    if partial_size > committed:
        with partial.open("r+b") as handle:
            handle.truncate(committed)
    observed_prefix, observed_bytes = _sha256_path(partial)
    if observed_bytes != committed or observed_prefix != prefix_digest:
        raise ValueError("partial artifact does not match committed resume digest")

    try:
        _download_with_curl(
            url, partial, start=committed, identity=identity,
            timeout_seconds=timeout_seconds,
        )
    except Exception:
        if partial.exists():
            current_digest, current_bytes = _sha256_path(partial)
            if current_bytes < committed or current_bytes > identity.content_length:
                raise
            _write_json_atomic(
                state_path,
                {
                    "schema": _HTTP_RESUME_SCHEMA,
                    **_identity_payload(identity),
                    "committed_bytes": current_bytes,
                    "prefix_sha256": current_digest,
                },
            )
        raise

    observed_identity = _inspect_with_curl(url, timeout_seconds=timeout_seconds)
    if _identity_payload(observed_identity) != _identity_payload(identity):
        raise ValueError("remote artifact identity changed during curl acquisition")
    digest, byte_count = _sha256_path(partial)
    if byte_count != identity.content_length:
        raise ValueError("remote stream ended before declared Content-Length")
    os.replace(partial, destination)
    _write_json_atomic(
        receipt_path,
        {
            "schema": _HTTP_RECEIPT_SCHEMA,
            **_identity_payload(identity),
            "sha256": digest,
            "byte_count": byte_count,
        },
    )
    state_path.unlink(missing_ok=True)
    return DownloadedHTTPArtifact(
        url=url, path=destination, sha256=digest, byte_count=byte_count,
        validator_kind=validator_kind, validator_value=validator_value,
    )


def download_by_year_artifact(
    year: int,
    destination: Path,
    *,
    timeout_seconds: float = 120.0,
) -> DownloadedHTTPArtifact:
    """Capture one GHCN-Daily by-year gzip through the curl-owned scale path."""
    return download_resumable_http_artifact(
        by_year_url(year),
        destination,
        timeout_seconds=timeout_seconds,
    )


@dataclass(frozen=True)
class GHCNStationMetadata:
    station_id: str
    latitude_deg: float
    longitude_deg: float
    elevation_m: float | None
    state: str
    name: str
    gsn_flag: str
    hcn_crn_flag: str
    wmo_id: str


@dataclass(frozen=True)
class GHCNInventoryRecord:
    station_id: str
    latitude_deg: float
    longitude_deg: float
    element: str
    first_year: int
    last_year: int


@dataclass(frozen=True)
class GHCNStationCatalogPayload:
    sha256: str
    byte_count: int
    records: tuple[GHCNStationMetadata, ...]


@dataclass(frozen=True)
class GHCNInventoryPayload:
    sha256: str
    byte_count: int
    records: tuple[GHCNInventoryRecord, ...]


def _digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _lines(payload: bytes, *, minimum_width: int) -> Iterable[str]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("GHCN bulk metadata must be valid UTF-8") from exc
    for line_number, raw in enumerate(text.splitlines(), start=1):
        if not raw:
            continue
        if len(raw) < minimum_width:
            raise ValueError(
                f"GHCN metadata line {line_number} is shorter than "
                f"{minimum_width} provider columns"
            )
        yield raw


def _finite(value: str, name: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} is not numeric: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def parse_station_catalog(payload: bytes) -> GHCNStationCatalogPayload:
    """Parse the complete ghcnd-stations.txt fixed-width provider artifact."""
    records: list[GHCNStationMetadata] = []
    seen: set[str] = set()
    for line in _lines(payload, minimum_width=71):
        padded = line.ljust(85)
        station_id = padded[0:11].strip()
        if not station_id:
            raise ValueError("GHCN station id is empty")
        if station_id in seen:
            raise ValueError(f"duplicate GHCN station id {station_id}")
        seen.add(station_id)
        latitude = _finite(padded[12:20].strip(), "latitude")
        longitude = _finite(padded[21:30].strip(), "longitude")
        if not -90.0 <= latitude <= 90.0 or not -180.0 <= longitude <= 180.0:
            raise ValueError(f"GHCN station {station_id} has invalid coordinates")
        elevation_raw = padded[31:37].strip()
        elevation = _finite(elevation_raw, "elevation")
        if elevation == -999.9:
            elevation = None
        records.append(GHCNStationMetadata(
            station_id=station_id,
            latitude_deg=latitude,
            longitude_deg=longitude,
            elevation_m=elevation,
            state=padded[38:40].strip(),
            name=padded[41:71].strip(),
            gsn_flag=padded[72:75].strip(),
            hcn_crn_flag=padded[76:79].strip(),
            wmo_id=padded[80:85].strip(),
        ))
    if not records:
        raise ValueError("GHCN station catalog contains no records")
    return GHCNStationCatalogPayload(
        sha256=_digest(payload),
        byte_count=len(payload),
        records=tuple(records),
    )


def parse_inventory(payload: bytes) -> GHCNInventoryPayload:
    """Parse ghcnd-inventory.txt without loading observation values."""
    records: list[GHCNInventoryRecord] = []
    seen: set[tuple[str, str]] = set()
    for line in _lines(payload, minimum_width=45):
        station_id = line[0:11].strip()
        element = line[31:35].strip()
        if not station_id or not element:
            raise ValueError("GHCN inventory station and element must be non-empty")
        key = (station_id, element)
        if key in seen:
            raise ValueError(f"duplicate GHCN inventory identity {key}")
        seen.add(key)
        latitude = _finite(line[12:20].strip(), "inventory latitude")
        longitude = _finite(line[21:30].strip(), "inventory longitude")
        first_year = int(line[36:40])
        last_year = int(line[41:45])
        if first_year > last_year:
            raise ValueError(f"GHCN inventory {key} has reversed year range")
        records.append(GHCNInventoryRecord(
            station_id=station_id,
            latitude_deg=latitude,
            longitude_deg=longitude,
            element=element,
            first_year=first_year,
            last_year=last_year,
        ))
    if not records:
        raise ValueError("GHCN inventory contains no records")
    return GHCNInventoryPayload(
        sha256=_digest(payload),
        byte_count=len(payload),
        records=tuple(records),
    )


def build_federated_stations(
    catalog: GHCNStationCatalogPayload,
    inventory: GHCNInventoryPayload,
    *,
    metadata_effective_date: str,
) -> tuple[FederatedStation, ...]:
    """Normalize provider metadata without inventing pre-snapshot relocation history."""
    inventory_by_station: dict[str, list[GHCNInventoryRecord]] = {}
    for item in inventory.records:
        inventory_by_station.setdefault(item.station_id, []).append(item)

    out: list[FederatedStation] = []
    for item in catalog.records:
        alias = ProviderAlias(SOURCE_ID, item.station_id)
        variables = tuple(sorted({
            record.element
            for record in inventory_by_station.get(item.station_id, ())
        }))
        out.append(FederatedStation(
            canonical_station_id=canonical_station_id(alias),
            aliases=(AliasBinding(alias, "root", catalog.sha256),),
            location_history=(
                StationLocationEpoch(
                    latitude_deg=item.latitude_deg,
                    longitude_deg=item.longitude_deg,
                    elevation_m=item.elevation_m,
                    valid_from=metadata_effective_date,
                    valid_to=None,
                    source_alias=alias,
                    evidence_digest=catalog.sha256,
                ),
            ),
            variable_ids=variables,
        ))
    return tuple(out)


def _location(station: FederatedStation, at_date: str):
    location = station.location_at(at_date)
    if location is None:
        raise ValueError(
            f"station {station.canonical_station_id} has no resolved location "
            f"at metadata effective date {at_date}"
        )
    return location


def adaptive_catalog_shards(
    stations: Sequence[FederatedStation],
    *,
    metadata_effective_date: str,
    max_station_records: int,
) -> tuple[StationCatalogShard, ...]:
    """Build bounded hierarchical spatial shards without changing topology semantics."""
    if max_station_records <= 0:
        raise ValueError("max_station_records must be positive")
    records = tuple(stations)
    if not records:
        return ()

    def split(
        subset: tuple[FederatedStation, ...],
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        key: str,
    ) -> list[StationCatalogShard]:
        if len(subset) <= max_station_records:
            return [StationCatalogShard(key, key, subset)]

        lat_mid = (lat_min + lat_max) / 2.0
        lon_mid = (lon_min + lon_max) / 2.0
        buckets: list[list[FederatedStation]] = [[], [], [], []]
        for station in subset:
            location = _location(station, metadata_effective_date)
            north = location.latitude_deg >= lat_mid
            east = location.longitude_deg >= lon_mid
            index = (2 if north else 0) + (1 if east else 0)
            buckets[index].append(station)

        nonempty = [bucket for bucket in buckets if bucket]
        if len(nonempty) == 1:
            ordered = sorted(subset, key=lambda item: item.canonical_station_id)
            return [
                StationCatalogShard(
                    f"{key}/id-{start // max_station_records:06d}",
                    f"{key}/id-{start // max_station_records:06d}",
                    tuple(ordered[start:start + max_station_records]),
                )
                for start in range(0, len(ordered), max_station_records)
            ]

        bounds = (
            (lat_min, lat_mid, lon_min, lon_mid),
            (lat_min, lat_mid, lon_mid, lon_max),
            (lat_mid, lat_max, lon_min, lon_mid),
            (lat_mid, lat_max, lon_mid, lon_max),
        )
        result: list[StationCatalogShard] = []
        for index, bucket in enumerate(buckets):
            if not bucket:
                continue
            a, b, c, d = bounds[index]
            result.extend(split(tuple(bucket), a, b, c, d, f"{key}{index}"))
        return result

    return tuple(split(records, -90.0, 90.0, -180.0, 180.0, "q"))


def catalog_shard_refs(
    shards: Sequence[StationCatalogShard],
) -> tuple[CatalogShardRef, ...]:
    refs: list[CatalogShardRef] = []
    for shard in shards:
        providers = sorted({
            binding.alias.source_id
            for station in shard.stations
            for binding in station.aliases
        })
        variables = sorted({
            variable
            for station in shard.stations
            for variable in station.variable_ids
        })
        refs.append(CatalogShardRef(
            shard_id=shard.shard_id,
            spatial_partition=shard.spatial_partition,
            digest=shard.digest(),
            station_count=len(shard.stations),
            provider_source_ids=tuple(providers),
            variable_ids=tuple(variables),
        ))
    return tuple(refs)



@dataclass(frozen=True)
class GHCNByYearRecord:
    station_id: str
    observation_date: str
    element: str
    value: int | None
    measurement_flag: str
    quality_flag: str
    source_flag: str
    observation_time: str


@dataclass(frozen=True, order=True)
class GHCNObservationPartitionKey:
    source_id: str
    spatial_partition: str
    year: int
    element: str


@dataclass(frozen=True)
class RoutedGHCNObservation:
    canonical_station_id: str
    observation_date: str
    element: str
    value: int | None
    measurement_flag: str
    quality_flag: str
    source_flag: str
    observation_time: str


@dataclass(frozen=True)
class GHCNRoutingSummary:
    row_count: int
    missing_value_count: int
    partition_count: int


class ObservationPartitionSink(Protocol):
    def write(
        self,
        key: GHCNObservationPartitionKey,
        record: RoutedGHCNObservation,
    ) -> None:
        ...


class StationShardLookup:
    """Bounded station-metadata lookup used while streaming observation rows."""

    def __init__(
        self,
        shards: Sequence[StationCatalogShard],
        *,
        source_id: str = SOURCE_ID,
    ) -> None:
        mapping: dict[str, tuple[str, str]] = {}
        for shard in shards:
            for station in shard.stations:
                for binding in station.aliases:
                    alias = binding.alias
                    if alias.source_id != source_id:
                        continue
                    value = (station.canonical_station_id, shard.spatial_partition)
                    previous = mapping.get(alias.provider_station_id)
                    if previous is not None and previous != value:
                        raise ValueError(
                            f"provider station {alias.provider_station_id!r} "
                            "resolves to multiple federation shards"
                        )
                    mapping[alias.provider_station_id] = value
        if not mapping:
            raise ValueError(
                f"no aliases for provider {source_id!r} exist in catalog shards"
            )
        self.source_id = source_id
        self._mapping = mapping

    def resolve(self, provider_station_id: str) -> tuple[str, str]:
        try:
            return self._mapping[provider_station_id]
        except KeyError as exc:
            raise ValueError(
                f"provider station {provider_station_id!r} is absent from the "
                "captured federation catalog; refresh metadata instead of "
                "silently dropping the observation"
            ) from exc


def _parse_yyyymmdd(raw: str, *, expected_year: int) -> str:
    if len(raw) != 8 or not raw.isdigit():
        raise ValueError(f"GHCN by-year date must be YYYYMMDD, got {raw!r}")
    year = int(raw[0:4])
    month = int(raw[4:6])
    day = int(raw[6:8])
    if year != expected_year:
        raise ValueError(
            f"GHCN by-year record year {year} does not match artifact year "
            f"{expected_year}"
        )
    try:
        return date(year, month, day).isoformat()
    except ValueError as exc:
        raise ValueError(f"invalid GHCN by-year date {raw!r}") from exc


def parse_by_year_row(
    row: Sequence[str],
    *,
    expected_year: int,
) -> GHCNByYearRecord:
    """Parse one documented by-year CSV row without interpreting climate units."""
    if len(row) != 8:
        raise ValueError(
            f"GHCN by-year row must contain 8 comma-separated fields, got {len(row)}"
        )
    station_id, raw_date, element, raw_value, mflag, qflag, sflag, obs_time = row
    if len(station_id) != 11 or not station_id.strip():
        raise ValueError("GHCN by-year station id must contain 11 characters")
    if len(element) != 4 or not element.strip():
        raise ValueError("GHCN by-year element must contain 4 characters")
    try:
        integer_value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"GHCN by-year value is not an integer: {raw_value!r}") from exc
    value = None if integer_value == -9999 else integer_value
    for name, flag in (
        ("measurement_flag", mflag),
        ("quality_flag", qflag),
        ("source_flag", sflag),
    ):
        if len(flag) > 1:
            raise ValueError(f"{name} must be blank or one character")
    if obs_time and (len(obs_time) != 4 or not obs_time.isdigit()):
        raise ValueError("observation_time must be blank or HHMM digits")
    return GHCNByYearRecord(
        station_id=station_id,
        observation_date=_parse_yyyymmdd(raw_date, expected_year=expected_year),
        element=element,
        value=value,
        measurement_flag=mflag,
        quality_flag=qflag,
        source_flag=sflag,
        observation_time=obs_time,
    )


def iter_by_year_records(
    lines: Iterable[str],
    *,
    expected_year: int,
) -> Iterable[GHCNByYearRecord]:
    """Stream provider rows; no year-sized list or station-time tensor is built."""
    reader = csv.reader(lines)
    for row_number, row in enumerate(reader, start=1):
        if not row:
            continue
        try:
            yield parse_by_year_row(row, expected_year=expected_year)
        except ValueError as exc:
            raise ValueError(
                f"invalid GHCN by-year row {row_number}: {exc}"
            ) from exc


def iter_gzip_by_year(
    path: Path,
    *,
    expected_year: int,
) -> Iterable[GHCNByYearRecord]:
    """Stream a captured provider .csv.gz artifact directly from disk."""
    def generate():
        with gzip.open(path, mode="rt", encoding="ascii", newline="") as handle:
            yield from iter_by_year_records(handle, expected_year=expected_year)
    return generate()


def stream_by_year_partitions(
    lines: Iterable[str],
    *,
    expected_year: int,
    lookup: StationShardLookup,
    sink: ObservationPartitionSink,
    elements: Sequence[str] | None = None,
) -> GHCNRoutingSummary:
    """Route one provider year to bounded partition sinks without row buffering.

    Storage format is deliberately sink-owned so maintained Parquet/Zarr/object-
    store writers can implement publication without changing provider parsing,
    station identity, partition keys, or missing/flag semantics.
    """
    selected = None if elements is None else frozenset(elements)
    if selected is not None and (
        not selected or any(len(item) != 4 for item in selected)
    ):
        raise ValueError("elements must be a non-empty sequence of 4-character ids")
    partition_keys: set[GHCNObservationPartitionKey] = set()
    row_count = 0
    missing = 0
    for record in iter_by_year_records(lines, expected_year=expected_year):
        if selected is not None and record.element not in selected:
            continue
        canonical_id, spatial_partition = lookup.resolve(record.station_id)
        key = GHCNObservationPartitionKey(
            source_id=lookup.source_id,
            spatial_partition=spatial_partition,
            year=expected_year,
            element=record.element,
        )
        sink.write(
            key,
            RoutedGHCNObservation(
                canonical_station_id=canonical_id,
                observation_date=record.observation_date,
                element=record.element,
                value=record.value,
                measurement_flag=record.measurement_flag,
                quality_flag=record.quality_flag,
                source_flag=record.source_flag,
                observation_time=record.observation_time,
            ),
        )
        partition_keys.add(key)
        row_count += 1
        missing += record.value is None
    return GHCNRoutingSummary(
        row_count=row_count,
        missing_value_count=missing,
        partition_count=len(partition_keys),
    )

def by_year_url(year: int) -> str:
    if not isinstance(year, int) or year < 0 or year > 9999:
        raise ValueError("year must be a four-digit non-negative integer")
    return f"{GHCN_BY_YEAR_URL}/{year:04d}.csv.gz"


def by_year_urls(start_year: int, end_year: int) -> tuple[str, ...]:
    if start_year > end_year:
        raise ValueError("start_year must not be after end_year")
    return tuple(by_year_url(year) for year in range(start_year, end_year + 1))
