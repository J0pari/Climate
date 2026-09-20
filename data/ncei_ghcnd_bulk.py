"""Bulk NCEI GHCN-Daily catalog/inventory adapter for station federation.

This is the scale path for provider discovery. It parses the provider's complete
station and inventory artifacts into provider-neutral federated station records.
Bounded spatial sharding and catalog-reference derivation are owned by
src.station_federation and re-exported here for compatibility. The subset REST
adapter remains useful for captured fixtures but is not the worldwide ingestion
loop.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
import gzip
import hashlib
import math
from typing import Iterable, Protocol, Sequence

from src.station_federation import (
    AliasBinding,
    FederatedStation,
    ProviderAlias,
    StationCatalogShard,
    StationLocationEpoch,
    adaptive_catalog_shards,
    canonical_station_id,
    catalog_shard_refs,
)

SOURCE_ID = "ncei.ghcnd.v3"
GHCN_BASE_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily"
GHCN_STATION_CATALOG_URL = f"{GHCN_BASE_URL}/ghcnd-stations.txt"
GHCN_INVENTORY_URL = f"{GHCN_BASE_URL}/ghcnd-inventory.txt"
GHCN_VERSION_URL = f"{GHCN_BASE_URL}/ghcnd-version.txt"
GHCN_BY_YEAR_URL = f"{GHCN_BASE_URL}/by_year"




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
    ) -> None:
        mapping: dict[str, tuple[str, str]] = {}
        for shard in shards:
            for station in shard.stations:
                for binding in station.aliases:
                    alias = binding.alias
                    if alias.source_id != SOURCE_ID:
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
                f"no aliases for provider {SOURCE_ID!r} exist in catalog shards"
            )
        self.source_id = SOURCE_ID
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
