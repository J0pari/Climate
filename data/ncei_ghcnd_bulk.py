"""Bulk NCEI GHCN-Daily catalog/inventory adapter for station federation.

This is the scale path for provider discovery. It parses the provider's complete
station and inventory artifacts and converts them into bounded federation shards.
The subset REST adapter remains useful for captured fixtures but is not the
worldwide ingestion loop.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Iterable, Sequence

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
        text = payload.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("GHCN bulk metadata must be ASCII") from exc
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


def by_year_url(year: int) -> str:
    if not isinstance(year, int) or year < 0 or year > 9999:
        raise ValueError("year must be a four-digit non-negative integer")
    return f"{GHCN_BY_YEAR_URL}/{year:04d}.csv.gz"


def by_year_urls(start_year: int, end_year: int) -> tuple[str, ...]:
    if start_year > end_year:
        raise ValueError("start_year must not be after end_year")
    return tuple(by_year_url(year) for year in range(start_year, end_year + 1))
