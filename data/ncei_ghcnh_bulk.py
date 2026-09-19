"""NCEI GHCN-Hourly station catalog federation adapter.

GHCNh is a distinct provider namespace in Climate even where NCEI documents
that a station shares the same managed GHCN identifier with GHCN-Daily.
Identity linkage is emitted as explicit digest-bound crosswalk evidence; no
name, coordinate, or proximity matching is performed here.

Climate consumes externally captured station-list and annual archive artifacts.
This module owns provider-specific archive-name validation, catalog federation,
and hourly observation routing semantics; generic network transfer remains
external.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import math
import re
from datetime import datetime
from typing import Iterable, Protocol, Sequence

from src.station_federation import (
    AliasBinding,
    CrossProviderAliasEvidence,
    FederatedStation,
    ProviderAlias,
    StationCatalogShard,
    StationLocationEpoch,
    apply_cross_provider_alias_evidence,
    canonical_station_id,
)


SOURCE_ID = "ncei.ghcnh.v1"
STATION_LIST_URL = (
    "https://www.ncei.noaa.gov/oa/global-historical-climatology-network/"
    "hourly/doc/ghcnh-station-list.txt"
)
ARCHIVE_BASE_URL = (
    "https://www.ncei.noaa.gov/oa/global-historical-climatology-network/"
    "hourly/archive/"
)
_ARCHIVE_NAME = re.compile(
    r"^ghcn-hourly_v1\.[A-Za-z0-9.]+_d[0-9]{4}_c[0-9]{8}\.tar\.gz$"
)


def validate_year_archive_url(archive_url: str) -> str:
    """Validate provider version/data-year/creation-date identity without fetching."""
    if not archive_url.startswith(ARCHIVE_BASE_URL):
        raise ValueError("GHCNh archive URL must use the official NCEI archive path")
    name = archive_url.removeprefix(ARCHIVE_BASE_URL)
    if "/" in name or _ARCHIVE_NAME.fullmatch(name) is None:
        raise ValueError(
            "GHCNh archive URL must name one versioned data-year/creation-date tar.gz"
        )
    return name

def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


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


def _finite_coordinate(value: str, *, name: str, lower: float, upper: float) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"GHCNh {name} is not numeric") from exc
    if not math.isfinite(parsed) or not lower <= parsed <= upper:
        raise ValueError(f"GHCNh {name} is outside [{lower}, {upper}]")
    return parsed


@dataclass(frozen=True)
class GHCNhStationRecord:
    station_id: str
    latitude_deg: float
    longitude_deg: float
    elevation_m: float | None
    state: str
    name: str
    gsn_flag: str
    hcn_crn_flag: str
    wmo_id: str
    icao: str


@dataclass(frozen=True)
class GHCNhStationCatalog:
    records: tuple[GHCNhStationRecord, ...]
    sha256: str
    source_url: str = STATION_LIST_URL


@dataclass(frozen=True, order=True)
class GHCNhObservationPartitionKey:
    source_id: str
    spatial_partition: str
    year: int


@dataclass(frozen=True)
class RoutedGHCNhObservation:
    canonical_station_id: str
    observation_datetime: str
    provider_fields: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class GHCNhRoutingSummary:
    row_count: int
    partition_count: int
    field_names: tuple[str, ...]


class GHCNhObservationSink(Protocol):
    def write(
        self,
        key: GHCNhObservationPartitionKey,
        record: RoutedGHCNhObservation,
    ) -> None:
        ...


class GHCNhAliasShardLookup:
    """Resolve captured GHCNh aliases to canonical station/shard identity."""

    def __init__(self, shards: Sequence[StationCatalogShard]) -> None:
        mapping: dict[str, tuple[str, str]] = {}
        for shard in shards:
            for station in shard.stations:
                for binding in station.aliases:
                    alias = binding.alias
                    if alias.source_id != SOURCE_ID:
                        continue
                    previous = mapping.get(alias.provider_station_id)
                    value = (station.canonical_station_id, shard.shard_id)
                    if previous is not None and previous != value:
                        raise ValueError(
                            "GHCNh alias resolves to multiple canonical stations/shards"
                        )
                    mapping[alias.provider_station_id] = value
        self._mapping = mapping

    def resolve(self, station_id: str) -> tuple[str, str]:
        try:
            return self._mapping[station_id]
        except KeyError as exc:
            raise ValueError(
                f"GHCNh station {station_id!r} is absent from the captured "
                "federation catalog"
            ) from exc


@dataclass(frozen=True)
class GHCNhFederationResult:
    stations: tuple[FederatedStation, ...]
    crosswalk_evidence_digest: str
    shared_station_count: int
    new_root_station_count: int


def parse_station_catalog(payload: bytes) -> GHCNhStationCatalog:
    """Parse the documented fixed-width ghcnh-station-list.txt format."""
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("GHCNh station list must be ASCII") from exc

    records: list[GHCNhStationRecord] = []
    seen: set[str] = set()
    for line_number, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            continue
        line = raw.rstrip("\r\n")
        if len(line) < 30:
            raise ValueError(
                f"GHCNh station-list line {line_number} is shorter than required fields"
            )
        station_id = line[0:11].strip()
        if len(station_id) != 11:
            raise ValueError(
                f"GHCNh station-list line {line_number} requires an 11-character ID"
            )
        if station_id in seen:
            raise ValueError(f"duplicate GHCNh station ID {station_id!r}")
        seen.add(station_id)

        latitude = _finite_coordinate(
            line[12:20].strip(),
            name="latitude",
            lower=-90.0,
            upper=90.0,
        )
        longitude = _finite_coordinate(
            line[21:30].strip(),
            name="longitude",
            lower=-180.0,
            upper=180.0,
        )
        elevation_text = line[31:37].strip() if len(line) >= 37 else ""
        if elevation_text in {"", "-999.9"}:
            elevation = None
        else:
            try:
                elevation = float(elevation_text)
            except ValueError as exc:
                raise ValueError("GHCNh elevation is not numeric") from exc
            if not math.isfinite(elevation):
                raise ValueError("GHCNh elevation must be finite")

        records.append(
            GHCNhStationRecord(
                station_id=station_id,
                latitude_deg=latitude,
                longitude_deg=longitude,
                elevation_m=elevation,
                state=line[38:40].strip() if len(line) >= 40 else "",
                name=line[41:71].strip() if len(line) >= 71 else "",
                gsn_flag=line[72:75].strip() if len(line) >= 75 else "",
                hcn_crn_flag=line[76:79].strip() if len(line) >= 79 else "",
                wmo_id=line[80:85].strip() if len(line) >= 85 else "",
                icao=line[86:90].strip() if len(line) >= 90 else "",
            )
        )
    if not records:
        raise ValueError("GHCNh station list contains no stations")
    return GHCNhStationCatalog(tuple(records), _sha256(payload))


def _daily_root_aliases(
    stations: Sequence[FederatedStation],
) -> dict[str, ProviderAlias]:
    out: dict[str, ProviderAlias] = {}
    for station in stations:
        roots = [
            binding.alias
            for binding in station.aliases
            if (
                binding.binding_method == "root"
                and binding.alias.source_id == "ncei.ghcnd.v3"
            )
        ]
        if not roots:
            continue
        if len(roots) != 1:
            raise ValueError("GHCN-Daily station has ambiguous root aliases")
        station_id = roots[0].provider_station_id
        if station_id in out:
            raise ValueError("GHCN-Daily root station ID is not unique")
        out[station_id] = roots[0]
    return out


def _crosswalk_digest(
    *,
    ghcnh_catalog_digest: str,
    daily_station_ids: Sequence[str],
    shared_station_ids: Sequence[str],
) -> str:
    return _sha256(
        _canonical_json(
            {
                "schema": "ncei-ghcnd-ghcnh-shared-ghcn-id-crosswalk/v1",
                "ghcnh_catalog_digest": ghcnh_catalog_digest,
                "ghcnd_station_ids_digest": _sha256(
                    _canonical_json(sorted(daily_station_ids))
                ),
                "shared_station_ids": sorted(shared_station_ids),
                "rule": (
                    "exact shared 11-character GHCN identifier only; "
                    "no name/location/proximity matching"
                ),
            }
        )
    )


def federate_station_catalog(
    existing: Sequence[FederatedStation],
    catalog: GHCNhStationCatalog,
    *,
    metadata_effective_date: str,
) -> GHCNhFederationResult:
    """Add GHCNh metadata using NCEI's documented shared-GHCN-ID semantics."""
    existing_records = tuple(existing)
    daily_roots = _daily_root_aliases(existing_records)
    hourly_by_id = {item.station_id: item for item in catalog.records}
    shared = sorted(set(daily_roots) & set(hourly_by_id))
    crosswalk_digest = _crosswalk_digest(
        ghcnh_catalog_digest=catalog.sha256,
        daily_station_ids=tuple(daily_roots),
        shared_station_ids=shared,
    )

    evidence = tuple(
        CrossProviderAliasEvidence(
            root_alias=daily_roots[station_id],
            alias=ProviderAlias(SOURCE_ID, station_id),
            evidence_digest=crosswalk_digest,
        )
        for station_id in shared
    )
    federated = list(
        apply_cross_provider_alias_evidence(existing_records, evidence)
    )

    for record in catalog.records:
        if record.station_id in daily_roots:
            continue
        alias = ProviderAlias(SOURCE_ID, record.station_id)
        federated.append(
            FederatedStation(
                canonical_station_id=canonical_station_id(alias),
                aliases=(
                    AliasBinding(alias, "root", catalog.sha256),
                ),
                location_history=(
                    StationLocationEpoch(
                        record.latitude_deg,
                        record.longitude_deg,
                        record.elevation_m,
                        metadata_effective_date,
                        None,
                        alias,
                        catalog.sha256,
                    ),
                ),
                variable_ids=(),
            )
        )

    ordered = tuple(sorted(federated, key=lambda item: item.canonical_station_id))
    return GHCNhFederationResult(
        stations=ordered,
        crosswalk_evidence_digest=crosswalk_digest,
        shared_station_count=len(shared),
        new_root_station_count=len(catalog.records) - len(shared),
    )

def stream_station_year_psv(
    lines: Iterable[str],
    *,
    expected_year: int,
    lookup: GHCNhAliasShardLookup,
    sink: GHCNhObservationSink,
    expected_station_id: str | None = None,
) -> GHCNhRoutingSummary:
    """Stream one GHCNh station/year PSV without normalizing provider fields."""
    if expected_year < 0:
        raise ValueError("expected_year must be non-negative")

    reader = csv.DictReader(lines, delimiter="|")
    field_names = tuple(reader.fieldnames or ())
    if not field_names:
        raise ValueError("GHCNh PSV requires a header row")
    if len(set(field_names)) != len(field_names):
        raise ValueError("GHCNh PSV header contains duplicate field names")
    for required in ("STATION", "DATE"):
        if required not in field_names:
            raise ValueError(f"GHCNh PSV header is missing required field {required!r}")

    row_count = 0
    partitions: set[GHCNhObservationPartitionKey] = set()
    for row_number, row in enumerate(reader, start=2):
        if None in row:
            raise ValueError(
                f"GHCNh PSV row {row_number} contains more values than the header"
            )
        if any(value is None for value in row.values()):
            raise ValueError(
                f"GHCNh PSV row {row_number} contains fewer values than the header"
            )
        station_id = row["STATION"].strip()
        timestamp = row["DATE"].strip()
        if not station_id:
            raise ValueError(f"GHCNh PSV row {row_number} has empty STATION")
        if expected_station_id is not None and station_id != expected_station_id:
            raise ValueError(
                f"GHCNh PSV row {row_number} station {station_id!r} does not "
                f"match archive member station {expected_station_id!r}"
            )
        if not timestamp:
            raise ValueError(f"GHCNh PSV row {row_number} has empty DATE")
        try:
            parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(
                f"GHCNh PSV row {row_number} DATE is not ISO-8601"
            ) from exc
        if parsed.year != expected_year:
            raise ValueError(
                f"GHCNh PSV row {row_number} year {parsed.year} does not match "
                f"artifact year {expected_year}"
            )

        canonical_station_id, shard_id = lookup.resolve(station_id)
        key = GHCNhObservationPartitionKey(
            source_id=SOURCE_ID,
            spatial_partition=shard_id,
            year=expected_year,
        )
        record = RoutedGHCNhObservation(
            canonical_station_id=canonical_station_id,
            observation_datetime=timestamp,
            provider_fields=tuple(
                (name, row[name])
                for name in field_names
            ),
        )
        sink.write(key, record)
        partitions.add(key)
        row_count += 1

    return GHCNhRoutingSummary(
        row_count=row_count,
        partition_count=len(partitions),
        field_names=field_names,
    )
