"""NCEI GHCN-Hourly station catalog federation adapter.

GHCNh is a distinct provider namespace in Climate even where NCEI documents
that a station shares the same managed GHCN identifier with GHCN-Daily.
Identity linkage is emitted as explicit digest-bound crosswalk evidence; no
name, coordinate, or proximity matching is performed here.

Catalog federation and raw hourly routing remain separate semantic layers.
Provider metadata snapshots stay distinct from resolved topology locations:
cross-provider snapshots never override another root provider's resolution, while
GHCNh-root stations advance explicit metadata-effective resolved epochs when the
provider's own current location changes. Annual PSV fields remain provider-native
observation payloads.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, replace
import hashlib
import json
import math
import re
from datetime import date, datetime
from pathlib import Path
import sqlite3
from typing import Iterable, Iterator, Protocol, Sequence

from src.station_federation import (
    AliasBinding,
    CrossProviderAliasEvidence,
    FederatedStation,
    ProviderAlias,
    ProviderLocationSnapshot,
    StationCatalogShard,
    StationLocationEpoch,
    StationSpatialBounds,
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
    """Validate exact provider archive identity without performing transport."""
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


class GHCNhRoutingLookup(Protocol):
    def resolve(self, station_id: str) -> tuple[str, str]:
        ...


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
                    value = (
                        station.canonical_station_id,
                        shard.spatial_partition,
                    )
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
    crosswalk_evidence: tuple[CrossProviderAliasEvidence, ...]
    shared_station_count: int
    new_root_station_count: int


def _parse_station_record_line(line: str) -> GHCNhStationRecord:
    if len(line) < 30:
        raise ValueError("GHCNh station-list line is shorter than required fields")
    station_id = line[0:11].strip()
    if len(station_id) != 11:
        raise ValueError("GHCNh station-list requires an 11-character ID")

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

    return GHCNhStationRecord(
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


def _iter_bounded_binary_lines(
    handle,
    *,
    max_line_bytes: int,
) -> Iterable[bytes]:
    if max_line_bytes <= 0:
        raise ValueError("max_line_bytes must be positive")
    while True:
        raw = handle.readline(max_line_bytes + 1)
        if not raw:
            return
        if len(raw) > max_line_bytes:
            raise ValueError(
                f"provider metadata line exceeds {max_line_bytes} bytes"
            )
        yield raw


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
        try:
            item = _parse_station_record_line(line)
        except ValueError as exc:
            raise ValueError(
                f"invalid GHCNh station-list line {line_number}: {exc}"
            ) from exc
        if item.station_id in seen:
            raise ValueError(f"duplicate GHCNh station ID {item.station_id!r}")
        seen.add(item.station_id)
        records.append(item)
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


def _ghcnh_root_aliases(
    stations: Sequence[FederatedStation],
) -> dict[str, ProviderAlias]:
    out: dict[str, ProviderAlias] = {}
    for station in stations:
        roots = [
            binding.alias
            for binding in station.aliases
            if (
                binding.binding_method == "root"
                and binding.alias.source_id == SOURCE_ID
            )
        ]
        if not roots:
            continue
        if len(roots) != 1:
            raise ValueError("GHCNh station has ambiguous root aliases")
        station_id = roots[0].provider_station_id
        if station_id in out:
            raise ValueError("GHCNh root station ID is not unique")
        out[station_id] = roots[0]
    return out


def _same_location(
    epoch: StationLocationEpoch,
    snapshot: ProviderLocationSnapshot,
) -> bool:
    return (
        epoch.latitude_deg == snapshot.latitude_deg
        and epoch.longitude_deg == snapshot.longitude_deg
        and epoch.elevation_m == snapshot.elevation_m
    )


def _refresh_ghcnh_root_location(
    station: FederatedStation,
    snapshot: ProviderLocationSnapshot,
) -> FederatedStation:
    roots = [
        binding.alias
        for binding in station.aliases
        if (
            binding.binding_method == "root"
            and binding.alias.source_id == SOURCE_ID
        )
    ]
    if not roots:
        return station
    if len(roots) != 1:
        raise ValueError("GHCNh station has ambiguous root aliases")
    if roots[0] != snapshot.source_alias:
        return station

    open_epochs = [
        epoch for epoch in station.location_history if epoch.valid_to is None
    ]
    if len(open_epochs) != 1:
        raise ValueError(
            "GHCNh root metadata refresh requires exactly one open resolved "
            "location epoch"
        )
    current = open_epochs[0]
    if current.source_alias != snapshot.source_alias:
        return station

    current_start = (
        datetime.fromisoformat(current.valid_from).date()
        if current.valid_from is not None
        else datetime.min.date()
    )
    target = datetime.fromisoformat(snapshot.metadata_effective_date).date()
    if target < current_start:
        raise ValueError(
            "GHCNh root metadata refresh is out-of-order relative to the current "
            "resolved location epoch"
        )

    replacement = StationLocationEpoch(
        snapshot.latitude_deg,
        snapshot.longitude_deg,
        snapshot.elevation_m,
        snapshot.metadata_effective_date,
        None,
        snapshot.source_alias,
        snapshot.evidence_digest,
    )
    epochs = list(station.location_history)
    index = epochs.index(current)
    if target == current_start:
        epochs[index] = replacement
    elif not _same_location(current, snapshot):
        epochs[index] = replace(
            current,
            valid_to=snapshot.metadata_effective_date,
        )
        epochs.append(replacement)
    return replace(station, location_history=tuple(epochs))


def _crosswalk_digest(station_id: str) -> str:
    """Stable identity evidence for one exact shared managed GHCN identifier.

    Provider catalog revisions are deliberately excluded: their current metadata
    identity is carried by ProviderLocationSnapshot.evidence_digest. Otherwise a
    daily provider refresh would look like a new station-identity relation.
    """
    return _sha256(
        _canonical_json(
            {
                "schema": "ncei-ghcnd-ghcnh-shared-ghcn-id-crosswalk/v2",
                "root_alias": {
                    "source_id": "ncei.ghcnd.v3",
                    "provider_station_id": station_id,
                },
                "alias": {
                    "source_id": SOURCE_ID,
                    "provider_station_id": station_id,
                },
                "authority": "NCEI managed GHCN identifier namespace",
                "rule": (
                    "exact shared 11-character GHCN identifier only; "
                    "no name/location/proximity matching"
                ),
            }
        )
    )


@dataclass(frozen=True)
class GHCNhMetadataSpoolSummary:
    path: str
    station_count: int
    matched_station_count: int
    source_digest: str
    source_byte_count: int
    max_database_bytes: int
    sqlite_cache_kib: int
    max_transaction_rows: int


class GHCNhMetadataSpool:
    """Byte-capped scratch index for bounded GHCNh federation and routing."""

    def __init__(
        self,
        path: Path,
        *,
        max_database_bytes: int,
        sqlite_cache_kib: int = 2048,
        max_transaction_rows: int = 1000,
        create: bool = False,
    ) -> None:
        if max_database_bytes <= 0:
            raise ValueError("max_database_bytes must be positive")
        if sqlite_cache_kib <= 0:
            raise ValueError("sqlite_cache_kib must be positive")
        if max_transaction_rows <= 0:
            raise ValueError("max_transaction_rows must be positive")
        self.path = Path(path)
        self.max_database_bytes = int(max_database_bytes)
        self.sqlite_cache_kib = int(sqlite_cache_kib)
        self.max_transaction_rows = int(max_transaction_rows)
        self._existing_scan_complete = False
        self._new_root_scan_complete = False
        if create and self.path.exists():
            raise ValueError(
                "GHCNh metadata spool path already exists; refuse implicit overwrite"
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path)
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.execute("PRAGMA journal_mode = MEMORY")
        self._db.execute("PRAGMA temp_store = MEMORY")
        self._db.execute("PRAGMA mmap_size = 0")
        self._db.execute(f"PRAGMA cache_size = {-self.sqlite_cache_kib}")
        page_size = int(self._db.execute("PRAGMA page_size").fetchone()[0])
        max_pages = self.max_database_bytes // page_size
        if max_pages < 8:
            self._db.close()
            if create and self.path.exists():
                self.path.unlink()
            raise ValueError(
                "max_database_bytes is too small for the bounded GHCNh metadata spool"
            )
        current_pages = int(self._db.execute("PRAGMA page_count").fetchone()[0])
        if current_pages > max_pages:
            self._db.close()
            raise ValueError(
                "existing GHCNh metadata spool exceeds max_database_bytes"
            )
        self._db.execute(f"PRAGMA max_page_count = {max_pages}")
        if create:
            try:
                self._db.executescript(
                    """
                    CREATE TABLE metadata (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    );
                    CREATE TABLE stations (
                        station_id TEXT PRIMARY KEY,
                        latitude_deg REAL NOT NULL,
                        longitude_deg REAL NOT NULL,
                        elevation_m REAL,
                        state TEXT NOT NULL,
                        name TEXT NOT NULL,
                        gsn_flag TEXT NOT NULL,
                        hcn_crn_flag TEXT NOT NULL,
                        wmo_id TEXT NOT NULL,
                        icao TEXT NOT NULL,
                        matched INTEGER NOT NULL DEFAULT 0,
                        canonical_station_id TEXT,
                        spatial_partition TEXT
                    );
                    CREATE INDEX stations_lat_lon
                        ON stations(latitude_deg, longitude_deg);
                    CREATE INDEX stations_matched
                        ON stations(matched, station_id);
                    """
                )
                self._db.commit()
            except sqlite3.OperationalError as exc:
                self._db.close()
                self.path.unlink(missing_ok=True)
                if "full" in str(exc).lower():
                    raise ValueError(
                        "GHCNh metadata spool exceeded max_database_bytes while "
                        "creating schema"
                    ) from exc
                raise

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> "GHCNhMetadataSpool":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _metadata(self, key: str) -> str:
        row = self._db.execute(
            "SELECT value FROM metadata WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            raise ValueError(f"GHCNh metadata spool is missing {key!r}")
        return str(row[0])

    def _commit(self) -> None:
        try:
            self._db.commit()
        except sqlite3.OperationalError as exc:
            if "full" in str(exc).lower():
                raise ValueError(
                    "GHCNh metadata spool exceeded max_database_bytes"
                ) from exc
            raise

    def station_count(self) -> int:
        return int(self._db.execute("SELECT COUNT(*) FROM stations").fetchone()[0])

    def matched_station_count(self) -> int:
        return int(
            self._db.execute(
                "SELECT COUNT(*) FROM stations WHERE matched = 1"
            ).fetchone()[0]
        )

    def summary(self) -> GHCNhMetadataSpoolSummary:
        return GHCNhMetadataSpoolSummary(
            path=str(self.path),
            station_count=self.station_count(),
            matched_station_count=self.matched_station_count(),
            source_digest=self._metadata("source_digest"),
            source_byte_count=int(self._metadata("source_byte_count")),
            max_database_bytes=self.max_database_bytes,
            sqlite_cache_kib=self.sqlite_cache_kib,
            max_transaction_rows=self.max_transaction_rows,
        )

    def _record(self, station_id: str) -> GHCNhStationRecord | None:
        row = self._db.execute(
            """
            SELECT station_id, latitude_deg, longitude_deg, elevation_m,
                   state, name, gsn_flag, hcn_crn_flag, wmo_id, icao
            FROM stations WHERE station_id = ?
            """,
            (station_id,),
        ).fetchone()
        if row is None:
            return None
        return GHCNhStationRecord(
            station_id=str(row[0]),
            latitude_deg=float(row[1]),
            longitude_deg=float(row[2]),
            elevation_m=None if row[3] is None else float(row[3]),
            state=str(row[4]),
            name=str(row[5]),
            gsn_flag=str(row[6]),
            hcn_crn_flag=str(row[7]),
            wmo_id=str(row[8]),
            icao=str(row[9]),
        )

    def _snapshot(
        self,
        record: GHCNhStationRecord,
        *,
        metadata_effective_date: str,
    ) -> ProviderLocationSnapshot:
        alias = ProviderAlias(SOURCE_ID, record.station_id)
        return ProviderLocationSnapshot(
            latitude_deg=record.latitude_deg,
            longitude_deg=record.longitude_deg,
            elevation_m=record.elevation_m,
            metadata_effective_date=metadata_effective_date,
            source_alias=alias,
            evidence_digest=self._metadata("source_digest"),
        )

    def _mark_matched(
        self,
        station_id: str,
        *,
        canonical_station_id_value: str,
        spatial_partition: str,
    ) -> None:
        try:
            self._db.execute(
                """
                UPDATE stations
                SET matched = 1,
                    canonical_station_id = ?,
                    spatial_partition = ?
                WHERE station_id = ?
                """,
                (
                    canonical_station_id_value,
                    spatial_partition,
                    station_id,
                ),
            )
        except sqlite3.OperationalError as exc:
            if "full" in str(exc).lower():
                raise ValueError(
                    "GHCNh metadata spool exceeded max_database_bytes"
                ) from exc
            raise

    def enrich_existing_shards(
        self,
        shards: Iterable[StationCatalogShard],
        *,
        metadata_effective_date: str,
        max_station_records: int,
    ) -> Iterator[StationCatalogShard]:
        """Enrich bounded existing shards without materializing the federation."""
        date.fromisoformat(metadata_effective_date)
        if max_station_records <= 0:
            raise ValueError("max_station_records must be positive")
        if self._existing_scan_complete:
            raise ValueError("existing federation scan already completed")
        source_digest = self._metadata("source_digest")
        for shard in shards:
            if len(shard.stations) > max_station_records:
                raise ValueError(
                    "incoming catalog shard exceeds max_station_records"
                )
            updated_stations: list[FederatedStation] = []
            for original in shard.stations:
                ghcnh_aliases = [
                    binding.alias
                    for binding in original.aliases
                    if binding.alias.source_id == SOURCE_ID
                ]
                if len(ghcnh_aliases) > 1:
                    raise ValueError(
                        "station has multiple GHCNh aliases before bounded refresh"
                    )

                station = original
                alias: ProviderAlias | None = (
                    ghcnh_aliases[0] if ghcnh_aliases else None
                )
                record: GHCNhStationRecord | None = (
                    self._record(alias.provider_station_id)
                    if alias is not None
                    else None
                )

                if alias is not None and record is None:
                    raise ValueError(
                        f"bound GHCNh alias {alias.provider_station_id!r} is absent "
                        "from the current provider catalog; station retirement "
                        "semantics are required before removing it"
                    )

                if alias is None:
                    daily_roots = [
                        binding.alias
                        for binding in station.aliases
                        if (
                            binding.binding_method == "root"
                            and binding.alias.source_id == "ncei.ghcnd.v3"
                        )
                    ]
                    if len(daily_roots) > 1:
                        raise ValueError(
                            "station has ambiguous GHCN-Daily root aliases"
                        )
                    if daily_roots:
                        candidate = self._record(
                            daily_roots[0].provider_station_id
                        )
                        if candidate is not None:
                            alias = ProviderAlias(
                                SOURCE_ID, candidate.station_id
                            )
                            evidence = CrossProviderAliasEvidence(
                                root_alias=daily_roots[0],
                                alias=alias,
                                evidence_digest=_crosswalk_digest(
                                    candidate.station_id
                                ),
                            )
                            station = apply_cross_provider_alias_evidence(
                                (station,), (evidence,)
                            )[0]
                            record = candidate

                if alias is not None and record is not None:
                    snapshot = self._snapshot(
                        record,
                        metadata_effective_date=metadata_effective_date,
                    )
                    snapshots = {
                        item.source_alias: item
                        for item in station.provider_location_snapshots
                    }
                    snapshots[alias] = snapshot
                    station = replace(
                        station,
                        provider_location_snapshots=tuple(
                            snapshots[key] for key in sorted(snapshots)
                        ),
                    )
                    station = _refresh_ghcnh_root_location(
                        station, snapshot
                    )
                    self._mark_matched(
                        alias.provider_station_id,
                        canonical_station_id_value=station.canonical_station_id,
                        spatial_partition=shard.spatial_partition,
                    )
                updated_stations.append(station)

            if not updated_stations:
                raise ValueError("existing catalog shard is unexpectedly empty")
            locations = [
                station.location_at(metadata_effective_date)
                for station in updated_stations
            ]
            if any(item is None for item in locations):
                raise ValueError(
                    "updated station shard lacks a resolved location at metadata date"
                )
            bounds = StationSpatialBounds(
                min(item.latitude_deg for item in locations if item is not None),
                max(item.latitude_deg for item in locations if item is not None),
                min(item.longitude_deg for item in locations if item is not None),
                max(item.longitude_deg for item in locations if item is not None),
            )
            self._commit()
            yield StationCatalogShard(
                shard.shard_id,
                shard.spatial_partition,
                tuple(updated_stations),
                bounds,
            )
        self._existing_scan_complete = True
        self._commit()

    def _unmatched_count(self, where_sql: str, params: tuple) -> int:
        row = self._db.execute(
            f"""
            SELECT COUNT(*) FROM stations
            WHERE matched = 0 AND ({where_sql})
            """,
            params,
        ).fetchone()
        return int(row[0])

    def _new_root_shard(
        self,
        station_ids: Sequence[str],
        *,
        shard_id: str,
        metadata_effective_date: str,
    ) -> StationCatalogShard:
        if not station_ids:
            raise ValueError("cannot materialize an empty GHCNh root shard")
        source_digest = self._metadata("source_digest")
        stations: list[FederatedStation] = []
        for station_id in station_ids:
            record = self._record(station_id)
            if record is None:
                raise ValueError("GHCNh metadata spool lost a station row")
            alias = ProviderAlias(SOURCE_ID, station_id)
            snapshot = self._snapshot(
                record,
                metadata_effective_date=metadata_effective_date,
            )
            station = FederatedStation(
                canonical_station_id=canonical_station_id(alias),
                aliases=(AliasBinding(alias, "root", source_digest),),
                location_history=(
                    StationLocationEpoch(
                        record.latitude_deg,
                        record.longitude_deg,
                        record.elevation_m,
                        metadata_effective_date,
                        None,
                        alias,
                        source_digest,
                    ),
                ),
                variable_ids=(),
                provider_location_snapshots=(snapshot,),
            )
            stations.append(station)
            self._mark_matched(
                station_id,
                canonical_station_id_value=station.canonical_station_id,
                spatial_partition=shard_id,
            )
        self._commit()
        bounds = StationSpatialBounds(
            min(item.location_history[-1].latitude_deg for item in stations),
            max(item.location_history[-1].latitude_deg for item in stations),
            min(item.location_history[-1].longitude_deg for item in stations),
            max(item.location_history[-1].longitude_deg for item in stations),
        )
        return StationCatalogShard(
            shard_id, shard_id, tuple(stations), bounds
        )

    def iter_new_root_shards(
        self,
        *,
        metadata_effective_date: str,
        max_station_records: int,
        max_shard_depth: int,
    ) -> Iterator[StationCatalogShard]:
        if not self._existing_scan_complete:
            raise ValueError(
                "consume enrich_existing_shards() completely before emitting "
                "new GHCNh roots"
            )
        self._new_root_scan_complete = False
        date.fromisoformat(metadata_effective_date)
        if max_station_records <= 0:
            raise ValueError("max_station_records must be positive")
        if max_shard_depth <= 0:
            raise ValueError("max_shard_depth must be positive")
        prefix = "ghcnh-" + self._metadata("source_digest")[7:19] + "/q"

        def recurse(
            where_sql: str,
            params: tuple,
            lat_min: float,
            lat_max: float,
            lon_min: float,
            lon_max: float,
            key: str,
            depth: int,
        ) -> Iterator[StationCatalogShard]:
            count = self._unmatched_count(where_sql, params)
            if count == 0:
                return
            if count <= max_station_records:
                rows = self._db.execute(
                    f"""
                    SELECT station_id FROM stations
                    WHERE matched = 0 AND ({where_sql})
                    ORDER BY station_id
                    LIMIT ?
                    """,
                    (*params, max_station_records),
                ).fetchall()
                yield self._new_root_shard(
                    [row[0] for row in rows],
                    shard_id=key,
                    metadata_effective_date=metadata_effective_date,
                )
                return
            if depth >= max_shard_depth:
                raise ValueError(
                    "GHCNh new-root sharding exceeded max_shard_depth"
                )

            lat_mid = (lat_min + lat_max) / 2.0
            lon_mid = (lon_min + lon_max) / 2.0
            quadrants = (
                (
                    f"({where_sql}) AND latitude_deg < ? AND longitude_deg < ?",
                    (*params, lat_mid, lon_mid),
                    lat_min, lat_mid, lon_min, lon_mid,
                ),
                (
                    f"({where_sql}) AND latitude_deg < ? AND longitude_deg >= ?",
                    (*params, lat_mid, lon_mid),
                    lat_min, lat_mid, lon_mid, lon_max,
                ),
                (
                    f"({where_sql}) AND latitude_deg >= ? AND longitude_deg < ?",
                    (*params, lat_mid, lon_mid),
                    lat_mid, lat_max, lon_min, lon_mid,
                ),
                (
                    f"({where_sql}) AND latitude_deg >= ? AND longitude_deg >= ?",
                    (*params, lat_mid, lon_mid),
                    lat_mid, lat_max, lon_mid, lon_max,
                ),
            )
            counts = [
                self._unmatched_count(item[0], item[1]) for item in quadrants
            ]
            nonempty = [i for i, value in enumerate(counts) if value]
            if len(nonempty) == 1:
                last_station_id = ""
                chunk = 0
                while True:
                    rows = self._db.execute(
                        f"""
                        SELECT station_id FROM stations
                        WHERE matched = 0 AND ({where_sql})
                          AND station_id > ?
                        ORDER BY station_id
                        LIMIT ?
                        """,
                        (*params, last_station_id, max_station_records),
                    ).fetchall()
                    if not rows:
                        break
                    ids = [row[0] for row in rows]
                    yield self._new_root_shard(
                        ids,
                        shard_id=f"{key}/id-{chunk:06d}",
                        metadata_effective_date=metadata_effective_date,
                    )
                    last_station_id = ids[-1]
                    chunk += 1
                return

            for index, item in enumerate(quadrants):
                if counts[index] == 0:
                    continue
                yield from recurse(
                    item[0], item[1],
                    item[2], item[3], item[4], item[5],
                    f"{key}{index}",
                    depth + 1,
                )

        yield from recurse(
            "1 = 1", (), -90.0, 90.0, -180.0, 180.0, prefix, 0
        )
        self._new_root_scan_complete = True

    def resolve(self, station_id: str) -> tuple[str, str]:
        if not self._existing_scan_complete or not self._new_root_scan_complete:
            raise ValueError(
                "GHCNh metadata spool routing is unavailable until existing "
                "enrichment and new-root sharding are fully consumed"
            )
        row = self._db.execute(
            """
            SELECT canonical_station_id, spatial_partition
            FROM stations WHERE station_id = ?
            """,
            (station_id,),
        ).fetchone()
        if row is None:
            raise ValueError(
                f"GHCNh station {station_id!r} is absent from the captured "
                "metadata spool"
            )
        if row[0] is None or row[1] is None:
            raise ValueError(
                f"GHCNh station {station_id!r} has not been assigned to the "
                "federation; complete bounded enrichment/new-root sharding first"
            )
        return str(row[0]), str(row[1])


def build_ghcnh_metadata_spool(
    station_list_path: Path,
    *,
    spool_path: Path,
    max_database_bytes: int,
    sqlite_cache_kib: int = 2048,
    max_transaction_rows: int = 1000,
    max_metadata_line_bytes: int = 4096,
) -> GHCNhMetadataSpool:
    """Stream one captured GHCNh station list into bounded scratch storage."""
    if max_metadata_line_bytes <= 0:
        raise ValueError("max_metadata_line_bytes must be positive")
    spool = GHCNhMetadataSpool(
        spool_path,
        max_database_bytes=max_database_bytes,
        sqlite_cache_kib=sqlite_cache_kib,
        max_transaction_rows=max_transaction_rows,
        create=True,
    )
    try:
        digest = hashlib.sha256()
        byte_count = 0
        pending = 0
        with station_list_path.open("rb") as handle:
            for line_number, raw_bytes in enumerate(
                _iter_bounded_binary_lines(
                    handle, max_line_bytes=max_metadata_line_bytes
                ),
                start=1,
            ):
                digest.update(raw_bytes)
                byte_count += len(raw_bytes)
                try:
                    line = raw_bytes.decode("ascii").rstrip("\r\n")
                except UnicodeDecodeError as exc:
                    raise ValueError(
                        f"invalid GHCNh station-list line {line_number}: "
                        "provider metadata must be ASCII"
                    ) from exc
                if not line.strip():
                    continue
                try:
                    item = _parse_station_record_line(line)
                    spool._db.execute(
                        """
                        INSERT INTO stations(
                            station_id, latitude_deg, longitude_deg, elevation_m,
                            state, name, gsn_flag, hcn_crn_flag, wmo_id, icao
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            item.station_id,
                            item.latitude_deg,
                            item.longitude_deg,
                            item.elevation_m,
                            item.state,
                            item.name,
                            item.gsn_flag,
                            item.hcn_crn_flag,
                            item.wmo_id,
                            item.icao,
                        ),
                    )
                except (ValueError, sqlite3.IntegrityError) as exc:
                    raise ValueError(
                        f"invalid GHCNh station-list line {line_number}: {exc}"
                    ) from exc
                pending += 1
                if pending >= max_transaction_rows:
                    spool._commit()
                    pending = 0
        spool._commit()
        if spool.station_count() == 0:
            raise ValueError("GHCNh station list contains no stations")
        spool._db.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            (
                ("source_digest", "sha256:" + digest.hexdigest()),
                ("source_byte_count", str(byte_count)),
            ),
        )
        spool._commit()
        return spool
    except sqlite3.OperationalError as exc:
        spool.close()
        spool_path.unlink(missing_ok=True)
        if "full" in str(exc).lower():
            raise ValueError(
                "GHCNh metadata spool exceeded max_database_bytes"
            ) from exc
        raise
    except Exception:
        spool.close()
        spool_path.unlink(missing_ok=True)
        raise


def federate_station_catalog(
    existing: Sequence[FederatedStation],
    catalog: GHCNhStationCatalog,
    *,
    metadata_effective_date: str,
) -> GHCNhFederationResult:
    """Add GHCNh metadata using NCEI's documented shared-GHCN-ID semantics."""
    existing_records = tuple(existing)
    daily_roots = _daily_root_aliases(existing_records)
    ghcnh_roots = _ghcnh_root_aliases(existing_records)
    hourly_by_id = {item.station_id: item for item in catalog.records}
    shared = sorted(set(daily_roots) & set(hourly_by_id))
    evidence = tuple(
        CrossProviderAliasEvidence(
            root_alias=daily_roots[station_id],
            alias=ProviderAlias(SOURCE_ID, station_id),
            evidence_digest=_crosswalk_digest(station_id),
        )
        for station_id in shared
    )
    federated = list(
        apply_cross_provider_alias_evidence(existing_records, evidence)
    )

    snapshot_by_alias = {
        ProviderAlias(SOURCE_ID, record.station_id): ProviderLocationSnapshot(
            latitude_deg=record.latitude_deg,
            longitude_deg=record.longitude_deg,
            elevation_m=record.elevation_m,
            metadata_effective_date=metadata_effective_date,
            source_alias=ProviderAlias(SOURCE_ID, record.station_id),
            evidence_digest=catalog.sha256,
        )
        for record in catalog.records
    }
    updated_existing: list[FederatedStation] = []
    for station in federated:
        snapshots = {
            snapshot.source_alias: snapshot
            for snapshot in station.provider_location_snapshots
        }
        for binding in station.aliases:
            current = snapshot_by_alias.get(binding.alias)
            if current is not None:
                snapshots[binding.alias] = current
        updated = replace(
            station,
            provider_location_snapshots=tuple(
                snapshots[alias] for alias in sorted(snapshots)
            ),
        )
        ghcnh_root = next(
            (
                binding.alias
                for binding in updated.aliases
                if (
                    binding.binding_method == "root"
                    and binding.alias.source_id == SOURCE_ID
                )
            ),
            None,
        )
        if ghcnh_root is not None:
            current_snapshot = snapshots.get(ghcnh_root)
            if current_snapshot is not None:
                updated = _refresh_ghcnh_root_location(
                    updated, current_snapshot
                )
        updated_existing.append(updated)
    federated = updated_existing

    new_root_station_count = 0
    for record in catalog.records:
        if (
            record.station_id in daily_roots
            or record.station_id in ghcnh_roots
        ):
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
                provider_location_snapshots=(
                    ProviderLocationSnapshot(
                        record.latitude_deg,
                        record.longitude_deg,
                        record.elevation_m,
                        metadata_effective_date,
                        alias,
                        catalog.sha256,
                    ),
                ),
            )
        )
        new_root_station_count += 1

    ordered = tuple(sorted(federated, key=lambda item: item.canonical_station_id))
    return GHCNhFederationResult(
        stations=ordered,
        crosswalk_evidence=evidence,
        shared_station_count=len(shared),
        new_root_station_count=new_root_station_count,
    )

def stream_station_year_psv(
    lines: Iterable[str],
    *,
    expected_year: int,
    lookup: GHCNhRoutingLookup,
    sink: GHCNhObservationSink,
    max_rows: int,
    max_partition_keys: int,
    expected_station_id: str | None = None,
) -> GHCNhRoutingSummary:
    """Stream one GHCNh station/year PSV without normalizing provider fields."""
    if expected_year < 0:
        raise ValueError("expected_year must be non-negative")
    if max_rows <= 0:
        raise ValueError("max_rows must be positive")
    if max_partition_keys <= 0:
        raise ValueError("max_partition_keys must be positive")

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
        if row_count >= max_rows:
            raise ValueError("GHCNh routing exceeded max_rows")
        if key not in partitions:
            if len(partitions) >= max_partition_keys:
                raise ValueError(
                    "GHCNh routing exceeded max_partition_keys"
                )
            partitions.add(key)
        sink.write(key, record)
        row_count += 1

    return GHCNhRoutingSummary(
        row_count=row_count,
        partition_count=len(partitions),
        field_names=field_names,
    )
