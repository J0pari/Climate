"""Bulk NCEI GHCN-Daily catalog/inventory adapter for station federation.

Small-fixture parsing helpers return in-memory tuples for direct verification.
The worldwide scale path is build_metadata_spool(): it streams captured station
and inventory artifacts into a caller-owned, byte-capped SQLite scratch database,
persists station-to-shard routing on disk, and materializes at most one configured
shard of federated station records at a time. Provider-neutral catalog-reference
and manifest semantics remain owned by src.station_federation. The subset REST
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
from pathlib import Path
import sqlite3
from typing import Iterable, Iterator, Protocol, Sequence

from src.station_federation import (
    AliasBinding,
    FederatedStation,
    ProviderAlias,
    ProviderLocationSnapshot,
    ProviderVariableAvailability,
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
        item = _parse_station_metadata_line(line)
        if item.station_id in seen:
            raise ValueError(f"duplicate GHCN station id {item.station_id}")
        seen.add(item.station_id)
        records.append(item)
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
        item = _parse_inventory_record_line(line)
        key = (item.station_id, item.element)
        if key in seen:
            raise ValueError(f"duplicate GHCN inventory identity {key}")
        seen.add(key)
        records.append(item)
    if not records:
        raise ValueError("GHCN inventory contains no records")
    return GHCNInventoryPayload(
        sha256=_digest(payload),
        byte_count=len(payload),
        records=tuple(records),
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


def _parse_station_metadata_line(line: str) -> GHCNStationMetadata:
    if len(line) < 71:
        raise ValueError("GHCN station metadata line is shorter than 71 provider columns")
    padded = line.ljust(85)
    station_id = padded[0:11].strip()
    if not station_id:
        raise ValueError("GHCN station id is empty")
    latitude = _finite(padded[12:20].strip(), "latitude")
    longitude = _finite(padded[21:30].strip(), "longitude")
    if not -90.0 <= latitude <= 90.0 or not -180.0 <= longitude <= 180.0:
        raise ValueError(f"GHCN station {station_id} has invalid coordinates")
    elevation_raw = padded[31:37].strip()
    elevation = _finite(elevation_raw, "elevation")
    if elevation == -999.9:
        elevation = None
    return GHCNStationMetadata(
        station_id=station_id,
        latitude_deg=latitude,
        longitude_deg=longitude,
        elevation_m=elevation,
        state=padded[38:40].strip(),
        name=padded[41:71].strip(),
        gsn_flag=padded[72:75].strip(),
        hcn_crn_flag=padded[76:79].strip(),
        wmo_id=padded[80:85].strip(),
    )


def _parse_inventory_record_line(line: str) -> GHCNInventoryRecord:
    if len(line) < 45:
        raise ValueError("GHCN inventory line is shorter than 45 provider columns")
    station_id = line[0:11].strip()
    element = line[31:35].strip()
    if not station_id or not element:
        raise ValueError("GHCN inventory station and element must be non-empty")
    latitude = _finite(line[12:20].strip(), "inventory latitude")
    longitude = _finite(line[21:30].strip(), "inventory longitude")
    first_year = int(line[36:40])
    last_year = int(line[41:45])
    if first_year > last_year:
        raise ValueError(
            f"GHCN inventory {(station_id, element)} has reversed year range"
        )
    return GHCNInventoryRecord(
        station_id=station_id,
        latitude_deg=latitude,
        longitude_deg=longitude,
        element=element,
        first_year=first_year,
        last_year=last_year,
    )


@dataclass(frozen=True)
class GHCNMetadataSpoolSummary:
    path: str
    station_count: int
    availability_count: int
    catalog_digest: str
    catalog_byte_count: int
    inventory_digest: str
    inventory_byte_count: int
    max_database_bytes: int
    sqlite_cache_kib: int
    max_transaction_rows: int


class GHCNMetadataSpool:
    """Rebuildable disk-backed metadata index for bounded worldwide ingestion.

    The main SQLite database is hard-capped through max_page_count. SQLite's
    rollback journal and temporary store are kept in memory, mmap is disabled,
    and the page cache is explicitly bounded. This is scratch execution state,
    not a durable scientific artifact; exact provider digests remain the authority.
    """

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
        self.path = path
        self.max_database_bytes = int(max_database_bytes)
        self.sqlite_cache_kib = int(sqlite_cache_kib)
        self.max_transaction_rows = int(max_transaction_rows)
        self._sharding_complete = False
        if create and path.exists():
            raise ValueError(
                "metadata spool path already exists; refuse implicit overwrite"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.execute("PRAGMA journal_mode = MEMORY")
        self._db.execute("PRAGMA temp_store = MEMORY")
        self._db.execute("PRAGMA mmap_size = 0")
        self._db.execute(f"PRAGMA cache_size = {-self.sqlite_cache_kib}")
        page_size = int(self._db.execute("PRAGMA page_size").fetchone()[0])
        max_pages = self.max_database_bytes // page_size
        if max_pages < 8:
            self._db.close()
            if create and path.exists():
                path.unlink()
            raise ValueError(
                "max_database_bytes is too small for the bounded metadata spool"
            )
        current_pages = int(self._db.execute("PRAGMA page_count").fetchone()[0])
        if current_pages > max_pages:
            self._db.close()
            raise ValueError("existing metadata spool exceeds max_database_bytes")
        self._db.execute(f"PRAGMA max_page_count = {max_pages}")
        if create:
            try:
                self._create_schema()
            except sqlite3.OperationalError as exc:
                self._db.close()
                if path.exists():
                    path.unlink()
                if "full" in str(exc).lower():
                    raise ValueError(
                        "metadata spool exceeded max_database_bytes while creating schema"
                    ) from exc
                raise

    def _create_schema(self) -> None:
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
                shard_id TEXT
            );
            CREATE TABLE availability (
                station_id TEXT NOT NULL,
                variable_id TEXT NOT NULL,
                first_year INTEGER NOT NULL,
                last_year INTEGER NOT NULL,
                PRIMARY KEY (station_id, variable_id),
                FOREIGN KEY (station_id) REFERENCES stations(station_id)
            );
            CREATE INDEX stations_lat_lon
                ON stations(latitude_deg, longitude_deg);
            CREATE INDEX stations_shard
                ON stations(shard_id);
            """
        )
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> "GHCNMetadataSpool":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _metadata(self, key: str) -> str:
        row = self._db.execute(
            "SELECT value FROM metadata WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            raise ValueError(f"metadata spool is missing {key!r}")
        return str(row[0])

    def _commit_batch(self) -> None:
        try:
            self._db.commit()
        except sqlite3.OperationalError as exc:
            if "full" in str(exc).lower():
                raise ValueError(
                    "metadata spool exceeded max_database_bytes"
                ) from exc
            raise

    def station_count(self) -> int:
        return int(self._db.execute("SELECT COUNT(*) FROM stations").fetchone()[0])

    def availability_count(self) -> int:
        return int(
            self._db.execute("SELECT COUNT(*) FROM availability").fetchone()[0]
        )

    def summary(self) -> GHCNMetadataSpoolSummary:
        return GHCNMetadataSpoolSummary(
            path=str(self.path),
            station_count=self.station_count(),
            availability_count=self.availability_count(),
            catalog_digest=self._metadata("catalog_digest"),
            catalog_byte_count=int(self._metadata("catalog_byte_count")),
            inventory_digest=self._metadata("inventory_digest"),
            inventory_byte_count=int(self._metadata("inventory_byte_count")),
            max_database_bytes=self.max_database_bytes,
            sqlite_cache_kib=self.sqlite_cache_kib,
            max_transaction_rows=self.max_transaction_rows,
        )

    def _station_from_row(
        self,
        row,
        *,
        catalog_digest: str,
        inventory_digest: str,
        metadata_effective_date: str,
    ) -> FederatedStation:
        station_id, lat, lon, elev = row[0], row[1], row[2], row[3]
        alias = ProviderAlias(SOURCE_ID, station_id)
        availability_rows = self._db.execute(
            """
            SELECT variable_id, first_year, last_year
            FROM availability
            WHERE station_id = ?
            ORDER BY variable_id
            """,
            (station_id,),
        ).fetchall()
        availability = tuple(
            ProviderVariableAvailability(
                variable_id=item[0],
                first_year=int(item[1]),
                last_year=int(item[2]),
                source_alias=alias,
                evidence_digest=inventory_digest,
            )
            for item in availability_rows
        )
        return FederatedStation(
            canonical_station_id=canonical_station_id(alias),
            aliases=(
                AliasBinding(alias, "root", catalog_digest),
            ),
            location_history=(
                StationLocationEpoch(
                    latitude_deg=float(lat),
                    longitude_deg=float(lon),
                    elevation_m=None if elev is None else float(elev),
                    valid_from=metadata_effective_date,
                    valid_to=None,
                    source_alias=alias,
                    evidence_digest=catalog_digest,
                ),
            ),
            variable_ids=tuple(item.variable_id for item in availability),
            provider_location_snapshots=(
                ProviderLocationSnapshot(
                    latitude_deg=float(lat),
                    longitude_deg=float(lon),
                    elevation_m=None if elev is None else float(elev),
                    metadata_effective_date=metadata_effective_date,
                    source_alias=alias,
                    evidence_digest=catalog_digest,
                ),
            ),
            provider_variable_availability=availability,
        )

    def _count_where(self, where_sql: str, params: tuple) -> int:
        row = self._db.execute(
            f"SELECT COUNT(*) FROM stations WHERE {where_sql}", params
        ).fetchone()
        return int(row[0])

    def _leaf(
        self,
        station_ids: Sequence[str],
        *,
        shard_id: str,
    ) -> StationCatalogShard:
        if not station_ids:
            raise ValueError("cannot materialize an empty metadata shard")
        rows = []
        for station_id in sorted(station_ids):
            row = self._db.execute(
                """
                SELECT station_id, latitude_deg, longitude_deg, elevation_m
                FROM stations
                WHERE station_id = ?
                """,
                (station_id,),
            ).fetchone()
            if row is None:
                raise ValueError("metadata spool lost station rows while sharding")
            rows.append(row)
        catalog_digest = self._metadata("catalog_digest")
        inventory_digest = self._metadata("inventory_digest")
        metadata_effective_date = self._metadata("metadata_effective_date")
        stations = tuple(
            self._station_from_row(
                row,
                catalog_digest=catalog_digest,
                inventory_digest=inventory_digest,
                metadata_effective_date=metadata_effective_date,
            )
            for row in rows
        )
        self._db.executemany(
            "UPDATE stations SET shard_id = ? WHERE station_id = ?",
            ((shard_id, station_id) for station_id in station_ids),
        )
        self._commit_batch()
        bounds = StationSpatialBounds(
            latitude_min_deg=min(item.location_history[-1].latitude_deg for item in stations),
            latitude_max_deg=max(item.location_history[-1].latitude_deg for item in stations),
            longitude_min_deg=min(item.location_history[-1].longitude_deg for item in stations),
            longitude_max_deg=max(item.location_history[-1].longitude_deg for item in stations),
        )
        return StationCatalogShard(shard_id, shard_id, stations, bounds)

    def iter_catalog_shards(
        self,
        *,
        max_station_records: int,
        max_shard_depth: int,
    ) -> Iterator[StationCatalogShard]:
        if max_station_records <= 0:
            raise ValueError("max_station_records must be positive")
        if max_shard_depth <= 0:
            raise ValueError("max_shard_depth must be positive")
        self._sharding_complete = False
        self._db.execute("UPDATE stations SET shard_id = NULL")
        self._commit_batch()

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
            count = self._count_where(where_sql, params)
            if count == 0:
                return
            if count <= max_station_records:
                ids = [
                    row[0]
                    for row in self._db.execute(
                        f"""
                        SELECT station_id FROM stations
                        WHERE {where_sql}
                        ORDER BY station_id
                        """,
                        params,
                    ).fetchall()
                ]
                yield self._leaf(ids, shard_id=key)
                return

            if depth >= max_shard_depth:
                raise ValueError(
                    "GHCN metadata sharding exceeded max_shard_depth"
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
                self._count_where(item[0], item[1]) for item in quadrants
            ]
            nonempty = [index for index, value in enumerate(counts) if value]
            if len(nonempty) == 1:
                last_station_id = ""
                chunk_index = 0
                while True:
                    rows = self._db.execute(
                        f"""
                        SELECT station_id FROM stations
                        WHERE ({where_sql}) AND station_id > ?
                        ORDER BY station_id
                        LIMIT ?
                        """,
                        (*params, last_station_id, max_station_records),
                    ).fetchall()
                    if not rows:
                        break
                    station_ids = [row[0] for row in rows]
                    yield self._leaf(
                        station_ids,
                        shard_id=f"{key}/id-{chunk_index:06d}",
                    )
                    last_station_id = station_ids[-1]
                    chunk_index += 1
                return

            for index, item in enumerate(quadrants):
                if counts[index] == 0:
                    continue
                yield from recurse(
                    item[0],
                    item[1],
                    item[2],
                    item[3],
                    item[4],
                    item[5],
                    f"{key}{index}",
                    depth + 1,
                )

        yield from recurse(
            "1 = 1", (), -90.0, 90.0, -180.0, 180.0, "q", 0
        )
        self._sharding_complete = True

    def resolve(self, provider_station_id: str) -> tuple[str, str]:
        if not self._sharding_complete:
            raise ValueError(
                "metadata spool routing is unavailable until "
                "iter_catalog_shards() is fully consumed"
            )
        row = self._db.execute(
            "SELECT shard_id FROM stations WHERE station_id = ?",
            (provider_station_id,),
        ).fetchone()
        if row is None:
            raise ValueError(
                f"provider station {provider_station_id!r} is absent from "
                "the captured metadata spool"
            )
        if row[0] is None:
            raise ValueError(
                "metadata spool has not assigned catalog shards; consume "
                "iter_catalog_shards() before routing observations"
            )
        alias = ProviderAlias(SOURCE_ID, provider_station_id)
        return canonical_station_id(alias), str(row[0])

    @property
    def source_id(self) -> str:
        return SOURCE_ID


def build_metadata_spool(
    catalog_path: Path,
    inventory_path: Path,
    *,
    spool_path: Path,
    metadata_effective_date: str,
    max_database_bytes: int,
    sqlite_cache_kib: int = 2048,
    max_transaction_rows: int = 1000,
    max_metadata_line_bytes: int = 4096,
) -> GHCNMetadataSpool:
    """Stream captured provider metadata into a hard-capped local scratch index."""
    date.fromisoformat(metadata_effective_date)
    if max_metadata_line_bytes <= 0:
        raise ValueError("max_metadata_line_bytes must be positive")
    spool = GHCNMetadataSpool(
        spool_path,
        max_database_bytes=max_database_bytes,
        sqlite_cache_kib=sqlite_cache_kib,
        max_transaction_rows=max_transaction_rows,
        create=True,
    )
    try:
        spool._db.execute(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            ("metadata_effective_date", metadata_effective_date),
        )
        spool._commit_batch()

        catalog_hash = hashlib.sha256()
        catalog_bytes = 0
        pending = 0
        with catalog_path.open("rb") as handle:
            for line_number, raw_bytes in enumerate(
                _iter_bounded_binary_lines(
                    handle, max_line_bytes=max_metadata_line_bytes
                ),
                start=1,
            ):
                catalog_hash.update(raw_bytes)
                catalog_bytes += len(raw_bytes)
                try:
                    line = raw_bytes.decode("utf-8").rstrip("\r\n")
                except UnicodeDecodeError as exc:
                    raise ValueError(
                        f"invalid GHCN station metadata line {line_number}: "
                        "metadata must be valid UTF-8"
                    ) from exc
                if not line:
                    continue
                try:
                    item = _parse_station_metadata_line(line)
                    spool._db.execute(
                        """
                        INSERT INTO stations(
                            station_id, latitude_deg, longitude_deg, elevation_m,
                            state, name, gsn_flag, hcn_crn_flag, wmo_id
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        ),
                    )
                except (ValueError, sqlite3.IntegrityError) as exc:
                    raise ValueError(
                        f"invalid GHCN station metadata line {line_number}: {exc}"
                    ) from exc
                pending += 1
                if pending >= max_transaction_rows:
                    spool._commit_batch()
                    pending = 0
        spool._commit_batch()
        if spool.station_count() == 0:
            raise ValueError("GHCN station catalog contains no records")
        catalog_digest = "sha256:" + catalog_hash.hexdigest()
        spool._db.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            (
                ("catalog_digest", catalog_digest),
                ("catalog_byte_count", str(catalog_bytes)),
            ),
        )
        spool._commit_batch()

        inventory_hash = hashlib.sha256()
        inventory_bytes = 0
        pending = 0
        with inventory_path.open("rb") as handle:
            for line_number, raw_bytes in enumerate(
                _iter_bounded_binary_lines(
                    handle, max_line_bytes=max_metadata_line_bytes
                ),
                start=1,
            ):
                inventory_hash.update(raw_bytes)
                inventory_bytes += len(raw_bytes)
                try:
                    line = raw_bytes.decode("utf-8").rstrip("\r\n")
                except UnicodeDecodeError as exc:
                    raise ValueError(
                        f"invalid GHCN inventory line {line_number}: "
                        "metadata must be valid UTF-8"
                    ) from exc
                if not line:
                    continue
                try:
                    item = _parse_inventory_record_line(line)
                    spool._db.execute(
                        """
                        INSERT INTO availability(
                            station_id, variable_id, first_year, last_year
                        ) VALUES (?, ?, ?, ?)
                        """,
                        (
                            item.station_id,
                            item.element,
                            item.first_year,
                            item.last_year,
                        ),
                    )
                except (ValueError, sqlite3.IntegrityError) as exc:
                    raise ValueError(
                        f"invalid GHCN inventory line {line_number}: {exc}"
                    ) from exc
                pending += 1
                if pending >= max_transaction_rows:
                    spool._commit_batch()
                    pending = 0
        spool._commit_batch()
        if spool.availability_count() == 0:
            raise ValueError("GHCN inventory contains no records")
        inventory_digest = "sha256:" + inventory_hash.hexdigest()
        spool._db.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            (
                ("inventory_digest", inventory_digest),
                ("inventory_byte_count", str(inventory_bytes)),
            ),
        )
        spool._commit_batch()
        return spool
    except sqlite3.OperationalError as exc:
        spool.close()
        if spool_path.exists():
            spool_path.unlink()
        if "full" in str(exc).lower():
            raise ValueError(
                "metadata spool exceeded max_database_bytes"
            ) from exc
        raise
    except Exception:
        spool.close()
        if spool_path.exists():
            spool_path.unlink()
        raise


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
        inventory_records = tuple(
            inventory_by_station.get(item.station_id, ())
        )
        variables = tuple(sorted({
            record.element for record in inventory_records
        }))
        availability = tuple(
            ProviderVariableAvailability(
                variable_id=record.element,
                first_year=record.first_year,
                last_year=record.last_year,
                source_alias=alias,
                evidence_digest=inventory.sha256,
            )
            for record in sorted(
                inventory_records,
                key=lambda value: value.element,
            )
        )
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
            provider_location_snapshots=(
                ProviderLocationSnapshot(
                    latitude_deg=item.latitude_deg,
                    longitude_deg=item.longitude_deg,
                    elevation_m=item.elevation_m,
                    metadata_effective_date=metadata_effective_date,
                    source_alias=alias,
                    evidence_digest=catalog.sha256,
                ),
            ),
            provider_variable_availability=availability,
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


class StationRoutingLookup(Protocol):
    source_id: str

    def resolve(self, provider_station_id: str) -> tuple[str, str]:
        ...


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
    lookup: StationRoutingLookup,
    sink: ObservationPartitionSink,
    max_rows: int,
    max_partition_keys: int,
    elements: Sequence[str] | None = None,
) -> GHCNRoutingSummary:
    """Route one provider year to bounded partition sinks without row buffering.

    Storage format is deliberately sink-owned so maintained Parquet/Zarr/object-
    store writers can implement publication without changing provider parsing,
    station identity, partition keys, or missing/flag semantics.
    """
    if max_rows <= 0:
        raise ValueError("max_rows must be positive")
    if max_partition_keys <= 0:
        raise ValueError("max_partition_keys must be positive")
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
        if row_count >= max_rows:
            raise ValueError("observation routing exceeded max_rows")
        if key not in partition_keys:
            if len(partition_keys) >= max_partition_keys:
                raise ValueError(
                    "observation routing exceeded max_partition_keys"
                )
            partition_keys.add(key)
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
