from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from src.station_federation import (
    adaptive_catalog_shards as generic_adaptive_catalog_shards,
    catalog_shard_refs as generic_catalog_shard_refs,
)
from data.ncei_ghcnd_bulk import (
    adaptive_catalog_shards,
    build_federated_stations,
    build_metadata_spool,
    GHCNMetadataSpool,
    by_year_url,
    by_year_urls,
    catalog_shard_refs,
    StationShardLookup,
    parse_inventory,
    parse_station_catalog,
    stream_by_year_partitions,
)


def station_line(station_id, lat, lon, elev, name, state="", wmo=""):
    return (
        f"{station_id:<11} {lat:8.4f} {lon:9.4f} {elev:6.1f} "
        f"{state:<2} {name:<30} {'':<3} {'':<3} {wmo:<5}"
    ) + "\n"


def inventory_line(station_id, lat, lon, element, first, last):
    return (
        f"{station_id:<11} {lat:8.4f} {lon:9.4f} "
        f"{element:<4} {first:4d} {last:4d}\n"
    )


class GHCNBulkFederationTests(unittest.TestCase):
    def payloads(self):
        catalog = (
            station_line("USW00000001", 40.0, -75.0, 10.0, "ALPHA", "PA", "12345")
            + station_line("USW00000002", -33.9, 151.2, 5.0, "BETA")
            + station_line("USW00000003", 51.5, -0.1, -999.9, "GAMMA")
        ).encode("ascii")
        inventory = (
            inventory_line("USW00000001", 40.0, -75.0, "TMAX", 1900, 2026)
            + inventory_line("USW00000001", 40.0, -75.0, "PRCP", 1900, 2026)
            + inventory_line("USW00000002", -33.9, 151.2, "TMIN", 1950, 2026)
            + inventory_line("USW00000003", 51.5, -0.1, "TAVG", 1880, 2026)
        ).encode("ascii")
        return parse_station_catalog(catalog), parse_inventory(inventory)

    def test_station_catalog_accepts_utf8_names_without_shifting_fixed_columns(self):
        payload = station_line(
            "BR000000001",
            -23.5505,
            -46.6333,
            760.0,
            "SÃO PAULO",
            "SP",
            "83781",
        ).encode("utf-8")
        catalog = parse_station_catalog(payload)
        self.assertEqual(catalog.records[0].name, "SÃO PAULO")
        self.assertEqual(catalog.records[0].state, "SP")
        self.assertEqual(catalog.records[0].wmo_id, "83781")

    def test_provider_fixed_width_catalog_and_inventory_parse(self):
        catalog, inventory = self.payloads()
        self.assertEqual(len(catalog.records), 3)
        self.assertEqual(catalog.records[0].wmo_id, "12345")
        self.assertIsNone(catalog.records[2].elevation_m)
        self.assertEqual(len(inventory.records), 4)
        self.assertTrue(catalog.sha256.startswith("sha256:"))
        self.assertTrue(inventory.sha256.startswith("sha256:"))

    def test_bulk_metadata_enters_federation_without_historical_location_guess(self):
        catalog, inventory = self.payloads()
        stations = build_federated_stations(
            catalog, inventory, metadata_effective_date="2026-09-18"
        )
        self.assertEqual(stations[0].variable_ids, ("PRCP", "TMAX"))
        self.assertIsNone(stations[0].location_at("2000-01-01"))
        self.assertIsNotNone(stations[0].location_at("2026-09-18"))
        self.assertEqual(len(stations[0].provider_location_snapshots), 1)
        snapshot = stations[0].provider_location_snapshots[0]
        self.assertEqual(snapshot.source_alias, stations[0].aliases[0].alias)
        self.assertEqual(snapshot.metadata_effective_date, "2026-09-18")
        self.assertEqual(snapshot.evidence_digest, catalog.sha256)
        availability = {
            item.variable_id: item
            for item in stations[0].provider_variable_availability
        }
        self.assertEqual(set(availability), {"PRCP", "TMAX"})
        self.assertEqual(availability["TMAX"].first_year, 1900)
        self.assertEqual(availability["TMAX"].last_year, 2026)
        self.assertEqual(
            availability["TMAX"].evidence_digest,
            inventory.sha256,
        )

    def test_bounded_metadata_spool_streams_catalog_and_routes_from_disk(self):
        catalog_text = (
            station_line("USW00000001", 40.0, -75.0, 10.0, "ALPHA")
            + station_line("USW00000002", -33.9, 151.2, 5.0, "BETA")
            + station_line("USW00000003", 51.5, -0.1, -999.9, "GAMMA")
        )
        inventory_text = (
            inventory_line("USW00000001", 40.0, -75.0, "TMAX", 1900, 2026)
            + inventory_line("USW00000001", 40.0, -75.0, "PRCP", 1950, 2026)
            + inventory_line("USW00000002", -33.9, 151.2, "TMIN", 1970, 2026)
            + inventory_line("USW00000003", 51.5, -0.1, "TAVG", 1880, 2026)
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog_path = root / "ghcnd-stations.txt"
            inventory_path = root / "ghcnd-inventory.txt"
            spool_path = root / "station-metadata.sqlite"
            catalog_path.write_text(catalog_text, encoding="utf-8")
            inventory_path.write_text(inventory_text, encoding="utf-8")

            with build_metadata_spool(
                catalog_path,
                inventory_path,
                spool_path=spool_path,
                metadata_effective_date="2026-09-18",
                max_database_bytes=1024 * 1024,
                sqlite_cache_kib=256,
                max_transaction_rows=2,
            ) as spool:
                summary = spool.summary()
                self.assertEqual(summary.station_count, 3)
                self.assertEqual(summary.availability_count, 4)
                self.assertLessEqual(spool_path.stat().st_size, 1024 * 1024)

                shards = list(
                    spool.iter_catalog_shards(max_station_records=1, max_shard_depth=16)
                )
                self.assertEqual(len(shards), 3)
                self.assertTrue(
                    all(len(shard.stations) <= 1 for shard in shards)
                )
                alpha = next(
                    station
                    for shard in shards
                    for station in shard.stations
                    if any(
                        binding.alias.provider_station_id
                        == "USW00000001"
                        for binding in station.aliases
                    )
                )
                availability = {
                    item.variable_id: item
                    for item in alpha.provider_variable_availability
                }
                self.assertEqual(availability["TMAX"].first_year, 1900)
                self.assertEqual(availability["PRCP"].first_year, 1950)
                self.assertEqual(
                    availability["TMAX"].evidence_digest,
                    summary.inventory_digest,
                )
                canonical_id, shard_id = spool.resolve("USW00000001")
                self.assertEqual(canonical_id, alpha.canonical_station_id)
                self.assertTrue(shard_id)

    def test_metadata_spool_refuses_unbudgeted_storage_and_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog_path = root / "ghcnd-stations.txt"
            inventory_path = root / "ghcnd-inventory.txt"
            spool_path = root / "station-metadata.sqlite"
            catalog_path.write_text(
                station_line(
                    "USW00000001", 40.0, -75.0, 10.0, "ALPHA"
                ),
                encoding="utf-8",
            )
            inventory_path.write_text(
                inventory_line(
                    "USW00000001", 40.0, -75.0, "TMAX", 1900, 2026
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "too small"):
                build_metadata_spool(
                    catalog_path,
                    inventory_path,
                    spool_path=spool_path,
                    metadata_effective_date="2026-09-18",
                    max_database_bytes=1024,
                )
            self.assertFalse(spool_path.exists())

            spool = build_metadata_spool(
                catalog_path,
                inventory_path,
                spool_path=spool_path,
                metadata_effective_date="2026-09-18",
                max_database_bytes=1024 * 1024,
            )
            spool.close()
            with self.assertRaisesRegex(ValueError, "implicit overwrite"):
                build_metadata_spool(
                    catalog_path,
                    inventory_path,
                    spool_path=spool_path,
                    metadata_effective_date="2026-09-18",
                    max_database_bytes=1024 * 1024,
                )

    def test_metadata_spool_rejects_duplicate_or_orphan_inventory_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog_path = root / "ghcnd-stations.txt"
            inventory_path = root / "ghcnd-inventory.txt"
            spool_path = root / "station-metadata.sqlite"
            catalog_path.write_text(
                station_line(
                    "USW00000001", 40.0, -75.0, 10.0, "ALPHA"
                ),
                encoding="utf-8",
            )
            duplicate = inventory_line(
                "USW00000001", 40.0, -75.0, "TMAX", 1900, 2026
            )
            inventory_path.write_text(
                duplicate + duplicate,
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "inventory line 2"):
                build_metadata_spool(
                    catalog_path,
                    inventory_path,
                    spool_path=spool_path,
                    metadata_effective_date="2026-09-18",
                    max_database_bytes=1024 * 1024,
                )
            self.assertFalse(spool_path.exists())

            inventory_path.write_text(
                inventory_line(
                    "USW99999999", 40.0, -75.0, "TMAX", 1900, 2026
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "inventory line 1"):
                build_metadata_spool(
                    catalog_path,
                    inventory_path,
                    spool_path=spool_path,
                    metadata_effective_date="2026-09-18",
                    max_database_bytes=1024 * 1024,
                )
            self.assertFalse(spool_path.exists())

    def test_metadata_spool_bounds_corrupt_line_allocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog_path = root / "ghcnd-stations.txt"
            inventory_path = root / "ghcnd-inventory.txt"
            spool_path = root / "station-metadata.sqlite"
            catalog_path.write_bytes(b"X" * 5000 + b"\n")
            inventory_path.write_text(
                inventory_line(
                    "USW00000001", 40.0, -75.0, "TMAX", 1900, 2026
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "exceeds 4096 bytes"):
                build_metadata_spool(
                    catalog_path,
                    inventory_path,
                    spool_path=spool_path,
                    metadata_effective_date="2026-09-18",
                    max_database_bytes=1024 * 1024,
                    max_metadata_line_bytes=4096,
                )
            self.assertFalse(spool_path.exists())

    def test_leaf_materialization_avoids_multi_bind_sqlite_dependency(self):
        class SingleBindExecuteConnection:
            def __init__(self, connection):
                self.connection = connection

            def execute(self, sql, params=()):
                if len(params) > 1:
                    raise AssertionError(
                        "single execute() must not depend on multi-bind SQL"
                    )
                return self.connection.execute(sql, params)

            def executemany(self, sql, params):
                return self.connection.executemany(sql, params)

            def commit(self):
                return self.connection.commit()

            def close(self):
                return self.connection.close()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spool = GHCNMetadataSpool(
                root / "metadata.sqlite",
                max_database_bytes=1024 * 1024,
                create=True,
            )
            spool._db.executemany(
                "INSERT INTO metadata(key, value) VALUES (?, ?)",
                (
                    ("catalog_digest", "sha256:" + "1" * 64),
                    ("catalog_byte_count", "1"),
                    ("inventory_digest", "sha256:" + "2" * 64),
                    ("inventory_byte_count", "1"),
                    ("metadata_effective_date", "2026-09-18"),
                ),
            )
            spool._db.executemany(
                """
                INSERT INTO stations(
                    station_id, latitude_deg, longitude_deg, elevation_m,
                    state, name, gsn_flag, hcn_crn_flag, wmo_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    (
                        "USW00000001",
                        40.0,
                        -75.0,
                        10.0,
                        "",
                        "ALPHA",
                        "",
                        "",
                        "",
                    ),
                    (
                        "USW00000002",
                        41.0,
                        -74.0,
                        11.0,
                        "",
                        "BETA",
                        "",
                        "",
                        "",
                    ),
                ),
            )
            spool._db.commit()
            spool._db = SingleBindExecuteConnection(spool._db)
            try:
                shard = spool._leaf(
                    ("USW00000001", "USW00000002"),
                    shard_id="q",
                )
                self.assertEqual(len(shard.stations), 2)
            finally:
                spool.close()

    def test_disk_lookup_is_unavailable_until_sharding_is_fully_consumed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog_path = root / "ghcnd-stations.txt"
            inventory_path = root / "ghcnd-inventory.txt"
            catalog_path.write_text(
                station_line(
                    "USW00000001", 40.0, -75.0, 10.0, "ALPHA"
                )
                + station_line(
                    "USW00000002", -33.9, 151.2, 5.0, "BETA"
                ),
                encoding="utf-8",
            )
            inventory_path.write_text(
                inventory_line(
                    "USW00000001", 40.0, -75.0, "TMAX", 1900, 2026
                )
                + inventory_line(
                    "USW00000002", -33.9, 151.2, "TMIN", 1950, 2026
                ),
                encoding="utf-8",
            )
            with build_metadata_spool(
                catalog_path,
                inventory_path,
                spool_path=root / "metadata.sqlite",
                metadata_effective_date="2026-09-18",
                max_database_bytes=1024 * 1024,
            ) as spool:
                scan = spool.iter_catalog_shards(
                    max_station_records=1,
                    max_shard_depth=16,
                )
                next(scan)
                with self.assertRaisesRegex(
                    ValueError, "fully consumed"
                ):
                    spool.resolve("USW00000001")
                scan.close()

    def test_disk_lookup_routes_observations_without_global_alias_dict(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog_path = root / "ghcnd-stations.txt"
            inventory_path = root / "ghcnd-inventory.txt"
            spool_path = root / "station-metadata.sqlite"
            catalog_path.write_text(
                station_line(
                    "USW00000001", 40.0, -75.0, 10.0, "ALPHA"
                )
                + station_line(
                    "USW00000002", -33.9, 151.2, 5.0, "BETA"
                ),
                encoding="utf-8",
            )
            inventory_path.write_text(
                inventory_line(
                    "USW00000001", 40.0, -75.0, "TMAX", 1900, 2026
                )
                + inventory_line(
                    "USW00000002", -33.9, 151.2, "TMIN", 1950, 2026
                ),
                encoding="utf-8",
            )
            writes = []

            class Sink:
                def write(self, key, record):
                    writes.append((key, record))

            with build_metadata_spool(
                catalog_path,
                inventory_path,
                spool_path=spool_path,
                metadata_effective_date="2026-09-18",
                max_database_bytes=1024 * 1024,
            ) as spool:
                list(spool.iter_catalog_shards(max_station_records=1, max_shard_depth=16))
                summary = stream_by_year_partitions(
                    [
                        "USW00000001,20240101,TMAX,123,,,S,0700\n",
                        "USW00000002,20240101,TMIN,45,,,S,0700\n",
                    ],
                    expected_year=2024,
                    lookup=spool,
                    sink=Sink(),
                max_rows=100,
                    max_partition_keys=10,
                )
            self.assertEqual(summary.row_count, 2)
            self.assertEqual(len(writes), 2)
            self.assertNotEqual(
                writes[0][0].spatial_partition,
                writes[1][0].spatial_partition,
            )

    def test_provider_module_reexports_provider_neutral_sharding(self):
        self.assertIs(adaptive_catalog_shards, generic_adaptive_catalog_shards)
        self.assertIs(catalog_shard_refs, generic_catalog_shard_refs)

    def test_adaptive_shards_are_bounded_and_content_addressed(self):
        catalog, inventory = self.payloads()
        stations = build_federated_stations(
            catalog, inventory, metadata_effective_date="2026-09-18"
        )
        shards = adaptive_catalog_shards(
            stations,
            metadata_effective_date="2026-09-18",
            max_station_records=1,
        )
        self.assertEqual(sum(len(shard.stations) for shard in shards), 3)
        self.assertTrue(all(len(shard.stations) <= 1 for shard in shards))
        refs = catalog_shard_refs(shards)
        self.assertEqual(len(refs), len(shards))
        self.assertTrue(all(ref.digest.startswith("sha256:") for ref in refs))

    def test_by_year_urls_are_provider_bulk_artifacts_not_station_requests(self):
        self.assertEqual(
            by_year_url(2024),
            "https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_year/2024.csv.gz",
        )
        self.assertEqual(
            by_year_urls(2023, 2024),
            (
                "https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_year/2023.csv.gz",
                "https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_year/2024.csv.gz",
            ),
        )


    def test_by_year_stream_routes_rows_without_year_materialization(self):
        catalog, inventory = self.payloads()
        stations = build_federated_stations(
            catalog, inventory, metadata_effective_date="2026-09-18"
        )
        shards = adaptive_catalog_shards(
            stations,
            metadata_effective_date="2026-09-18",
            max_station_records=1,
        )
        lookup = StationShardLookup(shards)
        writes = []
        class Sink:
            def write(self, key, record):
                writes.append((key, record))

        def lines():
            yield "USW00000001,20240101,TMAX,123,,,S,0700\n"
            self.assertEqual(len(writes), 1)
            yield "USW00000002,20240101,TMIN,-9999,,Q,S,\n"
            self.assertEqual(len(writes), 2)
            yield "USW00000001,20240102,PRCP,5,,,S,0700\n"

        summary = stream_by_year_partitions(
            lines(),
            expected_year=2024,
            lookup=lookup,
            sink=Sink(),
                max_rows=100,
                    max_partition_keys=10,
        )
        self.assertEqual(summary.row_count, 3)
        self.assertEqual(summary.missing_value_count, 1)
        self.assertEqual(summary.partition_count, 3)
        self.assertIsNone(writes[1][1].value)
        self.assertEqual(writes[1][1].quality_flag, "Q")
        self.assertEqual(writes[0][0].source_id, "ncei.ghcnd.v3")

    def test_by_year_row_budget_fails_before_sink_side_effect(self):
        catalog, inventory = self.payloads()
        stations = build_federated_stations(
            catalog, inventory, metadata_effective_date="2026-09-18"
        )
        lookup = StationShardLookup(
            adaptive_catalog_shards(
                stations,
                metadata_effective_date="2026-09-18",
                max_station_records=1,
            )
        )
        writes = []

        class Sink:
            def write(self, key, record):
                writes.append((key, record))

        with self.assertRaisesRegex(ValueError, "max_rows"):
            stream_by_year_partitions(
                [
                    "USW00000001,20240101,TMAX,100,,,S,0700\n",
                    "USW00000001,20240102,TMAX,101,,,S,0700\n",
                ],
                expected_year=2024,
                lookup=lookup,
                sink=Sink(),
                max_rows=1,
                max_partition_keys=10,
            )
        self.assertEqual(len(writes), 1)

    def test_by_year_partition_key_budget_fails_closed(self):
        catalog, inventory = self.payloads()
        stations = build_federated_stations(
            catalog, inventory, metadata_effective_date="2026-09-18"
        )
        lookup = StationShardLookup(
            adaptive_catalog_shards(
                stations,
                metadata_effective_date="2026-09-18",
                max_station_records=1,
            )
        )

        class Sink:
            def write(self, key, record):
                pass

        with self.assertRaisesRegex(ValueError, "max_partition_keys"):
            stream_by_year_partitions(
                [
                    "USW00000001,20240101,TMAX,100,,,S,0700\n",
                    "USW00000002,20240101,TMIN,100,,,S,0700\n",
                ],
                expected_year=2024,
                lookup=lookup,
                sink=Sink(),
                max_rows=100,
                max_partition_keys=1,
            )

    def test_by_year_unknown_station_fails_closed(self):
        catalog, inventory = self.payloads()
        stations = build_federated_stations(
            catalog, inventory, metadata_effective_date="2026-09-18"
        )
        lookup = StationShardLookup(
            adaptive_catalog_shards(
                stations,
                metadata_effective_date="2026-09-18",
                max_station_records=2,
            )
        )
        class Sink:
            def write(self, key, record):
                raise AssertionError("unknown station must refuse before sink write")
        with self.assertRaisesRegex(ValueError, "absent from the captured federation catalog"):
            stream_by_year_partitions(
                ["USW99999999,20240101,TMAX,100,,,S,0700\n"],
                expected_year=2024,
                lookup=lookup,
                sink=Sink(),
                max_rows=100,
                    max_partition_keys=10,
            )

    def test_by_year_wrong_artifact_year_refuses(self):
        catalog, inventory = self.payloads()
        stations = build_federated_stations(
            catalog, inventory, metadata_effective_date="2026-09-18"
        )
        lookup = StationShardLookup(
            adaptive_catalog_shards(
                stations,
                metadata_effective_date="2026-09-18",
                max_station_records=2,
            )
        )
        class Sink:
            def write(self, key, record):
                raise AssertionError("wrong-year row must not reach sink")
        with self.assertRaisesRegex(ValueError, "does not match artifact year"):
            stream_by_year_partitions(
                ["USW00000001,20250101,TMAX,100,,,S,0700\n"],
                expected_year=2024,
                lookup=lookup,
                sink=Sink(),
                max_rows=100,
                    max_partition_keys=10,
            )


if __name__ == "__main__":
    unittest.main()
