from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from data.ncei_ghcnh_partition import (
    MEDIA_TYPE,
    publish_year_archive,
)
from data.ncei_ghcnh_bulk import (
    ARCHIVE_BASE_URL,
    SOURCE_ID,
    GHCNhAliasShardLookup,
    build_ghcnh_metadata_spool,
    validate_year_archive_url,
    federate_station_catalog,
    parse_station_catalog,
    stream_station_year_psv,
)
from src.station_federation import (
    AliasBinding,
    FederatedStation,
    ProviderAlias,
    ProviderLocationSnapshot,
    StationCatalogShard,
    StationFederationManifest,
    StationLocationEpoch,
    adaptive_catalog_shards,
    canonical_station_id,
    catalog_shard_refs,
)


D1 = "sha256:" + "1" * 64


def station_line(
    station_id,
    lat,
    lon,
    elev,
    name,
    state="",
    gsn="",
    hcn="",
    wmo="",
    icao="",
):
    return (
        f"{station_id:<11} {lat:8.4f} {lon:9.4f} {elev:6.1f} "
        f"{state:<2} {name:<30} {gsn:<3} {hcn:<3} {wmo:<5} {icao:<4}"
    ) + "\n"


def daily_station(station_id: str) -> FederatedStation:
    alias = ProviderAlias("ncei.ghcnd.v3", station_id)
    return FederatedStation(
        canonical_station_id=canonical_station_id(alias),
        aliases=(AliasBinding(alias, "root", D1),),
        location_history=(
            StationLocationEpoch(
                40.0,
                -75.0,
                10.0,
                "2026-01-01",
                None,
                alias,
                D1,
            ),
        ),
        variable_ids=("TMAX",),
        provider_location_snapshots=(
            ProviderLocationSnapshot(
                40.0,
                -75.0,
                10.0,
                "2026-01-01",
                alias,
                D1,
            ),
        ),
    )


def write_year_archive(path: Path, members: dict[str, str]) -> None:
    with tarfile.open(path, mode="w:gz") as archive:
        for name, text in members.items():
            encoded = text.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(encoded)
            info.mtime = 0
            archive.addfile(info, io.BytesIO(encoded))


class GHCNhFederationTests(unittest.TestCase):
    def test_versioned_annual_archive_url_refuses_unversioned_name(self):
        with self.assertRaisesRegex(ValueError, "versioned"):
            validate_year_archive_url(ARCHIVE_BASE_URL + "latest.tar.gz")

    def test_versioned_annual_archive_url_preserves_provider_identity(self):
        name = "ghcn-hourly_v1.0.0_d2026_c20260918.tar.gz"
        self.assertEqual(
            validate_year_archive_url(ARCHIVE_BASE_URL + name),
            name,
        )
    def test_station_list_parses_documented_fixed_width_fields(self):
        payload = station_line(
            "USW00094846",
            41.98,
            -87.90,
            204.0,
            "CHICAGO OHARE",
            "IL",
            "GSN",
            "",
            "94846",
            "KORD",
        ).encode("ascii")
        catalog = parse_station_catalog(payload)
        self.assertEqual(len(catalog.records), 1)
        station = catalog.records[0]
        self.assertEqual(station.station_id, "USW00094846")
        self.assertEqual(station.wmo_id, "94846")
        self.assertEqual(station.icao, "KORD")
        self.assertTrue(catalog.sha256.startswith("sha256:"))

    def test_shared_ghcn_identifier_becomes_explicit_alias_not_new_root(self):
        daily = daily_station("USW00094846")
        catalog = parse_station_catalog(
            (
                station_line(
                    "USW00094846", 41.98, -87.90, 204.0, "CHICAGO OHARE", "IL"
                )
                + station_line(
                    "CAW00099999", 50.0, -100.0, 300.0, "HOURLY ONLY"
                )
            ).encode("ascii")
        )
        result = federate_station_catalog(
            (daily,),
            catalog,
            metadata_effective_date="2026-09-18",
        )
        self.assertEqual(result.shared_station_count, 1)
        self.assertEqual(result.new_root_station_count, 1)
        self.assertEqual(len(result.stations), 2)

        linked = next(
            item
            for item in result.stations
            if item.canonical_station_id == daily.canonical_station_id
        )
        hourly_alias = ProviderAlias(SOURCE_ID, "USW00094846")
        binding = next(item for item in linked.aliases if item.alias == hourly_alias)
        crosswalk = next(
            item for item in result.crosswalk_evidence if item.alias == hourly_alias
        )
        self.assertEqual(binding.binding_method, "provider_crosswalk")
        self.assertEqual(binding.evidence_digest, crosswalk.evidence_digest)
        hourly_snapshot = next(
            item
            for item in linked.provider_location_snapshots
            if item.source_alias == hourly_alias
        )
        self.assertEqual(hourly_snapshot.latitude_deg, 41.98)
        self.assertEqual(hourly_snapshot.longitude_deg, -87.90)
        self.assertEqual(hourly_snapshot.evidence_digest, catalog.sha256)
        self.assertEqual(
            linked.location_at("2026-09-18").latitude_deg,
            40.0,
        )

        hourly_only = next(
            item
            for item in result.stations
            if any(
                binding.alias == ProviderAlias(SOURCE_ID, "CAW00099999")
                and binding.binding_method == "root"
                for binding in item.aliases
            )
        )
        self.assertEqual(
            hourly_only.canonical_station_id,
            canonical_station_id(ProviderAlias(SOURCE_ID, "CAW00099999")),
        )

    def test_catalog_revision_updates_snapshot_without_rebinding_identity(self):
        daily = daily_station("USW00094846")
        first_catalog = parse_station_catalog(
            station_line(
                "USW00094846",
                41.98,
                -87.90,
                204.0,
                "CHICAGO OHARE",
            ).encode("ascii")
        )
        first = federate_station_catalog(
            (daily,),
            first_catalog,
            metadata_effective_date="2026-09-18",
        )
        second_catalog = parse_station_catalog(
            station_line(
                "USW00094846",
                41.99,
                -87.89,
                205.0,
                "CHICAGO OHARE",
            ).encode("ascii")
        )
        second = federate_station_catalog(
            first.stations,
            second_catalog,
            metadata_effective_date="2026-09-19",
        )

        first_station = first.stations[0]
        second_station = second.stations[0]
        hourly_alias = ProviderAlias(SOURCE_ID, "USW00094846")
        self.assertEqual(
            first_station.canonical_station_id,
            second_station.canonical_station_id,
        )
        self.assertEqual(first.crosswalk_evidence, second.crosswalk_evidence)
        first_binding = next(
            item for item in first_station.aliases if item.alias == hourly_alias
        )
        second_binding = next(
            item for item in second_station.aliases if item.alias == hourly_alias
        )
        self.assertEqual(first_binding, second_binding)

        hourly_snapshots = [
            item
            for item in second_station.provider_location_snapshots
            if item.source_alias == hourly_alias
        ]
        self.assertEqual(len(hourly_snapshots), 1)
        self.assertEqual(hourly_snapshots[0].latitude_deg, 41.99)
        self.assertEqual(hourly_snapshots[0].longitude_deg, -87.89)
        self.assertEqual(hourly_snapshots[0].elevation_m, 205.0)
        self.assertEqual(
            hourly_snapshots[0].metadata_effective_date,
            "2026-09-19",
        )
        self.assertEqual(
            hourly_snapshots[0].evidence_digest,
            second_catalog.sha256,
        )
        self.assertEqual(
            second_station.location_at("2026-09-19").latitude_deg,
            40.0,
        )

    def test_hourly_only_refresh_is_idempotent_bounded_and_advances_root_epoch(self):
        first_catalog = parse_station_catalog(
            station_line(
                "CAW00099999",
                50.0,
                -100.0,
                300.0,
                "HOURLY ONLY",
            ).encode("ascii")
        )
        first = federate_station_catalog(
            (),
            first_catalog,
            metadata_effective_date="2026-09-18",
        )
        self.assertEqual(first.new_root_station_count, 1)
        self.assertEqual(len(first.stations), 1)

        second_catalog = parse_station_catalog(
            station_line(
                "CAW00099999",
                50.1,
                -99.9,
                301.0,
                "HOURLY ONLY",
            ).encode("ascii")
        )
        second = federate_station_catalog(
            first.stations,
            second_catalog,
            metadata_effective_date="2026-09-19",
        )
        self.assertEqual(second.new_root_station_count, 0)
        self.assertEqual(len(second.stations), 1)
        item = second.stations[0]
        self.assertEqual(len(item.provider_location_snapshots), 1)
        self.assertEqual(len(item.location_history), 2)
        self.assertEqual(item.location_at("2026-09-18").latitude_deg, 50.0)
        self.assertEqual(item.location_at("2026-09-19").latitude_deg, 50.1)
        self.assertEqual(
            item.provider_location_snapshots[0].evidence_digest,
            second_catalog.sha256,
        )

        replay = federate_station_catalog(
            second.stations,
            second_catalog,
            metadata_effective_date="2026-09-19",
        )
        self.assertEqual(replay.new_root_station_count, 0)
        self.assertEqual(replay.stations, second.stations)

        with self.assertRaisesRegex(ValueError, "out-of-order"):
            federate_station_catalog(
                second.stations,
                first_catalog,
                metadata_effective_date="2026-09-17",
            )

    def test_catalog_federation_does_not_use_proximity_or_station_name(self):
        daily = daily_station("USW00000001")
        catalog = parse_station_catalog(
            station_line(
                "USW00000002",
                40.0,
                -75.0,
                10.0,
                "SAME PLACE AND NAME",
            ).encode("ascii")
        )
        result = federate_station_catalog(
            (daily,),
            catalog,
            metadata_effective_date="2026-09-18",
        )
        self.assertEqual(result.shared_station_count, 0)
        self.assertEqual(result.new_root_station_count, 1)
        self.assertEqual(len(result.stations), 2)

    def test_bounded_hourly_metadata_spool_enriches_and_routes_from_disk(self):
        daily = daily_station("USW00094846")
        existing = (
            StationCatalogShard(
                "daily-shard-id",
                "daily-spatial-partition",
                (daily,),
            ),
        )
        station_list = (
            station_line(
                "USW00094846", 41.98, -87.90, 204.0, "CHICAGO OHARE"
            )
            + station_line(
                "CAW00099999", 50.0, -100.0, 300.0, "HOURLY ONLY"
            )
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "ghcnh-station-list.txt"
            spool_path = root / "ghcnh.sqlite"
            source.write_text(station_list, encoding="ascii")
            with build_ghcnh_metadata_spool(
                source,
                spool_path=spool_path,
                max_database_bytes=1024 * 1024,
                sqlite_cache_kib=256,
                max_transaction_rows=1,
            ) as spool:
                enriched = list(
                    spool.enrich_existing_shards(
                        existing,
                        metadata_effective_date="2026-09-18",
                        max_station_records=1,
                    )
                )
                new_roots = list(
                    spool.iter_new_root_shards(
                        metadata_effective_date="2026-09-18",
                        max_station_records=1,
                        max_shard_depth=16,
                    )
                )
                self.assertEqual(len(enriched), 1)
                self.assertEqual(len(new_roots), 1)
                self.assertTrue(
                    all(
                        len(shard.stations) <= 1
                        for shard in (*enriched, *new_roots)
                    )
                )
                shared_id, shared_partition = spool.resolve(
                    "USW00094846"
                )
                self.assertEqual(shared_id, daily.canonical_station_id)
                self.assertEqual(
                    shared_partition, "daily-spatial-partition"
                )
                new_id, new_partition = spool.resolve("CAW00099999")
                self.assertEqual(
                    new_id,
                    canonical_station_id(
                        ProviderAlias(SOURCE_ID, "CAW00099999")
                    ),
                )
                self.assertTrue(new_partition.startswith("ghcnh-"))
                self.assertEqual(spool.summary().matched_station_count, 2)
                self.assertLessEqual(
                    spool_path.stat().st_size, 1024 * 1024
                )

    def test_bounded_hourly_spool_requires_complete_existing_scan(self):
        daily = daily_station("USW00094846")
        station_list = (
            station_line(
                "USW00094846", 41.98, -87.90, 204.0, "CHICAGO OHARE"
            )
            + station_line(
                "CAW00099999", 50.0, -100.0, 300.0, "HOURLY ONLY"
            )
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "ghcnh-station-list.txt"
            source.write_text(station_list, encoding="ascii")
            with build_ghcnh_metadata_spool(
                source,
                spool_path=root / "ghcnh.sqlite",
                max_database_bytes=1024 * 1024,
            ) as spool:
                scan = spool.enrich_existing_shards(
                    (
                        StationCatalogShard(
                            "daily-shard",
                            "daily-cell",
                            (daily,),
                        ),
                    ),
                    metadata_effective_date="2026-09-18",
                    max_station_records=1,
                )
                next(scan)
                with self.assertRaisesRegex(
                    ValueError, "consume enrich_existing_shards"
                ):
                    next(
                        spool.iter_new_root_shards(
                            metadata_effective_date="2026-09-18",
                            max_station_records=1,
                            max_shard_depth=16,
                        )
                    )
                scan.close()

    def test_bounded_hourly_spool_rejects_storage_and_line_overrun(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "ghcnh-station-list.txt"
            spool_path = root / "ghcnh.sqlite"
            source.write_text(
                station_line(
                    "USW00094846", 41.98, -87.90, 204.0, "CHICAGO OHARE"
                ),
                encoding="ascii",
            )
            with self.assertRaisesRegex(ValueError, "too small"):
                build_ghcnh_metadata_spool(
                    source,
                    spool_path=spool_path,
                    max_database_bytes=1024,
                )
            self.assertFalse(spool_path.exists())

            source.write_bytes(b"X" * 5000 + b"\n")
            with self.assertRaisesRegex(ValueError, "exceeds 4096 bytes"):
                build_ghcnh_metadata_spool(
                    source,
                    spool_path=spool_path,
                    max_database_bytes=1024 * 1024,
                    max_metadata_line_bytes=4096,
                )
            self.assertFalse(spool_path.exists())

    def test_in_memory_hourly_lookup_uses_spatial_partition_not_shard_id(self):
        station = daily_station("USW00094846")
        alias = ProviderAlias(SOURCE_ID, "USW00094846")
        linked = FederatedStation(
            canonical_station_id=station.canonical_station_id,
            aliases=(
                *station.aliases,
                AliasBinding(alias, "provider_crosswalk", D1),
            ),
            location_history=station.location_history,
            variable_ids=station.variable_ids,
            provider_location_snapshots=station.provider_location_snapshots,
        )
        lookup = GHCNhAliasShardLookup(
            (
                StationCatalogShard(
                    "physical-shard-id",
                    "semantic-spatial-partition",
                    (linked,),
                ),
            )
        )
        self.assertEqual(
            lookup.resolve("USW00094846")[1],
            "semantic-spatial-partition",
        )

    def test_second_provider_composes_through_provider_neutral_shards_and_manifest(self):
        catalog = parse_station_catalog(
            (
                station_line(
                    "USW00094846", 41.98, -87.90, 204.0, "CHICAGO OHARE"
                )
                + station_line(
                    "CAW00099999", 50.0, -100.0, 300.0, "HOURLY ONLY"
                )
            ).encode("ascii")
        )
        result = federate_station_catalog(
            (daily_station("USW00094846"),),
            catalog,
            metadata_effective_date="2026-09-18",
        )
        shards = adaptive_catalog_shards(
            result.stations,
            metadata_effective_date="2026-09-18",
            max_station_records=1,
        )
        self.assertEqual(sum(len(shard.stations) for shard in shards), 2)
        self.assertEqual(
            sum(
                shard.to_station_catalog("2026-09-18").station_count
                for shard in shards
            ),
            2,
        )

        lookup = GHCNhAliasShardLookup(shards)
        header = "STATION|DATE|temperature|temperature_QC|SOURCE|Remarks\n"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "ghcn-hourly_v1.0.0_d2026_c20260918.tar.gz"
            write_year_archive(
                archive,
                {
                    "GHCNh_USW00094846_2026.psv": (
                        header
                        + "USW00094846|2026-09-18T12:00:00Z|19.4|V020|USAF|raw\n"
                    ),
                    "GHCNh_CAW00099999_2026.psv": (
                        header
                        + "CAW00099999|2026-09-18T12:00:00Z||V030|NOAA|\n"
                    ),
                },
            )
            publication = publish_year_archive(
                archive,
                expected_year=2026,
                lookup=lookup,
                object_root=root / "store",
                max_source_bytes=1024 * 1024,
                max_rows=100,
                max_partitions=10,
                max_archive_members=10,
                max_psv_line_bytes=4096,
            )
            manifest = StationFederationManifest(
                "global-free-stations.v1",
                "1.0.0",
                D1,
                catalog_shard_refs(shards),
                publication.partitions,
            )
            coverage = manifest.coverage_manifest()

        self.assertEqual(
            set(coverage["provider_source_ids"]),
            {"ncei.ghcnd.v3", SOURCE_ID},
        )
        self.assertEqual(
            {item["source_id"] for item in coverage["observation_availability"]},
            {SOURCE_ID},
        )
        self.assertEqual(coverage["unresolved_geographic_partitions"], [])
        self.assertTrue(
            all(
                item["geographic_bounds"] is not None
                for item in coverage["catalog_availability"]
            )
        )
        self.assertTrue(
            all(
                item["geographic_bounds"] is not None
                for item in coverage["observation_availability"]
            )
        )
        self.assertTrue(
            any(
                set(item["provider_source_ids"])
                == {"ncei.ghcnd.v3", SOURCE_ID}
                for item in coverage["catalog_availability"]
            )
        )

    def test_psv_stream_preserves_raw_provider_qc_and_source_fields(self):
        daily = daily_station("USW00094846")
        catalog = parse_station_catalog(
            station_line(
                "USW00094846", 41.98, -87.90, 204.0, "CHICAGO OHARE"
            ).encode("ascii")
        )
        result = federate_station_catalog(
            (daily,),
            catalog,
            metadata_effective_date="2026-09-18",
        )
        shard = StationCatalogShard("cell-1", "cell-1", result.stations)
        lookup = GHCNhAliasShardLookup((shard,))
        writes = []

        class Sink:
            def write(self, key, record):
                writes.append((key, record))

        summary = stream_station_year_psv(
            [
                "STATION|DATE|temperature|temperature_QC|SOURCE|Remarks\n",
                "USW00094846|2026-09-18T12:00:00Z|19.4|V020|USAF|raw remark\n",
                "USW00094846|2026-09-18T13:00:00Z||V030|NOAA|\n",
            ],
            expected_year=2026,
            lookup=lookup,
            sink=Sink(),
            max_rows=100,
            max_partition_keys=10,
        )
        self.assertEqual(summary.row_count, 2)
        self.assertEqual(summary.partition_count, 1)
        self.assertEqual(writes[0][0].source_id, SOURCE_ID)
        fields = dict(writes[0][1].provider_fields)
        self.assertEqual(fields["temperature_QC"], "V020")
        self.assertEqual(fields["SOURCE"], "USAF")
        self.assertEqual(fields["Remarks"], "raw remark")
        self.assertEqual(dict(writes[1][1].provider_fields)["temperature"], "")

    def test_psv_budgets_fail_before_over_budget_sink_side_effect(self):
        result = federate_station_catalog(
            (daily_station("USW00094846"),),
            parse_station_catalog(
                station_line(
                    "USW00094846",
                    41.98,
                    -87.90,
                    204.0,
                    "CHICAGO OHARE",
                ).encode("ascii")
            ),
            metadata_effective_date="2026-09-18",
        )
        lookup = GHCNhAliasShardLookup(
            (
                StationCatalogShard(
                    "cell-1", "cell-1", result.stations
                ),
            )
        )
        writes = []

        class Sink:
            def write(self, key, record):
                writes.append((key, record))

        with self.assertRaisesRegex(ValueError, "max_rows"):
            stream_station_year_psv(
                [
                    "STATION|DATE|temperature\n",
                    "USW00094846|2026-09-18T12:00:00Z|19.4\n",
                    "USW00094846|2026-09-18T13:00:00Z|19.5\n",
                ],
                expected_year=2026,
                lookup=lookup,
                sink=Sink(),
                max_rows=1,
                max_partition_keys=10,
            )
        self.assertEqual(len(writes), 1)

    def test_psv_stream_rejects_unknown_station_and_wrong_year(self):
        station = daily_station("USW00094846")
        hourly_alias = ProviderAlias(SOURCE_ID, "USW00094846")
        linked = FederatedStation(
            canonical_station_id=station.canonical_station_id,
            aliases=(
                *station.aliases,
                AliasBinding(hourly_alias, "provider_crosswalk", D1),
            ),
            location_history=station.location_history,
            variable_ids=station.variable_ids,
            provider_location_snapshots=station.provider_location_snapshots,
        )
        lookup = GHCNhAliasShardLookup(
            (StationCatalogShard("cell-1", "cell-1", (linked,)),)
        )

        class Sink:
            def write(self, key, record):
                raise AssertionError("invalid row must not reach sink")

        with self.assertRaisesRegex(ValueError, "absent"):
            stream_station_year_psv(
                [
                    "STATION|DATE|temperature\n",
                    "USW00000000|2026-09-18T12:00:00Z|19.4\n",
                ],
                expected_year=2026,
                lookup=lookup,
                sink=Sink(),
            max_rows=100,
            max_partition_keys=10,
            )
        with self.assertRaisesRegex(ValueError, "does not match"):
            stream_station_year_psv(
                [
                    "STATION|DATE|temperature\n",
                    "USW00094846|2025-09-18T12:00:00Z|19.4\n",
                ],
                expected_year=2026,
                lookup=lookup,
                sink=Sink(),
            max_rows=100,
            max_partition_keys=10,
            )

    def test_annual_archive_publishes_content_addressed_raw_partitions(self):
        catalog = parse_station_catalog(
            (
                station_line(
                    "USW00094846", 41.98, -87.90, 204.0, "CHICAGO OHARE"
                )
                + station_line(
                    "CAW00099999", 50.0, -100.0, 300.0, "HOURLY ONLY"
                )
            ).encode("ascii")
        )
        result = federate_station_catalog(
            (daily_station("USW00094846"),),
            catalog,
            metadata_effective_date="2026-09-18",
        )
        shared = next(
            station
            for station in result.stations
            if any(
                binding.alias == ProviderAlias(SOURCE_ID, "USW00094846")
                for binding in station.aliases
            )
        )
        hourly_only = next(
            station
            for station in result.stations
            if any(
                binding.alias == ProviderAlias(SOURCE_ID, "CAW00099999")
                for binding in station.aliases
            )
        )
        lookup = GHCNhAliasShardLookup(
            (
                StationCatalogShard("cell-a", "cell-a", (shared,)),
                StationCatalogShard("cell-b", "cell-b", (hourly_only,)),
            )
        )
        header = "STATION|DATE|temperature|temperature_QC|SOURCE|Remarks\n"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "ghcn-hourly_v1.0.0_d2026_c20260918.tar.gz"
            write_year_archive(
                archive,
                {
                    "GHCNh_USW00094846_2026.psv": (
                        header
                        + "USW00094846|2026-09-18T12:00:00Z|19.4|V020|USAF|raw\n"
                    ),
                    "GHCNh_CAW00099999_2026.psv": (
                        header
                        + "CAW00099999|2026-09-18T12:00:00Z||V030|NOAA|\n"
                    ),
                },
            )
            publication = publish_year_archive(
                archive,
                expected_year=2026,
                lookup=lookup,
                object_root=root / "store",
                max_source_bytes=1024 * 1024,
                max_rows=100,
                max_partitions=10,
                max_archive_members=10,
                max_psv_line_bytes=4096,
            )
            self.assertEqual(publication.archive_member_count, 2)
            self.assertEqual(len(publication.partitions), 2)
            expected_source = "sha256:" + hashlib.sha256(
                archive.read_bytes()
            ).hexdigest()
            self.assertEqual(publication.source_revision, expected_source)
            self.assertEqual(
                publication.source_byte_count,
                archive.stat().st_size,
            )
            self.assertTrue(
                all(item.source_revision == expected_source for item in publication.partitions)
            )
            self.assertTrue(
                all(item.media_type == MEDIA_TYPE for item in publication.partitions)
            )
            self.assertTrue(all(item.row_count == 1 for item in publication.partitions))

            for item in publication.partitions:
                object_dir = (
                    root
                    / "store"
                    / "objects"
                    / item.digest.removeprefix("sha256:")
                )
                manifest = json.loads(
                    (object_dir / "manifest.json").read_text(encoding="utf-8")
                )
                self.assertIn("temperature_QC", manifest["provider_field_names"])
                self.assertIn("GHCNH_FIELD:temperature", manifest["variable_ids"])
                self.assertIn("GHCNH_FIELD:temperature_QC", manifest["variable_ids"])
                self.assertIn("GHCNH_FIELD:SOURCE", manifest["variable_ids"])
                row = json.loads(
                    (object_dir / "records.ndjson").read_text(encoding="utf-8")
                )
                fields = dict(row["provider_fields"])
                self.assertIn(fields["SOURCE"], {"USAF", "NOAA"})
                self.assertIn(fields["temperature_QC"], {"V020", "V030"})

            repeated = publish_year_archive(
                archive,
                expected_year=2026,
                lookup=lookup,
                object_root=root / "store",
                max_source_bytes=1024 * 1024,
                max_rows=100,
                max_partitions=10,
                max_archive_members=10,
                max_psv_line_bytes=4096,
            )
            self.assertEqual(repeated.partitions, publication.partitions)

    def test_annual_archive_source_byte_budget_fails_before_staging(self):
        result = federate_station_catalog(
            (daily_station("USW00094846"),),
            parse_station_catalog(
                station_line(
                    "USW00094846",
                    41.98,
                    -87.90,
                    204.0,
                    "CHICAGO OHARE",
                ).encode("ascii")
            ),
            metadata_effective_date="2026-09-18",
        )
        lookup = GHCNhAliasShardLookup(
            (StationCatalogShard("cell-a", "cell-a", result.stations),)
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "fixture.tar.gz"
            write_year_archive(
                archive,
                {
                    "GHCNh_USW00094846_2026.psv": (
                        "STATION|DATE|temperature\n"
                        "USW00094846|2026-09-18T12:00:00Z|19.4\n"
                    )
                },
            )
            store = root / "store"
            with self.assertRaisesRegex(ValueError, "max_source_bytes"):
                publish_year_archive(
                    archive,
                    expected_year=2026,
                    lookup=lookup,
                    object_root=store,
                    max_source_bytes=1,
                    max_rows=100,
                    max_partitions=10,
                    max_archive_members=10,
                    max_psv_line_bytes=4096,
                )
            self.assertFalse(store.exists())

    def test_annual_archive_member_budget_fails_closed(self):
        result = federate_station_catalog(
            (daily_station("USW00094846"),),
            parse_station_catalog(
                (
                    station_line(
                        "USW00094846",
                        41.98,
                        -87.90,
                        204.0,
                        "CHICAGO OHARE",
                    )
                    + station_line(
                        "CAW00099999",
                        50.0,
                        -100.0,
                        300.0,
                        "HOURLY ONLY",
                    )
                ).encode("ascii")
            ),
            metadata_effective_date="2026-09-18",
        )
        shards = adaptive_catalog_shards(
            result.stations,
            metadata_effective_date="2026-09-18",
            max_station_records=1,
        )
        lookup = GHCNhAliasShardLookup(shards)
        header = "STATION|DATE|temperature\n"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "fixture.tar.gz"
            write_year_archive(
                archive,
                {
                    "GHCNh_USW00094846_2026.psv": (
                        header
                        + "USW00094846|2026-09-18T12:00:00Z|19.4\n"
                    ),
                    "GHCNh_CAW00099999_2026.psv": (
                        header
                        + "CAW00099999|2026-09-18T12:00:00Z|18.0\n"
                    ),
                },
            )
            with self.assertRaisesRegex(
                ValueError, "max_archive_members"
            ):
                publish_year_archive(
                    archive,
                    expected_year=2026,
                    lookup=lookup,
                    object_root=root / "store",
                    max_source_bytes=1024 * 1024,
                    max_rows=100,
                    max_partitions=10,
                    max_archive_members=1,
                    max_psv_line_bytes=4096,
                )

    def test_annual_archive_refuses_member_station_mismatch(self):
        catalog = parse_station_catalog(
            station_line(
                "USW00094846", 41.98, -87.90, 204.0, "CHICAGO OHARE"
            ).encode("ascii")
        )
        result = federate_station_catalog(
            (daily_station("USW00094846"),),
            catalog,
            metadata_effective_date="2026-09-18",
        )
        lookup = GHCNhAliasShardLookup(
            (StationCatalogShard("cell-a", "cell-a", result.stations),)
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "fixture.tar.gz"
            write_year_archive(
                archive,
                {
                    "GHCNh_USW00094846_2026.psv": (
                        "STATION|DATE|temperature\n"
                        "USW00000000|2026-09-18T12:00:00Z|19.4\n"
                    )
                },
            )
            with self.assertRaisesRegex(ValueError, "archive member station"):
                publish_year_archive(
                    archive,
                    expected_year=2026,
                    lookup=lookup,
                    object_root=root / "store",
                    max_source_bytes=1024 * 1024,
                max_rows=100,
                max_partitions=10,
                max_archive_members=10,
                max_psv_line_bytes=4096,
                )

    def test_duplicate_station_identifier_fails_closed(self):
        line = station_line(
            "USW00094846", 41.98, -87.90, 204.0, "CHICAGO OHARE"
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_station_catalog((line + line).encode("ascii"))


if __name__ == "__main__":
    unittest.main()
