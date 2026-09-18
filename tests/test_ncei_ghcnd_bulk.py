from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

from data.ncei_ghcnd_bulk import (
    HTTPArtifactIdentity,
    adaptive_catalog_shards,
    build_federated_stations,
    by_year_url,
    by_year_urls,
    catalog_shard_refs,
    download_by_year_artifact,
    download_resumable_http_artifact,
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


class FakeRangeTransport:
    def __init__(
        self,
        payload: bytes,
        *,
        etag: str = '"fixture-v1"',
        last_modified: str = "Fri, 18 Sep 2026 00:00:00 GMT",
        accept_ranges: bool = True,
        fail_after_bytes: int | None = None,
    ) -> None:
        self.payload = payload
        self.etag = etag
        self.last_modified = last_modified
        self.accept_ranges = accept_ranges
        self.fail_after_bytes = fail_after_bytes
        self.starts: list[int] = []
        self.urls: list[str] = []

    def inspect(self, url: str, *, timeout_seconds: float) -> HTTPArtifactIdentity:
        self.urls.append(url)
        return HTTPArtifactIdentity(
            url=url,
            content_length=len(self.payload),
            etag=self.etag,
            last_modified=self.last_modified,
            accept_ranges=self.accept_ranges,
        )

    def iter_bytes(
        self,
        url: str,
        *,
        start: int,
        identity: HTTPArtifactIdentity,
        timeout_seconds: float,
        chunk_bytes: int,
    ):
        self.starts.append(start)
        sent = 0
        for offset in range(start, len(self.payload), chunk_bytes):
            chunk = self.payload[offset:offset + chunk_bytes]
            yield chunk
            sent += len(chunk)
            if (
                self.fail_after_bytes is not None
                and start + sent >= self.fail_after_bytes
            ):
                raise RuntimeError("simulated transport interruption")


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

    def test_resumable_download_continues_from_committed_prefix(self):
        payload = b"0123456789abcdef"
        transport = FakeRangeTransport(payload, fail_after_bytes=4)
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "2024.csv.gz"
            with self.assertRaisesRegex(RuntimeError, "simulated"):
                download_resumable_http_artifact(
                    "https://example.test/2024.csv.gz",
                    destination,
                    transport=transport,
                    chunk_bytes=4,
                )
            state = destination.with_name(destination.name + ".resume.json")
            self.assertTrue(state.is_file())
            self.assertEqual(
                __import__("json").loads(state.read_text())["committed_bytes"],
                4,
            )

            transport.fail_after_bytes = None
            result = download_resumable_http_artifact(
                "https://example.test/2024.csv.gz",
                destination,
                transport=transport,
                chunk_bytes=4,
            )
            self.assertEqual(destination.read_bytes(), payload)
            self.assertEqual(result.byte_count, len(payload))
            self.assertEqual(
                result.sha256,
                "sha256:" + hashlib.sha256(payload).hexdigest(),
            )
            self.assertEqual(transport.starts, [0, 4])
            self.assertFalse(state.exists())

            starts_before = list(transport.starts)
            repeated = download_resumable_http_artifact(
                "https://example.test/2024.csv.gz",
                destination,
                transport=transport,
                chunk_bytes=4,
            )
            self.assertEqual(repeated.sha256, result.sha256)
            self.assertEqual(transport.starts, starts_before)

    def test_resumable_download_refuses_remote_revision_change(self):
        transport = FakeRangeTransport(b"abcdefgh", fail_after_bytes=4)
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "artifact.bin"
            with self.assertRaises(RuntimeError):
                download_resumable_http_artifact(
                    "https://example.test/artifact.bin",
                    destination,
                    transport=transport,
                    chunk_bytes=4,
                )
            transport.fail_after_bytes = None
            transport.etag = '"fixture-v2"'
            with self.assertRaisesRegex(ValueError, "identity changed"):
                download_resumable_http_artifact(
                    "https://example.test/artifact.bin",
                    destination,
                    transport=transport,
                    chunk_bytes=4,
                )
            self.assertEqual(transport.starts, [0])

    def test_by_year_downloader_uses_provider_bulk_url_and_requires_ranges(self):
        transport = FakeRangeTransport(b"payload", accept_ranges=False)
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "byte-range"):
                download_by_year_artifact(
                    2024,
                    Path(temp) / "2024.csv.gz",
                    transport=transport,
                )
        self.assertEqual(
            transport.urls,
            ["https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_year/2024.csv.gz"],
        )

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
        )
        self.assertEqual(summary.row_count, 3)
        self.assertEqual(summary.missing_value_count, 1)
        self.assertEqual(summary.partition_count, 3)
        self.assertIsNone(writes[1][1].value)
        self.assertEqual(writes[1][1].quality_flag, "Q")
        self.assertEqual(writes[0][0].source_id, "ncei.ghcnd.v3")

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
            )


if __name__ == "__main__":
    unittest.main()
