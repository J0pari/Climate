from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from data.ncei_ghcnd_bulk import (
    adaptive_catalog_shards,
    build_federated_stations,
    by_year_url,
    by_year_urls,
    catalog_shard_refs,
    StationShardLookup,
    capture_http_artifact,
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

    def test_atomic_curl_capture_binds_exact_bytes_and_transport(self):
        payload = b"provider bytes\n"
        commands = []

        def fake_run(command, **kwargs):
            commands.append(tuple(command))
            if command[-1] == "--version":
                return subprocess.CompletedProcess(
                    command, 0, "curl 8.12.1 (fixture) libcurl/8.12.1\n", ""
                )
            target = Path(command[command.index("--output") + 1])
            target.write_bytes(payload)
            return subprocess.CompletedProcess(command, 0, "", "")

        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "artifact.txt"
            curl_path = Path(temp) / "curl"
            curl_path.write_bytes(b"exact curl executable fixture")
            with (
                patch(
                    "data.ncei_ghcnd_bulk._curl_executable",
                    return_value=str(curl_path),
                ),
                patch("data.ncei_ghcnd_bulk.subprocess.run", side_effect=fake_run),
            ):
                artifact = capture_http_artifact(
                    "https://example.test/provider-artifact",
                    destination,
                    timeout_seconds=10.0,
                )

            self.assertEqual(destination.read_bytes(), payload)
            expected = "sha256:" + hashlib.sha256(payload).hexdigest()
            self.assertEqual(artifact.sha256, expected)
            self.assertEqual(artifact.byte_count, len(payload))
            self.assertEqual(artifact.transport, "curl")
            self.assertEqual(
                artifact.transport_version,
                "curl 8.12.1 (fixture) libcurl/8.12.1",
            )
            self.assertEqual(artifact.transport_executable, curl_path.resolve())
            self.assertEqual(
                artifact.transport_executable_sha256,
                "sha256:" + hashlib.sha256(curl_path.read_bytes()).hexdigest(),
            )
            receipt = json.loads(artifact.receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(receipt["sha256"], expected)
            self.assertEqual(receipt["url"], "https://example.test/provider-artifact")

        transfer = next(command for command in commands if "--output" in command)
        self.assertNotIn("--head", transfer)
        self.assertNotIn("--continue-at", transfer)
        self.assertNotIn("If-Range", " ".join(transfer))

    def test_failed_curl_capture_publishes_nothing(self):
        def fake_run(command, **kwargs):
            if command[-1] == "--version":
                return subprocess.CompletedProcess(
                    command, 0, "curl 8.12.1 (fixture) libcurl/8.12.1\n", ""
                )
            target = Path(command[command.index("--output") + 1])
            target.write_bytes(b"incomplete")
            return subprocess.CompletedProcess(command, 22, "", "HTTP 503")

        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "artifact.txt"
            curl_path = Path(temp) / "curl"
            curl_path.write_bytes(b"exact curl executable fixture")
            with (
                patch(
                    "data.ncei_ghcnd_bulk._curl_executable",
                    return_value=str(curl_path),
                ),
                patch("data.ncei_ghcnd_bulk.subprocess.run", side_effect=fake_run),
            ):
                with self.assertRaisesRegex(RuntimeError, "curl artifact capture failed"):
                    capture_http_artifact(
                        "https://example.test/provider-artifact",
                        destination,
                    )
            self.assertFalse(destination.exists())
            self.assertFalse(
                destination.with_name(destination.name + ".artifact.json").exists()
            )
            self.assertFalse(
                destination.with_name(destination.name + ".capture.tmp").exists()
            )

    def test_capture_refuses_implicit_overwrite_before_invoking_transport(self):
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "artifact.txt"
            destination.write_bytes(b"existing")
            with patch("data.ncei_ghcnd_bulk.subprocess.run") as run:
                with self.assertRaisesRegex(ValueError, "refuse implicit reuse"):
                    capture_http_artifact(
                        "https://example.test/provider-artifact",
                        destination,
                    )
                run.assert_not_called()

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
