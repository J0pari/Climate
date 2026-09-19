from __future__ import annotations

import gzip
import json
from pathlib import Path
import tempfile
import unittest

import pyarrow.parquet as pq

from data.ncei_ghcnd_bulk import (
    GHCNObservationPartitionKey,
    adaptive_catalog_shards,
    build_federated_stations,
    catalog_shard_refs,
    parse_inventory,
    parse_station_catalog,
    StationShardLookup,
    stream_by_year_partitions,
)
from data.ncei_ghcnd_parquet import (
    GHCNParquetPartitionPublisher,
    publish_gzip_by_year,
)
from src.station_federation import StationFederationManifest


REGISTRY_DIGEST = "sha256:" + "a" * 64


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


def write_gzip(path: Path, text: str) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as handle:
            handle.write(text.encode("ascii"))


class GHCNParquetPublicationTests(unittest.TestCase):
    def substrate(self):
        catalog = parse_station_catalog(
            (
                station_line(
                    "USW00000001", 40.0, -75.0, 10.0, "ALPHA", "PA", "12345"
                )
                + station_line("USW00000002", -33.9, 151.2, 5.0, "BETA")
            ).encode("ascii")
        )
        inventory = parse_inventory(
            (
                inventory_line(
                    "USW00000001", 40.0, -75.0, "TMAX", 1900, 2026
                )
                + inventory_line(
                    "USW00000001", 40.0, -75.0, "PRCP", 1900, 2026
                )
                + inventory_line(
                    "USW00000002", -33.9, 151.2, "TMIN", 1950, 2026
                )
            ).encode("ascii")
        )
        stations = build_federated_stations(
            catalog, inventory, metadata_effective_date="2026-09-18"
        )
        shards = adaptive_catalog_shards(
            stations,
            metadata_effective_date="2026-09-18",
            max_station_records=1,
        )
        return StationShardLookup(shards), catalog_shard_refs(shards)

    def test_gzip_stream_publishes_content_addressed_parquet_partitions(self):
        lookup, catalog_refs = self.substrate()
        rows = (
            "USW00000001,20240101,TMAX,123,,,S,0700\n"
            "USW00000002,20240101,TMIN,-9999,,Q,S,\n"
            "USW00000001,20240102,TMAX,125,,,S,0700\n"
            "USW00000001,20240102,PRCP,5,,,S,0700\n"
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "2024.csv.gz"
            store = root / "store"
            write_gzip(source, rows)

            result = publish_gzip_by_year(
                source,
                expected_year=2024,
                lookup=lookup,
                root=store,
                batch_rows=1,
            )
            self.assertEqual(result.routing.row_count, 4)
            self.assertEqual(result.routing.missing_value_count, 1)
            self.assertEqual(result.routing.partition_count, 3)
            self.assertEqual(len(result.partitions), 3)
            self.assertTrue(result.source_revision.startswith("sha256:"))
            self.assertEqual(
                {ref.source_revision for ref in result.partitions},
                {result.source_revision},
            )

            manifest = StationFederationManifest(
                "global-free-stations.v1",
                "1.0.0",
                REGISTRY_DIGEST,
                catalog_refs,
                result.partitions,
            )
            self.assertEqual(
                manifest.coverage_manifest()["observation_partition_count"], 3
            )

            tmin = next(
                ref for ref in result.partitions if ref.variable_ids == ("TMIN",)
            )
            object_dir = store / "objects" / tmin.digest.removeprefix("sha256:")
            payload = json.loads((object_dir / "manifest.json").read_text())
            self.assertEqual(payload["row_count"], 1)
            self.assertEqual(payload["element"], "TMIN")
            self.assertIn("unit normalization is external", payload["value_semantics"])
            fragment = payload["storage"]["fragments"][0]["filename"]
            table = pq.read_table(object_dir / fragment).to_pydict()
            self.assertEqual(table["provider_value"], [None])
            self.assertEqual(table["quality_flag"], ["Q"])

            tmax = next(
                ref for ref in result.partitions if ref.variable_ids == ("TMAX",)
            )
            tmax_dir = store / "objects" / tmax.digest.removeprefix("sha256:")
            tmax_payload = json.loads((tmax_dir / "manifest.json").read_text())
            self.assertEqual(len(tmax_payload["storage"]["fragments"]), 2)

    def test_identical_republication_is_idempotent(self):
        lookup, _ = self.substrate()
        rows = (
            "USW00000001,20240101,TMAX,123,,,S,0700\n"
            "USW00000001,20240102,TMAX,125,,,S,0700\n"
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "2024.csv.gz"
            store = root / "store"
            write_gzip(source, rows)

            first = publish_gzip_by_year(
                source,
                expected_year=2024,
                lookup=lookup,
                root=store,
                batch_rows=1,
            )
            second = publish_gzip_by_year(
                source,
                expected_year=2024,
                lookup=lookup,
                root=store,
                batch_rows=1,
            )
            self.assertEqual(first.partitions, second.partitions)
            self.assertEqual(first.source_revision, second.source_revision)
            objects = [
                path for path in (store / "objects").iterdir() if path.is_dir()
            ]
            self.assertEqual(len(objects), len(first.partitions))

    def test_interrupted_publication_resumes_from_flushed_fragment_boundary(self):
        lookup, _ = self.substrate()
        rows = (
            "USW00000001,20240101,TMAX,123,,,S,0700\n"
            "USW00000001,20240102,TMAX,125,,,S,0700\n"
            "USW00000001,20240103,TMAX,127,,,S,0700\n"
            "USW00000001,20240104,TMAX,129,,,S,0700\n"
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "2024.csv.gz"
            store = root / "store"
            clean_store = root / "clean"
            write_gzip(source, rows)
            source_revision = "sha256:" + __import__("hashlib").sha256(
                source.read_bytes()
            ).hexdigest()

            partial = GHCNParquetPartitionPublisher(
                store,
                source_revision=source_revision,
                batch_rows=2,
            )
            stream_by_year_partitions(
                rows.splitlines(keepends=True)[:3],
                expected_year=2024,
                lookup=lookup,
                sink=partial,
            )
            partial.abort()
            checkpoints = list((store / ".staging").glob("*/checkpoint.json"))
            self.assertEqual(len(checkpoints), 1)
            checkpoint = json.loads(checkpoints[0].read_text())
            self.assertEqual(checkpoint["committed_rows"], 2)

            resumed = publish_gzip_by_year(
                source,
                expected_year=2024,
                lookup=lookup,
                root=store,
                batch_rows=2,
            )
            clean = publish_gzip_by_year(
                source,
                expected_year=2024,
                lookup=lookup,
                root=clean_store,
                batch_rows=2,
            )
            self.assertEqual(resumed.routing.row_count, 4)
            self.assertEqual(resumed.partitions, clean.partitions)
            self.assertEqual(list((store / ".staging").iterdir()), [])

    def test_resume_refuses_changed_resolved_prefix_identity(self):
        lookup, _ = self.substrate()
        rows = (
            "USW00000001,20240101,TMAX,123,,,S,0700\n"
            "USW00000001,20240102,TMAX,125,,,S,0700\n"
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "2024.csv.gz"
            write_gzip(source, rows)
            source_revision = "sha256:" + __import__("hashlib").sha256(
                source.read_bytes()
            ).hexdigest()
            partial = GHCNParquetPartitionPublisher(
                root / "store",
                source_revision=source_revision,
                batch_rows=2,
            )
            stream_by_year_partitions(
                rows.splitlines(keepends=True),
                expected_year=2024,
                lookup=lookup,
                sink=partial,
            )
            partial.abort()

            altered_lookup, _ = self.substrate()
            original_resolve = altered_lookup.resolve

            class AlteredLookup:
                source_id = altered_lookup.source_id

                def resolve(self, station_id):
                    canonical, shard = original_resolve(station_id)
                    return canonical + ".changed", shard

            with self.assertRaisesRegex(ValueError, "committed input prefix"):
                publish_gzip_by_year(
                    source,
                    expected_year=2024,
                    lookup=AlteredLookup(),
                    root=root / "store",
                    batch_rows=2,
                )

    def test_provider_revision_requires_explicit_supersession_in_manifest(self):
        lookup, catalog_refs = self.substrate()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "2024.csv.gz"
            store = root / "store"
            write_gzip(
                source, "USW00000001,20240101,TMAX,123,,,S,0700\n"
            )
            first = publish_gzip_by_year(
                source,
                expected_year=2024,
                lookup=lookup,
                root=store,
                batch_rows=8,
            )
            old = first.partitions[0]
            key = GHCNObservationPartitionKey(
                old.source_id,
                old.spatial_partition,
                2024,
                old.variable_ids[0],
            )

            write_gzip(
                source, "USW00000001,20240101,TMAX,124,,,S,0700\n"
            )
            second = publish_gzip_by_year(
                source,
                expected_year=2024,
                lookup=lookup,
                root=store,
                batch_rows=8,
                supersedes_by_key={key: (old.digest,)},
            )
            new = second.partitions[0]
            self.assertNotEqual(old.digest, new.digest)
            self.assertEqual(new.supersedes, (old.digest,))

            manifest = StationFederationManifest(
                "global-free-stations.v1",
                "1.0.0",
                REGISTRY_DIGEST,
                catalog_refs,
                (old, new),
            )
            self.assertEqual(
                manifest.active_observation_partitions()[0].digest,
                new.digest,
            )


if __name__ == "__main__":
    unittest.main()
