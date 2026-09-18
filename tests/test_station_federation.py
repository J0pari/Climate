from __future__ import annotations

import unittest

from src.station_federation import (
    AliasBinding,
    CatalogShardRef,
    FederatedStation,
    ObservationPartitionRef,
    ProviderAlias,
    StationCatalogShard,
    StationFederationManifest,
    StationLocationEpoch,
    canonical_station_id,
)


D1 = "sha256:" + "1" * 64
D2 = "sha256:" + "2" * 64
D3 = "sha256:" + "3" * 64
D4 = "sha256:" + "4" * 64
D5 = "sha256:" + "5" * 64
D6 = "sha256:" + "6" * 64


def station(provider="ncei.ghcnd.v3", provider_id="AAA", *, alias=None):
    root = ProviderAlias(provider, provider_id)
    aliases = [AliasBinding(root, "root", D1)]
    if alias is not None:
        aliases.append(AliasBinding(alias, "provider_crosswalk", D2))
    return FederatedStation(
        canonical_station_id=canonical_station_id(root),
        aliases=tuple(aliases),
        location_history=(
            StationLocationEpoch(
                40.0, -75.0, 10.0, None, "2000-01-01", root, D1
            ),
            StationLocationEpoch(
                40.1, -74.9, 11.0, "2000-01-01", None, root, D2
            ),
        ),
        variable_ids=("TMAX", "TMIN"),
    )


class StationFederationTests(unittest.TestCase):
    def test_alias_addition_does_not_mutate_canonical_station_identity(self):
        root_only = station()
        linked = station(alias=ProviderAlias("other.provider", "XYZ"))
        self.assertEqual(root_only.canonical_station_id, linked.canonical_station_id)
        self.assertEqual(len(linked.aliases), 2)

    def test_proximity_never_merges_station_identity(self):
        first = station(provider_id="AAA")
        second = station(provider_id="BBB")
        self.assertNotEqual(first.canonical_station_id, second.canonical_station_id)
        shard = StationCatalogShard("cell-1", "cell-1", (first, second))
        self.assertEqual(len(shard.stations), 2)

    def test_location_history_selects_date_without_overlapping_epochs(self):
        item = station()
        self.assertEqual(item.location_at("1999-12-31").latitude_deg, 40.0)
        self.assertEqual(item.location_at("2000-01-01").latitude_deg, 40.1)

    def test_catalog_shard_projects_to_same_canonical_station_substrate(self):
        shard = StationCatalogShard(
            "cell-1", "cell-1", (station(provider_id="B"), station(provider_id="A"))
        )
        catalog = shard.to_station_catalog("2024-01-01")
        self.assertEqual(catalog.station_count, 2)
        self.assertEqual(
            catalog.station_ids,
            tuple(sorted(catalog.station_ids)),
        )
        self.assertEqual(shard.station_variable_ids("2024-01-01"),
                         (("TMAX", "TMIN"), ("TMAX", "TMIN")))

    def test_partition_revision_requires_explicit_supersession(self):
        old = ObservationPartitionRef(
            "ncei.ghcnd.v3", "cell-1", "2024-01-01", "2024-12-31",
            ("TMAX",), D3, "provider-rev-1", "application/json", 100, 1000,
        )
        replacement = ObservationPartitionRef(
            "ncei.ghcnd.v3", "cell-1", "2024-01-01", "2024-12-31",
            ("TMAX",), D4, "provider-rev-2", "application/json", 101, 1010,
            supersedes=(D3,),
        )
        manifest = StationFederationManifest(
            "global-free-stations.v1", "1.0.0", D1,
            (CatalogShardRef("cell-1", "cell-1", D2, 2,
                             ("ncei.ghcnd.v3",), ("TMAX", "TMIN")),),
            (old, replacement),
        )
        self.assertEqual(
            [item.digest for item in manifest.active_observation_partitions()],
            [D4],
        )
        with self.assertRaisesRegex(ValueError, "multiple active revisions"):
            StationFederationManifest(
                "global-free-stations.v1", "1.0.0", D1,
                manifest.catalog_shards,
                (old, ObservationPartitionRef(
                    "ncei.ghcnd.v3", "cell-1", "2024-01-01", "2024-12-31",
                    ("TMAX",), D5, "provider-rev-2", "application/json", 101, 1010,
                )),
            )

    def test_tombstone_is_explicit_and_coverage_uses_active_data_only(self):
        data = ObservationPartitionRef(
            "ncei.ghcnd.v3", "cell-1", "2023-01-01", "2023-12-31",
            ("TMIN",), D3, "rev-1", "application/json", 10, 100,
        )
        tombstone = ObservationPartitionRef(
            "ncei.ghcnd.v3", "cell-1", "2023-01-01", "2023-12-31",
            ("TMIN",), D4, "rev-2", "application/json", 0, 20,
            kind="tombstone", supersedes=(D3,),
        )
        live = ObservationPartitionRef(
            "ncei.ghcnd.v3", "cell-1", "2024-01-01", "2024-12-31",
            ("TMAX",), D5, "rev-1", "application/json", 20, 200,
        )
        manifest = StationFederationManifest(
            "global-free-stations.v1", "1.0.0", D1,
            (CatalogShardRef("cell-1", "cell-1", D2, 2,
                             ("ncei.ghcnd.v3",), ("TMAX", "TMIN")),),
            (data, tombstone, live),
        )
        coverage = manifest.coverage_manifest()
        self.assertEqual(coverage["observation_partition_count"], 1)
        self.assertEqual(coverage["variables"], ["TMAX", "TMIN"])
        self.assertEqual(coverage["time_start"], "2024-01-01")
        self.assertEqual(
            [item.kind for item in manifest.active_observation_partitions(
                include_tombstones=True)],
            ["tombstone", "data"],
        )

    def test_unknown_superseded_digest_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "unknown digest"):
            StationFederationManifest(
                "global-free-stations.v1", "1.0.0", D1,
                (CatalogShardRef("cell-1", "cell-1", D2, 1,
                                 ("ncei.ghcnd.v3",), ("TMAX",),
                                 supersedes=(D6,)),),
                (),
            )


if __name__ == "__main__":
    unittest.main()
