from __future__ import annotations

import unittest

from src.station_federation import (
    AliasBinding,
    CrossProviderAliasEvidence,
    CatalogShardRef,
    FederatedStation,
    ObservationPartitionRef,
    ProviderAlias,
    ProviderLocationSnapshot,
    StationCatalogShard,
    StationFederationManifest,
    StationLocationEpoch,
    apply_cross_provider_alias_evidence,
    canonical_station_id,
    remove_cross_provider_alias_evidence,
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
        provider_location_snapshots=(
            ProviderLocationSnapshot(
                40.1,
                -74.9,
                11.0,
                "2024-01-01",
                root,
                D1,
            ),
        ),
    )


class StationFederationTests(unittest.TestCase):
    def test_crosswalk_evidence_is_explicit_idempotent_and_reversible(self):
        original = station()
        alias = ProviderAlias("other.provider", "XYZ")
        evidence = CrossProviderAliasEvidence(
            ProviderAlias("ncei.ghcnd.v3", "AAA"),
            alias,
            D3,
        )
        linked = apply_cross_provider_alias_evidence((original,), (evidence,))
        self.assertEqual(
            linked[0].canonical_station_id,
            original.canonical_station_id,
        )
        self.assertEqual(linked[0].aliases[-1].alias, alias)
        self.assertEqual(linked[0].aliases[-1].binding_method, "provider_crosswalk")
        self.assertEqual(linked[0].aliases[-1].evidence_digest, D3)
        self.assertEqual(
            apply_cross_provider_alias_evidence(linked, (evidence,)),
            linked,
        )
        self.assertEqual(
            remove_cross_provider_alias_evidence(linked, D3),
            (original,),
        )

    def test_crosswalk_alias_conflict_fails_closed(self):
        first = station(provider_id="AAA")
        second = station(
            provider_id="BBB",
            alias=ProviderAlias("other.provider", "XYZ"),
        )
        evidence = CrossProviderAliasEvidence(
            ProviderAlias("ncei.ghcnd.v3", "AAA"),
            ProviderAlias("other.provider", "XYZ"),
            D3,
        )
        with self.assertRaisesRegex(ValueError, "owned by another"):
            apply_cross_provider_alias_evidence(
                (first, second),
                (evidence,),
            )

    def test_crosswalk_requires_resolved_root_and_distinct_provider(self):
        with self.assertRaisesRegex(ValueError, "different provider"):
            CrossProviderAliasEvidence(
                ProviderAlias("ncei.ghcnd.v3", "AAA"),
                ProviderAlias("ncei.ghcnd.v3", "BBB"),
                D3,
            )
        evidence = CrossProviderAliasEvidence(
            ProviderAlias("ncei.ghcnd.v3", "MISSING"),
            ProviderAlias("other.provider", "XYZ"),
            D3,
        )
        with self.assertRaisesRegex(ValueError, "root alias does not resolve"):
            apply_cross_provider_alias_evidence((station(),), (evidence,))

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

    def test_provider_location_snapshot_does_not_override_resolved_topology(self):
        alias = ProviderAlias("other.provider", "XYZ")
        base = station(alias=alias)
        item = FederatedStation(
            canonical_station_id=base.canonical_station_id,
            aliases=base.aliases,
            location_history=base.location_history,
            variable_ids=base.variable_ids,
            provider_location_snapshots=(
                base.provider_location_snapshots[0],
                ProviderLocationSnapshot(
                    41.5,
                    -76.5,
                    50.0,
                    "2024-01-01",
                    alias,
                    D3,
                ),
            ),
        )
        self.assertEqual(item.location_at("2024-01-01").latitude_deg, 40.1)
        self.assertEqual(
            next(
                snapshot.latitude_deg
                for snapshot in item.provider_location_snapshots
                if snapshot.source_alias == alias
            ),
            41.5,
        )
        with self.assertRaisesRegex(ValueError, "at most one"):
            FederatedStation(
                canonical_station_id=base.canonical_station_id,
                aliases=base.aliases,
                location_history=base.location_history,
                variable_ids=base.variable_ids,
                provider_location_snapshots=(
                    ProviderLocationSnapshot(
                        41.5, -76.5, 50.0, "2024-01-01", alias, D3
                    ),
                    ProviderLocationSnapshot(
                        41.6, -76.4, 51.0, "2024-01-02", alias, D4
                    ),
                ),
            )

    def test_crosswalk_removal_refuses_bound_provider_snapshot(self):
        alias = ProviderAlias("other.provider", "XYZ")
        evidence = CrossProviderAliasEvidence(
            ProviderAlias("ncei.ghcnd.v3", "AAA"),
            alias,
            D3,
        )
        linked = apply_cross_provider_alias_evidence((station(),), (evidence,))
        item = linked[0]
        with_snapshot = FederatedStation(
            canonical_station_id=item.canonical_station_id,
            aliases=item.aliases,
            location_history=item.location_history,
            variable_ids=item.variable_ids,
            provider_location_snapshots=(
                *item.provider_location_snapshots,
                ProviderLocationSnapshot(
                    41.0, -76.0, 20.0, "2024-01-01", alias, D4
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, "provider location snapshot"):
            remove_cross_provider_alias_evidence((with_snapshot,), D3)

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

    def test_coverage_preserves_provider_variable_and_spatial_gaps(self):
        manifest = StationFederationManifest(
            "global-free-stations.v1", "1.0.0", D1,
            (
                CatalogShardRef(
                    "cell-east", "cell-east", D2, 8,
                    ("ncei.ghcnd.v3",), ("TMAX", "TMIN"),
                ),
                CatalogShardRef(
                    "cell-west", "cell-west", D3, 5,
                    ("ncei.ghcnh.v1",), ("TAVG",),
                ),
            ),
            (
                ObservationPartitionRef(
                    "ncei.ghcnd.v3", "cell-east",
                    "2024-01-01", "2024-12-31",
                    ("TMAX",), D4, "ghcnd-2024",
                    "application/json", 20, 200,
                ),
                ObservationPartitionRef(
                    "ncei.ghcnh.v1", "cell-west",
                    "2022-01-01", "2022-12-31",
                    ("TAVG",), D5, "ghcnh-2022",
                    "application/json", 15, 150,
                ),
            ),
        )
        coverage = manifest.coverage_manifest()

        self.assertEqual(
            coverage["catalog_availability"],
            [
                {
                    "shard_id": "cell-east",
                    "spatial_partition": "cell-east",
                    "digest": D2,
                    "station_count": 8,
                    "provider_source_ids": ["ncei.ghcnd.v3"],
                    "variable_ids": ["TMAX", "TMIN"],
                },
                {
                    "shard_id": "cell-west",
                    "spatial_partition": "cell-west",
                    "digest": D3,
                    "station_count": 5,
                    "provider_source_ids": ["ncei.ghcnh.v1"],
                    "variable_ids": ["TAVG"],
                },
            ],
        )
        self.assertEqual(
            coverage["observation_availability"],
            [
                {
                    "source_id": "ncei.ghcnd.v3",
                    "spatial_partition": "cell-east",
                    "time_start": "2024-01-01",
                    "time_end": "2024-12-31",
                    "variable_ids": ["TMAX"],
                    "digest": D4,
                    "source_revision": "ghcnd-2024",
                    "row_count": 20,
                },
                {
                    "source_id": "ncei.ghcnh.v1",
                    "spatial_partition": "cell-west",
                    "time_start": "2022-01-01",
                    "time_end": "2022-12-31",
                    "variable_ids": ["TAVG"],
                    "digest": D5,
                    "source_revision": "ghcnh-2022",
                    "row_count": 15,
                },
            ],
        )
        self.assertFalse(
            any(
                item["source_id"] == "ncei.ghcnh.v1"
                and "TMIN" in item["variable_ids"]
                for item in coverage["observation_availability"]
            )
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
