from __future__ import annotations

import unittest

from data.ncei_ghcnh_bulk import (
    SOURCE_ID,
    federate_station_catalog,
    parse_station_catalog,
)
from src.station_federation import (
    AliasBinding,
    FederatedStation,
    ProviderAlias,
    StationLocationEpoch,
    canonical_station_id,
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
    )


class GHCNhFederationTests(unittest.TestCase):
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
        self.assertEqual(binding.binding_method, "provider_crosswalk")
        self.assertEqual(binding.evidence_digest, result.crosswalk_evidence_digest)

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

    def test_duplicate_station_identifier_fails_closed(self):
        line = station_line(
            "USW00094846", 41.98, -87.90, 204.0, "CHICAGO OHARE"
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_station_catalog((line + line).encode("ascii"))


if __name__ == "__main__":
    unittest.main()
