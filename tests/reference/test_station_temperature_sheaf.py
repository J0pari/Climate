from __future__ import annotations

import hashlib
import json
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from reference.ncei_ghcnd import build_request_url, parse_daily_summaries
from reference.station_temperature_sheaf import (
    StationSnapshot,
    build_station_temperature_sheaf,
    extract_station_snapshot,
)


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = (
    ROOT
    / "fixtures"
    / "data_authorities"
    / "ncei-ghcnd-NYC-three-station-2024-01-01-to-2024-01-03.json"
)
STATIONS = ("USW00094728", "USW00014732", "USW00094789")
SOURCE_DIGEST = "sha256:45f8e9a08d61939ca639e599250bdf4fbc8a41986f52d7c7e023f1a947c8bd42"
COVERAGE_RADIUS_KM = 20.0
PROJECTED_CRS = "EPSG:32618"


def source_payload():
    raw = FIXTURE.read_bytes()
    assert "sha256:" + hashlib.sha256(raw).hexdigest() == SOURCE_DIGEST
    request_url = build_request_url(STATIONS, "2024-01-01", "2024-01-03")
    return parse_daily_summaries(raw, request_url=request_url)


def source_snapshot(date: str = "2024-01-02") -> StationSnapshot:
    return extract_station_snapshot(
        source_payload(),
        date=date,
        station_ids=STATIONS,
        variables=("TMAX", "TMIN"),
        accept_quality_flagged=False,
    )


def derived_snapshot(
    snapshot: StationSnapshot,
    *,
    station_id: str,
    variable: str,
    value: float | None,
    transform_id: str,
) -> StationSnapshot:
    stations = dict(snapshot.stations)
    station = stations[station_id]
    values = dict(station.values)
    values[variable] = value
    stations[station_id] = replace(station, values=values)
    return replace(
        snapshot,
        stations=stations,
        lineage=(*snapshot.lineage, transform_id),
    )


class StationTemperatureSheafTests(unittest.TestCase):
    def test_provider_fixture_identity_and_station_coordinates(self):
        payload = source_payload()
        self.assertEqual(payload.sha256, SOURCE_DIGEST)
        self.assertEqual(len(payload.records), 9)

        snapshot = source_snapshot()
        self.assertEqual(snapshot.source_id, "ncei.ghcnd.v3")
        self.assertEqual(snapshot.source_artifact_sha256, SOURCE_DIGEST)
        self.assertEqual(
            (snapshot.stations["USW00094728"].latitude_deg,
             snapshot.stations["USW00094728"].longitude_deg),
            (40.77898, -73.96925),
        )
        self.assertEqual(
            (snapshot.stations["USW00014732"].latitude_deg,
             snapshot.stations["USW00014732"].longitude_deg),
            (40.77945, -73.88027),
        )
        self.assertEqual(
            (snapshot.stations["USW00094789"].latitude_deg,
             snapshot.stations["USW00094789"].longitude_deg),
            (40.63915, -73.7639),
        )

    def test_geographic_cover_builds_triangle_and_functorial_real_cochain(self):
        sheaf = build_station_temperature_sheaf(
            source_snapshot(),
            coverage_radius_km=COVERAGE_RADIUS_KM,
            projected_crs=PROJECTED_CRS,
        )
        self.assertEqual(sheaf.base.dimension, 2)
        self.assertIn(tuple(sorted(STATIONS)), sheaf.base.simplices(2))
        sheaf.verify_functoriality()
        self.assertTrue(sheaf.d_squared_is_zero(0))
        np.testing.assert_array_equal(
            sheaf.compatibility_residual(),
            sheaf.graph_residual_baseline(),
        )
        self.assertEqual(sheaf.provider_quality_flag_count(), 0)

    def test_injected_fault_is_explicit_and_has_no_hidden_sheaf_advantage(self):
        clean = source_snapshot(date="2024-01-01")
        clean_sheaf = build_station_temperature_sheaf(
            clean,
            coverage_radius_km=COVERAGE_RADIUS_KM,
            projected_crs=PROJECTED_CRS,
        )
        fault = derived_snapshot(
            clean,
            station_id="USW00094728",
            variable="TMAX",
            value=18.3,
            transform_id="synthetic_fault:TMAX:+10C:USW00094728",
        )
        fault_sheaf = build_station_temperature_sheaf(
            fault,
            coverage_radius_km=COVERAGE_RADIUS_KM,
            projected_crs=PROJECTED_CRS,
        )

        self.assertEqual(clean_sheaf.provider_quality_flag_count(), 0)
        self.assertEqual(fault_sheaf.provider_quality_flag_count(), 0)
        self.assertGreater(fault_sheaf.compatibility_energy(), clean_sheaf.compatibility_energy())
        np.testing.assert_array_equal(
            fault_sheaf.compatibility_residual(),
            fault_sheaf.graph_residual_baseline(),
        )
        self.assertIn("synthetic_fault", fault.lineage[-1])

    def test_withheld_value_changes_stalks_without_imputation(self):
        source = source_snapshot()
        withheld = derived_snapshot(
            source,
            station_id="USW00094789",
            variable="TMAX",
            value=None,
            transform_id="withheld:TMAX:USW00094789",
        )
        sheaf = build_station_temperature_sheaf(
            withheld,
            coverage_radius_km=COVERAGE_RADIUS_KM,
            projected_crs=PROJECTED_CRS,
        )
        self.assertEqual(
            sheaf.stalk_bases[("USW00094789",)],
            ("TMIN",),
        )
        for simplex, basis in sheaf.stalk_bases.items():
            if "USW00094789" in simplex:
                self.assertNotIn("TMAX", basis)
        self.assertTrue(sheaf.d_squared_is_zero(0))

        source_sheaf = build_station_temperature_sheaf(
            source,
            coverage_radius_km=COVERAGE_RADIUS_KM,
            projected_crs=PROJECTED_CRS,
        )
        prediction = source_sheaf.nearest_neighbor_prediction("USW00094789", "TMAX")
        actual = source.stations["USW00094789"].values["TMAX"]
        self.assertIsNotNone(actual)
        self.assertGreaterEqual(abs(prediction - float(actual)), 0.0)
        self.assertIsNone(withheld.stations["USW00094789"].values["TMAX"])

    def test_quality_flag_policy_is_explicit(self):
        payload = source_payload()
        records = [dict(record) for record in payload.records]
        target = next(
            record
            for record in records
            if record["STATION"] == "USW00094728" and record["DATE"] == "2024-01-02"
        )
        target["TMAX_ATTRIBUTES"] = ",O,W"
        synthetic_bytes = json.dumps(records, separators=(",", ":")).encode("utf-8")
        altered = replace(
            payload,
            sha256="sha256:" + hashlib.sha256(synthetic_bytes).hexdigest(),
            byte_count=len(synthetic_bytes),
            records=tuple(records),
        )

        rejected = extract_station_snapshot(
            altered,
            date="2024-01-02",
            station_ids=STATIONS,
            variables=("TMAX", "TMIN"),
            accept_quality_flagged=False,
        )
        accepted = extract_station_snapshot(
            altered,
            date="2024-01-02",
            station_ids=STATIONS,
            variables=("TMAX", "TMIN"),
            accept_quality_flagged=True,
        )
        self.assertIsNone(rejected.stations["USW00094728"].values["TMAX"])
        self.assertEqual(accepted.stations["USW00094728"].values["TMAX"], 5.6)
        self.assertEqual(rejected.stations["USW00094728"].quality_flags["TMAX"], "O")

    def test_missing_station_record_fails_closed(self):
        payload = source_payload()
        records = tuple(
            record for record in payload.records if record["STATION"] != "USW00094789"
        )
        incomplete = replace(payload, records=records)
        with self.assertRaisesRegex(ValueError, "has no 2024-01-02 record"):
            extract_station_snapshot(
                incomplete,
                date="2024-01-02",
                station_ids=STATIONS,
                variables=("TMAX", "TMIN"),
                accept_quality_flagged=False,
            )


if __name__ == "__main__":
    unittest.main()
