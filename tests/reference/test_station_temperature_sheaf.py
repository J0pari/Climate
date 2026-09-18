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
    extract_station_snapshot,
    to_station_catalog,
    to_station_section,
)
from src.station_sheaf import (
    GlobalRadiusIndex,
    SparseRipsComplex,
    StationIdentitySheaf,
    concatenate_edge_chunks,
)


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "fixtures" / "data_authorities" / "ncei-ghcnd-NYC-three-station-2024-01-01-to-2024-01-03.json"
STATIONS = ("USW00094728", "USW00014732", "USW00094789")
SOURCE_DIGEST = "sha256:45f8e9a08d61939ca639e599250bdf4fbc8a41986f52d7c7e023f1a947c8bd42"
RIPS_RADIUS_M = 25_000.0


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


def derived_snapshot(snapshot, *, station_id, variable, value, transform_id):
    stations = dict(snapshot.stations)
    station = stations[station_id]
    values = dict(station.values)
    values[variable] = value
    stations[station_id] = replace(station, values=values)
    return replace(snapshot, stations=stations, lineage=(*snapshot.lineage, transform_id))


def canonical_fixture(snapshot):
    catalog = to_station_catalog(snapshot)
    section = to_station_section(snapshot)
    chunks = list(GlobalRadiusIndex(catalog).iter_edges(max_distance_m=RIPS_RADIUS_M))
    edges = concatenate_edge_chunks(chunks)
    complex_ = SparseRipsComplex.from_edge_chunks(catalog.station_ids, chunks)
    sheaf = StationIdentitySheaf(
        station_count=catalog.station_count,
        variables=section.variables,
    )
    return catalog, section, edges, complex_, sheaf


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

    def test_provider_fixture_uses_canonical_global_sparse_path(self):
        _, section, edges, complex_, sheaf = canonical_fixture(source_snapshot())
        self.assertEqual(len(edges), 3)
        self.assertEqual(list(complex_.iter_simplices(2)), [(0, 1, 2)])
        d0 = sheaf.coboundary_0(edges)
        self.assertEqual(d0.shape, (6, 6))
        self.assertEqual(d0.nnz, 12)
        residual = sheaf.residual(edges, section)
        self.assertEqual(residual.structural_row_count, 6)
        np.testing.assert_array_equal(residual.row_indices, np.arange(6))
        np.testing.assert_allclose(
            residual.values,
            np.asarray(d0 @ section.values.reshape(-1)).reshape(-1),
        )

    def test_injected_fault_is_explicit_on_canonical_path(self):
        clean = source_snapshot(date="2024-01-01")
        fault = derived_snapshot(
            clean,
            station_id="USW00094728",
            variable="TMAX",
            value=18.3,
            transform_id="synthetic_fault:TMAX:+10C:USW00094728",
        )
        _, clean_section, clean_edges, _, clean_sheaf = canonical_fixture(clean)
        _, fault_section, fault_edges, _, fault_sheaf = canonical_fixture(fault)
        self.assertGreater(
            fault_sheaf.residual(fault_edges, fault_section).energy(),
            clean_sheaf.residual(clean_edges, clean_section).energy(),
        )
        self.assertIn("synthetic_fault", fault.lineage[-1])

    def test_withheld_value_masks_rows_without_imputation_or_topology_change(self):
        source = source_snapshot()
        withheld = derived_snapshot(
            source,
            station_id="USW00094789",
            variable="TMAX",
            value=None,
            transform_id="withheld:TMAX:USW00094789",
        )
        _, source_section, source_edges, source_complex, source_sheaf = canonical_fixture(source)
        _, withheld_section, withheld_edges, withheld_complex, withheld_sheaf = canonical_fixture(withheld)
        self.assertEqual(source_complex.edge_count, withheld_complex.edge_count)
        np.testing.assert_array_equal(source_edges.tail, withheld_edges.tail)
        np.testing.assert_array_equal(source_edges.head, withheld_edges.head)
        self.assertEqual(source_sheaf.residual(source_edges, source_section).values.size, 6)
        self.assertEqual(withheld_sheaf.residual(withheld_edges, withheld_section).values.size, 4)
        self.assertFalse(withheld_section.observed[2, 0])

    def test_quality_flag_policy_is_explicit(self):
        payload = source_payload()
        records = [dict(record) for record in payload.records]
        target = next(
            record for record in records
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
        incomplete = replace(
            payload,
            records=tuple(
                record for record in payload.records
                if record["STATION"] != "USW00094789"
            ),
        )
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
