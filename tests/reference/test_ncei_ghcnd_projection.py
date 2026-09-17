from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np

from reference.ncei_ghcnd import build_request_url, parse_daily_summaries
from reference.ncei_ghcnd_projection import (
    CF_CONVENTIONS,
    PROJECTION_ID,
    netcdf_bytes,
    open_netcdf_bytes,
    project_single_station_temperature,
)


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = (
    ROOT
    / "fixtures"
    / "data_authorities"
    / "ncei-ghcnd-USW00094728-2024-01-01-to-2024-01-03.json"
)
SOURCE_DIGEST = "sha256:06e00068f7de97d76b8bc523d5acb22ec7b21631adfec0e0c1a284faf1ac1a46"


def source_payload():
    request_url = build_request_url(
        ["USW00094728"], "2024-01-01", "2024-01-03"
    )
    return parse_daily_summaries(FIXTURE.read_bytes(), request_url=request_url)


class NceiGhcndProjectionTests(unittest.TestCase):
    def test_projection_is_cf_aware_and_provenance_bound(self) -> None:
        dataset = project_single_station_temperature(source_payload())

        self.assertEqual(dataset.attrs["Conventions"], CF_CONVENTIONS)
        self.assertEqual(dataset.attrs["featureType"], "timeSeries")
        self.assertEqual(dataset.attrs["projection_id"], PROJECTION_ID)
        self.assertTrue(dataset.attrs["history"])
        self.assertEqual(dataset.attrs["source_id"], "ncei.ghcnd.v3")
        self.assertEqual(dataset.attrs["source_artifact_sha256"], SOURCE_DIGEST)
        self.assertEqual(dataset.attrs["source_artifact_bytes"], 2324)
        self.assertEqual(dataset.attrs["source_record_count"], 3)
        self.assertEqual(dataset.sizes["time"], 3)

        self.assertEqual(dataset["station_id"].item(), "USW00094728")
        self.assertEqual(dataset["station_id"].attrs["cf_role"], "timeseries_id")
        self.assertAlmostEqual(dataset["latitude"].item(), 40.77898)
        self.assertAlmostEqual(dataset["longitude"].item(), -73.96925)
        self.assertAlmostEqual(dataset["station_altitude"].item(), 42.7)
        self.assertEqual(dataset["latitude"].attrs["units"], "degrees_north")
        self.assertEqual(dataset["longitude"].attrs["units"], "degrees_east")
        self.assertEqual(dataset["station_altitude"].attrs["units"], "m")
        self.assertEqual(dataset["time"].attrs["units_metadata"], "leap_seconds: unknown")

        np.testing.assert_allclose(
            dataset["daily_maximum_air_temperature"].values,
            np.asarray([8.3, 5.6, 6.1]),
        )
        np.testing.assert_allclose(
            dataset["daily_minimum_air_temperature"].values,
            np.asarray([1.7, -1.6, 1.1]),
        )
        self.assertEqual(
            dataset["daily_maximum_air_temperature"].attrs["standard_name"],
            "air_temperature",
        )
        self.assertEqual(
            dataset["daily_maximum_air_temperature"].attrs["units"],
            "degree_Celsius",
        )
        self.assertEqual(
            dataset["daily_maximum_air_temperature"].attrs["units_metadata"],
            "temperature: on_scale",
        )
        self.assertEqual(
            dataset["daily_minimum_air_temperature"].attrs["units_metadata"],
            "temperature: on_scale",
        )
        self.assertEqual(
            dataset["daily_maximum_air_temperature"].attrs["ancillary_variables"],
            "daily_maximum_air_temperature_attributes",
        )
        self.assertEqual(
            dataset["daily_maximum_air_temperature_attributes"].values.tolist(),
            [",,W", ",,W", ",,W"],
        )

    def test_missing_observation_remains_missing(self) -> None:
        payload = source_payload()
        records = [dict(record) for record in payload.records]
        records[1].pop("TMAX")
        records[1].pop("TMAX_ATTRIBUTES", None)
        modified = replace(payload, records=tuple(records))

        dataset = project_single_station_temperature(modified)
        values = dataset["daily_maximum_air_temperature"].values
        self.assertEqual(values[0], 8.3)
        self.assertTrue(np.isnan(values[1]))
        self.assertEqual(values[2], 6.1)
        self.assertEqual(
            dataset["daily_maximum_air_temperature_attributes"].values[1], ""
        )
        self.assertIn("no climatology", dataset.attrs["missing_data_policy"])

    def test_inconsistent_station_location_is_rejected(self) -> None:
        payload = source_payload()
        records = [dict(record) for record in payload.records]
        records[2]["LATITUDE"] = "41.0"
        modified = replace(payload, records=tuple(records))
        with self.assertRaises(ValueError):
            project_single_station_temperature(modified)

    def test_netcdf_roundtrip_preserves_projection_identity(self) -> None:
        dataset = project_single_station_temperature(source_payload())
        encoded_once = netcdf_bytes(dataset)
        encoded_twice = netcdf_bytes(dataset)
        self.assertEqual(encoded_once, encoded_twice)
        import hashlib
        self.assertEqual(len(encoded_once), PROJECTION_BYTES)
        self.assertEqual(
            "sha256:" + hashlib.sha256(encoded_once).hexdigest(),
            PROJECTION_DIGEST,
        )

        with open_netcdf_bytes(encoded_once) as reopened:
            self.assertEqual(reopened.attrs["projection_id"], PROJECTION_ID)
            self.assertEqual(reopened.attrs["source_artifact_sha256"], SOURCE_DIGEST)
            self.assertEqual(reopened.attrs["featureType"], "timeSeries")
            self.assertTrue(reopened.attrs["history"])
            np.testing.assert_allclose(
                reopened["daily_maximum_air_temperature"].values,
                np.asarray([8.3, 5.6, 6.1]),
            )
            self.assertEqual(
                reopened["daily_maximum_air_temperature"].attrs["units_metadata"],
                "temperature: on_scale",
            )
            self.assertEqual(reopened["time"].attrs["units_metadata"], "leap_seconds: unknown")
            self.assertEqual(reopened["station_id"].item(), "USW00094728")

    def test_projection_rejects_duplicate_dates(self) -> None:
        payload = source_payload()
        records = [dict(record) for record in payload.records]
        records[2]["DATE"] = records[1]["DATE"]
        modified = replace(payload, records=tuple(records))
        with self.assertRaises(ValueError):
            project_single_station_temperature(modified)


if __name__ == "__main__":
    unittest.main()
