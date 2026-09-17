from __future__ import annotations

import unittest
from pathlib import Path

from reference.ncei_ghcnd import (
    build_request_url,
    parse_daily_summaries,
    request_parameters,
)
from reference.nsidc_sea_ice_index import (
    monthly_extent_url,
    parse_monthly_extent_csv,
)


ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "fixtures" / "data_authorities"
NCEI_FIXTURE = FIXTURES / "ncei-ghcnd-USW00094728-2024-01-01-to-2024-01-03.json"
NSIDC_FIXTURE = FIXTURES / "nsidc-G02135-v4-N-september.csv"


class NceiGhcndAdapterTests(unittest.TestCase):
    def test_live_fixture_preserves_exact_payload_identity(self) -> None:
        raw = NCEI_FIXTURE.read_bytes()
        request_url = build_request_url(
            ["USW00094728"], "2024-01-01", "2024-01-03"
        )
        parsed = parse_daily_summaries(raw, request_url=request_url)

        self.assertEqual(parsed.byte_count, 2324)
        self.assertEqual(
            parsed.sha256,
            "sha256:06e00068f7de97d76b8bc523d5acb22ec7b21631adfec0e0c1a284faf1ac1a46",
        )
        self.assertEqual(len(parsed.records), 3)
        self.assertEqual(
            [record["DATE"] for record in parsed.records],
            ["2024-01-01", "2024-01-02", "2024-01-03"],
        )
        self.assertEqual(
            {record["STATION"] for record in parsed.records}, {"USW00094728"}
        )
        self.assertEqual(parsed.records[0]["TMAX"], "8.3")
        self.assertEqual(parsed.records[0]["PRCP_ATTRIBUTES"], ",,W,2400")

    def test_request_contract_is_explicit_and_stable(self) -> None:
        params = request_parameters(
            ["USW00094728"], "2024-01-01", "2024-01-03"
        )
        self.assertEqual(params["dataset"], "daily-summaries")
        self.assertEqual(params["format"], "json")
        self.assertEqual(params["units"], "metric")
        self.assertEqual(params["includeAttributes"], "true")
        self.assertEqual(params["includeStationLocation"], "true")
        self.assertIn("stations=USW00094728", build_request_url(
            ["USW00094728"], "2024-01-01", "2024-01-03"
        ))

    def test_invalid_date_range_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            request_parameters(["USW00094728"], "2024-01-03", "2024-01-01")

    def test_non_array_payload_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_daily_summaries(b'{"STATION":"USW00094728"}', request_url="fixture")


class NsidcSeaIceIndexAdapterTests(unittest.TestCase):
    def test_live_fixture_preserves_exact_payload_identity_and_source_transition(self) -> None:
        raw = NSIDC_FIXTURE.read_bytes()
        source_url = monthly_extent_url("N", 9)
        parsed = parse_monthly_extent_csv(raw, source_url=source_url)

        self.assertEqual(parsed.byte_count, 2304)
        self.assertEqual(
            parsed.sha256,
            "sha256:bdeb4dea8b52f6dc8a8faecfd5d220dc758b6dd2ddea345d7b18897df0d9716f",
        )
        self.assertEqual(len(parsed.records), 47)
        self.assertEqual((parsed.records[0].year, parsed.records[0].month), (1979, 9))
        self.assertEqual(parsed.records[0].source_dataset, "NSIDC-0051")
        self.assertEqual(parsed.records[-2].source_dataset, "NSIDC-0051")
        self.assertEqual(parsed.records[-1].source_dataset, "NSIDC-0803")
        self.assertEqual(parsed.records[-1].year, 2025)
        self.assertEqual(parsed.records[-1].extent_million_km2, 4.75)
        self.assertEqual(parsed.records[-1].area_million_km2, 3.08)

    def test_provider_url_is_versioned_and_month_specific(self) -> None:
        self.assertEqual(
            monthly_extent_url("N", 9),
            "https://noaadata.apps.nsidc.org/NOAA/G02135/north/monthly/data/N_09_extent_v4.0.csv",
        )

    def test_stale_header_shape_is_rejected(self) -> None:
        stale = (
            "year,mo,data_type,region,extent,area\n"
            "2024,9,NSIDC-0051,N,4.35,2.91\n"
        ).encode("utf-8")
        with self.assertRaises(ValueError):
            parse_monthly_extent_csv(stale, source_url="fixture")

    def test_negative_extent_is_rejected(self) -> None:
        invalid = (
            "year,mo,source_dataset,region,extent,area\n"
            "2024,9,NSIDC-0051,N,-1.0,2.91\n"
        ).encode("utf-8")
        with self.assertRaises(ValueError):
            parse_monthly_extent_csv(invalid, source_url="fixture")


if __name__ == "__main__":
    unittest.main()
