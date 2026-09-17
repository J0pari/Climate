"""Narrow reference adapter for NOAA/NCEI GHCN-Daily REST retrieval.

The adapter owns provider-specific request and payload semantics only. It does
not invent station metadata, fill missing observations, homogenize records, or
choose a scientific aggregation. Live HTTP uses the maintained ``requests``
library; parsing and provenance helpers remain usable in network-free CI.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlencode

SOURCE_ID = "ncei.ghcnd.v3"
DATASET_ID = "daily-summaries"
DATA_ENDPOINT = "https://www.ncei.noaa.gov/access/services/data/v1"


@dataclass(frozen=True)
class DailySummariesPayload:
    request_url: str
    sha256: str
    byte_count: int
    records: tuple[Mapping[str, Any], ...]


def _iso_date(value: str) -> str:
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError as exc:
        raise ValueError(f"invalid ISO date {value!r}") from exc


def request_parameters(
    stations: Sequence[str],
    start_date: str,
    end_date: str,
) -> dict[str, str]:
    station_ids = [station.strip() for station in stations if station.strip()]
    if not station_ids:
        raise ValueError("at least one station identifier is required")
    if any("," in station for station in station_ids):
        raise ValueError("station identifiers must be supplied as separate values")
    start = _iso_date(start_date)
    end = _iso_date(end_date)
    if start > end:
        raise ValueError("start_date must not be after end_date")
    return {
        "dataset": DATASET_ID,
        "stations": ",".join(station_ids),
        "startDate": start,
        "endDate": end,
        "format": "json",
        "units": "metric",
        "includeAttributes": "true",
        "includeStationLocation": "true",
    }


def build_request_url(
    stations: Sequence[str],
    start_date: str,
    end_date: str,
) -> str:
    return f"{DATA_ENDPOINT}?{urlencode(request_parameters(stations, start_date, end_date))}"


def parse_daily_summaries(payload: bytes, *, request_url: str) -> DailySummariesPayload:
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("NCEI daily-summaries payload is not valid UTF-8 JSON") from exc
    if not isinstance(decoded, list):
        raise ValueError("NCEI daily-summaries response must be a JSON array")

    records: list[Mapping[str, Any]] = []
    for index, record in enumerate(decoded):
        if not isinstance(record, dict):
            raise ValueError(f"NCEI record {index} is not an object")
        station = record.get("STATION")
        observation_date = record.get("DATE")
        if not isinstance(station, str) or not station:
            raise ValueError(f"NCEI record {index} has no STATION identifier")
        if not isinstance(observation_date, str) or not observation_date:
            raise ValueError(f"NCEI record {index} has no DATE")
        records.append(record)

    return DailySummariesPayload(
        request_url=request_url,
        sha256="sha256:" + hashlib.sha256(payload).hexdigest(),
        byte_count=len(payload),
        records=tuple(records),
    )


def fetch_daily_summaries(
    stations: Sequence[str],
    start_date: str,
    end_date: str,
    *,
    timeout_seconds: float = 30.0,
) -> tuple[bytes, DailySummariesPayload]:
    """Fetch one explicit NCEI GHCN-Daily subset without imputation or fallback."""
    import requests

    params = request_parameters(stations, start_date, end_date)
    response = requests.get(DATA_ENDPOINT, params=params, timeout=timeout_seconds)
    response.raise_for_status()
    raw = response.content
    parsed = parse_daily_summaries(raw, request_url=response.url)
    return raw, parsed


def _main() -> int:
    parser = argparse.ArgumentParser(description="retrieve an explicit NCEI GHCN-Daily subset")
    parser.add_argument("--station", action="append", required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    raw, parsed = fetch_daily_summaries(args.station, args.start_date, args.end_date)
    args.output.write_bytes(raw)
    print(json.dumps({
        "source_id": SOURCE_ID,
        "request_url": parsed.request_url,
        "sha256": parsed.sha256,
        "byte_count": parsed.byte_count,
        "record_count": len(parsed.records),
        "output": str(args.output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
