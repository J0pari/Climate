"""Provider-specific reference parser for captured NOAA/NCEI GHCN-Daily payloads.

Climate owns only request-identity reconstruction and payload/schema semantics
here. Network acquisition is externally owned: experiments must import an
already captured immutable artifact and bind its source URL/digest rather than
turn this reference module into a data-transfer client.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
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
