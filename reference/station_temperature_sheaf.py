"""NCEI GHCN-Daily adapter into the canonical global station substrate.

Provider parsing and provenance remain provider-specific. Geographic topology,
sparse cochain assembly, sharding, and observation-chunk semantics live in
src.station_sheaf and are shared with large station-network execution.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence
from urllib.parse import parse_qs, urlparse

import numpy as np

from reference.ncei_ghcnd import DailySummariesPayload, SOURCE_ID
from src.station_sheaf import StationCatalog, StationSection, StationVariable


TEMPERATURE_VARIABLES = ("TMAX", "TMIN")
TEMPERATURE_UNIT = "degree_Celsius"


@dataclass(frozen=True)
class StationDay:
    station_id: str
    date: str
    latitude_deg: float
    longitude_deg: float
    values: Mapping[str, float | None]
    quality_flags: Mapping[str, str]


@dataclass(frozen=True)
class StationSnapshot:
    source_id: str
    source_artifact_sha256: str
    request_url: str
    date: str
    variables: tuple[str, ...]
    units: Mapping[str, str]
    stations: Mapping[str, StationDay]
    lineage: tuple[str, ...]


def _quality_flag(record: Mapping[str, object], variable: str) -> str:
    attributes = record.get(f"{variable}_ATTRIBUTES", "")
    if not isinstance(attributes, str):
        raise ValueError(f"{variable}_ATTRIBUTES must be a string when present")
    parts = attributes.split(",")
    return parts[1].strip() if len(parts) > 1 else ""


def _numeric_or_missing(record: Mapping[str, object], variable: str) -> float | None:
    raw = record.get(variable)
    if raw is None or raw == "":
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{variable} is not numeric: {raw!r}") from exc
    if not np.isfinite(value):
        raise ValueError(f"{variable} must be finite when present")
    return value


def extract_station_snapshot(
    payload: DailySummariesPayload,
    *,
    date: str,
    station_ids: Sequence[str],
    variables: Sequence[str],
    accept_quality_flagged: bool,
) -> StationSnapshot:
    requested = tuple(station_ids)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("station_ids must be a non-empty unique sequence")
    selected_variables = tuple(variables)
    if not selected_variables or len(set(selected_variables)) != len(selected_variables):
        raise ValueError("variables must be a non-empty unique sequence")
    if any(variable not in TEMPERATURE_VARIABLES for variable in selected_variables):
        raise ValueError(
            f"this NCEI temperature adapter supports only {TEMPERATURE_VARIABLES!r}"
        )

    query = parse_qs(urlparse(payload.request_url).query)
    if query.get("units") != ["metric"]:
        raise ValueError("temperature adapter requires the provider metric-unit request")
    requested_from_url = tuple(query.get("stations", [""])[0].split(","))
    if set(requested_from_url) != set(requested):
        raise ValueError("station_ids do not match the captured provider request")

    location_by_station: dict[str, tuple[float, float]] = {}
    selected_records: dict[str, Mapping[str, object]] = {}
    for record in payload.records:
        station_id = str(record["STATION"])
        if station_id not in requested:
            continue
        latitude = float(record["LATITUDE"])
        longitude = float(record["LONGITUDE"])
        if not np.isfinite(latitude) or not np.isfinite(longitude):
            raise ValueError(f"station {station_id} has non-finite coordinates")
        location = (latitude, longitude)
        previous = location_by_station.setdefault(station_id, location)
        if previous != location:
            raise ValueError(f"station {station_id} changes location inside the payload")
        if record["DATE"] == date:
            if station_id in selected_records:
                raise ValueError(f"duplicate station/date record for {station_id} {date}")
            selected_records[station_id] = record

    missing_records = sorted(set(requested) - set(selected_records))
    if missing_records:
        raise ValueError(
            f"captured provider payload has no {date} record for {missing_records}"
        )

    stations: dict[str, StationDay] = {}
    for station_id in requested:
        record = selected_records[station_id]
        values: dict[str, float | None] = {}
        quality_flags: dict[str, str] = {}
        for variable in selected_variables:
            flag = _quality_flag(record, variable)
            quality_flags[variable] = flag
            value = _numeric_or_missing(record, variable)
            if flag and not accept_quality_flagged:
                value = None
            values[variable] = value
        latitude, longitude = location_by_station[station_id]
        stations[station_id] = StationDay(
            station_id=station_id,
            date=date,
            latitude_deg=latitude,
            longitude_deg=longitude,
            values=values,
            quality_flags=quality_flags,
        )

    return StationSnapshot(
        source_id=SOURCE_ID,
        source_artifact_sha256=payload.sha256,
        request_url=payload.request_url,
        date=date,
        variables=selected_variables,
        units={variable: TEMPERATURE_UNIT for variable in selected_variables},
        stations=stations,
        lineage=("ncei_daily_summaries:station_day_selection",),
    )


def to_station_catalog(snapshot: StationSnapshot) -> StationCatalog:
    station_ids = tuple(snapshot.stations)
    return StationCatalog(
        station_ids=tuple(f"{snapshot.source_id}:{station_id}" for station_id in station_ids),
        latitude_deg=np.asarray(
            [snapshot.stations[station_id].latitude_deg for station_id in station_ids],
            dtype=np.float64,
        ),
        longitude_deg=np.asarray(
            [snapshot.stations[station_id].longitude_deg for station_id in station_ids],
            dtype=np.float64,
        ),
    )


def to_station_section(snapshot: StationSnapshot) -> StationSection:
    station_ids = tuple(snapshot.stations)
    variables = tuple(
        StationVariable(variable, snapshot.units[variable])
        for variable in snapshot.variables
    )
    values = np.empty((len(station_ids), len(variables)), dtype=np.float64)
    observed = np.empty_like(values, dtype=bool)
    for station_index, station_id in enumerate(station_ids):
        station = snapshot.stations[station_id]
        for variable_index, variable in enumerate(snapshot.variables):
            value = station.values[variable]
            observed[station_index, variable_index] = value is not None
            values[station_index, variable_index] = 0.0 if value is None else float(value)
    return StationSection(variables=variables, values=values, observed=observed)
