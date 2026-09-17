"""CF-aware single-station projection for explicit NCEI GHCN-Daily payloads.

Generic labeled arrays and netCDF serialization are delegated to xarray. This
module owns only the climate-facing projection semantics: one station, explicit
coordinates, provider provenance, daily temperature variables, provider
attribute strings as ancillary data, and refusal to impute missing values.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping

from reference.ncei_ghcnd import build_request_url, parse_daily_summaries

PROJECTION_ID = "ncei.ghcnd.single_station_temperature.v1"
CF_CONVENTIONS = "CF-1.11"


def _required_station_scalar(
    records: tuple[Mapping[str, Any], ...],
    key: str,
) -> str:
    values = {str(record.get(key, "")).strip() for record in records}
    values.discard("")
    if len(values) != 1 or any(str(record.get(key, "")).strip() == "" for record in records):
        raise ValueError(f"NCEI projection requires one consistent {key} value")
    return next(iter(values))


def _optional_numeric(record: Mapping[str, Any], key: str) -> float:
    import numpy as np

    value = record.get(key)
    if value is None or str(value).strip() == "":
        return float("nan")
    parsed = float(value)
    if not np.isfinite(parsed):
        raise ValueError(f"NCEI {key} contains a non-finite value")
    return parsed


def _optional_attribute(record: Mapping[str, Any], key: str) -> str:
    value = record.get(key)
    return "" if value is None else str(value)


def project_single_station_temperature(payload):
    """Project one parsed GHCN-Daily station payload to an xarray Dataset."""
    import numpy as np
    import xarray as xr

    if not payload.records:
        raise ValueError("NCEI projection requires at least one record")

    station_id = _required_station_scalar(payload.records, "STATION")
    latitude = float(_required_station_scalar(payload.records, "LATITUDE"))
    longitude = float(_required_station_scalar(payload.records, "LONGITUDE"))
    altitude_m = float(_required_station_scalar(payload.records, "ELEVATION"))
    if not all(np.isfinite(value) for value in (latitude, longitude, altitude_m)):
        raise ValueError("NCEI station coordinates must be finite")
    if not -90.0 <= latitude <= 90.0:
        raise ValueError("NCEI station latitude is outside [-90, 90]")
    if not -180.0 <= longitude <= 180.0:
        raise ValueError("NCEI station longitude is outside [-180, 180]")

    ordered = sorted(payload.records, key=lambda record: str(record["DATE"]))
    date_strings = [str(record["DATE"]) for record in ordered]
    if len(date_strings) != len(set(date_strings)):
        raise ValueError("NCEI projection contains duplicate station dates")
    try:
        times = np.asarray(date_strings, dtype="datetime64[D]")
    except ValueError as exc:
        raise ValueError("NCEI DATE values are not ISO calendar dates") from exc
    if times.size > 1 and not np.all(times[1:] > times[:-1]):
        raise ValueError("NCEI projection dates must be strictly increasing")

    tmax = np.asarray([_optional_numeric(record, "TMAX") for record in ordered], dtype=np.float64)
    tmin = np.asarray([_optional_numeric(record, "TMIN") for record in ordered], dtype=np.float64)
    tmax_attributes = np.asarray(
        [_optional_attribute(record, "TMAX_ATTRIBUTES") for record in ordered], dtype=str
    )
    tmin_attributes = np.asarray(
        [_optional_attribute(record, "TMIN_ATTRIBUTES") for record in ordered], dtype=str
    )

    lineage = [
        {
            "source": "DATE",
            "target": "time",
            "operation": "ISO calendar-date decoding without interpolation",
        },
        {
            "source": "TMAX",
            "target": "daily_maximum_air_temperature",
            "operation": "numeric parse in requested NCEI metric units; absent values remain missing",
        },
        {
            "source": "TMIN",
            "target": "daily_minimum_air_temperature",
            "operation": "numeric parse in requested NCEI metric units; absent values remain missing",
        },
        {
            "source": "TMAX_ATTRIBUTES,TMIN_ATTRIBUTES",
            "target": "provider ancillary variables",
            "operation": "verbatim preservation",
        },
        {
            "source": "LATITUDE,LONGITUDE,ELEVATION,STATION",
            "target": "scalar station coordinates",
            "operation": "numeric or identifier parse without spatial remapping",
        },
    ]

    dataset = xr.Dataset(
        data_vars={
            "daily_maximum_air_temperature": (
                "time",
                tmax,
                {
                    "standard_name": "air_temperature",
                    "long_name": "daily maximum air temperature",
                    "units": "degree_Celsius",
                    "cell_methods": "time: maximum",
                    "ancillary_variables": "daily_maximum_air_temperature_attributes",
                },
            ),
            "daily_minimum_air_temperature": (
                "time",
                tmin,
                {
                    "standard_name": "air_temperature",
                    "long_name": "daily minimum air temperature",
                    "units": "degree_Celsius",
                    "cell_methods": "time: minimum",
                    "ancillary_variables": "daily_minimum_air_temperature_attributes",
                },
            ),
            "daily_maximum_air_temperature_attributes": (
                "time",
                tmax_attributes,
                {
                    "long_name": "NCEI GHCN-Daily provider attributes for TMAX",
                    "comment": "Provider measurement, quality, source, and observation-time fields are preserved verbatim.",
                },
            ),
            "daily_minimum_air_temperature_attributes": (
                "time",
                tmin_attributes,
                {
                    "long_name": "NCEI GHCN-Daily provider attributes for TMIN",
                    "comment": "Provider measurement, quality, source, and observation-time fields are preserved verbatim.",
                },
            ),
        },
        coords={
            "time": (
                "time",
                times,
                {"standard_name": "time", "long_name": "date of daily summary", "axis": "T"},
            ),
            "latitude": (
                (),
                latitude,
                {"standard_name": "latitude", "units": "degrees_north", "axis": "Y"},
            ),
            "longitude": (
                (),
                longitude,
                {"standard_name": "longitude", "units": "degrees_east", "axis": "X"},
            ),
            "station_altitude": (
                (),
                altitude_m,
                {
                    "standard_name": "surface_altitude",
                    "units": "m",
                    "positive": "up",
                    "axis": "Z",
                },
            ),
            "station_id": (
                (),
                station_id,
                {"long_name": "NCEI GHCN-Daily station identifier", "cf_role": "timeseries_id"},
            ),
        },
        attrs={
            "Conventions": CF_CONVENTIONS,
            "featureType": "timeSeries",
            "title": "NCEI GHCN-Daily single-station temperature projection",
            "source": "NOAA NCEI Global Historical Climatology Network - Daily, daily-summaries API",
            "projection_id": PROJECTION_ID,
            "source_id": "ncei.ghcnd.v3",
            "source_request_url": payload.request_url,
            "source_artifact_sha256": payload.sha256,
            "source_artifact_bytes": payload.byte_count,
            "source_record_count": len(payload.records),
            "missing_data_policy": "Absent TMAX/TMIN observations remain NaN; no climatology, interpolation, or fallback value is applied.",
            "quality_control_policy": "NCEI provider attribute strings are retained as ancillary variables; this projection applies no local QC filtering.",
            "transformation_lineage": json.dumps(lineage, separators=(",", ":"), sort_keys=True),
        },
    )

    coordinate_names = "latitude longitude station_altitude station_id"
    for name in dataset.data_vars:
        dataset[name].encoding["coordinates"] = coordinate_names
    dataset["time"].encoding.update(
        {"units": "days since 1970-01-01 00:00:00", "calendar": "proleptic_gregorian"}
    )
    return dataset


def netcdf_bytes(dataset) -> bytes:
    """Serialize with xarray's maintained SciPy netCDF backend."""
    payload = dataset.to_netcdf(path=None, engine="scipy", format="NETCDF3_64BIT")
    return bytes(payload)


def netcdf_sha256(dataset) -> str:
    return "sha256:" + hashlib.sha256(netcdf_bytes(dataset)).hexdigest()


def open_netcdf_bytes(payload: bytes):
    """Decode projected bytes through xarray's CF-aware reader."""
    import xarray as xr

    return xr.open_dataset(BytesIO(payload), engine="scipy", decode_cf=True)


def _main() -> int:
    parser = argparse.ArgumentParser(description="project one NCEI GHCN-Daily station subset to CF-aware netCDF")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--station", required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    raw = args.input.read_bytes()
    request_url = build_request_url([args.station], args.start_date, args.end_date)
    parsed = parse_daily_summaries(raw, request_url=request_url)
    dataset = project_single_station_temperature(parsed)
    projected = netcdf_bytes(dataset)
    args.output.write_bytes(projected)
    print(json.dumps({
        "projection_id": PROJECTION_ID,
        "source_artifact_sha256": parsed.sha256,
        "projection_artifact_sha256": "sha256:" + hashlib.sha256(projected).hexdigest(),
        "projection_bytes": len(projected),
        "output": str(args.output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
