"""Provider-specific parser for captured NOAA@NSIDC Sea Ice Index monthly CSV files.

The provider publishes one extent/area CSV per hemisphere and month. Climate
retains the versioned URL/schema and parsing semantics only; acquisition is
externally owned and evidence consumes an already captured immutable artifact.
The parser does not infer missing months, smooth the time series, or reinterpret
sea-ice extent as concentration or area.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import dataclass

SOURCE_ID = "nsidc.sea_ice_index.v4"
BASE_URL = "https://noaadata.apps.nsidc.org/NOAA/G02135"
EXPECTED_COLUMNS = ("year", "mo", "source_dataset", "region", "extent", "area")


@dataclass(frozen=True)
class MonthlyExtentRecord:
    year: int
    month: int
    source_dataset: str
    region: str
    extent_million_km2: float
    area_million_km2: float


@dataclass(frozen=True)
class MonthlyExtentPayload:
    source_url: str
    sha256: str
    byte_count: int
    records: tuple[MonthlyExtentRecord, ...]


def monthly_extent_url(hemisphere: str, month: int) -> str:
    normalized = hemisphere.strip().upper()
    if normalized not in {"N", "S"}:
        raise ValueError("hemisphere must be N or S")
    if not 1 <= month <= 12:
        raise ValueError("month must be in 1..12")
    directory = "north" if normalized == "N" else "south"
    return f"{BASE_URL}/{directory}/monthly/data/{normalized}_{month:02d}_extent_v4.0.csv"


def parse_monthly_extent_csv(payload: bytes, *, source_url: str) -> MonthlyExtentPayload:
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Sea Ice Index CSV is not valid UTF-8 text") from exc

    reader = csv.DictReader(io.StringIO(text), skipinitialspace=True)
    if reader.fieldnames is None:
        raise ValueError("Sea Ice Index CSV has no header")
    normalized_fields = tuple(field.strip().lower() for field in reader.fieldnames)
    if normalized_fields != EXPECTED_COLUMNS:
        raise ValueError(
            f"Sea Ice Index CSV columns {normalized_fields!r} do not match {EXPECTED_COLUMNS!r}"
        )

    records: list[MonthlyExtentRecord] = []
    for line_number, raw in enumerate(reader, start=2):
        row = {str(key).strip().lower(): str(value).strip() for key, value in raw.items()}
        try:
            year = int(row["year"])
            month = int(row["mo"])
            extent = float(row["extent"])
            area = float(row["area"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid Sea Ice Index row at line {line_number}") from exc
        region = row["region"].upper()
        if region not in {"N", "S"}:
            raise ValueError(f"invalid Sea Ice Index region at line {line_number}: {region!r}")
        if not 1 <= month <= 12:
            raise ValueError(f"invalid Sea Ice Index month at line {line_number}: {month}")
        if extent < 0.0 or area < 0.0:
            raise ValueError(f"negative Sea Ice Index extent/area at line {line_number}")
        records.append(MonthlyExtentRecord(
            year=year,
            month=month,
            source_dataset=row["source_dataset"],
            region=region,
            extent_million_km2=extent,
            area_million_km2=area,
        ))

    if not records:
        raise ValueError("Sea Ice Index CSV contains no data rows")
    return MonthlyExtentPayload(
        source_url=source_url,
        sha256="sha256:" + hashlib.sha256(payload).hexdigest(),
        byte_count=len(payload),
        records=tuple(records),
    )
