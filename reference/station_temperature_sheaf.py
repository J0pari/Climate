"""Provider-backed station-temperature sheaf semantics for reference experiments.

This adapter composes existing repository authorities rather than replacing them:

* reference.ncei_ghcnd owns NCEI request/payload identity;
* reference.sheaf_cohomology.FiniteCover owns exact finite-cover nerves;
* pyproj owns geographic coordinate transformation;
* NumPy owns real cochain linear algebra;
* SciPy owns the nearest-neighbor interpolation baseline.

Climate-specific scope here is the declared station analysis support, temperature
stalk basis, missing/QC semantics, restriction maps, and provenance-preserving
composition. Synthetic fault or withholding transforms belong to tests and must
remain explicitly derived from a captured provider payload.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence
from urllib.parse import parse_qs, urlparse

import numpy as np
from pyproj import Transformer
from scipy.interpolate import NearestNDInterpolator

from reference.ncei_ghcnd import DailySummariesPayload, SOURCE_ID
from reference.sheaf_cohomology import FiniteCover, FiniteSimplicialComplex, Simplex

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


@dataclass(frozen=True)
class StationTemperatureSheaf:
    snapshot: StationSnapshot
    coverage_radius_km: float
    projected_crs: str
    projected_xy_m: Mapping[str, tuple[float, float]]
    cover: FiniteCover
    base: FiniteSimplicialComplex
    stalk_bases: Mapping[Simplex, tuple[str, ...]]
    restrictions: Mapping[tuple[Simplex, Simplex], np.ndarray]

    def cochain_dimension(self, degree: int) -> int:
        return sum(len(self.stalk_bases[s]) for s in self.base.simplices(degree))

    def coboundary_matrix(self, degree: int) -> np.ndarray:
        lower = self.base.simplices(degree)
        upper = self.base.simplices(degree + 1)
        row_count = sum(len(self.stalk_bases[s]) for s in upper)
        col_count = sum(len(self.stalk_bases[s]) for s in lower)
        matrix = np.zeros((row_count, col_count), dtype=np.float64)

        row_offset = 0
        for coface in upper:
            col_offset = 0
            for face in lower:
                if len(coface) == len(face) + 1 and set(face) < set(coface):
                    omitted = next(
                        index for index, vertex in enumerate(coface) if vertex not in face
                    )
                    sign = -1.0 if omitted % 2 else 1.0
                    block = self.restrictions[(face, coface)]
                    rows, cols = block.shape
                    matrix[
                        row_offset : row_offset + rows,
                        col_offset : col_offset + cols,
                    ] = sign * block
                col_offset += len(self.stalk_bases[face])
            row_offset += len(self.stalk_bases[coface])
        return matrix

    def verify_functoriality(self) -> None:
        for lower in self.base.faces:
            for middle in self.base.faces:
                if not set(lower) < set(middle):
                    continue
                for upper in self.base.faces:
                    if not set(middle) < set(upper):
                        continue
                    direct = self.restrictions[(lower, upper)]
                    composed = (
                        self.restrictions[(middle, upper)]
                        @ self.restrictions[(lower, middle)]
                    )
                    if not np.array_equal(direct, composed):
                        raise AssertionError(
                            f"restriction composition fails for {lower} < {middle} < {upper}"
                        )

    def d_squared_is_zero(self, degree: int) -> bool:
        first = self.coboundary_matrix(degree)
        second = self.coboundary_matrix(degree + 1)
        if second.shape[0] == 0:
            return True
        composite = second @ first
        return bool(np.array_equal(composite, np.zeros_like(composite)))

    def vertex_cochain(self) -> np.ndarray:
        values: list[float] = []
        for vertex in self.base.simplices(0):
            station = self.snapshot.stations[vertex[0]]
            for variable in self.stalk_bases[vertex]:
                value = station.values[variable]
                if value is None:
                    raise AssertionError(
                        f"vertex basis contains unavailable {variable} at {vertex[0]}"
                    )
                values.append(value)
        return np.asarray(values, dtype=np.float64)

    def compatibility_residual(self) -> np.ndarray:
        return self.coboundary_matrix(0) @ self.vertex_cochain()

    def compatibility_energy(self) -> float:
        residual = self.compatibility_residual()
        return float(residual @ residual)

    def global_section_dimension(self) -> int:
        d0 = self.coboundary_matrix(0)
        rank = int(np.linalg.matrix_rank(d0))
        return self.cochain_dimension(0) - rank

    def observed_assignment_is_compatible(self) -> bool:
        residual = self.compatibility_residual()
        return bool(np.array_equal(residual, np.zeros_like(residual)))

    def graph_residual_baseline(self) -> np.ndarray:
        residual: list[float] = []
        for edge in self.base.simplices(1):
            tail, head = edge
            for variable in self.stalk_bases[edge]:
                tail_value = self.snapshot.stations[tail].values[variable]
                head_value = self.snapshot.stations[head].values[variable]
                if tail_value is None or head_value is None:
                    raise AssertionError("edge basis contains unavailable value")
                residual.append(head_value - tail_value)
        return np.asarray(residual, dtype=np.float64)

    def provider_quality_flag_count(self) -> int:
        return sum(
            1
            for station in self.snapshot.stations.values()
            for variable in self.snapshot.variables
            if station.quality_flags[variable]
        )

    def nearest_neighbor_prediction(
        self, held_out_station_id: str, variable: str
    ) -> float:
        if held_out_station_id not in self.snapshot.stations:
            raise ValueError(f"unknown station {held_out_station_id!r}")
        points: list[tuple[float, float]] = []
        values: list[float] = []
        for station_id, station in self.snapshot.stations.items():
            if station_id == held_out_station_id:
                continue
            value = station.values.get(variable)
            if value is None:
                continue
            points.append(self.projected_xy_m[station_id])
            values.append(value)
        if not points:
            raise ValueError("interpolation baseline has no training observations")
        interpolator = NearestNDInterpolator(
            np.asarray(points, dtype=np.float64),
            np.asarray(values, dtype=np.float64),
        )
        prediction = interpolator(np.asarray([self.projected_xy_m[held_out_station_id]]))
        return float(np.asarray(prediction).reshape(-1)[0])


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
            f"station-temperature sheaf supports only {TEMPERATURE_VARIABLES!r}"
        )

    query = parse_qs(urlparse(payload.request_url).query)
    if query.get("units") != ["metric"]:
        raise ValueError("temperature sheaf requires the provider metric-unit request")
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
            quality_flag = _quality_flag(record, variable)
            quality_flags[variable] = quality_flag
            value = _numeric_or_missing(record, variable)
            if quality_flag and not accept_quality_flagged:
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


def build_station_temperature_sheaf(
    snapshot: StationSnapshot,
    *,
    coverage_radius_km: float,
    projected_crs: str,
) -> StationTemperatureSheaf:
    if not np.isfinite(coverage_radius_km) or coverage_radius_km <= 0.0:
        raise ValueError("coverage_radius_km must be finite and positive")
    if not projected_crs.strip():
        raise ValueError("projected_crs must be non-empty")

    transformer = Transformer.from_crs("EPSG:4326", projected_crs, always_xy=True)
    projected: dict[str, tuple[float, float]] = {}
    for station_id, station in snapshot.stations.items():
        x_m, y_m = transformer.transform(station.longitude_deg, station.latitude_deg)
        if not np.isfinite(x_m) or not np.isfinite(y_m):
            raise ValueError(f"projection produced non-finite coordinates for {station_id}")
        projected[station_id] = (float(x_m), float(y_m))

    radius_m = coverage_radius_km * 1000.0
    members: dict[str, set[str]] = {}
    for station_id, xy in projected.items():
        origin = np.asarray(xy, dtype=np.float64)
        support = {
            candidate_id
            for candidate_id, candidate_xy in projected.items()
            if np.linalg.norm(np.asarray(candidate_xy, dtype=np.float64) - origin)
            <= radius_m
        }
        if station_id not in support:
            raise AssertionError("station analysis support must contain its center")
        members[station_id] = support

    cover = FiniteCover.from_members(members)
    base = cover.nerve()

    stalk_bases: dict[Simplex, tuple[str, ...]] = {}
    for simplex in base.faces:
        basis = tuple(
            variable
            for variable in snapshot.variables
            if all(snapshot.stations[station].values[variable] is not None for station in simplex)
        )
        if not basis:
            raise ValueError(
                f"nerve simplex {simplex} has no shared present declared variable; "
                "missing data are not imputed into a nonzero stalk"
            )
        stalk_bases[simplex] = basis

    restrictions: dict[tuple[Simplex, Simplex], np.ndarray] = {}
    for face in base.faces:
        face_basis = stalk_bases[face]
        for coface in base.faces:
            if not set(face) < set(coface):
                continue
            coface_basis = stalk_bases[coface]
            matrix = np.zeros((len(coface_basis), len(face_basis)), dtype=np.float64)
            for row, variable in enumerate(coface_basis):
                try:
                    col = face_basis.index(variable)
                except ValueError as exc:
                    raise AssertionError(
                        f"coface variable {variable!r} is absent from face basis"
                    ) from exc
                matrix[row, col] = 1.0
            restrictions[(face, coface)] = matrix

    sheaf = StationTemperatureSheaf(
        snapshot=snapshot,
        coverage_radius_km=coverage_radius_km,
        projected_crs=projected_crs,
        projected_xy_m=projected,
        cover=cover,
        base=base,
        stalk_bases=stalk_bases,
        restrictions=restrictions,
    )
    sheaf.verify_functoriality()
    return sheaf
