"""Production-oriented global station-network and sheaf primitives.

The implementation path is global, sparse, locality-driven, and provider-neutral.
Small fixtures exercise these same primitives; they are not a separate topology
or linear-algebra implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np
from pyproj import Geod
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


_WGS84 = Geod(ellps="WGS84")


def _f64_1d(values, name: str) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64)
    if out.ndim != 1 or not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must be a finite one-dimensional array")
    out = np.array(out, copy=True)
    out.setflags(write=False)
    return out


def _i64_1d(values, name: str) -> np.ndarray:
    out = np.asarray(values, dtype=np.int64)
    if out.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    out = np.array(out, copy=True)
    out.setflags(write=False)
    return out


@dataclass(frozen=True)
class StationVariable:
    variable_id: str
    unit: str

    def __post_init__(self) -> None:
        if not self.variable_id.strip() or not self.unit.strip():
            raise ValueError("station variable id and unit must be non-empty")


@dataclass(frozen=True)
class StationCatalog:
    station_ids: tuple[str, ...]
    latitude_deg: np.ndarray
    longitude_deg: np.ndarray

    def __post_init__(self) -> None:
        ids = tuple(self.station_ids)
        if not ids or any(not value.strip() for value in ids):
            raise ValueError("station_ids must be non-empty strings")
        if len(set(ids)) != len(ids):
            raise ValueError("station_ids must be unique")
        lat = _f64_1d(self.latitude_deg, "latitude_deg")
        lon = _f64_1d(self.longitude_deg, "longitude_deg")
        if len(ids) != lat.size or len(ids) != lon.size:
            raise ValueError("coordinates must match station_ids length")
        if np.any((lat < -90.0) | (lat > 90.0)):
            raise ValueError("latitude_deg must lie in [-90, 90]")
        if np.any((lon < -180.0) | (lon > 180.0)):
            raise ValueError("longitude_deg must lie in [-180, 180]")
        object.__setattr__(self, "station_ids", ids)
        object.__setattr__(self, "latitude_deg", lat)
        object.__setattr__(self, "longitude_deg", lon)

    @property
    def station_count(self) -> int:
        return len(self.station_ids)

    def unit_vectors(self) -> np.ndarray:
        lat = np.deg2rad(self.latitude_deg)
        lon = np.deg2rad(self.longitude_deg)
        c = np.cos(lat)
        out = np.column_stack((c * np.cos(lon), c * np.sin(lon), np.sin(lat)))
        out.setflags(write=False)
        return out


@dataclass(frozen=True)
class StationEdgeChunk:
    tail: np.ndarray
    head: np.ndarray
    distance_m: np.ndarray

    def __post_init__(self) -> None:
        tail = _i64_1d(self.tail, "tail")
        head = _i64_1d(self.head, "head")
        distance = _f64_1d(self.distance_m, "distance_m")
        if not (tail.size == head.size == distance.size):
            raise ValueError("edge arrays must have equal length")
        if np.any(tail < 0) or np.any(head < 0) or np.any(tail >= head):
            raise ValueError("edges require deterministic non-negative tail < head orientation")
        if np.any(distance < 0.0):
            raise ValueError("distance_m must be non-negative")
        object.__setattr__(self, "tail", tail)
        object.__setattr__(self, "head", head)
        object.__setattr__(self, "distance_m", distance)

    def __len__(self) -> int:
        return int(self.tail.size)


class GlobalRadiusIndex:
    """WGS84 radius search using a 3-D candidate index and exact refinement."""

    def __init__(self, catalog: StationCatalog, *, leafsize: int = 64) -> None:
        if leafsize <= 0:
            raise ValueError("leafsize must be positive")
        self.catalog = catalog
        self._xyz = catalog.unit_vectors()
        self._tree = cKDTree(self._xyz, leafsize=leafsize, compact_nodes=True)

    @staticmethod
    def _candidate_chord(max_distance_m: float) -> float:
        if not np.isfinite(max_distance_m) or max_distance_m <= 0.0:
            raise ValueError("max_distance_m must be finite and positive")
        a = float(_WGS84.a)
        f = float(_WGS84.f)
        e2 = f * (2.0 - f)
        minimum_radius_m = a * (1.0 - e2)
        angle = min(np.pi, max_distance_m / minimum_radius_m)
        return float(2.0 * np.sin(0.5 * angle))

    def _refine(self, tail: np.ndarray, head: np.ndarray, limit: float):
        if tail.size == 0:
            return None
        _, _, distance = _WGS84.inv(
            self.catalog.longitude_deg[tail],
            self.catalog.latitude_deg[tail],
            self.catalog.longitude_deg[head],
            self.catalog.latitude_deg[head],
        )
        distance = np.asarray(distance, dtype=np.float64)
        keep = distance <= limit
        if not np.any(keep):
            return None
        return StationEdgeChunk(tail[keep], head[keep], distance[keep])

    def iter_edges(
        self,
        *,
        max_distance_m: float,
        query_chunk_size: int = 65_536,
        candidate_batch_size: int = 262_144,
        workers: int = 1,
        owner_start: int = 0,
        owner_stop: int | None = None,
    ) -> Iterator[StationEdgeChunk]:
        if query_chunk_size <= 0 or candidate_batch_size <= 0 or workers == 0:
            raise ValueError("chunk sizes must be positive and workers non-zero")
        n = self.catalog.station_count
        stop_owner = n if owner_stop is None else owner_stop
        if owner_start < 0 or stop_owner < owner_start or stop_owner > n:
            raise ValueError("invalid owner range")
        chord = self._candidate_chord(max_distance_m)
        tails: list[int] = []
        heads: list[int] = []

        def flush():
            if not tails:
                return
            tail = np.asarray(tails, dtype=np.int64)
            head = np.asarray(heads, dtype=np.int64)
            tails.clear()
            heads.clear()
            refined = self._refine(tail, head, max_distance_m)
            if refined is not None:
                yield refined

        for start in range(owner_start, stop_owner, query_chunk_size):
            stop = min(stop_owner, start + query_chunk_size)
            neighborhoods = self._tree.query_ball_point(
                self._xyz[start:stop],
                r=chord,
                workers=workers,
                return_sorted=True,
            )
            for offset, candidates in enumerate(neighborhoods):
                i = start + offset
                for candidate in candidates:
                    j = int(candidate)
                    if j <= i:
                        continue
                    tails.append(i)
                    heads.append(j)
                    if len(tails) >= candidate_batch_size:
                        yield from flush()
        yield from flush()


@dataclass(frozen=True)
class SparseRipsComplex:
    station_ids: tuple[str, ...]
    adjacency: csr_matrix

    @classmethod
    def from_edge_chunks(cls, station_ids: Sequence[str], edge_chunks) -> "SparseRipsComplex":
        ids = tuple(station_ids)
        if not ids:
            raise ValueError("station_ids must be non-empty")
        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        for chunk in edge_chunks:
            if len(chunk) == 0:
                continue
            if int(chunk.head.max()) >= len(ids):
                raise ValueError("edge references station outside station_ids")
            rows.extend((chunk.tail, chunk.head))
            cols.extend((chunk.head, chunk.tail))
        if not rows:
            return cls(ids, csr_matrix((len(ids), len(ids)), dtype=np.uint8))
        row = np.concatenate(rows)
        col = np.concatenate(cols)
        matrix = coo_matrix(
            (np.ones(row.size, dtype=np.uint8), (row, col)),
            shape=(len(ids), len(ids)),
        ).tocsr()
        matrix.sum_duplicates()
        matrix.data[:] = 1
        matrix.sort_indices()
        return cls(ids, matrix)

    @property
    def edge_count(self) -> int:
        return int(self.adjacency.nnz // 2)

    def iter_simplices(self, degree: int) -> Iterator[tuple[int, ...]]:
        if degree < 0:
            return
        target = degree + 1
        n = len(self.station_ids)
        if target == 1:
            for vertex in range(n):
                yield (vertex,)
            return
        forward = []
        for vertex in range(n):
            start, stop = self.adjacency.indptr[vertex : vertex + 2]
            neighbors = self.adjacency.indices[start:stop]
            forward.append(neighbors[neighbors > vertex])

        def extend(prefix: tuple[int, ...], candidates: np.ndarray):
            if len(prefix) == target:
                yield prefix
                return
            if candidates.size < target - len(prefix):
                return
            for position, vertex in enumerate(candidates):
                remainder = candidates[position + 1 :]
                if len(prefix) + 1 == target:
                    yield (*prefix, int(vertex))
                else:
                    common = np.intersect1d(
                        remainder, forward[int(vertex)], assume_unique=True
                    )
                    yield from extend((*prefix, int(vertex)), common)

        yield from extend((), np.arange(n, dtype=np.int64))


@dataclass(frozen=True)
class StationSection:
    """One bounded time/aggregation slice; missingness is separate from values."""

    variables: tuple[StationVariable, ...]
    values: np.ndarray
    observed: np.ndarray

    def __post_init__(self) -> None:
        variables = tuple(self.variables)
        if not variables or len({item.variable_id for item in variables}) != len(variables):
            raise ValueError("section variables must be non-empty and unique")
        values = np.asarray(self.values, dtype=np.float64)
        observed = np.asarray(self.observed, dtype=bool)
        if values.ndim != 2 or observed.shape != values.shape:
            raise ValueError("values and observed must share (station, variable) shape")
        if values.shape[1] != len(variables):
            raise ValueError("section variable dimension does not match variables")
        if not np.all(np.isfinite(values[observed])):
            raise ValueError("observed values must be finite")
        values = np.array(values, copy=True)
        observed = np.array(observed, copy=True)
        values.setflags(write=False)
        observed.setflags(write=False)
        object.__setattr__(self, "variables", variables)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "observed", observed)

    @property
    def station_count(self) -> int:
        return int(self.values.shape[0])


@dataclass(frozen=True)
class ObservedResidual:
    row_indices: np.ndarray
    values: np.ndarray
    structural_row_count: int

    def __post_init__(self) -> None:
        rows = _i64_1d(self.row_indices, "row_indices")
        values = _f64_1d(self.values, "values")
        if rows.size != values.size or self.structural_row_count < 0:
            raise ValueError("invalid residual dimensions")
        if rows.size and (int(rows.min()) < 0 or int(rows.max()) >= self.structural_row_count):
            raise ValueError("row index outside structural row space")
        object.__setattr__(self, "row_indices", rows)
        object.__setattr__(self, "values", values)

    def energy(self) -> float:
        return float(self.values @ self.values)


class StationIdentitySheaf:
    """Sparse d0 for same-variable identity restrictions across station edges."""

    def __init__(self, *, station_count: int, variables: Sequence[StationVariable]) -> None:
        variables = tuple(variables)
        if station_count <= 0 or not variables:
            raise ValueError("station_count and variables must be non-empty")
        if len({item.variable_id for item in variables}) != len(variables):
            raise ValueError("variable_ids must be unique")
        self.station_count = station_count
        self.variables = variables

    @property
    def variable_count(self) -> int:
        return len(self.variables)

    def coboundary_0(self, edges: StationEdgeChunk) -> csr_matrix:
        if len(edges) == 0:
            return csr_matrix((0, self.station_count * self.variable_count), dtype=np.float64)
        if int(edges.head.max()) >= self.station_count:
            raise ValueError("edge references station outside sheaf")
        edge_index = np.repeat(np.arange(len(edges), dtype=np.int64), self.variable_count)
        variable_index = np.tile(np.arange(self.variable_count, dtype=np.int64), len(edges))
        rows = edge_index * self.variable_count + variable_index
        tail_cols = edges.tail[edge_index] * self.variable_count + variable_index
        head_cols = edges.head[edge_index] * self.variable_count + variable_index
        matrix = coo_matrix(
            (
                np.concatenate((-np.ones(rows.size), np.ones(rows.size))),
                (np.concatenate((rows, rows)), np.concatenate((tail_cols, head_cols))),
            ),
            shape=(len(edges) * self.variable_count, self.station_count * self.variable_count),
        ).tocsr()
        matrix.sort_indices()
        return matrix

    def residual(self, edges: StationEdgeChunk, section: StationSection) -> ObservedResidual:
        if section.station_count != self.station_count or section.variables != self.variables:
            raise ValueError("section schema does not match sheaf")
        rows = len(edges) * self.variable_count
        if len(edges) == 0:
            return ObservedResidual(np.empty(0, np.int64), np.empty(0), rows)
        if int(edges.head.max()) >= self.station_count:
            raise ValueError("edge references station outside sheaf")
        active = section.observed[edges.tail] & section.observed[edges.head]
        difference = section.values[edges.head] - section.values[edges.tail]
        flat = active.reshape(-1)
        return ObservedResidual(
            np.flatnonzero(flat).astype(np.int64, copy=False),
            difference.reshape(-1)[flat],
            rows,
        )


def concatenate_edge_chunks(chunks: Sequence[StationEdgeChunk]) -> StationEdgeChunk:
    if not chunks:
        return StationEdgeChunk(np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0))
    return StationEdgeChunk(
        np.concatenate([chunk.tail for chunk in chunks]),
        np.concatenate([chunk.head for chunk in chunks]),
        np.concatenate([chunk.distance_m for chunk in chunks]),
    )


Simplex = tuple[int, ...]


def codimension_one_faces(simplex: Simplex) -> tuple[Simplex, ...]:
    if len(simplex) < 2:
        return ()
    return tuple(
        simplex[:omitted] + simplex[omitted + 1 :]
        for omitted in range(len(simplex))
    )


@dataclass(frozen=True)
class CochainIndex:
    """Partition-local sparse cochain layout.

    A partition owns a bounded simplex set. Halo faces may be added explicitly,
    so distributed cochain assembly does not require global simplex tables.
    """

    simplices: tuple[Simplex, ...]
    bases: tuple[tuple[int, ...], ...]
    offsets: tuple[int, ...]
    simplex_position: Mapping[Simplex, int]
    total_dimension: int

    def basis(self, simplex: Simplex) -> tuple[int, ...]:
        try:
            return self.bases[self.simplex_position[simplex]]
        except KeyError as exc:
            raise ValueError(f"simplex {simplex!r} is absent from cochain index") from exc

    def offset(self, simplex: Simplex) -> int:
        try:
            return self.offsets[self.simplex_position[simplex]]
        except KeyError as exc:
            raise ValueError(f"simplex {simplex!r} is absent from cochain index") from exc


@dataclass(frozen=True)
class GlobalSectionReport:
    """Exact extension state for a partial station section."""

    exists: bool
    unique: bool
    structural_dimension: int
    determined_components: int
    free_components: int
    conflicting_components: int


class StationSchemaSheaf:
    """Heterogeneous station-variable sheaf over a sparse locality complex.

    Each station declares the subset of the normalized variable schema that it
    can carry structurally. A simplex stalk is the intersection of its vertex
    schemas. Restrictions are sparse coordinate-selection maps, so identity and
    composition are exact by construction while allowing heterogeneous station
    capabilities.
    """

    def __init__(
        self,
        *,
        variables: Sequence[StationVariable],
        station_variable_ids: Sequence[Sequence[str]],
    ) -> None:
        self.variables = tuple(variables)
        if not self.variables:
            raise ValueError("variables must be non-empty")
        variable_ids = tuple(item.variable_id for item in self.variables)
        if len(set(variable_ids)) != len(variable_ids):
            raise ValueError("variable_ids must be unique")
        lookup = {variable_id: index for index, variable_id in enumerate(variable_ids)}
        schemas: list[tuple[int, ...]] = []
        for station_index, schema in enumerate(station_variable_ids):
            names = tuple(schema)
            if len(set(names)) != len(names):
                raise ValueError(f"station {station_index} schema contains duplicate variable ids")
            unknown = sorted(set(names) - set(lookup))
            if unknown:
                raise ValueError(f"station {station_index} schema contains unknown variables {unknown}")
            schemas.append(tuple(sorted(lookup[name] for name in names)))
        if not schemas:
            raise ValueError("station_variable_ids must be non-empty")
        self.station_bases = tuple(schemas)

    @property
    def station_count(self) -> int:
        return len(self.station_bases)

    def stalk_basis(self, simplex: Simplex) -> tuple[int, ...]:
        if not simplex:
            raise ValueError("simplex must be non-empty")
        if any(vertex < 0 or vertex >= self.station_count for vertex in simplex):
            raise ValueError("simplex references station outside sheaf")
        shared = set(self.station_bases[simplex[0]])
        for vertex in simplex[1:]:
            shared.intersection_update(self.station_bases[vertex])
        return tuple(sorted(shared))

    def cochain_index(self, simplices: Sequence[Simplex]) -> CochainIndex:
        ordered = tuple(simplices)
        if len(set(ordered)) != len(ordered):
            raise ValueError("cochain simplices must be unique")
        bases: list[tuple[int, ...]] = []
        offsets: list[int] = []
        positions: dict[Simplex, int] = {}
        offset = 0
        for position, simplex in enumerate(ordered):
            canonical = tuple(sorted(simplex))
            if canonical != simplex or len(set(simplex)) != len(simplex):
                raise ValueError(f"simplex {simplex!r} is not strictly ordered")
            positions[simplex] = position
            basis = self.stalk_basis(simplex)
            bases.append(basis)
            offsets.append(offset)
            offset += len(basis)
        return CochainIndex(
            simplices=ordered,
            bases=tuple(bases),
            offsets=tuple(offsets),
            simplex_position=positions,
            total_dimension=offset,
        )

    def required_faces(self, cofaces: Sequence[Simplex]) -> tuple[Simplex, ...]:
        faces = {
            face
            for coface in cofaces
            for face in codimension_one_faces(coface)
        }
        return tuple(sorted(faces))

    def restriction_matrix(self, face: Simplex, coface: Simplex) -> csr_matrix:
        if not set(face) < set(coface):
            raise ValueError("restriction requires a strict face inclusion")
        face_basis = self.stalk_basis(face)
        coface_basis = self.stalk_basis(coface)
        face_columns = {variable: column for column, variable in enumerate(face_basis)}
        rows = np.arange(len(coface_basis), dtype=np.int64)
        cols = np.asarray([face_columns[variable] for variable in coface_basis], dtype=np.int64)
        return coo_matrix(
            (np.ones(len(coface_basis), dtype=np.float64), (rows, cols)),
            shape=(len(coface_basis), len(face_basis)),
        ).tocsr()

    def coboundary(
        self,
        *,
        lower: CochainIndex,
        upper: CochainIndex,
    ) -> csr_matrix:
        """Assemble a sparse oriented coboundary on an explicit partition/halo."""
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        for coface in upper.simplices:
            upper_basis = upper.basis(coface)
            upper_offset = upper.offset(coface)
            for omitted, face in enumerate(codimension_one_faces(coface)):
                if face not in lower.simplex_position:
                    raise ValueError(
                        f"lower cochain index lacks required halo face {face!r}"
                    )
                lower_basis = lower.basis(face)
                lower_columns = {variable: column for column, variable in enumerate(lower_basis)}
                lower_offset = lower.offset(face)
                sign = -1.0 if omitted % 2 else 1.0
                for row_local, variable in enumerate(upper_basis):
                    rows.append(upper_offset + row_local)
                    cols.append(lower_offset + lower_columns[variable])
                    data.append(sign)
        matrix = coo_matrix(
            (np.asarray(data, dtype=np.float64), (rows, cols)),
            shape=(upper.total_dimension, lower.total_dimension),
        ).tocsr()
        matrix.sort_indices()
        return matrix


    def global_section_report(
        self,
        complex_: SparseRipsComplex,
        section: StationSection,
    ) -> GlobalSectionReport:
        """Evaluate exact global-section extension without dense nullspace algebra.

        For coordinate-selection restrictions, each normalized variable forms
        an identity sheaf on the induced subgraph of stations that structurally
        support it. Kernel dimension is therefore the number of connected
        components over those variable-specific subgraphs. A partial observed
        section extends exactly when observed values agree inside every such
        component. Components with no observed anchor remain free rather than
        being filled by interpolation or a default.
        """
        if section.station_count != self.station_count:
            raise ValueError("section station dimension does not match sheaf")
        if section.variables != self.variables:
            raise ValueError("section variable schema does not match sheaf")
        if complex_.vertex_count != self.station_count:
            raise ValueError("complex vertex count does not match sheaf")

        determined = 0
        free = 0
        conflicts = 0
        dimension = 0

        for variable_index in range(len(self.variables)):
            supported = np.asarray(
                [variable_index in basis for basis in self.station_bases],
                dtype=bool,
            )
            if np.any(section.observed[:, variable_index] & ~supported):
                raise ValueError(
                    "section marks an observation present where the station schema "
                    "does not support that variable"
                )
            vertices = np.flatnonzero(supported)
            if vertices.size == 0:
                continue
            subgraph = complex_.adjacency[vertices][:, vertices]
            component_count, labels = connected_components(
                subgraph,
                directed=False,
                return_labels=True,
            )
            dimension += int(component_count)
            for component in range(component_count):
                members = vertices[labels == component]
                mask = section.observed[members, variable_index]
                values = section.values[members[mask], variable_index]
                if values.size == 0:
                    free += 1
                elif np.all(values == values[0]):
                    determined += 1
                else:
                    conflicts += 1

        return GlobalSectionReport(
            exists=conflicts == 0,
            unique=conflicts == 0 and free == 0,
            structural_dimension=dimension,
            determined_components=determined,
            free_components=free,
            conflicting_components=conflicts,
        )

    def degree_operator(
        self,
        complex_: SparseRipsComplex,
        degree: int,
    ) -> tuple[CochainIndex, CochainIndex, csr_matrix]:
        """Convenience assembly for bounded single-partition witnesses."""
        lower_simplices = tuple(complex_.iter_simplices(degree))
        upper_simplices = tuple(complex_.iter_simplices(degree + 1))
        lower = self.cochain_index(lower_simplices)
        upper = self.cochain_index(upper_simplices)
        return lower, upper, self.coboundary(lower=lower, upper=upper)
