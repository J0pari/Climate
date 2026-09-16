"""Exact finite-complex reference for staged sheaf/cohomology realization.

The reference deliberately separates mathematical realization from climate
interpretation.  It currently provides:

* finite abstract simplicial complexes;
* exact nerves of declared finite covers of a discrete support set;
* finite-dimensional cellular sheaves over GF(2) with explicit stalks and
  restriction maps;
* functoriality checks for restriction composition;
* block coboundary matrices and executable d^2 = 0 witnesses;
* cohomology dimensions computed by exact rank arithmetic.

None of those facts imply that a station network has been modeled by the right
cover, that climate measurements form the right stalks, or that a cohomology
class detects a scientifically meaningful defect.  Those remain separate open
obligations in ``methods/sheaf-realization.v1.json``.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Mapping


Simplex = tuple[str, ...]
Matrix = list[list[int]]
RestrictionKey = tuple[Simplex, Simplex]


def _canonical_simplex(vertices: Iterable[str]) -> Simplex:
    simplex = tuple(sorted(set(vertices)))
    if not simplex:
        raise ValueError("a simplex must contain at least one vertex")
    return simplex


def _matrix_product_mod2(left: Matrix, right: Matrix) -> Matrix:
    """Return ``left @ right`` over GF(2)."""
    if not left:
        return []
    if not right:
        return [[] for _ in left]
    shared = len(left[0])
    if any(len(row) != shared for row in left):
        raise ValueError("left matrix is ragged")
    if len(right) != shared:
        raise ValueError("matrix dimensions do not compose")
    width = len(right[0]) if right else 0
    if any(len(row) != width for row in right):
        raise ValueError("right matrix is ragged")
    return [
        [
            sum(left[i][k] * right[k][j] for k in range(shared)) % 2
            for j in range(width)
        ]
        for i in range(len(left))
    ]


def rank_mod2(matrix: Matrix) -> int:
    """Exact Gaussian-elimination rank over GF(2)."""
    if not matrix:
        return 0
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError("matrix is ragged")
    work = [[entry & 1 for entry in row] for row in matrix]
    rank = 0
    pivot_col = 0
    while rank < len(work) and pivot_col < width:
        pivot = next((r for r in range(rank, len(work)) if work[r][pivot_col]), None)
        if pivot is None:
            pivot_col += 1
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        for r in range(len(work)):
            if r != rank and work[r][pivot_col]:
                work[r] = [a ^ b for a, b in zip(work[r], work[rank])]
        rank += 1
        pivot_col += 1
    return rank


@dataclass(frozen=True)
class FiniteSimplicialComplex:
    """Finite abstract simplicial complex, stored with all non-empty faces."""

    faces: frozenset[Simplex]

    @classmethod
    def from_maximal_simplices(
        cls, maximal_simplices: Iterable[Iterable[str]]
    ) -> "FiniteSimplicialComplex":
        faces: set[Simplex] = set()
        for raw in maximal_simplices:
            maximal = _canonical_simplex(raw)
            for size in range(1, len(maximal) + 1):
                faces.update(tuple(face) for face in combinations(maximal, size))
        if not faces:
            raise ValueError("complex must contain at least one simplex")
        return cls(frozenset(faces))

    @property
    def dimension(self) -> int:
        return max(len(simplex) - 1 for simplex in self.faces)

    def simplices(self, degree: int) -> tuple[Simplex, ...]:
        if degree < 0:
            return ()
        return tuple(sorted(s for s in self.faces if len(s) == degree + 1))


@dataclass(frozen=True)
class FiniteCover:
    """Declared finite cover of a discrete support set.

    This is an exact combinatorial reference, not yet a geographic station
    coverage model.  Each cover member is represented by the support atoms it
    contains; a nerve simplex exists exactly when the corresponding members
    have non-empty common intersection.
    """

    members: Mapping[str, frozenset[str]]

    @classmethod
    def from_members(
        cls, members: Mapping[str, Iterable[str]]
    ) -> "FiniteCover":
        if not members:
            raise ValueError("finite cover must contain at least one member")
        clean: dict[str, frozenset[str]] = {}
        for name, support in members.items():
            if not isinstance(name, str) or not name:
                raise ValueError("cover member names must be non-empty strings")
            atoms = frozenset(str(atom) for atom in support)
            if not atoms:
                raise ValueError(f"cover member {name!r} has empty support")
            clean[name] = atoms
        return cls(clean)

    def nerve(self) -> FiniteSimplicialComplex:
        names = sorted(self.members)
        intersecting: list[tuple[str, ...]] = []
        for size in range(1, len(names) + 1):
            for candidate in combinations(names, size):
                common = set(self.members[candidate[0]])
                for name in candidate[1:]:
                    common.intersection_update(self.members[name])
                if common:
                    intersecting.append(candidate)
        return FiniteSimplicialComplex.from_maximal_simplices(intersecting)


@dataclass(frozen=True)
class CellularSheafGF2:
    """Finite-dimensional cellular sheaf on a simplicial complex over GF(2).

    ``stalk_dimensions[sigma]`` gives ``dim F(sigma)``.  For every strict face
    inclusion ``sigma < tau``, ``restrictions[(sigma, tau)]`` is the matrix of
    ``F(sigma <= tau): F(sigma) -> F(tau)``.  All comparable-pair maps are
    explicit so composition can be checked rather than inferred from names.
    """

    base: FiniteSimplicialComplex
    stalk_dimensions: Mapping[Simplex, int]
    restrictions: Mapping[RestrictionKey, Matrix]

    def __post_init__(self) -> None:
        faces = set(self.base.faces)
        dimensions = dict(self.stalk_dimensions)
        if set(dimensions) != faces:
            missing = sorted(faces - set(dimensions))
            extra = sorted(set(dimensions) - faces)
            raise ValueError(f"stalk dimensions do not match base; missing={missing}, extra={extra}")
        for simplex, dimension in dimensions.items():
            if not isinstance(dimension, int) or dimension <= 0:
                raise ValueError(f"stalk dimension for {simplex} must be a positive integer")

        required: set[RestrictionKey] = {
            (face, coface)
            for face in faces
            for coface in faces
            if set(face) < set(coface)
        }
        supplied = set(self.restrictions)
        if supplied != required:
            missing = sorted(required - supplied)
            extra = sorted(supplied - required)
            raise ValueError(f"restriction map surface mismatch; missing={missing}, extra={extra}")

        for (face, coface), matrix in self.restrictions.items():
            rows = dimensions[coface]
            cols = dimensions[face]
            if len(matrix) != rows or any(len(row) != cols for row in matrix):
                raise ValueError(
                    f"restriction {face}->{coface} must have shape {rows}x{cols}"
                )
            if any(entry not in {0, 1} for row in matrix for entry in row):
                raise ValueError(f"restriction {face}->{coface} contains non-GF(2) entries")

        for lower, middle in required:
            for upper in faces:
                if set(middle) < set(upper):
                    direct = self.restrictions[(lower, upper)]
                    composed = _matrix_product_mod2(
                        self.restrictions[(middle, upper)],
                        self.restrictions[(lower, middle)],
                    )
                    if direct != composed:
                        raise ValueError(
                            f"restriction composition fails for {lower} < {middle} < {upper}"
                        )

    @classmethod
    def constant_rank_one(cls, base: FiniteSimplicialComplex) -> "CellularSheafGF2":
        dimensions = {simplex: 1 for simplex in base.faces}
        restrictions: dict[RestrictionKey, Matrix] = {}
        for face in base.faces:
            for coface in base.faces:
                if set(face) < set(coface):
                    restrictions[(face, coface)] = [[1]]
        return cls(base, dimensions, restrictions)

    def cochain_dimension(self, degree: int) -> int:
        return sum(self.stalk_dimensions[s] for s in self.base.simplices(degree))

    def coboundary_matrix(self, degree: int) -> Matrix:
        """Block matrix of d^degree over GF(2).

        Characteristic two removes orientation signs.  Each codimension-one
        incidence contributes the corresponding restriction-map block.
        """
        lower = self.base.simplices(degree)
        upper = self.base.simplices(degree + 1)
        row_count = sum(self.stalk_dimensions[s] for s in upper)
        col_count = sum(self.stalk_dimensions[s] for s in lower)
        matrix = [[0 for _ in range(col_count)] for _ in range(row_count)]

        row_offset = 0
        for coface in upper:
            col_offset = 0
            for face in lower:
                if set(face) < set(coface):
                    block = self.restrictions[(face, coface)]
                    for i, row in enumerate(block):
                        for j, value in enumerate(row):
                            matrix[row_offset + i][col_offset + j] = value
                col_offset += self.stalk_dimensions[face]
            row_offset += self.stalk_dimensions[coface]
        return matrix

    def d_squared_is_zero(self, degree: int) -> bool:
        first = self.coboundary_matrix(degree)
        second = self.coboundary_matrix(degree + 1)
        if not second:
            return True
        composite = _matrix_product_mod2(second, first)
        return all(entry == 0 for row in composite for entry in row)

    def verify_complex(self) -> None:
        for degree in range(max(0, self.base.dimension - 1)):
            if not self.d_squared_is_zero(degree):
                raise AssertionError(f"d^{degree + 1} o d^{degree} != 0")

    def cohomology_dimension(self, degree: int) -> int:
        """Return dim H^degree = dim ker d_degree - dim im d_(degree-1)."""
        cochains = self.cochain_dimension(degree)
        if cochains == 0:
            return 0
        outgoing_rank = rank_mod2(self.coboundary_matrix(degree))
        incoming_rank = (
            rank_mod2(self.coboundary_matrix(degree - 1)) if degree > 0 else 0
        )
        dimension = cochains - outgoing_rank - incoming_rank
        if dimension < 0:
            raise AssertionError("cohomology dimension became negative")
        return dimension

    def cohomology_dimensions(self) -> tuple[int, ...]:
        return tuple(
            self.cohomology_dimension(k) for k in range(self.base.dimension + 1)
        )


@dataclass(frozen=True)
class ConstantCellularSheafGF2:
    """Compatibility wrapper for the constant rank-one cellular sheaf."""

    base: FiniteSimplicialComplex

    @property
    def _generic(self) -> CellularSheafGF2:
        return CellularSheafGF2.constant_rank_one(self.base)

    def coboundary_matrix(self, degree: int) -> Matrix:
        return self._generic.coboundary_matrix(degree)

    def betti_number(self, degree: int) -> int:
        return self._generic.cohomology_dimension(degree)

    def betti_numbers(self) -> tuple[int, ...]:
        return self._generic.cohomology_dimensions()

    def verify_complex(self) -> None:
        self._generic.verify_complex()
