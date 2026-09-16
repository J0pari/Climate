"""Exact finite-complex reference for the first sheaf/cohomology realization layer.

This module implements simplicial cohomology over GF(2), equivalently the
cohomology of the constant rank-one cellular sheaf on a finite simplicial
complex.  It is deliberately narrower than a climate-data sheaf: there are no
station measurements, learned restriction maps, interpolation semantics, or
empirical claims here.

The purpose is to establish a non-negotiable mathematical kernel for later
work: a real complex, linear coboundaries, d^2 = 0, and Betti numbers computed
as dimensions of cohomology groups rather than threshold counts.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Sequence


Simplex = tuple[str, ...]
Matrix = list[list[int]]


def _canonical_simplex(vertices: Iterable[str]) -> Simplex:
    simplex = tuple(sorted(set(vertices)))
    if not simplex:
        raise ValueError("a simplex must contain at least one vertex")
    return simplex


def _matrix_product_mod2(left: Matrix, right: Matrix) -> Matrix:
    """Return left @ right over GF(2)."""
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

    def coboundary_matrix(self, degree: int) -> Matrix:
        """Matrix of d^degree: C^degree -> C^(degree+1) over GF(2).

        With constant rank-one stalks and identity restrictions, orientation
        signs disappear in characteristic two.  An entry is one exactly when
        the lower-dimensional simplex is a codimension-one face of the upper.
        """
        lower = self.simplices(degree)
        upper = self.simplices(degree + 1)
        if not upper:
            return []
        lower_sets = [set(simplex) for simplex in lower]
        return [
            [int(face.issubset(set(simplex))) for face in lower_sets]
            for simplex in upper
        ]

    def d_squared_is_zero(self, degree: int) -> bool:
        first = self.coboundary_matrix(degree)
        second = self.coboundary_matrix(degree + 1)
        if not second:
            return True
        composite = _matrix_product_mod2(second, first)
        return all(entry == 0 for row in composite for entry in row)


@dataclass(frozen=True)
class ConstantCellularSheafGF2:
    """Constant rank-one cellular sheaf on a finite simplicial complex."""

    base: FiniteSimplicialComplex

    def coboundary_matrix(self, degree: int) -> Matrix:
        return self.base.coboundary_matrix(degree)

    def betti_number(self, degree: int) -> int:
        """dim H^degree = dim ker d_degree - dim im d_(degree-1)."""
        cochains = len(self.base.simplices(degree))
        if cochains == 0:
            return 0
        outgoing_rank = rank_mod2(self.coboundary_matrix(degree))
        incoming_rank = (
            rank_mod2(self.coboundary_matrix(degree - 1)) if degree > 0 else 0
        )
        betti = cochains - outgoing_rank - incoming_rank
        if betti < 0:
            raise AssertionError("cohomology dimension became negative")
        return betti

    def betti_numbers(self) -> tuple[int, ...]:
        return tuple(self.betti_number(k) for k in range(self.base.dimension + 1))

    def verify_complex(self) -> None:
        for degree in range(max(0, self.base.dimension - 1)):
            if not self.base.d_squared_is_zero(degree):
                raise AssertionError(f"d^{degree + 1} o d^{degree} != 0")
