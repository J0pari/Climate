from __future__ import annotations

import unittest

import numpy as np

from src.station_sheaf import (
    GlobalRadiusIndex,
    SparseRipsComplex,
    StationCatalog,
    StationIdentitySheaf,
    StationSchemaSheaf,
    StationSection,
    StationVariable,
    concatenate_edge_chunks,
)


class GlobalStationSheafTests(unittest.TestCase):
    def test_antimeridian_global_index(self):
        catalog = StationCatalog(
            ("west", "east", "far"),
            np.array([0.0, 0.0, 0.0]),
            np.array([179.9, -179.9, 170.0]),
        )
        edges = concatenate_edge_chunks(
            list(GlobalRadiusIndex(catalog).iter_edges(max_distance_m=30_000.0, query_chunk_size=1))
        )
        self.assertEqual(list(zip(edges.tail.tolist(), edges.head.tolist())), [(0, 1)])
        self.assertLess(edges.distance_m[0], 25_000.0)

    def test_chunk_and_owner_partition_invariance(self):
        catalog = StationCatalog(
            ("a", "b", "c", "d"),
            np.array([35.0, 35.02, 35.04, 35.06]),
            np.array([-80.0, -80.02, -80.04, -80.06]),
        )
        index = GlobalRadiusIndex(catalog)
        whole = concatenate_edge_chunks(
            list(index.iter_edges(max_distance_m=10_000.0, query_chunk_size=1))
        )
        large = concatenate_edge_chunks(
            list(index.iter_edges(max_distance_m=10_000.0, query_chunk_size=4))
        )
        np.testing.assert_array_equal(whole.tail, large.tail)
        np.testing.assert_array_equal(whole.head, large.head)
        left = concatenate_edge_chunks(
            list(index.iter_edges(max_distance_m=10_000.0, owner_start=0, owner_stop=2))
        )
        right = concatenate_edge_chunks(
            list(index.iter_edges(max_distance_m=10_000.0, owner_start=2, owner_stop=4))
        )
        partitioned = sorted(
            list(zip(left.tail.tolist(), left.head.tolist()))
            + list(zip(right.tail.tolist(), right.head.tolist()))
        )
        self.assertEqual(partitioned, sorted(zip(whole.tail.tolist(), whole.head.tolist())))

    def test_higher_simplices_are_local(self):
        catalog = StationCatalog(
            ("a", "b", "c", "d"),
            np.array([0.0, 0.0, 0.05, 20.0]),
            np.array([0.0, 0.05, 0.0, 20.0]),
        )
        chunks = list(GlobalRadiusIndex(catalog).iter_edges(max_distance_m=9_000.0))
        complex_ = SparseRipsComplex.from_edge_chunks(catalog.station_ids, chunks)
        self.assertEqual(complex_.edge_count, 3)
        self.assertEqual(list(complex_.iter_simplices(2)), [(0, 1, 2)])

    def test_sparse_coboundary_and_missingness(self):
        catalog = StationCatalog(
            ("a", "b"),
            np.array([10.0, 10.01]),
            np.array([10.0, 10.01]),
        )
        edges = concatenate_edge_chunks(
            list(GlobalRadiusIndex(catalog).iter_edges(max_distance_m=5_000.0))
        )
        variables = (
            StationVariable("air_temperature", "K"),
            StationVariable("precipitation_amount", "mm"),
        )
        sheaf = StationIdentitySheaf(station_count=2, variables=variables)
        d0 = sheaf.coboundary_0(edges)
        self.assertEqual(d0.nnz, 2 * d0.shape[0])
        section = StationSection(
            variables,
            np.array([[280.0, 1.5], [282.0, 999.0]]),
            np.array([[True, True], [True, False]]),
        )
        residual = sheaf.residual(edges, section)
        np.testing.assert_array_equal(residual.row_indices, np.array([0]))
        np.testing.assert_allclose(residual.values, np.array([2.0]))
        self.assertEqual(residual.energy(), 4.0)

    def test_invalid_catalog_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            StationCatalog(("a", "a"), np.array([0.0, 1.0]), np.array([0.0, 1.0]))
        with self.assertRaisesRegex(ValueError, "latitude"):
            StationCatalog(("a",), np.array([91.0]), np.array([0.0]))


    def test_simplex_owner_ranges_partition_higher_cells(self):
        catalog = StationCatalog(
            station_ids=("a", "b", "c", "d"),
            latitude_deg=np.array([0.0, 0.0, 0.05, 20.0]),
            longitude_deg=np.array([0.0, 0.05, 0.0, 20.0]),
        )
        chunks = list(GlobalRadiusIndex(catalog).iter_edges(max_distance_m=9_000.0))
        complex_ = SparseRipsComplex.from_edge_chunks(catalog.station_ids, chunks)
        whole = list(complex_.iter_simplices(2))
        partitioned = (
            list(complex_.iter_simplices(2, owner_start=0, owner_stop=1))
            + list(complex_.iter_simplices(2, owner_start=1, owner_stop=4))
        )
        self.assertEqual(sorted(partitioned), sorted(whole))

    def test_heterogeneous_restrictions_are_functorial_and_sparse_d_squared_zero(self):
        catalog = StationCatalog(
            station_ids=("a", "b", "c"),
            latitude_deg=np.array([0.0, 0.0, 0.05]),
            longitude_deg=np.array([0.0, 0.05, 0.0]),
        )
        chunks = list(GlobalRadiusIndex(catalog).iter_edges(max_distance_m=9_000.0))
        complex_ = SparseRipsComplex.from_edge_chunks(catalog.station_ids, chunks)
        variables = (
            StationVariable("temperature", "K"),
            StationVariable("precipitation", "mm"),
            StationVariable("pressure", "Pa"),
        )
        sheaf = StationSchemaSheaf(
            variables=variables,
            station_variable_ids=(
                ("temperature", "precipitation"),
                ("temperature", "precipitation", "pressure"),
                ("temperature", "pressure"),
            ),
        )

        direct = sheaf.restriction_matrix((1,), (0, 1, 2))
        via_edge = (
            sheaf.restriction_matrix((0, 1), (0, 1, 2))
            @ sheaf.restriction_matrix((1,), (0, 1))
        )
        np.testing.assert_array_equal(direct.toarray(), via_edge.toarray())

        c0, c1, d0 = sheaf.degree_operator(complex_, 0)
        lower1, c2, d1 = sheaf.degree_operator(complex_, 1)
        self.assertEqual(c1.simplices, lower1.simplices)
        self.assertEqual(d0.nnz, 10)
        self.assertEqual(d1.nnz, 3)
        composite = d1 @ d0
        self.assertEqual(composite.nnz, 0)


    def test_global_section_extension_is_component_local_and_never_imputes(self):
        catalog = StationCatalog(
            station_ids=("a", "b", "c"),
            latitude_deg=np.array([0.0, 0.0, 0.05]),
            longitude_deg=np.array([0.0, 0.05, 0.0]),
        )
        chunks = list(GlobalRadiusIndex(catalog).iter_edges(max_distance_m=9_000.0))
        complex_ = SparseRipsComplex.from_edge_chunks(catalog.station_ids, chunks)
        variables = (
            StationVariable("temperature", "K"),
            StationVariable("precipitation", "mm"),
            StationVariable("pressure", "Pa"),
        )
        sheaf = StationSchemaSheaf(
            variables=variables,
            station_variable_ids=(
                ("temperature", "precipitation"),
                ("temperature", "precipitation", "pressure"),
                ("temperature", "pressure"),
            ),
        )
        compatible = StationSection(
            variables=variables,
            values=np.array(
                [
                    [280.0, 1.0, 0.0],
                    [280.0, 1.0, 100000.0],
                    [280.0, 0.0, 100000.0],
                ]
            ),
            observed=np.array(
                [
                    [True, True, False],
                    [True, True, True],
                    [True, False, True],
                ]
            ),
        )
        report = sheaf.global_section_report(complex_, compatible)
        self.assertTrue(report.exists)
        self.assertTrue(report.unique)
        self.assertEqual(report.structural_dimension, 3)
        self.assertEqual(report.determined_components, 3)
        self.assertEqual(report.free_components, 0)

        unanchored = StationSection(
            variables=variables,
            values=compatible.values,
            observed=np.array(
                [
                    [True, True, False],
                    [True, True, False],
                    [True, False, False],
                ]
            ),
        )
        report = sheaf.global_section_report(complex_, unanchored)
        self.assertTrue(report.exists)
        self.assertFalse(report.unique)
        self.assertEqual(report.free_components, 1)

        conflicting_values = np.array(compatible.values, copy=True)
        conflicting_values[2, 0] = 281.0
        conflicting = StationSection(
            variables=variables,
            values=conflicting_values,
            observed=compatible.observed,
        )
        report = sheaf.global_section_report(complex_, conflicting)
        self.assertFalse(report.exists)
        self.assertEqual(report.conflicting_components, 1)

if __name__ == "__main__":
    unittest.main()
