use std::collections::BTreeMap;

use climate_geometric_framework::sheaf::{EdgeRestriction, RealCellularSheaf, VertexStalk};
use nalgebra::DMatrix;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}

#[derive(Debug, Deserialize)]
struct Case {
    case_id: String,
    vertices: Vec<Vertex>,
    edges: Vec<Edge>,
    section: Vec<f64>,
}

#[derive(Debug, Deserialize)]
struct Vertex {
    id: String,
    dimension: usize,
}

#[derive(Debug, Deserialize)]
struct Edge {
    id: String,
    tail: String,
    head: String,
    edge_dimension: usize,
    tail_to_edge: Vec<Vec<f64>>,
    head_to_edge: Vec<Vec<f64>>,
}

fn fixture() -> Fixture {
    serde_json::from_str(include_str!(
        "../fixtures/sheaf/synthetic-linear-v1.json"
    ))
    .expect("synthetic sheaf fixture must parse")
}

fn dense(rows: &[Vec<f64>]) -> DMatrix<f64> {
    assert!(!rows.is_empty());
    let cols = rows[0].len();
    assert!(cols > 0);
    assert!(rows.iter().all(|row| row.len() == cols));
    let values: Vec<f64> = rows.iter().flat_map(|row| row.iter().copied()).collect();
    DMatrix::from_row_slice(rows.len(), cols, &values)
}

fn build_sheaf(case: &Case) -> RealCellularSheaf {
    let vertices = case
        .vertices
        .iter()
        .map(|vertex| VertexStalk::new(&vertex.id, vertex.dimension))
        .collect();
    let edges = case
        .edges
        .iter()
        .map(|edge| {
            EdgeRestriction::new(
                &edge.id,
                &edge.tail,
                &edge.head,
                edge.edge_dimension,
                dense(&edge.tail_to_edge),
                dense(&edge.head_to_edge),
            )
        })
        .collect();
    RealCellularSheaf::new(vertices, edges)
        .unwrap_or_else(|error| panic!("{}: invalid fixture sheaf: {error}", case.case_id))
}

fn vertex_offsets(case: &Case) -> BTreeMap<&str, (usize, usize)> {
    let mut offsets = BTreeMap::new();
    let mut offset = 0usize;
    for vertex in &case.vertices {
        offsets.insert(vertex.id.as_str(), (offset, vertex.dimension));
        offset += vertex.dimension;
    }
    assert_eq!(offset, case.section.len(), "{}: malformed section", case.case_id);
    offsets
}

fn apply(matrix: &[Vec<f64>], vector: &[f64]) -> Vec<f64> {
    matrix
        .iter()
        .map(|row| {
            assert_eq!(row.len(), vector.len());
            row.iter().zip(vector).map(|(a, b)| a * b).sum()
        })
        .collect()
}

/// Strong baseline: calculate local restriction residuals directly from the
/// fixture, without using the canonical sheaf's assembled coboundary matrix.
fn independent_local_residual_energy(case: &Case) -> f64 {
    let offsets = vertex_offsets(case);
    let mut energy = 0.0;

    for edge in &case.edges {
        let (tail_offset, tail_dimension) = offsets[edge.tail.as_str()];
        let (head_offset, head_dimension) = offsets[edge.head.as_str()];
        let tail = &case.section[tail_offset..tail_offset + tail_dimension];
        let head = &case.section[head_offset..head_offset + head_dimension];
        let tail_image = apply(&edge.tail_to_edge, tail);
        let head_image = apply(&edge.head_to_edge, head);

        assert_eq!(tail_image.len(), edge.edge_dimension);
        assert_eq!(head_image.len(), edge.edge_dimension);
        for (tail_value, head_value) in tail_image.iter().zip(&head_image) {
            let residual = head_value - tail_value;
            energy += residual * residual;
        }
    }

    energy
}

fn matrix_close(left: &DMatrix<f64>, right: &DMatrix<f64>, tolerance: f64) {
    assert_eq!(left.shape(), right.shape());
    for row in 0..left.nrows() {
        for col in 0..left.ncols() {
            assert!(
                (left[(row, col)] - right[(row, col)]).abs() <= tolerance,
                "matrix mismatch at ({row},{col}): {} vs {}",
                left[(row, col)],
                right[(row, col)]
            );
        }
    }
}

#[test]
fn global_energy_matches_independent_local_residual_baseline() {
    let fixture = fixture();
    assert!(!fixture.cases.is_empty(), "fixture must be non-vacuous");

    for case in &fixture.cases {
        let sheaf = build_sheaf(case);
        let canonical = sheaf.compatibility_energy(&case.section).unwrap();
        let baseline = independent_local_residual_energy(case);
        assert!(
            (canonical - baseline).abs() <= 1e-12,
            "{}: canonical energy {canonical} != independent local baseline {baseline}",
            case.case_id
        );
    }
}

#[test]
fn identity_restrictions_reduce_exactly_to_scalar_graph_laplacian() {
    let fixture = fixture();
    let case = fixture
        .cases
        .iter()
        .find(|case| case.case_id == "identity_path_clean")
        .expect("identity path case must exist");
    let sheaf = build_sheaf(case);
    let expected = DMatrix::from_row_slice(
        3,
        3,
        &[
            1.0, -1.0, 0.0,
            -1.0, 2.0, -1.0,
            0.0, -1.0, 1.0,
        ],
    );
    matrix_close(&sheaf.laplacian_0(), &expected, 1e-12);
}

#[test]
fn adding_a_known_kernel_vector_preserves_the_residual() {
    let fixture = fixture();
    let case = fixture
        .cases
        .iter()
        .find(|case| case.case_id == "nonconstant_projection_fault")
        .expect("nonconstant projection fault case must exist");
    let sheaf = build_sheaf(case);

    // The second coordinate of vertex a is annihilated by the declared
    // restriction [1, 0], so changing only that coordinate is a known ker(D)
    // perturbation for this fixture.
    let original = sheaf.compatibility_residual(&case.section).unwrap();
    let mut translated = case.section.clone();
    translated[1] += 17.0;
    let shifted = sheaf.compatibility_residual(&translated).unwrap();

    assert_eq!(original.len(), shifted.len());
    for (left, right) in original.iter().zip(shifted.iter()) {
        assert!((left - right).abs() <= 1e-12);
    }
}

#[test]
fn reversing_all_edge_orientations_preserves_laplacian_and_spectrum() {
    let fixture = fixture();
    for case in &fixture.cases {
        let forward = build_sheaf(case);
        let reversed_edges = case
            .edges
            .iter()
            .map(|edge| {
                EdgeRestriction::new(
                    format!("{}_reversed", edge.id),
                    &edge.head,
                    &edge.tail,
                    edge.edge_dimension,
                    dense(&edge.head_to_edge),
                    dense(&edge.tail_to_edge),
                )
            })
            .collect();
        let vertices = case
            .vertices
            .iter()
            .map(|vertex| VertexStalk::new(&vertex.id, vertex.dimension))
            .collect();
        let reversed = RealCellularSheaf::new(vertices, reversed_edges).unwrap();

        matrix_close(&forward.laplacian_0(), &reversed.laplacian_0(), 1e-12);
        let forward_report = forward.spectrum_report();
        let reversed_report = reversed.spectrum_report();
        assert_eq!(
            forward_report.global_section_nullity,
            reversed_report.global_section_nullity
        );
        assert!(
            (forward_report.largest_singular_value - reversed_report.largest_singular_value)
                .abs()
                <= 1e-12
        );
    }
}
