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
    expected: Expected,
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

#[derive(Debug, Deserialize)]
struct Expected {
    compatibility_energy: f64,
    global_section_nullity: usize,
}

fn dense(rows: &[Vec<f64>]) -> DMatrix<f64> {
    assert!(!rows.is_empty());
    let cols = rows[0].len();
    assert!(cols > 0);
    assert!(rows.iter().all(|row| row.len() == cols));
    let values: Vec<f64> = rows.iter().flat_map(|row| row.iter().copied()).collect();
    DMatrix::from_row_slice(rows.len(), cols, &values)
}

#[test]
fn synthetic_fixture_matches_canonical_real_sheaf_operator() {
    let fixture: Fixture = serde_json::from_str(include_str!(
        "../fixtures/sheaf/synthetic-linear-v1.json"
    ))
    .expect("synthetic sheaf fixture must parse");

    assert!(!fixture.cases.is_empty(), "fixture must be non-vacuous");

    for case in fixture.cases {
        let vertices = case
            .vertices
            .into_iter()
            .map(|vertex| VertexStalk::new(vertex.id, vertex.dimension))
            .collect();
        let edges = case
            .edges
            .into_iter()
            .map(|edge| {
                EdgeRestriction::new(
                    edge.id,
                    edge.tail,
                    edge.head,
                    edge.edge_dimension,
                    dense(&edge.tail_to_edge),
                    dense(&edge.head_to_edge),
                )
            })
            .collect();

        let sheaf = RealCellularSheaf::new(vertices, edges)
            .unwrap_or_else(|error| panic!("{}: invalid fixture sheaf: {error}", case.case_id));
        let energy = sheaf
            .compatibility_energy(&case.section)
            .unwrap_or_else(|error| panic!("{}: invalid fixture section: {error}", case.case_id));
        let report = sheaf.spectrum_report();

        assert!(
            (energy - case.expected.compatibility_energy).abs() <= 1e-12,
            "{}: compatibility energy {} != expected {}",
            case.case_id,
            energy,
            case.expected.compatibility_energy
        );
        assert_eq!(
            report.global_section_nullity,
            case.expected.global_section_nullity,
            "{}: unexpected global-section nullity",
            case.case_id
        );
    }
}
