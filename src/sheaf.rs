//! Real-valued finite cellular-sheaf operators for Climate diagnostics.
//!
//! This module is deliberately mathematical/numerical rather than interpretive.
//! A sheaf is supplied as finite-dimensional vertex and edge stalks together
//! with explicit linear restriction maps. From those data we derive the
//! degree-zero coboundary `D`, compatibility residuals `D x`, the sheaf
//! Laplacian `L0 = D^T D`, and SVD-based nullity/gap diagnostics.
//!
//! No threshold in this module means "fault", "tipping", or scientific
//! significance. Policy and empirical interpretation belong to callers and
//! experiments. The ordinary scalar graph Laplacian is a strict special case:
//! rank-one stalks with identity restrictions at both ends of every edge.

use std::collections::{BTreeMap, BTreeSet};

use nalgebra::{DMatrix, DVector};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VertexStalk {
    pub id: String,
    pub dimension: usize,
}

impl VertexStalk {
    pub fn new(id: impl Into<String>, dimension: usize) -> Self {
        Self {
            id: id.into(),
            dimension,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EdgeRestriction {
    pub id: String,
    pub tail: String,
    pub head: String,
    pub edge_dimension: usize,
    /// Linear map F(tail) -> F(edge), shape edge_dimension x dim(tail).
    pub tail_to_edge: DMatrix<f64>,
    /// Linear map F(head) -> F(edge), shape edge_dimension x dim(head).
    pub head_to_edge: DMatrix<f64>,
}

impl EdgeRestriction {
    pub fn new(
        id: impl Into<String>,
        tail: impl Into<String>,
        head: impl Into<String>,
        edge_dimension: usize,
        tail_to_edge: DMatrix<f64>,
        head_to_edge: DMatrix<f64>,
    ) -> Self {
        Self {
            id: id.into(),
            tail: tail.into(),
            head: head.into(),
            edge_dimension,
            tail_to_edge,
            head_to_edge,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SheafSpectrumReport {
    pub vertex_degrees_of_freedom: usize,
    pub edge_degrees_of_freedom: usize,
    pub numerical_rank: usize,
    /// Numerical dimension of ker(D), i.e. compatible/global sections at the
    /// supplied numerical precision.
    pub global_section_nullity: usize,
    pub numerical_rank_tolerance: f64,
    pub largest_singular_value: f64,
    pub smallest_positive_singular_value: Option<f64>,
    /// Smallest positive eigenvalue of D^T D, obtained as sigma_min_positive^2.
    pub smallest_positive_laplacian_eigenvalue: Option<f64>,
}

#[derive(Debug, Error, Clone, PartialEq)]
pub enum SheafError {
    #[error("a cellular sheaf requires at least one vertex stalk")]
    NoVertices,
    #[error("{kind} id must be non-empty")]
    EmptyId { kind: &'static str },
    #[error("duplicate vertex id {id:?}")]
    DuplicateVertex { id: String },
    #[error("duplicate edge id {id:?}")]
    DuplicateEdge { id: String },
    #[error("vertex {id:?} has invalid zero-dimensional stalk")]
    ZeroVertexDimension { id: String },
    #[error("edge {id:?} has invalid zero-dimensional stalk")]
    ZeroEdgeDimension { id: String },
    #[error("edge {edge:?} references unknown {endpoint} vertex {vertex:?}")]
    UnknownVertex {
        edge: String,
        endpoint: &'static str,
        vertex: String,
    },
    #[error("edge {edge:?} is a self-loop on {vertex:?}; simplicial graph edges require distinct endpoints")]
    SelfLoop { edge: String, vertex: String },
    #[error(
        "restriction {edge:?}:{endpoint} has shape {actual_rows}x{actual_cols}; expected {expected_rows}x{expected_cols}"
    )]
    RestrictionShape {
        edge: String,
        endpoint: &'static str,
        expected_rows: usize,
        expected_cols: usize,
        actual_rows: usize,
        actual_cols: usize,
    },
    #[error("restriction {edge:?}:{endpoint} contains non-finite value at row {row}, column {col}: {value}")]
    NonFiniteRestriction {
        edge: String,
        endpoint: &'static str,
        row: usize,
        col: usize,
        value: f64,
    },
    #[error("section has length {actual}; expected {expected} vertex degrees of freedom")]
    SectionDimension { expected: usize, actual: usize },
    #[error("section contains non-finite value at index {index}: {value}")]
    NonFiniteSection { index: usize, value: f64 },
}

#[derive(Debug, Clone)]
pub struct RealCellularSheaf {
    vertices: Vec<VertexStalk>,
    edges: Vec<EdgeRestriction>,
    vertex_offsets: BTreeMap<String, usize>,
    vertex_dimensions: BTreeMap<String, usize>,
    vertex_dof: usize,
    edge_dof: usize,
}

impl RealCellularSheaf {
    pub fn new(
        vertices: Vec<VertexStalk>,
        edges: Vec<EdgeRestriction>,
    ) -> Result<Self, SheafError> {
        if vertices.is_empty() {
            return Err(SheafError::NoVertices);
        }

        let mut vertex_offsets = BTreeMap::new();
        let mut vertex_dimensions = BTreeMap::new();
        let mut vertex_dof = 0usize;

        for vertex in &vertices {
            if vertex.id.is_empty() {
                return Err(SheafError::EmptyId { kind: "vertex" });
            }
            if vertex.dimension == 0 {
                return Err(SheafError::ZeroVertexDimension {
                    id: vertex.id.clone(),
                });
            }
            if vertex_offsets.contains_key(&vertex.id) {
                return Err(SheafError::DuplicateVertex {
                    id: vertex.id.clone(),
                });
            }
            vertex_offsets.insert(vertex.id.clone(), vertex_dof);
            vertex_dimensions.insert(vertex.id.clone(), vertex.dimension);
            vertex_dof += vertex.dimension;
        }

        let mut seen_edges = BTreeSet::new();
        let mut edge_dof = 0usize;
        for edge in &edges {
            if edge.id.is_empty() {
                return Err(SheafError::EmptyId { kind: "edge" });
            }
            if !seen_edges.insert(edge.id.clone()) {
                return Err(SheafError::DuplicateEdge {
                    id: edge.id.clone(),
                });
            }
            if edge.edge_dimension == 0 {
                return Err(SheafError::ZeroEdgeDimension {
                    id: edge.id.clone(),
                });
            }
            if edge.tail == edge.head {
                return Err(SheafError::SelfLoop {
                    edge: edge.id.clone(),
                    vertex: edge.tail.clone(),
                });
            }

            let tail_dimension = *vertex_dimensions.get(&edge.tail).ok_or_else(|| {
                SheafError::UnknownVertex {
                    edge: edge.id.clone(),
                    endpoint: "tail",
                    vertex: edge.tail.clone(),
                }
            })?;
            let head_dimension = *vertex_dimensions.get(&edge.head).ok_or_else(|| {
                SheafError::UnknownVertex {
                    edge: edge.id.clone(),
                    endpoint: "head",
                    vertex: edge.head.clone(),
                }
            })?;

            Self::validate_restriction(
                edge,
                "tail",
                tail_dimension,
                &edge.tail_to_edge,
            )?;
            Self::validate_restriction(
                edge,
                "head",
                head_dimension,
                &edge.head_to_edge,
            )?;
            edge_dof += edge.edge_dimension;
        }

        Ok(Self {
            vertices,
            edges,
            vertex_offsets,
            vertex_dimensions,
            vertex_dof,
            edge_dof,
        })
    }

    fn validate_restriction(
        edge: &EdgeRestriction,
        endpoint: &'static str,
        vertex_dimension: usize,
        matrix: &DMatrix<f64>,
    ) -> Result<(), SheafError> {
        if matrix.nrows() != edge.edge_dimension || matrix.ncols() != vertex_dimension {
            return Err(SheafError::RestrictionShape {
                edge: edge.id.clone(),
                endpoint,
                expected_rows: edge.edge_dimension,
                expected_cols: vertex_dimension,
                actual_rows: matrix.nrows(),
                actual_cols: matrix.ncols(),
            });
        }
        for row in 0..matrix.nrows() {
            for col in 0..matrix.ncols() {
                let value = matrix[(row, col)];
                if !value.is_finite() {
                    return Err(SheafError::NonFiniteRestriction {
                        edge: edge.id.clone(),
                        endpoint,
                        row,
                        col,
                        value,
                    });
                }
            }
        }
        Ok(())
    }

    pub fn vertex_degrees_of_freedom(&self) -> usize {
        self.vertex_dof
    }

    pub fn edge_degrees_of_freedom(&self) -> usize {
        self.edge_dof
    }

    pub fn vertices(&self) -> &[VertexStalk] {
        &self.vertices
    }

    pub fn edges(&self) -> &[EdgeRestriction] {
        &self.edges
    }

    /// Construct the oriented degree-zero coboundary D.
    ///
    /// For edge e = (tail -> head), the edge block is
    ///     -R_tail * x_tail + R_head * x_head.
    /// Reversing edge orientation multiplies that row block by -1 and therefore
    /// leaves D^T D and all singular values invariant.
    pub fn coboundary_0(&self) -> DMatrix<f64> {
        let mut matrix = DMatrix::<f64>::zeros(self.edge_dof, self.vertex_dof);
        let mut row_offset = 0usize;

        for edge in &self.edges {
            let tail_offset = self.vertex_offsets[&edge.tail];
            let head_offset = self.vertex_offsets[&edge.head];
            let tail_dimension = self.vertex_dimensions[&edge.tail];
            let head_dimension = self.vertex_dimensions[&edge.head];

            for row in 0..edge.edge_dimension {
                for col in 0..tail_dimension {
                    matrix[(row_offset + row, tail_offset + col)] =
                        -edge.tail_to_edge[(row, col)];
                }
                for col in 0..head_dimension {
                    matrix[(row_offset + row, head_offset + col)] =
                        edge.head_to_edge[(row, col)];
                }
            }
            row_offset += edge.edge_dimension;
        }
        matrix
    }

    pub fn laplacian_0(&self) -> DMatrix<f64> {
        let d0 = self.coboundary_0();
        d0.transpose() * d0
    }

    fn validated_section(&self, section: &[f64]) -> Result<DVector<f64>, SheafError> {
        if section.len() != self.vertex_dof {
            return Err(SheafError::SectionDimension {
                expected: self.vertex_dof,
                actual: section.len(),
            });
        }
        for (index, &value) in section.iter().enumerate() {
            if !value.is_finite() {
                return Err(SheafError::NonFiniteSection { index, value });
            }
        }
        Ok(DVector::from_column_slice(section))
    }

    /// Exact numerical compatibility residual D x for a supplied local section.
    pub fn compatibility_residual(&self, section: &[f64]) -> Result<DVector<f64>, SheafError> {
        let section = self.validated_section(section)?;
        Ok(self.coboundary_0() * section)
    }

    /// Squared residual norm ||D x||^2. This is a measured incompatibility
    /// energy, not a probability or policy decision.
    pub fn compatibility_energy(&self, section: &[f64]) -> Result<f64, SheafError> {
        let residual = self.compatibility_residual(section)?;
        Ok(residual.dot(&residual))
    }

    /// SVD-derived structural report for D. The tolerance scales with machine
    /// precision, matrix size, and sigma_max; no fixed singular-value cutoff is
    /// embedded here.
    pub fn spectrum_report(&self) -> SheafSpectrumReport {
        let d0 = self.coboundary_0();
        let singular_values = d0.clone().svd(false, false).singular_values;
        let largest = singular_values.iter().copied().fold(0.0_f64, f64::max);
        let tolerance = f64::EPSILON * d0.nrows().max(d0.ncols()) as f64 * largest;
        let numerical_rank = singular_values
            .iter()
            .filter(|&&sigma| sigma > tolerance)
            .count();
        let smallest_positive = singular_values
            .iter()
            .copied()
            .filter(|&sigma| sigma > tolerance)
            .fold(None, |acc: Option<f64>, sigma| {
                Some(acc.map_or(sigma, |current| current.min(sigma)))
            });

        SheafSpectrumReport {
            vertex_degrees_of_freedom: self.vertex_dof,
            edge_degrees_of_freedom: self.edge_dof,
            numerical_rank,
            global_section_nullity: self.vertex_dof - numerical_rank,
            numerical_rank_tolerance: tolerance,
            largest_singular_value: largest,
            smallest_positive_singular_value: smallest_positive,
            smallest_positive_laplacian_eigenvalue: smallest_positive.map(|sigma| sigma * sigma),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar_edge(id: &str, tail: &str, head: &str) -> EdgeRestriction {
        EdgeRestriction::new(
            id,
            tail,
            head,
            1,
            DMatrix::from_row_slice(1, 1, &[1.0]),
            DMatrix::from_row_slice(1, 1, &[1.0]),
        )
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
    fn scalar_constant_sheaf_recovers_graph_laplacian() {
        let sheaf = RealCellularSheaf::new(
            vec![VertexStalk::new("a", 1), VertexStalk::new("b", 1)],
            vec![scalar_edge("ab", "a", "b")],
        )
        .unwrap();

        matrix_close(
            &sheaf.coboundary_0(),
            &DMatrix::from_row_slice(1, 2, &[-1.0, 1.0]),
            1e-12,
        );
        matrix_close(
            &sheaf.laplacian_0(),
            &DMatrix::from_row_slice(2, 2, &[1.0, -1.0, -1.0, 1.0]),
            1e-12,
        );
        assert_eq!(sheaf.compatibility_energy(&[3.0, 3.0]).unwrap(), 0.0);
        assert_eq!(sheaf.compatibility_energy(&[3.0, 4.0]).unwrap(), 1.0);
    }

    #[test]
    fn nonconstant_restrictions_change_what_compatibility_means() {
        let sheaf = RealCellularSheaf::new(
            vec![VertexStalk::new("a", 2), VertexStalk::new("b", 1)],
            vec![EdgeRestriction::new(
                "ab",
                "a",
                "b",
                1,
                DMatrix::from_row_slice(1, 2, &[1.0, 0.0]),
                DMatrix::from_row_slice(1, 1, &[1.0]),
            )],
        )
        .unwrap();

        assert_eq!(sheaf.compatibility_energy(&[5.0, 99.0, 5.0]).unwrap(), 0.0);
        assert_eq!(sheaf.compatibility_energy(&[4.0, 99.0, 5.0]).unwrap(), 1.0);
        let report = sheaf.spectrum_report();
        assert_eq!(report.numerical_rank, 1);
        assert_eq!(report.global_section_nullity, 2);
    }

    #[test]
    fn orientation_changes_d_but_not_laplacian_or_spectrum() {
        let forward = RealCellularSheaf::new(
            vec![VertexStalk::new("a", 1), VertexStalk::new("b", 1)],
            vec![EdgeRestriction::new(
                "ab",
                "a",
                "b",
                1,
                DMatrix::from_row_slice(1, 1, &[2.0]),
                DMatrix::from_row_slice(1, 1, &[3.0]),
            )],
        )
        .unwrap();
        let reversed = RealCellularSheaf::new(
            vec![VertexStalk::new("a", 1), VertexStalk::new("b", 1)],
            vec![EdgeRestriction::new(
                "ab",
                "b",
                "a",
                1,
                DMatrix::from_row_slice(1, 1, &[3.0]),
                DMatrix::from_row_slice(1, 1, &[2.0]),
            )],
        )
        .unwrap();

        matrix_close(&forward.laplacian_0(), &reversed.laplacian_0(), 1e-12);
        let a = forward.spectrum_report();
        let b = reversed.spectrum_report();
        assert!((a.largest_singular_value - b.largest_singular_value).abs() < 1e-12);
        assert_eq!(a.global_section_nullity, b.global_section_nullity);
    }

    #[test]
    fn disconnected_components_increase_global_section_nullity_for_constant_sheaf() {
        let sheaf = RealCellularSheaf::new(
            vec![
                VertexStalk::new("a", 1),
                VertexStalk::new("b", 1),
                VertexStalk::new("c", 1),
            ],
            vec![scalar_edge("ab", "a", "b")],
        )
        .unwrap();
        let report = sheaf.spectrum_report();
        assert_eq!(report.numerical_rank, 1);
        assert_eq!(report.global_section_nullity, 2);
    }

    #[test]
    fn compatibility_energy_matches_quadratic_form() {
        let sheaf = RealCellularSheaf::new(
            vec![VertexStalk::new("a", 1), VertexStalk::new("b", 1)],
            vec![scalar_edge("ab", "a", "b")],
        )
        .unwrap();
        let x = DVector::from_column_slice(&[2.0, 5.0]);
        let quadratic = (x.transpose() * sheaf.laplacian_0() * &x)[(0, 0)];
        let energy = sheaf.compatibility_energy(x.as_slice()).unwrap();
        assert!((quadratic - energy).abs() < 1e-12);
    }

    #[test]
    fn malformed_restrictions_and_sections_fail_closed() {
        let error = RealCellularSheaf::new(
            vec![VertexStalk::new("a", 2), VertexStalk::new("b", 1)],
            vec![EdgeRestriction::new(
                "ab",
                "a",
                "b",
                1,
                DMatrix::from_row_slice(1, 1, &[1.0]),
                DMatrix::from_row_slice(1, 1, &[1.0]),
            )],
        )
        .unwrap_err();
        assert!(matches!(error, SheafError::RestrictionShape { endpoint: "tail", .. }));

        let sheaf = RealCellularSheaf::new(
            vec![VertexStalk::new("a", 1), VertexStalk::new("b", 1)],
            vec![scalar_edge("ab", "a", "b")],
        )
        .unwrap();
        assert!(matches!(
            sheaf.compatibility_energy(&[1.0]),
            Err(SheafError::SectionDimension { .. })
        ));
        assert!(matches!(
            sheaf.compatibility_energy(&[1.0, f64::NAN]),
            Err(SheafError::NonFiniteSection { .. })
        ));
    }
}
