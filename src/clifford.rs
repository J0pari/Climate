//! Sparse Euclidean Clifford algebra `Cl(n, 0)` for experimental representations.
//!
//! This is the actual algebraic kernel: basis vectors square to `+1`, distinct
//! generators anticommute, basis blades multiply by the Clifford geometric
//! product, and sparse multivectors compose bilinearly. Climate interpretation
//! and claims of incremental value over complex/spectral representations are
//! deliberately outside this module.

use std::collections::BTreeMap;
use thiserror::Error;

const MAX_GENERATORS: u8 = 16;

#[derive(Debug, Error, Clone, PartialEq)]
pub enum CliffordError {
    #[error("Clifford generator count {generators} exceeds supported sparse-kernel limit {max}")]
    TooManyGenerators { generators: u8, max: u8 },
    #[error("basis-vector index {index} is outside Cl({generators},0)")]
    InvalidGenerator { index: u8, generators: u8 },
    #[error("basis blade mask {mask:#x} uses generators outside Cl({generators},0)")]
    InvalidBlade { mask: u64, generators: u8 },
    #[error("multivectors belong to different Clifford algebras: {left} and {right} generators")]
    AlgebraMismatch { left: u8, right: u8 },
    #[error("non-finite multivector coefficient for blade {mask:#x}: {coefficient}")]
    NonFiniteCoefficient { mask: u64, coefficient: f64 },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Multivector {
    generators: u8,
    terms: BTreeMap<u64, f64>,
}

impl Multivector {
    pub fn generators(&self) -> u8 {
        self.generators
    }

    pub fn terms(&self) -> &BTreeMap<u64, f64> {
        &self.terms
    }

    pub fn coefficient(&self, blade_mask: u64) -> f64 {
        self.terms.get(&blade_mask).copied().unwrap_or(0.0)
    }

    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EuclideanClifford {
    generators: u8,
}

impl EuclideanClifford {
    pub fn new(generators: u8) -> Result<Self, CliffordError> {
        if generators > MAX_GENERATORS {
            return Err(CliffordError::TooManyGenerators {
                generators,
                max: MAX_GENERATORS,
            });
        }
        Ok(Self { generators })
    }

    pub fn generators(&self) -> u8 {
        self.generators
    }

    pub fn basis_blade_count(&self) -> usize {
        1usize << self.generators
    }

    pub fn scalar(&self, value: f64) -> Result<Multivector, CliffordError> {
        self.multivector([(0, value)])
    }

    pub fn basis_vector(&self, index: u8) -> Result<Multivector, CliffordError> {
        if index >= self.generators {
            return Err(CliffordError::InvalidGenerator {
                index,
                generators: self.generators,
            });
        }
        self.basis_blade(1u64 << index, 1.0)
    }

    pub fn basis_blade(&self, mask: u64, coefficient: f64) -> Result<Multivector, CliffordError> {
        self.multivector([(mask, coefficient)])
    }

    pub fn multivector(
        &self,
        terms: impl IntoIterator<Item = (u64, f64)>,
    ) -> Result<Multivector, CliffordError> {
        let valid_mask = if self.generators == 64 {
            u64::MAX
        } else if self.generators == 0 {
            0
        } else {
            (1u64 << self.generators) - 1
        };
        let mut normalized = BTreeMap::new();
        for (mask, coefficient) in terms {
            if mask & !valid_mask != 0 {
                return Err(CliffordError::InvalidBlade {
                    mask,
                    generators: self.generators,
                });
            }
            if !coefficient.is_finite() {
                return Err(CliffordError::NonFiniteCoefficient { mask, coefficient });
            }
            if coefficient != 0.0 {
                *normalized.entry(mask).or_insert(0.0) += coefficient;
            }
        }
        normalized.retain(|_, coefficient| *coefficient != 0.0);
        Ok(Multivector {
            generators: self.generators,
            terms: normalized,
        })
    }

    pub fn geometric_product(
        &self,
        left: &Multivector,
        right: &Multivector,
    ) -> Result<Multivector, CliffordError> {
        self.require_same_algebra(left, right)?;
        let mut terms = BTreeMap::<u64, f64>::new();
        for (&left_mask, &left_coefficient) in &left.terms {
            for (&right_mask, &right_coefficient) in &right.terms {
                let (sign, mask) = blade_geometric_product(left_mask, right_mask);
                *terms.entry(mask).or_insert(0.0) +=
                    sign * left_coefficient * right_coefficient;
            }
        }
        self.multivector(terms)
    }

    pub fn exterior_product(
        &self,
        left: &Multivector,
        right: &Multivector,
    ) -> Result<Multivector, CliffordError> {
        self.require_same_algebra(left, right)?;
        let mut terms = BTreeMap::<u64, f64>::new();
        for (&left_mask, &left_coefficient) in &left.terms {
            for (&right_mask, &right_coefficient) in &right.terms {
                if left_mask & right_mask != 0 {
                    continue;
                }
                let (sign, mask) = blade_geometric_product(left_mask, right_mask);
                *terms.entry(mask).or_insert(0.0) +=
                    sign * left_coefficient * right_coefficient;
            }
        }
        self.multivector(terms)
    }

    fn require_same_algebra(
        &self,
        left: &Multivector,
        right: &Multivector,
    ) -> Result<(), CliffordError> {
        if left.generators != right.generators {
            return Err(CliffordError::AlgebraMismatch {
                left: left.generators,
                right: right.generators,
            });
        }
        if left.generators != self.generators {
            return Err(CliffordError::AlgebraMismatch {
                left: self.generators,
                right: left.generators,
            });
        }
        Ok(())
    }
}

/// Product of canonical Euclidean basis blades.
///
/// The output blade is XOR of generator masks. The sign is the parity of the
/// swaps required to move the concatenated generators back to canonical order;
/// duplicate generators cancel via `e_i^2 = +1`.
fn blade_geometric_product(left: u64, right: u64) -> (f64, u64) {
    let mut swaps = 0u32;
    let mut remaining = left;
    while remaining != 0 {
        let index = remaining.trailing_zeros();
        remaining &= remaining - 1;
        let lower_mask = if index == 0 { 0 } else { (1u64 << index) - 1 };
        swaps += (right & lower_mask).count_ones();
    }
    let sign = if swaps % 2 == 0 { 1.0 } else { -1.0 };
    (sign, left ^ right)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn product(algebra: &EuclideanClifford, a: u64, b: u64) -> Multivector {
        let a = algebra.basis_blade(a, 1.0).unwrap();
        let b = algebra.basis_blade(b, 1.0).unwrap();
        algebra.geometric_product(&a, &b).unwrap()
    }

    #[test]
    fn generators_square_to_positive_one_and_distinct_generators_anticommute() {
        let algebra = EuclideanClifford::new(4).unwrap();
        for i in 0..4 {
            let ei = algebra.basis_vector(i).unwrap();
            let square = algebra.geometric_product(&ei, &ei).unwrap();
            assert_eq!(square.coefficient(0), 1.0);
            assert_eq!(square.terms().len(), 1);

            for j in (i + 1)..4 {
                let ej = algebra.basis_vector(j).unwrap();
                let ij = algebra.geometric_product(&ei, &ej).unwrap();
                let ji = algebra.geometric_product(&ej, &ei).unwrap();
                let mask = (1u64 << i) | (1u64 << j);
                assert_eq!(ij.coefficient(mask), 1.0);
                assert_eq!(ji.coefficient(mask), -1.0);
            }
        }
    }

    #[test]
    fn bivector_square_is_minus_one() {
        let algebra = EuclideanClifford::new(2).unwrap();
        let e0e1 = product(&algebra, 0b01, 0b10);
        let square = algebra.geometric_product(&e0e1, &e0e1).unwrap();
        assert_eq!(square.coefficient(0), -1.0);
        assert_eq!(square.terms().len(), 1);
    }

    #[test]
    fn geometric_product_is_associative_on_every_basis_triple_in_cl4() {
        let algebra = EuclideanClifford::new(4).unwrap();
        for a in 0u64..16 {
            for b in 0u64..16 {
                for c in 0u64..16 {
                    let av = algebra.basis_blade(a, 1.0).unwrap();
                    let bv = algebra.basis_blade(b, 1.0).unwrap();
                    let cv = algebra.basis_blade(c, 1.0).unwrap();
                    let ab = algebra.geometric_product(&av, &bv).unwrap();
                    let bc = algebra.geometric_product(&bv, &cv).unwrap();
                    let left = algebra.geometric_product(&ab, &cv).unwrap();
                    let right = algebra.geometric_product(&av, &bc).unwrap();
                    assert_eq!(left, right, "associativity failed for {a:#x}, {b:#x}, {c:#x}");
                }
            }
        }
    }

    #[test]
    fn exterior_product_is_alternating_on_generators() {
        let algebra = EuclideanClifford::new(3).unwrap();
        let e0 = algebra.basis_vector(0).unwrap();
        let e1 = algebra.basis_vector(1).unwrap();
        assert!(algebra.exterior_product(&e0, &e0).unwrap().is_zero());
        let e0e1 = algebra.exterior_product(&e0, &e1).unwrap();
        let e1e0 = algebra.exterior_product(&e1, &e0).unwrap();
        assert_eq!(e0e1.coefficient(0b11), 1.0);
        assert_eq!(e1e0.coefficient(0b11), -1.0);
    }

    #[test]
    fn simple_multivector_identity_uses_full_geometric_product() {
        let algebra = EuclideanClifford::new(1).unwrap();
        let one_plus_e = algebra.multivector([(0, 1.0), (1, 1.0)]).unwrap();
        let one_minus_e = algebra.multivector([(0, 1.0), (1, -1.0)]).unwrap();
        let product = algebra
            .geometric_product(&one_plus_e, &one_minus_e)
            .unwrap();
        assert!(product.is_zero());
    }

    #[test]
    fn invalid_basis_and_nonfinite_coefficients_fail_closed() {
        let algebra = EuclideanClifford::new(2).unwrap();
        assert_eq!(
            algebra.basis_vector(2),
            Err(CliffordError::InvalidGenerator {
                index: 2,
                generators: 2,
            })
        );
        assert!(matches!(
            algebra.basis_blade(0b100, 1.0),
            Err(CliffordError::InvalidBlade { .. })
        ));
        assert!(matches!(
            algebra.scalar(f64::NAN),
            Err(CliffordError::NonFiniteCoefficient { .. })
        ));
    }
}
