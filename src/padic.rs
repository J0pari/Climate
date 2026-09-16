//! Exact finite-precision p-adic primitives for experimental representations.
//!
//! A value here is a residue class in `Z / p^n Z`, represented by its `n`
//! low-order base-`p` digits. For two classes at the same prime and precision,
//! the valuation of their difference is the first digit position at which they
//! differ; the induced distance is `p^(-v_p(x-y))`, with identical classes at
//! distance zero.
//!
//! This module establishes the mathematics of the representation only. It does
//! **not** assert that any climate variable, region, or teleconnection has a
//! physically meaningful p-adic encoding. Encoding choices and empirical value
//! belong to separately falsifiable experiments.

use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum PadicError {
    #[error("p-adic base must be prime and at least 2; got {prime}")]
    NonPrimeBase { prime: u32 },
    #[error("finite p-adic residue precision must be at least one digit")]
    EmptyDigits,
    #[error("digit at index {index} is {digit}, but base {prime} permits only 0..={max_digit}")]
    InvalidDigit {
        index: usize,
        digit: u32,
        prime: u32,
        max_digit: u32,
    },
    #[error("p-adic comparison requires the same prime base; got {left} and {right}")]
    PrimeMismatch { left: u32, right: u32 },
    #[error("p-adic comparison requires the same finite precision; got {left} and {right}")]
    PrecisionMismatch { left: usize, right: usize },
    #[error("truncation precision must lie in 1..={available}; got {requested}")]
    InvalidTruncation { requested: usize, available: usize },
}

/// An exact symbolic p-adic distance at finite precision.
///
/// Keeping the distance in this form avoids pretending that a floating-point
/// approximation to `p^(-v)` is the mathematical object itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PadicDistance {
    Zero,
    Power { prime: u32, valuation: usize },
}

impl PadicDistance {
    pub fn as_f64(self) -> f64 {
        match self {
            Self::Zero => 0.0,
            Self::Power { prime, valuation } => (prime as f64).powi(-(valuation as i32)),
        }
    }
}

/// A residue class in `Z / p^n Z`, encoded low-order digit first.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FinitePadicResidue {
    prime: u32,
    digits: Vec<u32>,
}

impl FinitePadicResidue {
    pub fn new(prime: u32, digits: Vec<u32>) -> Result<Self, PadicError> {
        if !is_prime(prime) {
            return Err(PadicError::NonPrimeBase { prime });
        }
        if digits.is_empty() {
            return Err(PadicError::EmptyDigits);
        }
        for (index, &digit) in digits.iter().enumerate() {
            if digit >= prime {
                return Err(PadicError::InvalidDigit {
                    index,
                    digit,
                    prime,
                    max_digit: prime - 1,
                });
            }
        }
        Ok(Self { prime, digits })
    }

    pub fn prime(&self) -> u32 {
        self.prime
    }

    pub fn precision(&self) -> usize {
        self.digits.len()
    }

    pub fn digits(&self) -> &[u32] {
        &self.digits
    }

    /// Canonical quotient projection `Z / p^n Z -> Z / p^m Z` for `m <= n`.
    pub fn truncate(&self, precision: usize) -> Result<Self, PadicError> {
        if precision == 0 || precision > self.precision() {
            return Err(PadicError::InvalidTruncation {
                requested: precision,
                available: self.precision(),
            });
        }
        Ok(Self {
            prime: self.prime,
            digits: self.digits[..precision].to_vec(),
        })
    }

    /// `v_p(x-y)` for distinct finite residue classes.
    ///
    /// `None` means the two classes are identical modulo `p^n`; otherwise the
    /// returned index is the first low-order digit at which they differ.
    pub fn valuation_of_difference(&self, other: &Self) -> Result<Option<usize>, PadicError> {
        self.require_comparable(other)?;
        Ok(self
            .digits
            .iter()
            .zip(other.digits.iter())
            .position(|(left, right)| left != right))
    }

    /// Exact finite-precision p-adic metric on `Z / p^n Z`.
    pub fn distance(&self, other: &Self) -> Result<PadicDistance, PadicError> {
        let valuation = self.valuation_of_difference(other)?;
        Ok(match valuation {
            None => PadicDistance::Zero,
            Some(valuation) => PadicDistance::Power {
                prime: self.prime,
                valuation,
            },
        })
    }

    /// Whether the two classes agree modulo `p^k`.
    pub fn congruent_mod_power(&self, other: &Self, k: usize) -> Result<bool, PadicError> {
        self.require_comparable(other)?;
        if k > self.precision() {
            return Err(PadicError::InvalidTruncation {
                requested: k,
                available: self.precision(),
            });
        }
        if k == 0 {
            return Ok(true);
        }
        Ok(self.digits[..k] == other.digits[..k])
    }

    fn require_comparable(&self, other: &Self) -> Result<(), PadicError> {
        if self.prime != other.prime {
            return Err(PadicError::PrimeMismatch {
                left: self.prime,
                right: other.prime,
            });
        }
        if self.precision() != other.precision() {
            return Err(PadicError::PrecisionMismatch {
                left: self.precision(),
                right: other.precision(),
            });
        }
        Ok(())
    }
}

fn is_prime(value: u32) -> bool {
    if value < 2 {
        return false;
    }
    if value == 2 || value == 3 {
        return true;
    }
    if value % 2 == 0 || value % 3 == 0 {
        return false;
    }
    let mut divisor = 5u32;
    while divisor.saturating_mul(divisor) <= value {
        if value % divisor == 0 || value % (divisor + 2) == 0 {
            return false;
        }
        divisor += 6;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn code(prime: u32, digits: &[u32]) -> FinitePadicResidue {
        FinitePadicResidue::new(prime, digits.to_vec()).unwrap()
    }

    fn binary_residue(value: u32, precision: usize) -> FinitePadicResidue {
        let digits = (0..precision).map(|bit| (value >> bit) & 1).collect();
        FinitePadicResidue::new(2, digits).unwrap()
    }

    #[test]
    fn construction_enforces_actual_base_p_digits() {
        assert_eq!(
            FinitePadicResidue::new(4, vec![0, 1]),
            Err(PadicError::NonPrimeBase { prime: 4 })
        );
        assert_eq!(
            FinitePadicResidue::new(3, vec![]),
            Err(PadicError::EmptyDigits)
        );
        assert_eq!(
            FinitePadicResidue::new(3, vec![0, 3]),
            Err(PadicError::InvalidDigit {
                index: 1,
                digit: 3,
                prime: 3,
                max_digit: 2,
            })
        );
    }

    #[test]
    fn valuation_and_metric_match_known_p_adic_cases() {
        let base = code(3, &[1, 2, 0]);
        let differs_at_zero = code(3, &[2, 2, 0]);
        let differs_at_one = code(3, &[1, 0, 0]);
        let differs_at_two = code(3, &[1, 2, 1]);

        assert_eq!(base.valuation_of_difference(&base).unwrap(), None);
        assert_eq!(base.distance(&base).unwrap(), PadicDistance::Zero);
        assert_eq!(
            base.distance(&differs_at_zero).unwrap(),
            PadicDistance::Power {
                prime: 3,
                valuation: 0,
            }
        );
        assert_eq!(
            base.distance(&differs_at_one).unwrap(),
            PadicDistance::Power {
                prime: 3,
                valuation: 1,
            }
        );
        assert_eq!(
            base.distance(&differs_at_two).unwrap(),
            PadicDistance::Power {
                prime: 3,
                valuation: 2,
            }
        );
    }

    #[test]
    fn metric_is_symmetric_and_identity_is_exact_zero() {
        let a = code(5, &[1, 4, 2]);
        let b = code(5, &[1, 4, 3]);
        assert_eq!(a.distance(&a).unwrap(), PadicDistance::Zero);
        assert_eq!(a.distance(&b).unwrap(), b.distance(&a).unwrap());
    }

    #[test]
    fn exhaustive_small_space_satisfies_strong_triangle_inequality() {
        let residues: Vec<_> = (0..8).map(|value| binary_residue(value, 3)).collect();
        for x in &residues {
            for y in &residues {
                for z in &residues {
                    let dxz = x.distance(z).unwrap().as_f64();
                    let dxy = x.distance(y).unwrap().as_f64();
                    let dyz = y.distance(z).unwrap().as_f64();
                    assert!(
                        dxz <= dxy.max(dyz),
                        "ultrametric inequality failed: d(x,z)={dxz}, d(x,y)={dxy}, d(y,z)={dyz}"
                    );
                }
            }
        }
    }

    #[test]
    fn quotient_projection_preserves_low_order_congruence_and_is_nonexpansive() {
        let x = code(3, &[1, 2, 0, 2]);
        let y = code(3, &[1, 2, 1, 0]);
        let original = x.distance(&y).unwrap().as_f64();
        let x2 = x.truncate(2).unwrap();
        let y2 = y.truncate(2).unwrap();

        assert!(x.congruent_mod_power(&y, 2).unwrap());
        assert_eq!(x2, y2);
        assert!(x2.distance(&y2).unwrap().as_f64() <= original);
    }

    #[test]
    fn incomparable_prime_or_precision_fails_closed() {
        let p3 = code(3, &[1, 0]);
        let p5 = code(5, &[1, 0]);
        let short = code(3, &[1]);

        assert_eq!(
            p3.distance(&p5),
            Err(PadicError::PrimeMismatch { left: 3, right: 5 })
        );
        assert_eq!(
            p3.distance(&short),
            Err(PadicError::PrecisionMismatch { left: 2, right: 1 })
        );
    }
}
