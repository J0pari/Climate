import Mathlib.Data.Matrix.Basic
import Mathlib.Basic.Real.Basic
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith

set_option autoImplicit false

namespace ClimateFormal.SheafScalar

/-- For a scalar rank-one identity restriction on one oriented edge, the
compatibility energy expands to the ordinary graph quadratic form. -/
theorem scalar_edge_quadratic_form (a b : ℝ) :
    (-a + b)^2 = a^2 - 2*a*b + b^2 := by
  ring

/-- Reversing the edge orientation flips the residual sign but preserves its
squared energy. -/
theorem scalar_edge_orientation_invariant (a b : ℝ) :
    (-a + b)^2 = (-b + a)^2 := by
  ring

/-- Zero scalar compatibility energy is equivalent to exact agreement of the
endpoint values. This is an iff, not a one-way diagnostic implication. -/
theorem scalar_edge_zero_energy_iff_compatible (a b : ℝ) :
    (-a + b)^2 = 0 ↔ a = b := by
  constructor
  · intro h
    have hz : -a + b = 0 := sq_eq_zero_iff.mp h
    linarith
  · intro h
    subst b
    simp

/-- Simultaneously reversing every oriented residual row changes `D` to `-D`
but leaves the complete Gram operator `DᵀD` unchanged. This is the
operator-level orientation invariance used by the real cellular-sheaf
Laplacian, not merely the scalar one-edge identity. -/
theorem matrix_gram_orientation_invariant {m n : Type*} [Fintype m]
    (D : Matrix m n ℝ) :
    (-D).transpose * (-D) = D.transpose * D := by
  simp

#print axioms scalar_edge_quadratic_form
#print axioms scalar_edge_orientation_invariant
#print axioms scalar_edge_zero_energy_iff_compatible
#print axioms matrix_gram_orientation_invariant

end ClimateFormal.SheafScalar
