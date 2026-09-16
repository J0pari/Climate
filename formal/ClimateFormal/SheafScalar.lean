import Mathlib

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
endpoint values.  This is an iff, not a one-way diagnostic implication. -/
theorem scalar_edge_zero_energy_iff_compatible (a b : ℝ) :
    (-a + b)^2 = 0 ↔ a = b := by
  constructor
  · intro h
    nlinarith [sq_nonneg (-a + b)]
  · intro h
    subst b
    ring

#print axioms scalar_edge_quadratic_form
#print axioms scalar_edge_orientation_invariant
#print axioms scalar_edge_zero_energy_iff_compatible

end ClimateFormal.SheafScalar
