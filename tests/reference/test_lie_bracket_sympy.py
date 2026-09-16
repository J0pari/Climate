from __future__ import annotations

import unittest

import sympy as sp

from reference.lie_bracket_sympy import directional_derivative, is_zero_vector, lie_bracket


class LieBracketReferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.x, self.y = sp.symbols("x y", real=True)
        self.coords = (self.x, self.y)

    def assert_vector_equal(self, actual: sp.Matrix, expected: sp.Matrix) -> None:
        self.assertTrue(is_zero_vector((actual - expected).applyfunc(sp.simplify)))

    def test_coordinate_translations_commute(self) -> None:
        tx = (sp.Integer(1), sp.Integer(0))
        ty = (sp.Integer(0), sp.Integer(1))
        self.assertTrue(is_zero_vector(lie_bracket(self.coords, tx, ty)))

    def test_translation_rotation_bracket_is_second_translation(self) -> None:
        tx = (sp.Integer(1), sp.Integer(0))
        rotation = (-self.y, self.x)
        self.assert_vector_equal(
            lie_bracket(self.coords, tx, rotation),
            sp.Matrix([0, 1]),
        )

    def test_antisymmetry_for_polynomial_vector_fields(self) -> None:
        left = (self.x**2 + self.y, self.x * self.y)
        right = (self.y**2, self.x - self.y)
        residual = lie_bracket(self.coords, left, right) + lie_bracket(
            self.coords, right, left
        )
        self.assertTrue(is_zero_vector(residual))

    def test_jacobi_identity_for_polynomial_vector_fields(self) -> None:
        x_field = (self.x**2 + self.y, self.x * self.y)
        y_field = (self.y**2, self.x - self.y)
        z_field = (self.x + self.y, self.x**2 - self.y)

        yz = lie_bracket(self.coords, y_field, z_field)
        zx = lie_bracket(self.coords, z_field, x_field)
        xy = lie_bracket(self.coords, x_field, y_field)
        jacobi = (
            lie_bracket(self.coords, x_field, tuple(yz))
            + lie_bracket(self.coords, y_field, tuple(zx))
            + lie_bracket(self.coords, z_field, tuple(xy))
        )
        self.assertTrue(is_zero_vector(jacobi))

    def test_linear_vector_fields_reduce_to_matrix_commutator(self) -> None:
        a11, a12, a21, a22 = sp.symbols("a11 a12 a21 a22", real=True)
        b11, b12, b21, b22 = sp.symbols("b11 b12 b21 b22", real=True)
        a = sp.Matrix([[a11, a12], [a21, a22]])
        b = sp.Matrix([[b11, b12], [b21, b22]])
        point = sp.Matrix([self.x, self.y])
        left = a * point
        right = b * point
        expected = (b * a - a * b) * point
        self.assert_vector_equal(
            lie_bracket(self.coords, tuple(left), tuple(right)),
            expected,
        )

    def test_leibniz_rule_distinguishes_lie_bracket_from_black_box_commutator_score(self) -> None:
        left = (sp.Integer(1), self.x)
        right = (self.y, self.x**2)
        scalar = self.x + 2 * self.y
        scaled_right = tuple(sp.expand(scalar * component) for component in right)

        actual = lie_bracket(self.coords, left, scaled_right)
        base = lie_bracket(self.coords, left, right)
        x_on_scalar = directional_derivative(self.coords, left, scalar)
        expected = scalar * base + x_on_scalar * sp.Matrix(right)
        self.assert_vector_equal(actual, expected)

    def test_dimension_mismatch_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            lie_bracket(self.coords, (1,), (1, 0))


if __name__ == "__main__":
    unittest.main()
