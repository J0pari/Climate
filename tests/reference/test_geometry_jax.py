"""Independent JAX automatic-differentiation witnesses for analytic geometry fixtures."""
from __future__ import annotations

import math
import unittest
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import sympy as sp

from reference.geometry_sympy import load_fixture_manifest, parse_metric_fixture


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "fixtures" / "geometry" / "analytic-fixtures.json"

REPRESENTATIVE_POINTS = {
    "flat.cartesian.2d": (0.31, -0.27),
    "flat.polar.2d": (1.2, 0.7),
    "flat.quadratic_shear.2d": (0.31, -0.27),
    "sphere.radius2.2d": (1.0, 0.7),
    "hyperbolic.poincare.radius1.2d": (0.2, -0.25),
}


def _metric_callable(fixture: dict, *, precision: str):
    if precision == "float64":
        dtype = jnp.float64
    elif precision == "float32":
        dtype = jnp.float32
    else:
        raise ValueError(f"unsupported JAX geometry precision: {precision}")

    coordinates, metric = parse_metric_fixture(fixture)
    elements = [
        [sp.lambdify(coordinates, metric[i, j], modules="jax") for j in range(metric.cols)]
        for i in range(metric.rows)
    ]

    def evaluate(point: jax.Array) -> jax.Array:
        args = tuple(point[index] for index in range(point.shape[0]))
        return jnp.stack(
            [
                jnp.stack(
                    [
                        jnp.asarray(elements[i][j](*args), dtype=dtype)
                        for j in range(metric.cols)
                    ]
                )
                for i in range(metric.rows)
            ]
        )

    return coordinates, metric, evaluate, dtype


def _jax_metric_jet(
    fixture: dict,
    point: tuple[float, ...],
    *,
    precision: str = "float64",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coordinates, metric, evaluate, dtype = _metric_callable(
        fixture,
        precision=precision,
    )
    values = np.asarray(point, dtype=float)
    if values.shape != (len(coordinates),) or not np.isfinite(values).all():
        raise ValueError("JAX metric-derivative point must be finite and match fixture dimension")

    argument = jnp.asarray(values, dtype=dtype)
    metric_value = np.asarray(evaluate(argument))
    first_raw = np.asarray(jax.jacfwd(evaluate)(argument))
    second_raw = np.asarray(jax.jacfwd(jax.jacfwd(evaluate))(argument))

    first = np.moveaxis(first_raw, -1, 0)
    second = np.transpose(second_raw, (2, 3, 0, 1))
    if (
        metric_value.shape != metric.shape
        or first.shape != (len(coordinates), metric.rows, metric.cols)
        or second.shape
        != (len(coordinates), len(coordinates), metric.rows, metric.cols)
    ):
        raise RuntimeError("JAX metric derivative returned an unexpected tensor shape")
    if not (
        np.isfinite(metric_value).all()
        and np.isfinite(first).all()
        and np.isfinite(second).all()
    ):
        raise RuntimeError("JAX metric derivative produced non-finite values")
    return metric_value, first, second


def _exact_metric_jet(
    fixture: dict,
    point: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coordinates, metric = parse_metric_fixture(fixture)
    substitutions = {
        coordinate: sp.Float(value, 40)
        for coordinate, value in zip(coordinates, point, strict=True)
    }
    n = len(coordinates)
    metric_value = np.empty((n, n), dtype=float)
    first = np.empty((n, n, n), dtype=float)
    second = np.empty((n, n, n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            metric_value[i, j] = float(sp.N(metric[i, j].subs(substitutions), 30))
            for k in range(n):
                first[k, i, j] = float(
                    sp.N(sp.diff(metric[i, j], coordinates[k]).subs(substitutions), 30)
                )
                for l in range(n):
                    second[k, l, i, j] = float(
                        sp.N(
                            sp.diff(
                                metric[i, j],
                                coordinates[k],
                                coordinates[l],
                            ).subs(substitutions),
                            30,
                        )
                    )
    return metric_value, first, second


class GeometryJaxDerivativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixtures = {
            item["fixture_id"]: item
            for item in load_fixture_manifest(MANIFEST)["fixtures"]
        }

    def assert_jet_close(
        self,
        fixture_id: str,
        point: tuple[float, float],
        *,
        rtol: float = 5e-12,
        atol: float = 5e-12,
    ) -> None:
        fixture = self.fixtures[fixture_id]
        observed = _jax_metric_jet(fixture, point, precision="float64")
        expected = _exact_metric_jet(fixture, point)
        for observed_component, expected_component in zip(
            observed,
            expected,
            strict=True,
        ):
            np.testing.assert_allclose(
                observed_component,
                expected_component,
                rtol=rtol,
                atol=atol,
            )

    def test_fp64_jacfwd_agrees_with_exact_sympy_on_all_analytic_fixtures(self) -> None:
        for fixture_id, point in REPRESENTATIVE_POINTS.items():
            with self.subTest(fixture_id=fixture_id):
                self.assert_jet_close(fixture_id, point)

    def test_fp64_jacfwd_agrees_at_multiple_nonlinear_points(self) -> None:
        cases = {
            "sphere.radius2.2d": (
                (0.35, 0.1),
                (1.4, 2.1),
                (math.pi - 0.35, 5.2),
            ),
            "hyperbolic.poincare.radius1.2d": (
                (-0.45, 0.20),
                (0.10, 0.10),
                (0.40, -0.30),
            ),
        }
        for fixture_id, points in cases.items():
            for point in points:
                with self.subTest(fixture_id=fixture_id, point=point):
                    self.assert_jet_close(fixture_id, point)

    def test_fp32_remains_an_explicit_distinct_precision(self) -> None:
        metric, first, second = _jax_metric_jet(
            self.fixtures["sphere.radius2.2d"],
            (1.0, 0.7),
            precision="float32",
        )
        self.assertEqual(metric.dtype, np.dtype("float32"))
        self.assertEqual(first.dtype, np.dtype("float32"))
        self.assertEqual(second.dtype, np.dtype("float32"))

    def test_nonfinite_derivative_output_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            _jax_metric_jet(
                self.fixtures["hyperbolic.poincare.radius1.2d"],
                (1.0, 0.0),
                precision="float64",
            )

    def test_nonfinite_point_and_unknown_precision_fail_closed(self) -> None:
        fixture = self.fixtures["flat.cartesian.2d"]
        with self.assertRaisesRegex(ValueError, "finite"):
            _jax_metric_jet(fixture, (float("nan"), 0.0))
        with self.assertRaisesRegex(ValueError, "unsupported"):
            _jax_metric_jet(fixture, (0.0, 0.0), precision="float16")


if __name__ == "__main__":
    unittest.main()
