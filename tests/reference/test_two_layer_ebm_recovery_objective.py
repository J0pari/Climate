from __future__ import annotations

import math
import unittest

import numpy as np

from reference.information_geometry_ebm_recovery import load_recovery_fixture
from reference.two_layer_ebm_recovery_objective import (
    gaussian_nll_per_observation,
    log_parameter_bounds,
    parameter_vector,
    protocol_temperature_outputs,
    standardized_gaussian_temperature_jacobian,
    standardized_gaussian_temperature_residual,
)
from reference.two_layer_energy_balance import (
    load_fixture as load_ebm_fixture,
    parameters_from_fixture,
)
from reference.two_layer_forcing_protocols import load_protocol_fixture


class TwoLayerEbmRecoveryObjectiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ebm = load_ebm_fixture()
        self.protocols = load_protocol_fixture()
        self.recovery = load_recovery_fixture()
        self.truth = parameters_from_fixture(self.ebm)
        self.truth_log = np.log(parameter_vector(self.truth))
        self.discovery_ids = list(self.recovery["discovery_protocol_ids"])
        self.samples_per_segment = int(self.recovery["samples_per_segment"])
        self.noise_std = float(self.recovery["observation_noise_std_k"])
        self.observed = protocol_temperature_outputs(
            self.truth,
            self.protocols,
            self.discovery_ids,
            samples_per_segment=self.samples_per_segment,
        )

    def test_noiseless_truth_has_zero_standardized_residual(self) -> None:
        residual = standardized_gaussian_temperature_residual(
            self.truth_log,
            self.observed,
            self.protocols,
            self.discovery_ids,
            samples_per_segment=self.samples_per_segment,
            noise_std_k=self.noise_std,
        )
        np.testing.assert_allclose(residual, 0.0, atol=1e-13, rtol=0.0)

    def test_gaussian_nll_normalization_is_explicit(self) -> None:
        value = gaussian_nll_per_observation(
            np.zeros(self.observed.size, dtype=float),
            self.noise_std,
        )
        expected = 0.5 * (
            math.log(2.0 * math.pi) + 2.0 * math.log(self.noise_std)
        )
        self.assertAlmostEqual(value, expected, places=14)

    def test_common_log_bounds_contain_every_preregistered_start(self) -> None:
        lower, upper = log_parameter_bounds(
            self.truth,
            float(self.recovery["log_parameter_bound_factor"]),
        )
        for offset in self.recovery["start_log_parameter_offsets"]:
            start = self.truth_log + np.asarray(offset, dtype=float)
            self.assertTrue(np.all(start >= lower))
            self.assertTrue(np.all(start <= upper))

    def test_objective_jacobian_matches_independent_directional_difference(self) -> None:
        step = float(self.recovery["centered_log_jacobian_step"])
        jacobian = standardized_gaussian_temperature_jacobian(
            self.truth_log,
            self.observed.size,
            self.protocols,
            self.discovery_ids,
            samples_per_segment=self.samples_per_segment,
            noise_std_k=self.noise_std,
            log_step=step,
        )
        direction = np.array([0.31, -0.27, 0.19, 0.41], dtype=float)
        direction /= np.linalg.norm(direction)
        directional_step = 2.0e-5
        plus = standardized_gaussian_temperature_residual(
            self.truth_log + directional_step * direction,
            self.observed,
            self.protocols,
            self.discovery_ids,
            samples_per_segment=self.samples_per_segment,
            noise_std_k=self.noise_std,
        )
        minus = standardized_gaussian_temperature_residual(
            self.truth_log - directional_step * direction,
            self.observed,
            self.protocols,
            self.discovery_ids,
            samples_per_segment=self.samples_per_segment,
            noise_std_k=self.noise_std,
        )
        finite_difference = (plus - minus) / (2.0 * directional_step)
        np.testing.assert_allclose(
            jacobian @ direction,
            finite_difference,
            rtol=2.0e-6,
            atol=2.0e-7,
        )

    def test_missing_protocol_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not resolve"):
            protocol_temperature_outputs(
                self.truth,
                self.protocols,
                ["not-a-protocol"],
                samples_per_segment=self.samples_per_segment,
            )


if __name__ == "__main__":
    unittest.main()
