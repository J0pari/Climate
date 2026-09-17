from __future__ import annotations

import unittest

import numpy as np

from reference.two_layer_energy_balance import load_fixture
from reference.two_layer_observation_information import (
    gaussian_mean_fisher_reference,
    load_observation_fixture,
    observation_jacobian,
    observation_noise_stddev,
)
from reference.two_layer_dynamic_observability import (
    linear_observability_report,
    sampled_initial_state_fisher,
    two_layer_observability_report,
)


class TwoLayerDynamicObservabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ebm = load_fixture()
        cls.observations = load_observation_fixture()

    def test_single_time_sample_reduces_to_instantaneous_fisher(self) -> None:
        channels = ("surface_temperature", "ocean_heat_uptake")
        instantaneous = gaussian_mean_fisher_reference(
            observation_jacobian(self.ebm, channels),
            observation_noise_stddev(self.observations, channels),
        )
        sampled = sampled_initial_state_fisher(
            self.ebm,
            self.observations,
            channels,
            (0.0,),
        )
        np.testing.assert_allclose(sampled["fisher"], instantaneous["fisher"], rtol=0.0, atol=0.0)
        self.assertEqual(sampled["rank"], instantaneous["rank"])
        self.assertEqual(sampled["nullity"], instantaneous["nullity"])

    def test_repeated_same_time_adds_precision_without_rank(self) -> None:
        channels = ("surface_temperature",)
        single = sampled_initial_state_fisher(
            self.ebm,
            self.observations,
            channels,
            (0.0,),
        )
        repeated = sampled_initial_state_fisher(
            self.ebm,
            self.observations,
            channels,
            (0.0, 0.0),
        )
        self.assertEqual(single["rank"], 1)
        self.assertEqual(repeated["rank"], 1)
        np.testing.assert_allclose(repeated["fisher"], 2.0 * single["fisher"], rtol=2e-14, atol=2e-14)

    def test_each_single_channel_is_dynamically_observable_in_the_coupled_control(self) -> None:
        for channel in (
            "surface_temperature",
            "toa_imbalance",
            "ocean_heat_uptake",
        ):
            structural = two_layer_observability_report(self.ebm, (channel,))
            sampled = sampled_initial_state_fisher(
                self.ebm,
                self.observations,
                (channel,),
                (0.0, 10.0),
            )
            self.assertEqual(structural["rank"], 2, msg=channel)
            self.assertEqual(structural["nullity"], 0, msg=channel)
            self.assertEqual(sampled["rank"], 2, msg=channel)
            self.assertEqual(sampled["nullity"], 0, msg=channel)
            self.assertTrue(np.isfinite(sampled["condition_number_2"]), msg=channel)

    def test_sample_order_does_not_change_initial_state_fisher(self) -> None:
        first = sampled_initial_state_fisher(
            self.ebm,
            self.observations,
            ("surface_temperature",),
            (0.0, 1.0, 10.0),
        )
        permuted = sampled_initial_state_fisher(
            self.ebm,
            self.observations,
            ("surface_temperature",),
            (10.0, 0.0, 1.0),
        )
        np.testing.assert_allclose(first["fisher"], permuted["fisher"], rtol=2e-14, atol=2e-14)

    def test_adding_independent_sample_is_information_monotone(self) -> None:
        first = sampled_initial_state_fisher(
            self.ebm,
            self.observations,
            ("surface_temperature",),
            (0.0,),
        )
        expanded = sampled_initial_state_fisher(
            self.ebm,
            self.observations,
            ("surface_temperature",),
            (0.0, 10.0),
        )
        increment = expanded["fisher"] - first["fisher"]
        eigenvalues = np.linalg.eigvalsh(increment)
        self.assertGreaterEqual(float(eigenvalues.min()), -2e-13)

    def test_decoupled_scalar_sensor_is_a_true_unobservable_counterexample(self) -> None:
        report = linear_observability_report(
            np.diag([-1.0, -2.0]),
            np.array([[1.0, 0.0]], dtype=float),
        )
        self.assertEqual(report["rank"], 1)
        self.assertEqual(report["nullity"], 1)

    def test_invalid_sample_times_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "nonempty"):
            sampled_initial_state_fisher(
                self.ebm,
                self.observations,
                ("surface_temperature",),
                (),
            )
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            sampled_initial_state_fisher(
                self.ebm,
                self.observations,
                ("surface_temperature",),
                (0.0, -1.0),
            )


if __name__ == "__main__":
    unittest.main()
