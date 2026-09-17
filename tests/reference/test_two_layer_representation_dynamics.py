from __future__ import annotations

import unittest

import numpy as np

from reference.two_layer_dynamic_observability import two_layer_observability_report
from reference.two_layer_energy_balance import load_fixture
from reference.two_layer_representation_dynamics import (
    analyze_representations,
    decay_timescales_years,
    representation_pairs,
)


class TwoLayerRepresentationDynamicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = load_fixture()
        cls.result = analyze_representations(cls.fixture, dt_years=1.0)

    def test_full_rank_views_preserve_both_exact_thermal_modes(self) -> None:
        exact = np.asarray(self.result["exact_timescales_years"])
        for name in (
            "temperature_state",
            "heat_flux",
            "thermal_modes",
            "temperature_plus_flux",
            "all_full_rank_views",
        ):
            diagnostics = self.result["representations"][name]
            recovered = np.asarray(diagnostics["dmd_timescales_years"])
            self.assertEqual(diagnostics["matrix_rank"], 2, msg=name)
            self.assertEqual(recovered.shape, (2,), msg=name)
            np.testing.assert_allclose(
                recovered,
                exact,
                rtol=2e-11,
                atol=2e-11,
                err_msg=name,
            )
            self.assertLess(
                diagnostics["one_step_relative_error"], 2e-12, msg=name
            )

    def test_redundant_concatenations_add_dimensions_but_not_dynamical_rank(self) -> None:
        temperature_flux = self.result["representations"]["temperature_plus_flux"]
        all_views = self.result["representations"]["all_full_rank_views"]

        self.assertEqual(temperature_flux["dimension"], 4)
        self.assertEqual(temperature_flux["matrix_rank"], 2)
        self.assertEqual(temperature_flux["redundant_dimension_count"], 2)
        self.assertGreater(temperature_flux["condition_number"], 1e12)

        self.assertEqual(all_views["dimension"], 6)
        self.assertEqual(all_views["matrix_rank"], 2)
        self.assertEqual(all_views["redundant_dimension_count"], 4)
        self.assertGreater(all_views["condition_number"], 1e12)

    def test_scalar_views_are_structurally_unable_to_resolve_two_modes(self) -> None:
        for name in ("surface_temperature_scalar", "toa_imbalance_scalar"):
            diagnostics = self.result["representations"][name]
            self.assertEqual(diagnostics["dimension"], 1, msg=name)
            self.assertEqual(diagnostics["matrix_rank"], 1, msg=name)
            self.assertEqual(diagnostics["redundant_dimension_count"], 0, msg=name)
            self.assertEqual(diagnostics["resolved_mode_count"], 1, msg=name)
            self.assertEqual(len(diagnostics["dmd_timescales_years"]), 1, msg=name)

    def test_scalar_markov_closure_is_not_dynamic_unobservability(self) -> None:
        cases = (
            ("surface_temperature_scalar", ("surface_temperature",)),
            ("toa_imbalance_scalar", ("toa_imbalance",)),
        )
        for representation_name, channels in cases:
            dmd = self.result["representations"][representation_name]
            observability = two_layer_observability_report(self.fixture, channels)

            self.assertEqual(dmd["dimension"], 1, msg=representation_name)
            self.assertEqual(dmd["resolved_mode_count"], 1, msg=representation_name)
            self.assertEqual(observability["rank"], 2, msg=representation_name)
            self.assertEqual(observability["nullity"], 0, msg=representation_name)

    def test_paired_view_samples_share_identical_physical_realizations(self) -> None:
        pairs = representation_pairs(self.fixture, dt_years=1.0)
        sample_count = pairs["temperature_state"][0].shape[0]
        for x, y in pairs.values():
            self.assertEqual(x.shape[0], sample_count)
            self.assertEqual(y.shape[0], sample_count)
            self.assertTrue(np.isfinite(x).all())
            self.assertTrue(np.isfinite(y).all())

    def test_invalid_decay_spectrum_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly between zero and one"):
            decay_timescales_years(np.array([1.0]), 1.0)
        with self.assertRaisesRegex(ValueError, "expected to be real"):
            decay_timescales_years(np.array([0.9 + 0.1j]), 1.0)


if __name__ == "__main__":
    unittest.main()
