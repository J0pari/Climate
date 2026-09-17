from __future__ import annotations

import unittest

import numpy as np

from reference.two_layer_energy_balance import (
    TwoLayerParameters,
    advance_constant_forcing,
    energy_budget_residual_w_m2,
    equilibrium_state_k,
    flux_coordinates_w_m2,
    forcing_from_fixture,
    heat_capacity_metric,
    initial_state_from_fixture,
    load_fixture,
    modal_coordinates,
    mode_basis,
    ocean_heat_uptake_w_m2,
    parameters_from_fixture,
    reservoir_storage_fluxes_w_m2,
    state_from_flux_coordinates_k,
    state_from_modal_coordinates_k,
    system_matrix,
    tendency_k_per_year,
    toa_imbalance_w_m2,
    trajectory_constant_forcing,
)


class TwoLayerEnergyBalanceReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = load_fixture()
        cls.parameters = parameters_from_fixture(cls.fixture)
        cls.forcing = forcing_from_fixture(cls.fixture)
        cls.initial = initial_state_from_fixture(cls.fixture)

    def test_fixture_parameters_are_literature_fixed_values(self) -> None:
        self.assertEqual(self.parameters.surface_heat_capacity_w_yr_m2_k, 7.3)
        self.assertEqual(self.parameters.deep_heat_capacity_w_yr_m2_k, 106.0)
        self.assertEqual(self.parameters.ocean_heat_exchange_w_m2_k, 0.73)
        self.assertEqual(self.parameters.climate_feedback_w_m2_k, 1.13)
        self.assertEqual(self.forcing, 1.0)

    def test_global_energy_budget_closes_and_exchange_cancels(self) -> None:
        state = np.array([0.42, 0.11])
        storage = reservoir_storage_fluxes_w_m2(state, self.parameters, self.forcing)
        ocean_flux = ocean_heat_uptake_w_m2(state, self.parameters)
        toa = toa_imbalance_w_m2(state, self.parameters, self.forcing)

        self.assertAlmostEqual(storage[0], toa - ocean_flux, places=14)
        self.assertAlmostEqual(storage[1], ocean_flux, places=14)
        self.assertAlmostEqual(storage.sum(), toa, places=14)
        self.assertAlmostEqual(
            energy_budget_residual_w_m2(state, self.parameters, self.forcing),
            0.0,
            places=14,
        )

    def test_equilibrium_has_zero_tendency_and_zero_ocean_exchange(self) -> None:
        equilibrium = equilibrium_state_k(self.parameters, self.forcing)
        expected = self.forcing / self.parameters.climate_feedback_w_m2_k
        np.testing.assert_allclose(equilibrium, [expected, expected], rtol=0.0, atol=0.0)
        np.testing.assert_allclose(
            tendency_k_per_year(equilibrium, self.parameters, self.forcing),
            [0.0, 0.0],
            rtol=0.0,
            atol=2e-17,
        )
        self.assertAlmostEqual(
            ocean_heat_uptake_w_m2(equilibrium, self.parameters), 0.0, places=14
        )

    def test_exact_constant_forcing_advance_has_semigroup_property(self) -> None:
        state = np.array([0.18, -0.04])
        first = advance_constant_forcing(state, self.parameters, self.forcing, 7.25)
        composed = advance_constant_forcing(first, self.parameters, self.forcing, 19.5)
        direct = advance_constant_forcing(state, self.parameters, self.forcing, 26.75)
        np.testing.assert_allclose(composed, direct, rtol=3e-14, atol=3e-14)

    def test_flux_representation_round_trip_is_exact_to_roundoff(self) -> None:
        state = np.array([0.51, 0.17])
        flux = flux_coordinates_w_m2(state, self.parameters, self.forcing)
        recovered = state_from_flux_coordinates_k(flux, self.parameters, self.forcing)
        np.testing.assert_allclose(recovered, state, rtol=0.0, atol=2e-16)

    def test_system_matrix_is_self_adjoint_in_heat_capacity_metric(self) -> None:
        matrix = system_matrix(self.parameters)
        metric = heat_capacity_metric(self.parameters)
        np.testing.assert_allclose(
            metric @ matrix,
            matrix.T @ metric,
            rtol=2e-14,
            atol=2e-14,
        )

    def test_modal_basis_is_heat_capacity_orthonormal_and_diagonalizes_dynamics(self) -> None:
        timescales, basis = mode_basis(self.parameters)
        metric = heat_capacity_metric(self.parameters)
        matrix = system_matrix(self.parameters)
        decay_rates = -1.0 / timescales

        np.testing.assert_allclose(
            basis.T @ metric @ basis,
            np.eye(2),
            rtol=2e-13,
            atol=2e-13,
        )
        np.testing.assert_allclose(
            matrix @ basis,
            basis @ np.diag(decay_rates),
            rtol=2e-13,
            atol=2e-13,
        )

    def test_modal_representation_round_trip_and_timescale_order(self) -> None:
        state = np.array([0.37, 0.09])
        timescales, basis = mode_basis(self.parameters)
        coordinates = modal_coordinates(state, self.parameters, self.forcing)
        recovered = state_from_modal_coordinates_k(
            coordinates, self.parameters, self.forcing
        )

        self.assertEqual(basis.shape, (2, 2))
        self.assertTrue(np.all(timescales > 0.0))
        self.assertLess(timescales[0], timescales[1])
        np.testing.assert_allclose(recovered, state, rtol=0.0, atol=2e-15)

    def test_modal_coordinates_are_heat_capacity_weighted_projection(self) -> None:
        state = np.array([0.37, 0.09])
        equilibrium = equilibrium_state_k(self.parameters, self.forcing)
        _, basis = mode_basis(self.parameters)
        metric = heat_capacity_metric(self.parameters)
        coordinates = modal_coordinates(state, self.parameters, self.forcing)
        expected = basis.T @ metric @ (state - equilibrium)
        np.testing.assert_allclose(coordinates, expected, rtol=2e-14, atol=2e-14)

    def test_direct_trajectory_matches_exact_pointwise_advance(self) -> None:
        times = np.array([0.0, 1.0, 10.0, 100.0, 500.0])
        trajectory = trajectory_constant_forcing(
            self.initial, self.parameters, self.forcing, times
        )
        for index, time in enumerate(times):
            expected = advance_constant_forcing(
                self.initial, self.parameters, self.forcing, float(time)
            )
            np.testing.assert_allclose(
                trajectory[index], expected, rtol=2e-14, atol=2e-14
            )

    def test_invalid_physical_parameters_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            TwoLayerParameters(
                surface_heat_capacity_w_yr_m2_k=0.0,
                deep_heat_capacity_w_yr_m2_k=106.0,
                ocean_heat_exchange_w_m2_k=0.73,
                climate_feedback_w_m2_k=1.13,
            )


if __name__ == "__main__":
    unittest.main()
