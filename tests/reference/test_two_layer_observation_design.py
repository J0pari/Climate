from __future__ import annotations

from copy import deepcopy
import unittest

import numpy as np

from reference.two_layer_energy_balance import load_fixture
from reference.two_layer_observation_design import (
    analyze_observation_design,
    design_dominates,
    load_design_fixture,
)
from reference.two_layer_observation_information import load_observation_fixture


class TwoLayerObservationDesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ebm = load_fixture()
        cls.observations = load_observation_fixture()
        cls.design_control = load_design_fixture()
        cls.result = analyze_observation_design(
            cls.ebm,
            cls.observations,
            cls.design_control,
        )

    def _by_channels(self, result=None):
        payload = self.result if result is None else result
        return {
            frozenset(item["channels"]): item
            for item in payload["designs"]
        }

    def test_design_control_is_equal_cost_and_explicitly_synthetic(self) -> None:
        self.assertEqual(
            self.design_control["provenance"]["kind"],
            "synthetic_structural_control",
        )
        costs = list(self.design_control["channel_costs"].values())
        self.assertEqual(len(costs), 3)
        self.assertTrue(all(cost == costs[0] for cost in costs))
        self.assertGreater(costs[0], 0.0)

    def test_all_nonempty_channel_subsets_are_evaluated(self) -> None:
        self.assertEqual(len(self.result["designs"]), 7)
        expected = {
            frozenset(("surface_temperature",)),
            frozenset(("toa_imbalance",)),
            frozenset(("ocean_heat_uptake",)),
            frozenset(("surface_temperature", "toa_imbalance")),
            frozenset(("surface_temperature", "ocean_heat_uptake")),
            frozenset(("toa_imbalance", "ocean_heat_uptake")),
            frozenset(("surface_temperature", "toa_imbalance", "ocean_heat_uptake")),
        }
        self.assertEqual(set(self._by_channels()), expected)

    def test_rank_exposes_redundant_surface_and_toa_information(self) -> None:
        designs = self._by_channels()
        self.assertEqual(designs[frozenset(("surface_temperature",))]["state_rank"], 1)
        self.assertEqual(designs[frozenset(("toa_imbalance",))]["state_rank"], 1)
        self.assertEqual(designs[frozenset(("ocean_heat_uptake",))]["state_rank"], 1)
        self.assertEqual(
            designs[frozenset(("surface_temperature", "toa_imbalance"))]["state_rank"],
            1,
        )
        self.assertEqual(
            designs[frozenset(("surface_temperature", "ocean_heat_uptake"))]["state_rank"],
            2,
        )
        self.assertEqual(
            designs[frozenset(("toa_imbalance", "ocean_heat_uptake"))]["state_rank"],
            2,
        )
        self.assertEqual(
            designs[
                frozenset(("surface_temperature", "toa_imbalance", "ocean_heat_uptake"))
            ]["state_rank"],
            2,
        )

    def test_added_independent_channels_never_reduce_fisher_information(self) -> None:
        designs = self._by_channels()
        for smaller_channels, smaller in designs.items():
            smaller_fisher = np.asarray(smaller["state_fisher"], dtype=float)
            for larger_channels, larger in designs.items():
                if smaller_channels < larger_channels:
                    increment = np.asarray(larger["state_fisher"], dtype=float) - smaller_fisher
                    eigenvalues = np.linalg.eigvalsh(increment)
                    self.assertGreaterEqual(
                        float(eigenvalues.min()),
                        -2e-12,
                        msg=f"{sorted(smaller_channels)} -> {sorted(larger_channels)}",
                    )

    def test_pareto_projection_contains_exactly_nondominated_designs(self) -> None:
        designs = self.result["designs"]
        frontier = set(self.result["pareto_design_ids"])
        self.assertTrue(frontier)
        for design in designs:
            dominators = [
                other
                for other in designs
                if other["design_id"] != design["design_id"]
                and design_dominates(other, design)
            ]
            if design["design_id"] in frontier:
                self.assertEqual(dominators, [], msg=design["design_id"])
            else:
                self.assertTrue(dominators, msg=design["design_id"])

    def test_uniform_cost_rescaling_preserves_pareto_membership(self) -> None:
        scaled = analyze_observation_design(
            self.ebm,
            self.observations,
            self.design_control,
            cost_multiplier=7.0,
        )
        self.assertEqual(self.result["pareto_design_ids"], scaled["pareto_design_ids"])
        base = {item["design_id"]: item for item in self.result["designs"]}
        changed = {item["design_id"]: item for item in scaled["designs"]}
        for design_id in base:
            self.assertAlmostEqual(changed[design_id]["cost"], 7.0 * base[design_id]["cost"])
            np.testing.assert_allclose(
                changed[design_id]["state_fisher"],
                base[design_id]["state_fisher"],
                rtol=0.0,
                atol=0.0,
            )

    def test_uniform_noise_rescaling_preserves_frontier_and_scales_information(self) -> None:
        doubled_noise = analyze_observation_design(
            self.ebm,
            self.observations,
            self.design_control,
            noise_multiplier=2.0,
        )
        self.assertEqual(self.result["pareto_design_ids"], doubled_noise["pareto_design_ids"])
        base = {item["design_id"]: item for item in self.result["designs"]}
        changed = {item["design_id"]: item for item in doubled_noise["designs"]}
        for design_id in base:
            self.assertEqual(changed[design_id]["state_rank"], base[design_id]["state_rank"])
            np.testing.assert_allclose(
                changed[design_id]["state_fisher"],
                np.asarray(base[design_id]["state_fisher"]) / 4.0,
                rtol=2e-14,
                atol=2e-14,
            )
            self.assertAlmostEqual(
                changed[design_id]["fast_mode_information"],
                base[design_id]["fast_mode_information"] / 4.0,
                places=12,
            )
            self.assertAlmostEqual(
                changed[design_id]["slow_mode_information"],
                base[design_id]["slow_mode_information"] / 4.0,
                places=12,
            )
            self.assertAlmostEqual(
                changed[design_id]["normalized_cross_mode_coupling"],
                base[design_id]["normalized_cross_mode_coupling"],
                places=12,
            )

    def test_cost_coverage_and_multipliers_fail_closed(self) -> None:
        missing = deepcopy(self.design_control)
        del missing["channel_costs"]["ocean_heat_uptake"]
        with self.assertRaisesRegex(ValueError, "exactly cover"):
            analyze_observation_design(self.ebm, self.observations, missing)
        with self.assertRaisesRegex(ValueError, "cost_multiplier"):
            analyze_observation_design(
                self.ebm,
                self.observations,
                self.design_control,
                cost_multiplier=0.0,
            )
        with self.assertRaisesRegex(ValueError, "noise_multiplier"):
            analyze_observation_design(
                self.ebm,
                self.observations,
                self.design_control,
                noise_multiplier=float("nan"),
            )


if __name__ == "__main__":
    unittest.main()
