from __future__ import annotations

import json
import unittest

from reference.multirepresentation_structure_evaluation import (
    _expected_coarse_observational_structure,
    coarse_observational_structure,
    load_evaluation_fixture,
    select_world_dimensions,
)
from reference.multirepresentation_worlds import generate_worlds, load_fixture


class MultirepresentationCoarseStructureSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.world_fixture = load_fixture()
        cls.evaluation_fixture = load_evaluation_fixture()
        cls.dimension_policy = cls.evaluation_fixture["dimension_selector"]
        cls.structure_policy = cls.evaluation_fixture[
            "observational_structure_selector"
        ]
        cls.discovery = generate_worlds(cls.world_fixture)
        cls.confirmation = generate_worlds(
            cls.world_fixture,
            seed_offset=cls.evaluation_fixture["confirmation_seed_offset"],
        )

    def test_policy_declares_coarse_observational_boundary(self) -> None:
        self.assertEqual(
            self.structure_policy["family"],
            "dependence_plus_intrinsic_dimension",
        )
        boundary = self.structure_policy["interpretation_boundary"]
        for excluded in ("product", "fiber", "quotient", "stratified", "topology"):
            self.assertIn(excluded, boundary)

    def test_no_cross_view_evidence_abstains_even_if_dimensions_suggest_overlap(self) -> None:
        decision = coarse_observational_structure(
            {"decision": "abstain_no_cross_view_evidence"},
            {
                "status": "selected",
                "selected_shared_dimension": 2,
                "selected_private_dimensions": [0, 0],
            },
            self.structure_policy,
        )
        self.assertEqual(decision["status"], "abstained")
        self.assertEqual(decision["decision"], "abstain_no_cross_view_evidence")

    def test_shared_only_and_shared_plus_private_are_distinguished(self) -> None:
        shared_only = coarse_observational_structure(
            {"decision": "shared_coordinate_supported"},
            {
                "status": "selected",
                "selected_shared_dimension": 1,
                "selected_private_dimensions": [0, 0],
            },
            self.structure_policy,
        )
        shared_private = coarse_observational_structure(
            {"decision": "shared_coordinate_supported"},
            {
                "status": "selected",
                "selected_shared_dimension": 1,
                "selected_private_dimensions": [1, 2],
            },
            self.structure_policy,
        )
        self.assertEqual(
            shared_only["decision"],
            "shared_only_dimension_structure_supported",
        )
        self.assertEqual(
            shared_private["decision"],
            "shared_plus_private_dimension_structure_supported",
        )

    def test_dependence_dimension_conflict_abstains(self) -> None:
        decision = coarse_observational_structure(
            {"decision": "shared_coordinate_supported"},
            {
                "status": "selected",
                "selected_shared_dimension": 0,
                "selected_private_dimensions": [2, 2],
            },
            self.structure_policy,
        )
        self.assertEqual(decision["status"], "abstained")
        self.assertEqual(
            decision["decision"],
            "abstain_dependence_dimension_conflict",
        )

    def test_authority_scores_only_identifiable_coarse_classes(self) -> None:
        for world in self.discovery.values():
            ground_truth = world.ground_truth
            if ground_truth["shared_evaluation_target_name"] is None:
                expected = "abstain_no_cross_view_evidence"
            elif [int(value) for value in ground_truth["private_dimensions"]] == [0, 0]:
                expected = "shared_only_dimension_structure_supported"
            else:
                expected = "shared_plus_private_dimension_structure_supported"
            self.assertEqual(
                _expected_coarse_observational_structure(world),
                expected,
            )

    def test_dimension_composition_is_stable_on_discovery_and_confirmation(self) -> None:
        for world_id in sorted(self.discovery):
            expected = _expected_coarse_observational_structure(
                self.discovery[world_id]
            )
            dependence = {
                "decision": (
                    "abstain_no_cross_view_evidence"
                    if expected == "abstain_no_cross_view_evidence"
                    else "shared_coordinate_supported"
                )
            }
            for role, worlds in (
                ("discovery", self.discovery),
                ("confirmation", self.confirmation),
            ):
                dimensions = select_world_dimensions(
                    worlds[world_id], self.dimension_policy
                )
                selected = coarse_observational_structure(
                    dependence, dimensions, self.structure_policy
                )
                self.assertEqual(
                    selected["decision"],
                    expected,
                    msg=(world_id, role, selected),
                )

    def test_outputs_are_json_portable(self) -> None:
        values = [
            coarse_observational_structure(
                {"decision": "shared_coordinate_supported"},
                {
                    "status": "selected",
                    "selected_shared_dimension": 1,
                    "selected_private_dimensions": [1, 1],
                },
                self.structure_policy,
            ),
            coarse_observational_structure(
                {"decision": "abstain_no_cross_view_evidence"},
                {
                    "status": "selected",
                    "selected_shared_dimension": 0,
                    "selected_private_dimensions": [2, 2],
                },
                self.structure_policy,
            ),
        ]
        json.dumps(values, sort_keys=True, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
