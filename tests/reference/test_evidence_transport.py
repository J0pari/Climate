from __future__ import annotations

import json
from pathlib import Path
import unittest

from reference.evidence_transport import (
    BridgeSpec,
    EvidenceSource,
    compare_bridge,
    require_direct_evidence_class,
    synthesized_source_scope,
)


FIXTURE = Path("fixtures/evaluation/cross-model-evidence-transport-v1.json")


def load_fixture():
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    sources = []
    for item in payload["sources"]:
        obj = item["transferable_object"]
        sources.append(
            EvidenceSource(
                source_id=item["source_id"],
                model_class=item["model_class"],
                domain_of_validity=item["domain_of_validity"],
                object_kind=obj["kind"],
                unit=obj["unit"],
                coordinate=tuple(float(value) for value in obj["coordinate"]),
                values=tuple(float(value) for value in obj["values"]),
            )
        )
    item = payload["bridge"]
    bridge = BridgeSpec(
        bridge_id=item["bridge_id"],
        semantic_version=item["semantic_version"],
        source_id=item["source_id"],
        target_id=item["target_id"],
        transferable_object=item["transferable_object"],
        comparison_metric=item["comparison_metric"],
        max_normalized_rmse=float(item["max_normalized_rmse"]),
        bridge_assumption=item["bridge_assumption"],
    )
    return payload, tuple(sources), bridge


class EvidenceTransportTests(unittest.TestCase):
    def test_declared_intervention_response_bridge_is_compatible(self) -> None:
        _, sources, bridge = load_fixture()
        result = compare_bridge(sources, bridge)
        self.assertTrue(result.compatible)
        self.assertLessEqual(result.normalized_rmse, bridge.max_normalized_rmse)
        self.assertEqual(result.source.model_class, "idealized_model")
        self.assertEqual(result.target.model_class, "learned_emulator")
        self.assertEqual(
            result.interpretation,
            "bridge_compatibility_only_no_evidence_promotion",
        )

    def test_bridge_preserves_source_class_and_scope(self) -> None:
        _, sources, _ = load_fixture()
        scope = synthesized_source_scope(sources)
        self.assertEqual([item["model_class"] for item in scope], [
            "learned_emulator",
            "idealized_model",
        ])
        self.assertTrue(all(item["domain_of_validity"] for item in scope))
        self.assertNotEqual(scope[0]["domain_of_validity"], scope[1]["domain_of_validity"])

    def test_emulator_agreement_cannot_promote_parent_gcm_validation(self) -> None:
        _, sources, _ = load_fixture()
        with self.assertRaisesRegex(ValueError, "parent_gcm.*absent"):
            require_direct_evidence_class(sources, "parent_gcm")

    def test_multimodel_agreement_cannot_promote_observational_support(self) -> None:
        ensemble = EvidenceSource(
            source_id="ensemble-only",
            model_class="model_ensemble",
            domain_of_validity="Synthetic multimodel comparison only.",
            object_kind="intervention_response",
            unit="K",
            coordinate=(0.0, 1.0),
            values=(0.0, 1.0),
        )
        with self.assertRaisesRegex(ValueError, "observation.*absent"):
            require_direct_evidence_class((ensemble,), "observation")

    def test_direct_class_requirement_returns_only_actual_sources(self) -> None:
        _, sources, _ = load_fixture()
        matched = require_direct_evidence_class(sources, "learned_emulator")
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0].source_id, "emulator-response")

    def test_bridge_accepts_single_pass_source_iterable(self) -> None:
        _, sources, bridge = load_fixture()
        result = compare_bridge((item for item in sources), bridge)
        self.assertTrue(result.compatible)

    def test_duplicate_source_identity_fails_closed(self) -> None:
        _, sources, bridge = load_fixture()
        duplicate = EvidenceSource(
            source_id=sources[0].source_id,
            model_class=sources[1].model_class,
            domain_of_validity=sources[1].domain_of_validity,
            object_kind=sources[1].object_kind,
            unit=sources[1].unit,
            coordinate=sources[1].coordinate,
            values=sources[1].values,
        )
        with self.assertRaisesRegex(ValueError, "source_id values must be unique"):
            compare_bridge((sources[0], duplicate), bridge)

    def test_bridge_rejects_unit_substitution(self) -> None:
        _, sources, bridge = load_fixture()
        altered = EvidenceSource(
            source_id=sources[1].source_id,
            model_class=sources[1].model_class,
            domain_of_validity=sources[1].domain_of_validity,
            object_kind=sources[1].object_kind,
            unit="W m-2",
            coordinate=sources[1].coordinate,
            values=sources[1].values,
        )
        with self.assertRaisesRegex(ValueError, "same transferable-object unit"):
            compare_bridge((sources[0], altered), bridge)

    def test_fixture_names_forbidden_promotions_explicitly(self) -> None:
        payload, _, _ = load_fixture()
        pairs = {
            (item["from_model_class"], item["requested_model_class"])
            for item in payload["forbidden_promotions"]
        }
        self.assertIn(("learned_emulator", "parent_gcm"), pairs)
        self.assertIn(("model_ensemble", "observation"), pairs)


if __name__ == "__main__":
    unittest.main()
