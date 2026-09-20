import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from architecture import codespaces_campaign


class CodespacesCampaignTests(unittest.TestCase):
    def test_discovers_exact_canonical_runtime_experiment_frontier(self) -> None:
        experiments = codespaces_campaign.discover_supported_experiments()
        self.assertEqual(
            set(experiments),
            {
                "multirepresentation.ebm_dynamics.v1",
                "multirepresentation.ebm_forced_ood.v1",
                "multirepresentation.ebm_observation_degradation.v1",
                "multirepresentation.ebm_regime_feedback.v1",
                "multirepresentation.ebm_stochastic_statistics.v1",
                "physics.two_layer_ebm.forcing_protocols.v1",
                "physics.two_layer_ebm.parameter_identifiability.v1",
            },
        )
        for experiment_id, path in experiments.items():
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["experiment_id"], experiment_id)
            self.assertEqual(path.parent, codespaces_campaign.EXPERIMENTS_DIR)

    def test_campaign_id_rejects_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "campaign id"):
                codespaces_campaign.run_campaign(
                    [],
                    campaign_id="../escape",
                    artifact_root=Path(tmp),
                    allow_dirty=False,
                )

    def test_codespaces_requires_committed_resource_authorization(self) -> None:
        with patch.dict(
            os.environ,
            {"CODESPACES": "true", "CODESPACE_NAME": "space-a"},
            clear=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "resource"):
                codespaces_campaign._codespaces_session("campaign-1", None)

    def test_safe_component_preserves_experiment_identity_characters(self) -> None:
        self.assertEqual(
            codespaces_campaign._safe_component("physics.two_layer_ebm.forcing_protocols.v1"),
            "physics.two_layer_ebm.forcing_protocols.v1",
        )


if __name__ == "__main__":
    unittest.main()
