"""Witness the public Commons -> Climate read-execution harness."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from architecture import commons_live_witness


class CommonsLiveWitnessTests(unittest.TestCase):
    def _fixture(self, root: Path, revision: str) -> tuple[Path, str]:
        experiment = root / "experiments" / "fixture.json"
        experiment.parent.mkdir(parents=True)
        experiment.write_text(json.dumps({
            "experiment_id": "fixture.experiment.v1",
        }), encoding="utf-8")
        output_rel = "run-artifacts/commons/fixture-scope"
        output = root / output_rel
        output.mkdir(parents=True)
        (output / "outcome.json").write_text(json.dumps({
            "experiment_id": "fixture.experiment.v1",
            "runs": [{
                "run_id": "fixture-run",
                "repository_revision": revision,
            }],
        }), encoding="utf-8")
        return experiment, output_rel

    def test_public_lifecycle_and_climate_outcome_complete_witness(self):
        revision = "a" * 40
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment, output_rel = self._fixture(root, revision)
            ack = {
                "jobId": "job-1",
                "status": "queued",
                "outputDir": output_rel,
            }
            with patch.object(commons_live_witness, "ROOT", root), \
                 patch.object(
                     commons_live_witness.commons_control,
                     "submit_cpu_experiment",
                     return_value=ack,
                 ), \
                 patch.object(
                     commons_live_witness.commons_control,
                     "inspect_job",
                     side_effect=[
                         {"status": "queued", "exitCode": None},
                         {"status": "done", "exitCode": 0},
                     ],
                 ), \
                 patch.object(commons_live_witness.time, "sleep"):
                result = commons_live_witness.run_live_witness(
                    experiment_path=experiment,
                    repository_revision=revision,
                    run_scope="fixture-scope",
                    ram_mib=1024,
                    max_minutes=1.0,
                    priority=0,
                    poll_seconds=0.01,
                    timeout_seconds=5.0,
                )
            self.assertEqual(result["jobStatus"], "done")
            self.assertEqual(result["experimentId"], "fixture.experiment.v1")

    def test_failed_commons_job_never_becomes_climate_evidence(self):
        revision = "b" * 40
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment, output_rel = self._fixture(root, revision)
            with patch.object(commons_live_witness, "ROOT", root), \
                 patch.object(
                     commons_live_witness.commons_control,
                     "submit_cpu_experiment",
                     return_value={
                         "jobId": "job-2",
                         "status": "queued",
                         "outputDir": output_rel,
                     },
                 ), \
                 patch.object(
                     commons_live_witness.commons_control,
                     "inspect_job",
                     return_value={"status": "failed", "exitCode": 1},
                 ):
                with self.assertRaises(commons_live_witness.LiveWitnessError):
                    commons_live_witness.run_live_witness(
                        experiment_path=experiment,
                        repository_revision=revision,
                        run_scope="fixture-scope",
                        ram_mib=1024,
                        max_minutes=1.0,
                        priority=0,
                    )

    def test_outcome_revision_mismatch_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment, output_rel = self._fixture(root, "c" * 40)
            with self.assertRaisesRegex(
                commons_live_witness.LiveWitnessError,
                "different repository revision",
            ):
                commons_live_witness.validate_outcome(
                    experiment_path=experiment,
                    output_dir=root / output_rel,
                    repository_revision="d" * 40,
                )


if __name__ == "__main__":
    unittest.main()
