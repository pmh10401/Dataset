from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.exp400_402.run_pipeline import run_pipeline
from experiments.exp400_402.test_evaluation import labels_fixture
from experiments.exp400_402.test_prediction import exp292_fixture, exp397_fixture


def fixture_loader():
    return exp292_fixture(), exp397_fixture(), {"fixture": "a" * 64}


class PipelineTests(unittest.TestCase):
    def test_smoke_pipeline_separates_prediction_and_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = run_pipeline(
                "all",
                root,
                input_loader=fixture_loader,
                label_loader=lambda _datasets: labels_fixture(),
                full_run=False,
            )
            self.assertEqual(result["predict"]["datasets"], 3)
            prediction = (root / "exp400_shadow_ranking_replay.jsonl").read_text(
                encoding="utf-8"
            )
            self.assertNotIn("is_true_anomaly", prediction)
            self.assertNotIn('"f1"', prediction)
            consistency = json.loads(
                (root / "independent_consistency_report.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(consistency["mismatch_count"], 0)
            readiness = json.loads(
                (root / "exp402_readiness_report.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertFalse(readiness["prospective_upstream_available"])
            self.assertFalse(readiness["prospective_validation_completed"])


if __name__ == "__main__":
    unittest.main()
