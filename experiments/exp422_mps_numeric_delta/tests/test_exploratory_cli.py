"""Boundary tests for the review-only continuation runner."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from experiments.exp422_mps_numeric_delta.tests.test_manifest import _core
import scratch.run_exp422_ir_exploratory_score as cli


class ExploratoryPayloadTests(unittest.TestCase):
    def test_payload_marks_review_only_and_preserves_failed_core(self) -> None:
        class Score:
            stage = type("Stage", (), {"reason": "score_parity_gate_failed"})()
            reports = ()

        with patch.object(cli, "_score_payload", return_value={"stage": {"state": "failed"}}):
            payload = cli._exploratory_payload(
                _core(),
                Score(),
                {"core_gate_overridden": True, "review_only": True},
            )
        self.assertEqual(payload["experiment_id"], "Exp422-IR-E")
        self.assertEqual(payload["state"], "review_only")
        self.assertTrue(payload["core_gate_overridden"])
        self.assertEqual(payload["core_stage"]["stage"]["state"], "failed")
        self.assertTrue(payload["provenance"]["review_only"])


if __name__ == "__main__":
    unittest.main()
