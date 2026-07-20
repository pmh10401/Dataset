from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.exp255_change_point_features.run_pipeline import parser, write_status


class ChangePointPipelineTest(unittest.TestCase):
    def test_cli_stages(self) -> None:
        choices = parser()._option_string_actions["--stage"].choices
        self.assertEqual(set(choices), {"audit", "cache", "predict", "evaluate", "all"})

    def test_initial_status_is_zero_and_label_free(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            heartbeat = root / "heartbeat.json"
            external = root / "external.json"
            with patch(
                "experiments.exp255_change_point_features.run_pipeline.HEARTBEAT_PATH",
                heartbeat,
            ), patch(
                "experiments.exp255_change_point_features.run_pipeline.EXTERNAL_STATUS_PATH",
                external,
            ):
                write_status(root, "running", "cache", 0, 1117)
            status = json.loads((root / "status.json").read_text())
            self.assertEqual(status["datasets_done"], 0)
            self.assertEqual(status["errors"], 0)
            self.assertEqual(status["progress_percent"], 0.0)
            self.assertFalse(status["contains_test_labels"])
            self.assertEqual(status, json.loads(heartbeat.read_text()))

    def test_dashboard_registration(self) -> None:
        root = Path(__file__).resolve().parents[3]
        dashboard = (root / "serve_rank_dashboard.py").read_text()
        self.assertIn("experiment_255_change_point_features", dashboard)
        self.assertIn("Exp255 - change-point features", dashboard)


if __name__ == "__main__":
    unittest.main()
