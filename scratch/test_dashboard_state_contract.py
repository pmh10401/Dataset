from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_dashboard_module():
    spec = importlib.util.spec_from_file_location(
        "dashboard_under_test", ROOT / "serve_rank_dashboard.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class DashboardStateContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dashboard = load_dashboard_module()

    def test_completed_queue_state_overrides_planned_catalog_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            detail = Path(directory) / "result.csv"
            detail.write_text("dataset_name\nexample\n", encoding="utf-8")
            status = self.dashboard.experiment_history_status(
                "experiment_x",
                {"history_status": "planned", "detail_csv": detail},
                {"completed": ["experiment_x"], "queue": [], "running": None},
            )
        self.assertEqual(status["history_status"], "completed_valid")

    def test_no_running_experiment_does_not_report_last_completed_as_active(self):
        paths = self.dashboard.active_paths(
            {
                "running": None,
                "history": [{"id": "experiment_185_final_independent_reaggregation"}],
                "completed": ["experiment_185_final_independent_reaggregation"],
            }
        )
        self.assertIsNone(paths["active_id"])
        self.assertEqual(paths["last_completed_id"], "experiment_185_final_independent_reaggregation")
        self.assertEqual(paths["id"], "experiment_185_final_independent_reaggregation")

    def test_pid_metadata_accepts_legacy_and_json_formats(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pid.json"
            path.write_text("123\n", encoding="utf-8")
            self.assertEqual(self.dashboard.read_pid_metadata(path)["pid"], 123)
            path.write_text('{"pid": 456, "instance_id": "abc"}\n', encoding="utf-8")
            metadata = self.dashboard.read_pid_metadata(path)
            self.assertEqual(metadata["pid"], 456)
            self.assertEqual(metadata["instance_id"], "abc")


if __name__ == "__main__":
    unittest.main()
