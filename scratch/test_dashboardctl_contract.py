from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_controller_module():
    spec = importlib.util.spec_from_file_location(
        "dashboardctl_under_test", ROOT / "dashboardctl.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class DashboardControllerContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.controller = load_controller_module()

    def test_pid_metadata_accepts_legacy_and_json_formats(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pid"
            path.write_text("123\n", encoding="utf-8")
            self.assertEqual(self.controller.read_pid_metadata(path)["pid"], 123)
            path.write_text('{"pid": 456, "instance_id": "new"}\n', encoding="utf-8")
            self.assertEqual(self.controller.read_pid_metadata(path)["instance_id"], "new")

    def test_health_url_is_explicit(self):
        self.assertEqual(self.controller.health_url(8765), "http://127.0.0.1:8765/api/healthz")

    def test_health_payload_must_match_expected_pid(self):
        self.assertTrue(self.controller.health_matches_process({"server": {"pid": 99}}, 99))
        self.assertFalse(self.controller.health_matches_process({"server": {"pid": 100}}, 99))

    def test_listener_pid_only_counts_when_it_is_the_dashboard(self):
        original_listener = self.controller.port_listener_pid
        original_command = self.controller.command_for_pid
        self.addCleanup(setattr, self.controller, "port_listener_pid", original_listener)
        self.addCleanup(setattr, self.controller, "command_for_pid", original_command)
        self.controller.port_listener_pid = lambda _port: 456
        self.controller.command_for_pid = lambda _pid: "python serve_rank_dashboard.py"
        self.assertEqual(self.controller.listener_dashboard_pid(), 456)
        self.controller.command_for_pid = lambda _pid: "python unrelated.py"
        self.assertIsNone(self.controller.listener_dashboard_pid())


if __name__ == "__main__":
    unittest.main()
