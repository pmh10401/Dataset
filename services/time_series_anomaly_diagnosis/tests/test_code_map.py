import json
from pathlib import Path
import subprocess
import sys
import unittest

from services.time_series_anomaly_diagnosis.tools.build_code_map import build_code_map

SERVICE_ROOT = Path(__file__).resolve().parents[1]


class CodeMapTest(unittest.TestCase):
    def test_committed_code_map_is_current(self) -> None:
        subprocess.run(
            [sys.executable, str(SERVICE_ROOT / "tools/build_code_map.py"), "--validate"],
            check=True,
        )
        payload = json.loads((SERVICE_ROOT / "docs/code_map_v1.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["schema"], "time_series_anomaly_diagnosis_code_map_v1")
        self.assertEqual(payload["unknown_relationships"], [])

    def test_web_application_calls_application_server(self) -> None:
        payload = build_code_map()
        nodes = {node["id"]: node for node in payload["nodes"]}

        self.assertIn("web_application", nodes)
        self.assertEqual(nodes["web_application"]["path"], "web/application.js")
        self.assertIn(
            {"from": "web_application", "to": "application_server", "type": "calls"},
            payload["edges"],
        )

    def test_service_launcher_starts_application_server(self) -> None:
        payload = build_code_map()
        nodes = {node["id"]: node for node in payload["nodes"]}

        self.assertEqual(nodes["service_launcher"]["path"], "service_launcher.py")
        self.assertIn(
            {"from": "service_launcher", "to": "application_server", "type": "starts"},
            payload["edges"],
        )

    def test_application_server_ranks_packaged_review_evidence(self) -> None:
        payload = build_code_map()
        self.assertIn(
            {
                "from": "application_server",
                "to": "multi_evidence_review",
                "type": "ranks review candidates with",
            },
            payload["edges"],
        )
        self.assertIn(
            {
                "from": "model_resources",
                "to": "multi_evidence_review",
                "type": "parses review evidence with",
            },
            payload["edges"],
        )


if __name__ == "__main__":
    unittest.main()
