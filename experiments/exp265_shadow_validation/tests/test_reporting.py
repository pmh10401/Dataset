from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.exp265_shadow_validation.reporting import report_sha256, write_report


def report_response(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "collection_mode": "prospective_shadow",
        "policy_version": "exp265-shadow-v1",
        "generated_at": "2026-07-21T00:00:00+00:00",
        "collected_from": "2026-07-20T00:00:00+00:00",
        "collected_to": "2026-07-21T00:00:00+00:00",
        "config_sha256": ["a" * 64],
        "runs_collected": 3,
        "target_runs": 100,
        "progress_percent": 3,
        "candidates": 2,
        "reviewed_candidates": 2,
        "candidate_precision": 0.5,
        "fully_annotated_runs": 0,
        "recall": None,
        "f1": None,
        "metric_scope": "candidate_feedback_only",
        "autonomous_scope": "Hard_only",
        "automatic_policy_changed": False,
        "deployment_validated": False,
        "prospective": True,
        "retrospective": False,
        "lanes": {
            "Hard": {"candidates": 1}, "Shadow": {"candidates": 1},
            "Priority": {"candidates": 0}, "Standard": {"candidates": 0},
        },
    }
    payload.update(overrides)
    payload["report_sha256"] = report_sha256(payload)
    return payload


class ReportingTest(unittest.TestCase):
    def test_candidate_only_report_omits_recall_and_f1(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = write_report(Path(directory), report_response(), api_url="https://dashboard.example/api/shadow/report")
            text = paths.markdown.read_text(encoding="utf-8")
            self.assertIn("Recall: 계산 불가", text)
            self.assertIn("F1: 계산 불가", text)

    def test_report_never_selects_policy_at_100_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = write_report(Path(directory), report_response(runs_collected=100, progress_percent=100), api_url="https://dashboard.example/api/shadow/report")
            self.assertIs(result.payload["automatic_policy_changed"], False)

    def test_rejects_recall_without_full_annotation(self) -> None:
        payload = report_response(recall=0.8, f1=0.7)
        payload["report_sha256"] = report_sha256(payload)
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, "fully annotated"):
            write_report(Path(directory), payload, api_url="https://dashboard.example/api/shadow/report")

    def test_rejects_deployment_claim(self) -> None:
        payload = report_response(deployment_validated=True)
        payload["report_sha256"] = report_sha256(payload)
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, "deployment"):
            write_report(Path(directory), payload, api_url="https://dashboard.example/api/shadow/report")


if __name__ == "__main__":
    unittest.main()
