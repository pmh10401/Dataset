from __future__ import annotations

from pathlib import Path
import unittest


_REPO_ROOT = Path(__file__).resolve().parents[3]
PREREGISTRATION = _REPO_ROOT / "docs" / "EXP409_PREREGISTRATION_KO.md"
RUNBOOK = _REPO_ROOT / "docs" / "EXP409_RUNBOOK_KO.md"


class DocumentationTests(unittest.TestCase):
    def test_preregistration_contains_every_frozen_boundary(self) -> None:
        text = PREREGISTRATION.read_text(encoding="utf-8")
        for required in (
            "Exp409",
            "two_new_plus_existing_independent",
            "0.005",
            "100",
            "20260727",
            "0.90",
            "20",
            "review-only",
            "not_evaluated",
            "prospective_validated: false",
            "deployment_eligible: false",
            "operational_policy_changed: false",
            "TEST label",
            "새 데이터가 현재 존재하지 않는다",
            "tail replacement",
        ):
            with self.subTest(required=required):
                self.assertIn(required, text)

    def test_runbook_has_only_smoke_command_and_stop_conditions(self) -> None:
        text = RUNBOOK.read_text(encoding="utf-8")
        for required in (
            "--representative-smoke",
            "--dataset-limit 8",
            "TIMESERIES_DATA_ROOT",
            "exp409_prospective_readiness_smoke",
            "exp409_provenance_manifest.json",
            "예산",
            "lane",
            "TEST label",
            "재사용하지",
        ):
            with self.subTest(required=required):
                self.assertIn(required, text)
        for forbidden in (
            "--stage all",
            "--resume",
            "--reset",
            "--device",
            "--variant",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, text)


if __name__ == "__main__":
    unittest.main()
