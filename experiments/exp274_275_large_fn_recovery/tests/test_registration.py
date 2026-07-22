from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
IDS = (
    "experiment_274_large_train_multisource_fn_recovery",
    "experiment_275_recovery_shadow_precision_guard",
)


class RegistrationTests(unittest.TestCase):
    def test_runner_and_dashboard_register_both_experiments(self) -> None:
        for relative in ("run_rank_experiments_sequential.py", "serve_rank_dashboard.py"):
            source = (ROOT / relative).read_text(encoding="utf-8")
            for experiment_id in IDS:
                self.assertIn(experiment_id, source)

    def test_dashboard_keeps_hard_unchanged_language(self) -> None:
        source = (ROOT / "serve_rank_dashboard.py").read_text(encoding="utf-8")
        self.assertIn("Exp253 Hard 변경 없음", source)
        self.assertIn("검토 전용", source)


if __name__ == "__main__":
    unittest.main()
