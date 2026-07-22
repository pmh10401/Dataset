import unittest

import run_rank_experiments_sequential as runner
import serve_rank_dashboard as dashboard


IDS = (
    "experiment_278_train_risk_resolution_audit",
    "experiment_279_train_calibration_stability",
    "experiment_280_event_merge_applicability_audit",
    "experiment_281_train_only_two_stage_verifier",
)


class RegistrationTest(unittest.TestCase):
    def test_queue_and_dashboard_use_exact_sequential_order(self) -> None:
        self.assertEqual(runner.EXP278_281_BATCH, IDS)
        registered = {item["id"] for item in runner.EXPERIMENTS}
        for experiment_id in IDS:
            self.assertIn(experiment_id, registered)
            self.assertIn(experiment_id, dashboard.EXPERIMENTS)
            self.assertFalse(dashboard.EXPERIMENTS[experiment_id]["comparison_eligible"])


if __name__ == "__main__":
    unittest.main()
