import unittest

import run_rank_experiments_sequential as runner
import serve_rank_dashboard as dashboard


IDS = (
    "experiment_276_recovery_shadow_confidence_freeze",
    "experiment_277_recovery_shadow_confidence_evaluation",
)


class RegistrationTest(unittest.TestCase):
    def test_runner_and_dashboard_register_both_experiments(self) -> None:
        runner_ids = {item["id"] for item in runner.EXPERIMENTS}
        self.assertTrue(set(IDS).issubset(runner_ids))
        self.assertEqual(runner.EXP276_277_BATCH, IDS)
        for experiment_id in IDS:
            self.assertIn(experiment_id, dashboard.EXPERIMENTS)
            self.assertFalse(dashboard.EXPERIMENTS[experiment_id]["comparison_eligible"])


if __name__ == "__main__":
    unittest.main()
