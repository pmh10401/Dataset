import unittest

import run_rank_experiments_sequential as runner
import serve_rank_dashboard as dashboard


IDS = (
    "experiment_282_prototype_distance_cache",
    "experiment_283_prototype_distance_prediction",
    "experiment_284_prototype_distance_evaluation",
)


class RegistrationTest(unittest.TestCase):
    def test_sequential_registration(self) -> None:
        self.assertEqual(runner.EXP282_284_BATCH, IDS)
        registered = {item["id"] for item in runner.EXPERIMENTS}
        for experiment_id in IDS:
            self.assertIn(experiment_id, registered); self.assertIn(experiment_id, dashboard.EXPERIMENTS)
            self.assertFalse(dashboard.EXPERIMENTS[experiment_id]["comparison_eligible"])


if __name__ == "__main__": unittest.main()
