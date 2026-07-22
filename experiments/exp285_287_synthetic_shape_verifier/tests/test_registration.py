import unittest
import run_rank_experiments_sequential as runner
import serve_rank_dashboard as dashboard


IDS = ("experiment_285_synthetic_shape_generation_audit", "experiment_286_synthetic_shape_prediction", "experiment_287_synthetic_shape_evaluation")


class RegistrationTest(unittest.TestCase):
    def test_registration(self):
        self.assertEqual(runner.EXP285_287_BATCH, IDS)
        for experiment_id in IDS: self.assertIn(experiment_id, dashboard.EXPERIMENTS)


if __name__ == "__main__": unittest.main()
