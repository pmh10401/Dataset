from __future__ import annotations

import unittest


EXPECTED_EXP267_273_BATCH = (
    "experiment_267_zero_support_fn_diagnostic",
    "experiment_268_multiscale_scattering_deviation",
    "experiment_269_ordered_path_signature_deviation",
    "experiment_270a_rff_normal_forecast",
    "experiment_270b_causal_tcn_normal_forecast",
    "experiment_270c_forecast_risk_audit",
    "experiment_271_new_source_qualification",
    "experiment_272_recovery_review_selector",
    "experiment_273_autonomous_promotion_audit",
)


class DashboardRegistrationTests(unittest.TestCase):
    def test_queue_order_is_sequential(self) -> None:
        from run_rank_experiments_sequential import EXP267_273_BATCH

        self.assertEqual(tuple(EXP267_273_BATCH), EXPECTED_EXP267_273_BATCH)

    def test_dashboard_has_separate_metric_scope_and_not_activated_label(self) -> None:
        from serve_rank_dashboard import EXPERIMENTS

        for experiment_id in EXPECTED_EXP267_273_BATCH:
            self.assertIn(experiment_id, EXPERIMENTS)
            self.assertIn("autonomous_metric_scope", EXPERIMENTS[experiment_id])
            self.assertIn("review_metric_scope", EXPERIMENTS[experiment_id])
        self.assertNotIn("activated", EXPERIMENTS[EXPECTED_EXP267_273_BATCH[-1]]["status_reason"].lower())


if __name__ == "__main__":
    unittest.main()
