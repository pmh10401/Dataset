import unittest

import numpy as np

from services.time_series_anomaly_diagnosis.routing_policy import (
    FEATURE_KEYS,
    MethodMetric,
    RouterTrainingRecord,
    extract_train_features,
    fit_train_feature_router,
)
from services.time_series_anomaly_diagnosis.score_sources import METHOD_NAMES


def record(group_id, x, multi=(0.8, 1), hydra=(0.8, 1), mini=(0.8, 0)):
    features = dict(zip(FEATURE_KEYS, (x, 20, x, x, x, x, x), strict=True))
    values = (mini, multi, hydra)
    return RouterTrainingRecord(
        group_id,
        features,
        tuple(MethodMetric(name, auc, zf) for name, (auc, zf) in zip(METHOD_NAMES, values, strict=True)),
    )


class RoutingPolicyTest(unittest.TestCase):
    def test_feature_extractor_matches_frozen_fixed_and_ragged_values(self):
        fixtures = (
            (
                [np.array([0.0, 1.0, 2.0, 3.0]), np.array([1.0, 3.0, 2.0, 5.0])],
                (4.0, 2.0, 0.4947493252961065, 0.5163977794932556,
                 -0.38485596708539216, 0.5745944049138345, 2 / 3),
            ),
            (
                [np.array([0.0, 1.0, 4.0]), np.array([2.0, 0.0, 2.0, 6.0, 3.0]),
                 np.array([-1.0, 2.0, 1.0, 0.0])],
                (4.0, 3.0, 0.42055519773251016, 0.7844465853775776,
                 0.02929687499357314, 1.4601922675905243, 0.5),
            ),
        )
        for rows, expected in fixtures:
            actual = extract_train_features(rows)
            self.assertEqual(tuple(actual), FEATURE_KEYS)
            np.testing.assert_allclose(tuple(actual.values()), expected, rtol=0, atol=1e-12)

    def test_feature_extractor_rejects_invalid_rows(self):
        for rows in (
            [np.array([1.0, 2.0])],
            [np.array([1.0]), np.array([1.0, 2.0])],
            [np.array([1.0, np.nan]), np.array([1.0, 2.0])],
            [np.array([[1.0, 2.0]]), np.array([1.0, 2.0])],
        ):
            with self.assertRaises(ValueError):
                extract_train_features(rows)

    def test_router_is_anonymous_immutable_and_name_invariant(self):
        router = fit_train_feature_router([record(i, float(i)) for i in range(20)])
        self.assertEqual(router.input_keys, FEATURE_KEYS)
        self.assertNotIn("dataset_name", router.input_keys)
        self.assertNotIn("family_name", router.input_keys)
        features = extract_train_features([np.arange(4.0), np.arange(4.0) + 1])
        self.assertEqual(router.route(features), router.route(dict(features)))
        self.assertEqual(router.route(features).selected_method, "multirocket_local_gap")
        with self.assertRaises(Exception):
            router.input_keys = ()

    def test_router_deduplicates_groups_before_support_and_preserves_ties(self):
        duplicated = [
            record(0, 0.0, multi=(0.8, 20), hydra=(0.8, 0)),
            record(0, 0.0, multi=(0.8, 0), hydra=(0.8, 0)),
            *(record(i, 0.0, multi=(0.8, 0), hydra=(0.8, 0)) for i in range(1, 20)),
        ]
        decision = fit_train_feature_router(duplicated).route(record(99, 0.0).features)
        self.assertEqual(decision.support, 20)
        self.assertEqual(decision.selected_method, "multirocket_local_gap")
        self.assertEqual(
            fit_train_feature_router([record(i, float(i)) for i in range(40)])
            .route(record(99, 0.0).features).support,
            31,
        )
        with self.assertRaisesRegex(ValueError, "20"):
            fit_train_feature_router([record(i // 2, 0.0) for i in range(38)]).route(
                record(99, 0.0).features
            )

    def test_router_imputes_and_scales_and_rejects_all_nonfinite_column(self):
        rows = [record(i, float(i)) for i in range(20)]
        rows[0] = RouterTrainingRecord(
            rows[0].group_id,
            {**rows[0].features, "skewness": np.nan},
            rows[0].metrics,
        )
        decision = fit_train_feature_router(rows).route(
            {**record(99, 10.0).features, "skewness": np.inf}
        )
        self.assertEqual(decision.support, 20)
        broken = [
            RouterTrainingRecord(row.group_id, {**row.features, "kurtosis": np.nan}, row.metrics)
            for row in rows
        ]
        with self.assertRaisesRegex(ValueError, "finite training value"):
            fit_train_feature_router(broken)

    def test_router_enforces_each_gate_boundary(self):
        base = [record(i, float(i)) for i in range(20)]
        self.assertEqual(fit_train_feature_router(base).route(base[0].features).selected_method,
                         "multirocket_local_gap")

        auc_rate = [record(i, float(i), multi=((0.7 if i < 3 else 0.8), 1)) for i in range(20)]
        self.assertEqual(fit_train_feature_router(auc_rate).route(base[0].features).selected_method,
                         "hydra_local_gap")
        auc_boundary = [record(i, float(i), multi=((0.7 if i < 2 else 0.8), 1))
                        for i in range(20)]
        boundary = fit_train_feature_router(auc_boundary).route(base[0].features)
        self.assertEqual(boundary.selected_method, "multirocket_local_gap")
        self.assertEqual(boundary.candidate_evidence[0].auc_non_degradation_rate, 0.90)
        self.assertEqual(boundary.candidate_evidence[0].auc_gain_quantile, 0.0)
        zf_rate = [record(i, float(i), mini=(0.8, 1), multi=(0.8, (0 if i < 3 else 2)))
                   for i in range(20)]
        failed = fit_train_feature_router(zf_rate).route(base[0].features)
        self.assertEqual(failed.selected_method, "minirocket_local_gap")
        self.assertEqual(failed.candidate_evidence[0].zf_non_degradation_rate, 0.85)
        zf_boundary = [record(i, float(i), mini=(0.8, 1), multi=(0.8, (0 if i < 2 else 2)))
                       for i in range(20)]
        boundary = fit_train_feature_router(zf_boundary).route(base[0].features)
        self.assertEqual(boundary.candidate_evidence[0].zf_non_degradation_rate, 0.90)
        self.assertEqual(boundary.selected_method, "multirocket_local_gap")
        zero_mean = [record(i, float(i), multi=(0.8, 0)) for i in range(20)]
        self.assertEqual(fit_train_feature_router(zero_mean).route(base[0].features).selected_method,
                         "hydra_local_gap")

    def test_router_selects_by_mean_zf_then_quantile_then_fixed_order(self):
        rows = [record(i, float(i), multi=(0.9, 2), hydra=(0.9, 1)) for i in range(20)]
        self.assertEqual(fit_train_feature_router(rows).route(rows[0].features).selected_method,
                         "multirocket_local_gap")
        rows = [record(i, float(i), multi=(0.85, 1), hydra=(0.9, 1)) for i in range(20)]
        self.assertEqual(fit_train_feature_router(rows).route(rows[0].features).selected_method,
                         "hydra_local_gap")
        rows = [record(i, float(i), multi=(0.9, 1), hydra=(0.9, 1)) for i in range(20)]
        self.assertEqual(fit_train_feature_router(rows).route(rows[0].features).selected_method,
                         "multirocket_local_gap")

    def test_training_record_validation_rejects_identity_and_metric_leaks(self):
        good = record(0, 0.0)
        with self.assertRaises(ValueError):
            RouterTrainingRecord(-1, good.features, good.metrics)
        with self.assertRaises(ValueError):
            MethodMetric("unknown", 0.5, 0)
        with self.assertRaises(ValueError):
            MethodMetric(METHOD_NAMES[0], 1.1, 0)
        with self.assertRaises(ValueError):
            MethodMetric(METHOD_NAMES[0], 0.5, -1)


if __name__ == "__main__":
    unittest.main()
