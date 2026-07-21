from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from experiments.exp239_248_diverse_score_sources.contracts import DatasetBundle
from experiments.exp267_273_zero_support_fn_recovery.config import SEED


def bundle(*, n_train: int = 100, target_length: int = 24) -> DatasetBundle:
    generator = np.random.default_rng(2706)
    time = np.linspace(0.0, 4.0 * np.pi, target_length)
    train = np.vstack(
        [
            np.sin(time + offset)
            + 0.06 * np.sin(3.0 * time + offset)
            + generator.normal(0.0, 0.02, target_length)
            for offset in np.linspace(0.0, 0.3, n_train)
        ]
    )
    normal = np.sin(time) + 0.06 * np.sin(3.0 * time)
    unusual = normal.copy()
    unusual[-8:] += 1.25
    return DatasetBundle(
        dataset="tcn-demo",
        train_normal=train.astype(np.float32),
        test_values=np.vstack((normal, unusual)).astype(np.float32),
        target_length=target_length,
    )


class CausalTCNNormalForecastTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_tcn_abstains_below_100_train_rows(self) -> None:
        from experiments.exp267_273_zero_support_fn_recovery.forecast_tcn_source import (
            CausalTCNNormalForecast,
        )

        result = CausalTCNNormalForecast().score(bundle(n_train=99), SEED, self.path)

        self.assertTrue(result.abstained)
        self.assertFalse(result.eligible)
        self.assertEqual(result.abstention_reason, "n_train_below_100")
        self.assertFalse(result.provenance["contains_test_labels"])

    def test_prediction_prefix_is_not_affected_by_future_mutation(self) -> None:
        from experiments.exp267_273_zero_support_fn_recovery.forecast_tcn_source import (
            build_tcn_model,
            set_deterministic_seed,
        )

        set_deterministic_seed(SEED)
        model = build_tcn_model().eval()
        values = torch.randn(2, 1, 32)
        original = model(values).detach().cpu().numpy()
        changed = values.clone()
        changed[..., -8:] += 100.0
        mutated = model(changed).detach().cpu().numpy()

        np.testing.assert_allclose(original[..., :-8], mutated[..., :-8], atol=1e-6)

    def test_frozen_network_shape_and_architecture(self) -> None:
        from experiments.exp267_273_zero_support_fn_recovery.forecast_tcn_source import (
            TCN_CHANNELS,
            TCN_DILATIONS,
            TCN_DROPOUT,
            TCN_KERNEL_SIZE,
            TCN_PATCH_SIZE,
            build_tcn_model,
        )

        model = build_tcn_model()
        output = model(torch.zeros(3, 1, 19))

        self.assertEqual(TCN_CHANNELS, 32)
        self.assertEqual(TCN_KERNEL_SIZE, 3)
        self.assertEqual(TCN_DILATIONS, (1, 2, 4, 8))
        self.assertEqual(TCN_DROPOUT, 0.1)
        self.assertEqual(TCN_PATCH_SIZE, 8)
        self.assertEqual(output.shape, (3, 8, 19))

    def test_train_only_fold_plan_is_deterministic_and_query_excluded(self) -> None:
        from experiments.exp267_273_zero_support_fn_recovery.forecast_tcn_source import (
            deterministic_tcn_fold_plan,
        )

        first = deterministic_tcn_fold_plan(100, SEED)
        second = deterministic_tcn_fold_plan(100, SEED)

        self.assertEqual(len(first), 5)
        seen_queries = []
        for (fit_a, query_a), (fit_b, query_b) in zip(first, second):
            np.testing.assert_array_equal(fit_a, fit_b)
            np.testing.assert_array_equal(query_a, query_b)
            self.assertFalse(np.intersect1d(fit_a, query_a).size)
            seen_queries.extend(query_a.tolist())
        self.assertEqual(sorted(seen_queries), list(range(100)))

    def test_fixed_seed_recreates_the_same_initial_weights(self) -> None:
        from experiments.exp267_273_zero_support_fn_recovery.forecast_tcn_source import (
            build_tcn_model,
            set_deterministic_seed,
        )

        set_deterministic_seed(SEED)
        first = build_tcn_model().state_dict()
        set_deterministic_seed(SEED)
        second = build_tcn_model().state_dict()

        for name in first:
            torch.testing.assert_close(first[name], second[name])

    def test_seed_score_variants_are_one_evidence_family(self) -> None:
        from experiments.exp267_273_zero_support_fn_recovery.forecast_tcn_source import (
            TCN_EVIDENCE_FAMILY,
            score_variants,
        )

        first = np.asarray([0.2, 0.8, 0.4])
        second = np.asarray([0.6, 0.4, 0.4])
        variants = score_variants(first, second)

        np.testing.assert_allclose(variants["seed_median"], [0.4, 0.6, 0.4])
        np.testing.assert_allclose(variants["seed_intersection"], [0.2, 0.4, 0.4])
        self.assertEqual(TCN_EVIDENCE_FAMILY, "normal_forecast")

    def test_device_benchmark_is_cached_before_reuse(self) -> None:
        from experiments.exp267_273_zero_support_fn_recovery.forecast_tcn_source import (
            benchmark_device,
        )

        first = benchmark_device(self.path, requested="cpu")
        with patch(
            "experiments.exp267_273_zero_support_fn_recovery.forecast_tcn_source._benchmark_device_once"
        ) as once:
            second = benchmark_device(self.path, requested="cpu")

        once.assert_not_called()
        self.assertEqual(first, second)
        self.assertEqual(first["selected_device"], "cpu")
        self.assertTrue((self.path / "tcn_device_benchmark.json").exists())

    def test_direct_mps_request_without_passing_benchmark_remains_cpu(self) -> None:
        from experiments.exp267_273_zero_support_fn_recovery.forecast_tcn_source import (
            CausalTCNNormalForecast,
            choose_device,
        )

        self.assertEqual(choose_device("mps"), "cpu")
        self.assertEqual(choose_device("mps", {"selected_device": "mps"}), "cpu")
        source = CausalTCNNormalForecast(device="mps")
        self.assertEqual(source.device, "cpu")
        self.assertIsNone(source.benchmark_record)

    def test_passing_mps_benchmark_is_frozen_and_used_for_training_and_scoring(self) -> None:
        from experiments.exp267_273_zero_support_fn_recovery import (
            forecast_tcn_source as tcn_module,
        )
        from experiments.exp267_273_zero_support_fn_recovery.forecast_tcn_source import (
            CausalTCNNormalForecast,
            benchmark_record_hash,
        )

        record = {
            "selected_device": "mps",
            "mps_available": True,
            "mps_finite": True,
            "mps_close_to_cpu": True,
            "mps_faster_than_cpu": True,
            "benchmark_config_hash": "frozen-benchmark-config",
            "mps_median_seconds": 0.01,
            "synthetic_rows": 100,
            "synthetic_target_length": 128,
        }
        fake = {
            20260721: (np.linspace(0.0, 1.0, 100), np.asarray([0.1, 0.9]), (5,) * 6),
            20261721: (np.linspace(0.2, 0.8, 100), np.asarray([0.3, 0.7]), (6,) * 6),
        }
        with patch.object(tcn_module, "_device_available", return_value=True), patch.object(
            tcn_module,
            "_score_one_seed",
            side_effect=lambda **kwargs: fake[kwargs["seed"]],
        ) as one_seed:
            result = CausalTCNNormalForecast(
                device="mps", benchmark_record=record
            ).score(bundle(n_train=100, target_length=16), SEED, self.path)

        self.assertTrue(all(call.kwargs["device"] == "mps" for call in one_seed.call_args_list))
        self.assertEqual(result.provenance["selected_device"], "mps")
        self.assertEqual(result.provenance["benchmark_record"], record)
        self.assertEqual(
            result.provenance["benchmark_record_hash"], benchmark_record_hash(record)
        )
        self.assertTrue(result.provenance["runtime_estimate"]["available"])
        self.assertEqual(
            result.provenance["runtime_estimate"]["selected_device"], "mps"
        )

    def test_large_train_shape_uses_bounded_batch_provenance_without_sampling(self) -> None:
        from experiments.exp267_273_zero_support_fn_recovery.forecast_tcn_source import (
            TCN_BATCH_SIZE,
            estimate_tcn_batch_bytes,
            tcn_compute_provenance,
        )

        provenance = tcn_compute_provenance(n_train=18_378, target_length=4_000)

        self.assertEqual(TCN_BATCH_SIZE, 32)
        self.assertEqual(provenance["n_train"], 18_378)
        self.assertEqual(provenance["target_length"], 4_000)
        self.assertFalse(provenance["train_rows_sampled"])
        self.assertFalse(provenance["target_length_resampled"])
        self.assertEqual(provenance["maximum_training_batch_rows"], 32)
        self.assertGreater(estimate_tcn_batch_bytes(32, 4_000), 0)

    def test_train_only_compute_guard_abstains_before_test_access_or_training(self) -> None:
        from experiments.exp267_273_zero_support_fn_recovery import (
            forecast_tcn_source as tcn_module,
        )
        from experiments.exp267_273_zero_support_fn_recovery.forecast_tcn_source import (
            CausalTCNNormalForecast,
        )

        class ForbiddenTestValues:
            def __array__(self, *args, **kwargs):
                raise AssertionError("TEST values must not be read by the TRAIN guard")

        train = np.zeros((1_251, 4_000), dtype=np.float32)
        oversized = DatasetBundle(
            dataset="oversized-train-only-shape",
            train_normal=train,
            test_values=ForbiddenTestValues(),  # type: ignore[arg-type]
            target_length=4_000,
        )
        with patch.object(tcn_module, "_score_one_seed") as one_seed:
            result = CausalTCNNormalForecast().score(oversized, SEED, self.path)

        one_seed.assert_not_called()
        self.assertTrue(result.abstained)
        self.assertFalse(result.eligible)
        self.assertEqual(result.abstention_reason, "train_work_exceeds_5000000")
        self.assertEqual(result.provenance["train_elements"], 5_004_000)
        self.assertEqual(result.provenance["tcn_max_train_elements"], 5_000_000)
        self.assertFalse(result.provenance["contains_test_labels"])

    def test_score_uses_no_test_labels_and_records_two_seed_scores(self) -> None:
        from experiments.exp267_273_zero_support_fn_recovery import (
            forecast_tcn_source as tcn_module,
        )
        from experiments.exp267_273_zero_support_fn_recovery.forecast_tcn_source import (
            CausalTCNNormalForecast,
        )

        source_bundle = bundle(n_train=100, target_length=16)
        fake = {
            20260721: (np.linspace(0.0, 1.0, 100), np.asarray([0.1, 0.9]), (5,) * 6),
            20261721: (np.linspace(0.2, 0.8, 100), np.asarray([0.3, 0.7]), (6,) * 6),
        }
        with patch.object(tcn_module, "_score_one_seed", side_effect=lambda **kwargs: fake[kwargs["seed"]]):
            result = CausalTCNNormalForecast().score(source_bundle, SEED, self.path)

        self.assertFalse(result.abstained)
        self.assertEqual(result.variant, "seed_median")
        np.testing.assert_allclose(result.train_oof, [0.1 + (0.8 / 99) * i for i in range(100)])
        np.testing.assert_allclose(result.test_scores, [0.2, 0.8])
        self.assertEqual(result.provenance["seed_order"], (20260721, 20261721))
        self.assertEqual(result.provenance["evidence_family"], "normal_forecast")
        self.assertFalse(result.provenance["contains_test_labels"])
        self.assertNotIn("y_test", result.provenance)

    def test_test_length_mismatch_is_rejected_before_training(self) -> None:
        from experiments.exp267_273_zero_support_fn_recovery.forecast_tcn_source import (
            CausalTCNNormalForecast,
        )

        source_bundle = bundle(n_train=100, target_length=16)
        malformed = DatasetBundle(
            dataset=source_bundle.dataset,
            train_normal=source_bundle.train_normal,
            test_values=source_bundle.test_values[:, :-1],
            target_length=source_bundle.target_length,
        )
        with self.assertRaisesRegex(ValueError, "target length"):
            CausalTCNNormalForecast().score(malformed, SEED, self.path)


if __name__ == "__main__":
    unittest.main()
