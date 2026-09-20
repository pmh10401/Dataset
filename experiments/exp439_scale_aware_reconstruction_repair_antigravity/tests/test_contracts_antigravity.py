import unittest

import numpy as np

from experiments.exp439_scale_aware_reconstruction_repair_antigravity.contracts_antigravity import (
    BackendBatchEvidence,
    BackendDecision,
    ChannelCalibration,
    Exp439DatasetResult,
    Exp439RunSummary,
    InstanceChannelEvidence,
    InstanceScoreEvidence,
    PreparedTrainGroup,
    RawValueSeries,
    reject_label_like_fields_antigravity,
)


class Exp439ContractsAntigravityTests(unittest.TestCase):
    def test_raw_series_copies_float32_values_and_keeps_nonfinite_for_audit(self) -> None:
        source = np.asarray([1.0, np.nan, 3.0], dtype=np.float64)
        row = RawValueSeries(
            dataset="D0001",
            instance_index=7,
            split="TRAIN",
            values=source,
            values_blob_sha256="a" * 64,
        )
        source[0] = 99.0
        self.assertEqual(row.values.dtype, np.float32)
        self.assertTrue(row.values.flags.c_contiguous)
        self.assertFalse(row.values.flags.writeable)
        self.assertEqual(float(row.values[0]), 1.0)
        self.assertTrue(np.isnan(row.values[1]))
        with self.assertRaises(ValueError):
            row.values[0] = 2.0

    def test_prepared_group_requires_finite_exact_length_readonly_matrix(self) -> None:
        rows = tuple(
            RawValueSeries("D0001", index, "TRAIN", np.arange(16) + index, f"{index + 1:064x}")
            for index in range(4)
        )
        group = PreparedTrainGroup(dataset="D0001", raw_length=16, rows=rows)
        self.assertEqual(group.matrix.shape, (4, 16))
        self.assertEqual(group.matrix.dtype, np.float32)
        self.assertFalse(group.matrix.flags.writeable)
        with self.assertRaisesRegex(ValueError, "same dataset and raw length"):
            PreparedTrainGroup(dataset="D0001", raw_length=15, rows=rows)

    def test_channel_calibration_enforces_available_and_unavailable_shapes(self) -> None:
        available = ChannelCalibration(
            channel="high_frequency",
            references=np.ones((4, 16), dtype=np.float32),
            center=np.asarray([0.0], dtype=np.float32),
            scale=np.asarray([1.0], dtype=np.float32),
            oof_scores=np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
            threshold=0.4,
            available=True,
            reason=None,
        )
        self.assertFalse(available.references.flags.writeable)
        with self.assertRaisesRegex(ValueError, "unavailable calibration"):
            ChannelCalibration(
                channel="low_frequency",
                references=None,
                center=None,
                scale=None,
                oof_scores=None,
                threshold=None,
                available=False,
                reason=None,
            )

    def test_instance_and_dataset_results_keep_channels_separate(self) -> None:
        channels = tuple(
            InstanceChannelEvidence(
                channel=name,
                score=float(index + 1),
                threshold=0.5,
                abnormal=True,
                available=True,
                reason=None,
            )
            for index, name in enumerate(
                ("high_frequency", "low_frequency", "amplitude_complexity")
            )
        )
        evidence = InstanceScoreEvidence(
            dataset="D0001",
            instance_index=5,
            raw_length=16,
            channels=channels,
            outside_all_frozen_lanes=True,
        )
        result = Exp439DatasetResult(
            dataset="D0001",
            evidence=(evidence,),
            abstentions=(),
            train_exclusions=(),
        )
        self.assertEqual(result.candidate_coordinates["high_frequency"], (("D0001", 5),))
        self.assertEqual(result.candidate_coordinates["low_frequency"], (("D0001", 5),))
        self.assertEqual(result.candidate_coordinates["amplitude_complexity"], (("D0001", 5),))
        self.assertFalse(hasattr(evidence, "combined_abnormal"))
        self.assertFalse(hasattr(evidence, "vote_count"))

    def test_backend_and_summary_contracts_are_strict(self) -> None:
        batch = BackendBatchEvidence(
            backend="mlx",
            batch_size=16,
            finite=True,
            candidate_parity=True,
            maximum_absolute_error=1e-6,
            cpu_median_seconds=2.0,
            device_median_seconds=1.0,
            speedup=2.0,
            passed=True,
            reason=None,
        )
        decision = BackendDecision(
            selected_backend="mlx",
            selected_batch_size=16,
            evidence=(batch,),
            fallback_reason=None,
        )
        summary = Exp439RunSummary(
            total_datasets=1117,
            completed_datasets=1117,
            successful_datasets=1117,
            abstained_datasets=0,
            failed_datasets=0,
            per_channel_candidate_counts={
                "high_frequency": 1,
                "low_frequency": 2,
                "amplitude_complexity": 3,
            },
            backend=decision,
            prediction_label_access_count=0,
        )
        self.assertEqual(summary.total_candidate_count, 6)
        with self.assertRaisesRegex(ValueError, "prediction label access"):
            Exp439RunSummary(
                total_datasets=1,
                completed_datasets=1,
                successful_datasets=1,
                abstained_datasets=0,
                failed_datasets=0,
                per_channel_candidate_counts={
                    "high_frequency": 0,
                    "low_frequency": 0,
                    "amplitude_complexity": 0,
                },
                backend=BackendDecision("cpu", None, (), "optional backends unavailable"),
                prediction_label_access_count=1,
            )

    def test_label_like_fields_are_rejected_recursively(self) -> None:
        reject_label_like_fields_antigravity({"dataset": "D0001", "nested": {"score": 1.0}})
        for key in ("label", "labels_blob", "anomaly_positions", "test_f1"):
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, "forbidden prediction field"):
                    reject_label_like_fields_antigravity({"nested": {key: 1}})


if __name__ == "__main__":
    unittest.main()
