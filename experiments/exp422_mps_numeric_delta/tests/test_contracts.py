from dataclasses import FrozenInstanceError
import math
import unittest

from experiments.exp422_mps_numeric_delta.contracts import (
    BatchKey,
    CoreParityReport,
    StageResult,
    TensorDelta,
    TimingEvidence,
)


class Exp422ContractTests(unittest.TestCase):
    def test_old_exp421_batches_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            BatchKey(1, 41701, 41711)

    def test_batch_key_requires_exp417_frozen_seeds(self) -> None:
        with self.assertRaises(ValueError):
            BatchKey(16, 41704, 41711)
        with self.assertRaises(ValueError):
            BatchKey(16, 41701, 41714)

    def test_timing_includes_transfer_compute_and_sync(self) -> None:
        with self.assertRaises(ValueError):
            TimingEvidence("mps", 0.2, 0.2, 0.2, 0.1)

    def test_records_are_immutable_and_reject_nonfinite_values(self) -> None:
        delta = TensorDelta(
            name="scores",
            finite=True,
            max_absolute_error=0.0,
            mean_absolute_error=0.0,
            max_normalized_error=0.0,
            max_ulp=0,
            cpu_sha256="a" * 64,
            mps_sha256="b" * 64,
            failing_index=None,
        )
        with self.assertRaises(FrozenInstanceError):
            delta.max_ulp = 1  # type: ignore[misc]
        with self.assertRaises(ValueError):
            TensorDelta(
                "scores", True, math.nan, 0.0, 0.0, 0, "a" * 64, "b" * 64, None
            )

    def test_reports_require_lowercase_digests_and_explicit_stage_states(self) -> None:
        delta = TensorDelta("scores", True, 0.0, 0.0, 0.0, 0, "a" * 64, "b" * 64, None)
        cpu_timing = TimingEvidence("cpu", 0.1, 0.2, 0.1, 0.4)
        mps_timing = TimingEvidence("mps", 0.05, 0.1, 0.05, 0.2)
        report = CoreParityReport(
            BatchKey(16, 41701, 41711), (delta,), cpu_timing, mps_timing, True, 1.0
        )
        self.assertTrue(report.passes)
        self.assertEqual(StageResult.failed("mps_unavailable").state, "failed")
        self.assertEqual(StageResult.skipped("core_failed").state, "skipped")
        with self.assertRaises(ValueError):
            TensorDelta("scores", True, 0.0, 0.0, 0.0, 0, "A" * 64, "b" * 64, None)
        with self.assertRaises(ValueError):
            StageResult("pending", None)

    def test_core_report_rejects_spearman_below_frozen_gate(self) -> None:
        delta = TensorDelta("scores", True, 0.0, 0.0, 0.0, 0, "a" * 64, "b" * 64, None)
        cpu_timing = TimingEvidence("cpu", 0.1, 0.2, 0.1, 0.4)
        mps_timing = TimingEvidence("mps", 0.05, 0.1, 0.05, 0.2)
        report = CoreParityReport(
            BatchKey(16, 41701, 41711), (delta,), cpu_timing, mps_timing, True, 0.998
        )
        self.assertFalse(report.passes)

    def test_stage_result_requires_reason_for_nonpassed_states(self) -> None:
        with self.assertRaises(ValueError):
            StageResult("passed", "unexpected")
        with self.assertRaises(ValueError):
            StageResult("failed", None)
        with self.assertRaises(ValueError):
            StageResult("skipped", None)


if __name__ == "__main__":
    unittest.main()
