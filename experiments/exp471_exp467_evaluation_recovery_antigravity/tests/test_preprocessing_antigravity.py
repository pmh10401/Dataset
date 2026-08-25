"""Deterministic preprocessing parallelism and live ETA evidence for Exp471."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from paper.equipment_anomaly_detection.contracts_paper import RawDataset
from paper.equipment_anomaly_detection.sources_paper.low_frequency_paper import (
    score_low_frequency,
)

try:
    from experiments.exp471_exp467_evaluation_recovery_antigravity.preprocessing_antigravity import (
        Exp471PreprocessingObserverAntigravity,
        append_preprocessing_progress_antigravity,
        instrument_captured_preprocessing_antigravity,
        latest_preprocessing_progress_antigravity,
        select_preprocessing_workers_antigravity,
    )
except ImportError:
    Exp471PreprocessingObserverAntigravity = None
    append_preprocessing_progress_antigravity = None
    instrument_captured_preprocessing_antigravity = None
    latest_preprocessing_progress_antigravity = None
    select_preprocessing_workers_antigravity = None


def _values_antigravity(dataset: str) -> RawDataset:
    x = np.arange(64, dtype=np.float32)
    train = tuple(
        (np.sin(x / (index + 2)) + index * 0.03).astype(np.float32)
        for index in range(5)
    )
    test = (
        (9.0 * np.sin(x / 4)).astype(np.float32),
        np.full(64, np.nan, dtype=np.float32),
        np.arange(32, dtype=np.float32),
    )
    return RawDataset(
        dataset,
        64,
        tuple(range(5)),
        (b"x",) * 5,
        train,
        (0, 1, 2),
        (b"y",) * 3,
        test,
    )


class PreprocessingTestsAntigravity(unittest.TestCase):
    def test_parallel_observer_preserves_exact_result_order_and_reports_global_eta(self) -> None:
        """A parallel completion race must not reorder scores or reset global ETA."""

        if Exp471PreprocessingObserverAntigravity is None:
            self.fail("Exp471 preprocessing observer is missing")
        values = {
            dataset: _values_antigravity(dataset)
            for dataset in ("A", "B", "C")
        }
        hard = {dataset: frozenset() for dataset in values}
        expected = score_low_frequency(values, hard)
        events: list[dict[str, object]] = []
        observer = Exp471PreprocessingObserverAntigravity(
            score_low_frequency,
            condition_order=("condition-a", "condition-b"),
            workers=2,
            progress=events.append,
            qualify_parallel=False,
            acceleration_backend="numba_serial_safe",
        )

        first = observer.score_antigravity(values, hard)
        second = observer.score_antigravity(values, hard)

        self.assertEqual(first, expected)
        self.assertEqual(second, expected)
        completed = [
            event["completed"]
            for event in events
            if event["status"] == "running"
        ]
        self.assertEqual(completed, [1, 2, 3, 4, 5, 6])
        self.assertTrue(all(event["total"] == 6 for event in events))
        self.assertEqual(events[-1]["status"], "complete")
        self.assertEqual(events[-1]["remaining"], 0)
        self.assertEqual(events[-1]["eta_seconds"], 0.0)
        self.assertEqual(events[-1]["condition"], "condition-b")
        self.assertIn("logical_cpu_count", events[-1])
        self.assertEqual(events[-1].get("logical_cpu_count"), 3)
        self.assertEqual(events[-1].get("requested_workers"), 2)
        self.assertEqual(events[-1]["workers"], 2)
        self.assertTrue(events[-1]["byte_parity"])
        self.assertTrue(events[-1]["parallel_accepted"])
        self.assertEqual(events[-1].get("acceleration_backend"), "numba_serial_safe")
        self.assertTrue(
            all(
                event["eta_seconds"] is None
                or (
                    type(event["eta_seconds"]) is float
                    and math.isfinite(event["eta_seconds"])
                    and event["eta_seconds"] >= 0.0
                )
                for event in events
            )
        )

    def test_captured_instrumentation_restores_frozen_scorer_after_failure(self) -> None:
        """An interrupted observed run must not leave a replaced scientific scorer."""

        if instrument_captured_preprocessing_antigravity is None:
            self.fail("Exp471 captured preprocessing instrumentation is missing")
        from experiments.exp467_numba_calibration_resume_antigravity import (
            verifier_antigravity as verifier,
        )

        compose = verifier.compose_all_conditions_antigravity
        compose_condition = compose.__globals__["compose_condition_score_antigravity"]
        scoring_globals = compose_condition.__globals__
        original = scoring_globals["score_low_frequency"]
        values = {"A": _values_antigravity("A")}
        hard = {"A": frozenset()}
        expected = original(values, hard)
        observer = Exp471PreprocessingObserverAntigravity(
            original,
            condition_order=("condition-a",),
            workers=1,
            progress=lambda _event: None,
            qualify_parallel=False,
        )
        session = SimpleNamespace(verifier=verifier)

        with self.assertRaisesRegex(RuntimeError, "public interruption"):
            with instrument_captured_preprocessing_antigravity(
                session, observer
            ):
                observed = scoring_globals["score_low_frequency"](values, hard)
                self.assertEqual(observed, expected)
                raise RuntimeError("public interruption")

        self.assertIs(scoring_globals["score_low_frequency"], original)
        self.assertEqual(original(values, hard), expected)

    def test_worker_gate_requires_exact_result_and_at_least_1_1x_speedup(self) -> None:
        """A slower or non-identical parallel smoke must retain one worker."""

        if select_preprocessing_workers_antigravity is None:
            self.fail("Exp471 preprocessing worker gate is missing")
        canonical = ("exact", 1, 2, 3)
        accepted = select_preprocessing_workers_antigravity(
            canonical,
            canonical,
            single_seconds=11.0,
            parallel_seconds=5.0,
            requested_workers=11,
        )
        slower = select_preprocessing_workers_antigravity(
            canonical,
            canonical,
            single_seconds=5.0,
            parallel_seconds=6.0,
            requested_workers=11,
        )
        changed = select_preprocessing_workers_antigravity(
            canonical,
            ("changed", 1, 2, 3),
            single_seconds=11.0,
            parallel_seconds=5.0,
            requested_workers=11,
        )

        self.assertEqual(accepted["selected_workers"], 11)
        self.assertTrue(accepted["parallel_accepted"])
        self.assertEqual(slower["selected_workers"], 1)
        self.assertFalse(slower["parallel_accepted"])
        self.assertEqual(changed["selected_workers"], 1)
        self.assertFalse(changed["byte_parity"])

    def test_preprocessing_log_round_trips_canonical_latest_record(self) -> None:
        """A live status reader must recover the latest complete ETA record."""

        if (
            append_preprocessing_progress_antigravity is None
            or latest_preprocessing_progress_antigravity is None
        ):
            self.fail("Exp471 preprocessing progress log is missing")
        with tempfile.TemporaryDirectory(
            prefix="exp471_preprocessing_log_antigravity_"
        ) as temporary:
            root = Path(temporary).resolve()
            first = {
                "schema": "exp471-preprocessing-progress-antigravity.v1",
                "stage": "score-input",
                "phase": "low-frequency",
                "condition": "condition-a",
                "dataset": "A",
                "completed": 1,
                "total": 2,
                "remaining": 1,
                "percent": 50.0,
                "elapsed_seconds": 2.0,
                "items_per_second": 0.5,
                "eta_seconds": 2.0,
                "logical_cpu_count": 3,
                "requested_workers": 2,
                "workers": 2,
                "worker_speedup": 1.2,
                "byte_parity": True,
                "parallel_accepted": True,
                "status": "running",
            }
            final = {
                **first,
                "schema": "exp471-preprocessing-progress-antigravity.v2",
                "dataset": "B",
                "completed": 2,
                "remaining": 0,
                "percent": 100.0,
                "elapsed_seconds": 4.0,
                "eta_seconds": 0.0,
                "acceleration_backend": "numba_serial_safe",
                "status": "complete",
            }

            try:
                append_preprocessing_progress_antigravity(root, first)
                append_preprocessing_progress_antigravity(root, final)
            except ValueError as error:
                self.fail(f"preprocessing provenance fields were rejected: {error}")

            latest = latest_preprocessing_progress_antigravity(root)
            self.assertEqual(
                latest,
                {
                    **final,
                    "timestamp_utc": latest["timestamp_utc"],
                },
            )
            self.assertTrue(
                (root / "preprocessing_progress_antigravity.jsonl").is_file()
            )


if __name__ == "__main__":
    unittest.main()
