from __future__ import annotations

import json
import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.exp226_235_hard_alert_quality import run_pipeline
from experiments.exp226_235_hard_alert_quality.contracts import PredictionRow
from experiments.exp226_235_hard_alert_quality.contracts import ScoreBundle
from experiments.exp226_235_hard_alert_quality.baseline_exp137 import BaselineLanes
from experiments.exp226_235_hard_alert_quality.run_pipeline import (
    freeze_prediction_rows,
    generate_rows_for_dataset,
    load_prediction_checkpoint,
    load_verified_prediction_rows,
    parse_experiment_range,
    save_prediction_checkpoint,
    write_pipeline_status,
)
from experiments.exp226_235_hard_alert_quality.reporting import (
    evaluate_prediction_rows,
    summarize_evaluated_rows,
)
from experiments.exp226_235_hard_alert_quality.independent_reaggregate import (
    reaggregate_result_rows,
)


class PipelineArtifactTests(unittest.TestCase):
    def _rows(self) -> list[PredictionRow]:
        return [
            PredictionRow(
                experiment_id="experiment_227",
                dataset="B",
                variant="dual",
                hard_indices=(2,),
                autonomous_alert=True,
            ),
            PredictionRow(
                experiment_id="experiment_227",
                dataset="A",
                variant="dual",
                hard_indices=(1,),
                autonomous_alert=True,
            ),
        ]

    def test_freeze_is_deterministic_and_sorted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = freeze_prediction_rows(self._rows(), root / "one.jsonl", root / "one.json")
            second = freeze_prediction_rows(list(reversed(self._rows())), root / "two.jsonl", root / "two.json")
            self.assertEqual(first["prediction_sha256"], second["prediction_sha256"])
            self.assertEqual((root / "one.jsonl").read_bytes(), (root / "two.jsonl").read_bytes())

    def test_verified_loader_rejects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prediction = root / "rows.jsonl"
            manifest = root / "manifest.json"
            freeze_prediction_rows(self._rows(), prediction, manifest)
            prediction.write_text(prediction.read_text() + "{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_verified_prediction_rows(prediction, manifest)

    def test_prediction_payload_has_no_evaluation_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prediction = root / "rows.jsonl"
            manifest = root / "manifest.json"
            freeze_prediction_rows(self._rows(), prediction, manifest)
            text = prediction.read_text(encoding="utf-8")
            for forbidden in ('"tp"', '"fp"', '"f1"', '"y_test"', '"labels"'):
                self.assertNotIn(forbidden, text)

    def test_checkpoint_requires_exact_contract_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.json"
            save_prediction_checkpoint(
                path,
                experiment=227,
                config_hash="config-a",
                input_hash="input-a",
                completed_datasets=["A"],
                rows=self._rows(),
            )
            checkpoint = load_prediction_checkpoint(
                path, experiment=227, config_hash="config-a", input_hash="input-a"
            )
            self.assertEqual(checkpoint["completed_datasets"], ["A"])
            with self.assertRaisesRegex(ValueError, "checkpoint contract mismatch"):
                load_prediction_checkpoint(
                    path, experiment=227, config_hash="config-b", input_hash="input-a"
                )


class PipelineStatusTests(unittest.TestCase):
    def test_experiment_range_is_numeric_and_bounded(self) -> None:
        self.assertEqual(parse_experiment_range("226-235"), tuple(range(226, 236)))
        with self.assertRaises(ValueError):
            parse_experiment_range("225-235")

    def test_status_write_replaces_json_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "status.json"
            write_pipeline_status(path, {"active_experiment": 226, "status": "running"})
            write_pipeline_status(path, {"active_experiment": 227, "status": "running"})
            self.assertEqual(json.loads(path.read_text())["active_experiment"], 227)

    def test_runtime_status_uses_dashboard_timestamp_field(self) -> None:
        source = inspect.getsource(run_pipeline._write_runtime_status)
        self.assertIn("updated_unix", source)

    def test_dashboard_catalog_contains_exp226_through_exp235(self) -> None:
        from serve_rank_dashboard import EXPERIMENTS

        for number in range(226, 236):
            self.assertIn(f"experiment_{number}_hard_alert_quality", EXPERIMENTS)


class ExperimentRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.baseline = BaselineLanes(
            dataset="demo",
            test_size=4,
            hard=frozenset({0, 1, 2}),
            standard_review=frozenset(),
            priority_review=frozenset(),
            hard_tp=0,
            hard_fp=3,
        )
        self.scores = ScoreBundle(
            dataset="demo",
            train_oof={
                "temporal_random_kernel": np.linspace(0.0, 1.0, 100),
                "canonical_normal_state": np.linspace(0.0, 1.0, 100),
                "frequency_image": np.linspace(0.0, 1.0, 100),
            },
            test_scores={
                "temporal_random_kernel": np.asarray([2.0, 2.0, 0.0, 0.0]),
                "canonical_normal_state": np.asarray([2.0, 0.0, 2.0, 0.0]),
                "frequency_image": np.asarray([2.0, 2.0, 2.0, 0.0]),
            },
            provenance={"n_train": 100, "n_test": 4},
        )

    def test_exp227_produces_partitioned_variants(self) -> None:
        rows = generate_rows_for_dataset(227, self.baseline, self.scores)
        by_variant = {row.variant: row for row in rows}
        self.assertEqual(by_variant["dual_only"].hard_indices, (0,))
        self.assertEqual(by_variant["single_or_dual"].hard_indices, (0, 1, 2))

    def test_exp230_is_explicitly_blocked_without_seed_cache(self) -> None:
        rows = generate_rows_for_dataset(230, self.baseline, self.scores)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].evidence["status"], "blocked_missing_frozen_seed_repetitions")
        self.assertFalse(rows[0].autonomous_alert)


class EvaluationBoundaryTests(unittest.TestCase):
    def test_hard_and_review_metrics_are_separate(self) -> None:
        rows = [
            {
                "experiment_id": "experiment_227",
                "dataset": "demo",
                "variant": "dual_only",
                "hard_indices": [0, 1],
                "review_indices": [2],
                "research_indices": [],
            }
        ]
        evaluated = evaluate_prediction_rows(
            rows, label_loader=lambda _: np.asarray([1, 0, 1, 0], dtype=np.int8)
        )
        self.assertEqual(evaluated[0]["hard_tp"], 1)
        self.assertEqual(evaluated[0]["hard_fp"], 1)
        self.assertEqual(evaluated[0]["review_tp"], 1)
        self.assertEqual(evaluated[0]["review_fp"], 0)
        summary = summarize_evaluated_rows(evaluated)
        self.assertEqual(summary[0]["hard_alerts"], 2)
        self.assertEqual(summary[0]["review_candidates"], 1)

    def test_independent_reaggregate_matches_frozen_summary_contract(self) -> None:
        rows = [
            {
                "experiment_id": "experiment_227",
                "dataset": "A",
                "variant": "dual_only",
                "hard_tp": "2",
                "hard_fp": "1",
                "hard_fn": "1",
                "hard_f1": "0.6666666666666666",
                "review_tp": "1",
                "review_fp": "1",
                "research_tp": "0",
                "research_fp": "0",
            },
            {
                "experiment_id": "experiment_227",
                "dataset": "B",
                "variant": "dual_only",
                "hard_tp": "1",
                "hard_fp": "0",
                "hard_fn": "1",
                "hard_f1": "0.6666666666666666",
                "review_tp": "0",
                "review_fp": "1",
                "research_tp": "1",
                "research_fp": "0",
            },
        ]
        summary = reaggregate_result_rows(rows)[0]
        self.assertEqual(summary["datasets"], 2)
        self.assertEqual(summary["hard_alerts"], 4)
        self.assertEqual(summary["hard_tp"], 3)
        self.assertEqual(summary["hard_fp"], 1)
        self.assertAlmostEqual(summary["hard_precision"], 0.75)
        self.assertEqual(summary["review_candidates"], 3)
        self.assertEqual(summary["research_candidates"], 1)


if __name__ == "__main__":
    unittest.main()
