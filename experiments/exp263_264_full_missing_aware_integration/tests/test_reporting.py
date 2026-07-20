from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path


def _metrics(selected: list[int], truth: set[int]) -> dict[str, float | int]:
    selected_set = set(selected)
    tp = len(selected_set & truth)
    fp = len(selected_set - truth)
    fn = len(truth - selected_set)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "alerts": len(selected_set),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "f1": f1,
    }


def _result_row(
    dataset: str,
    truth: set[int],
    *,
    baseline_hard: list[int],
    baseline_standard: list[int],
    baseline_priority: list[int],
    final_hard: list[int],
    final_standard: list[int],
    final_priority: list[int],
    sources: dict[str, list[int]],
    n_train: int = 100,
    quality_flags: list[int] = [],
    promoted: list[int] = [],
    quality_blocked: list[int] = [],
) -> dict[str, object]:
    row: dict[str, object] = {
        "dataset": dataset,
        "n_train": n_train,
        "n_test": 4,
        "true_anomaly_count": len(truth),
        "baseline_hard_indices": baseline_hard,
        "baseline_standard_indices": baseline_standard,
        "baseline_priority_indices": baseline_priority,
        "final_hard_indices": final_hard,
        "final_standard_indices": final_standard,
        "final_priority_indices": final_priority,
        "promoted_indices": promoted,
        "quality_blocked_indices": quality_blocked,
        "quality_flag_indices": quality_flags,
        "source_indices": sources,
    }
    lanes = {
        "baseline_hard": baseline_hard,
        "autonomous_hard": final_hard,
        "standard_review": final_standard,
        "priority_review": final_priority,
        "review": [*final_standard, *final_priority],
    }
    for lane, indices in lanes.items():
        for name, value in _metrics(indices, truth).items():
            row[f"{lane}_{name}"] = value
    return row


def _write_results(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, separators=(",", ":"))
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )


class IndependentReportingTest(unittest.TestCase):
    def test_generates_independent_diagnostics_and_manifest_without_self_hash(self) -> None:
        from experiments.exp263_264_full_missing_aware_integration.reporting import (
            generate_report,
            reaggregate_results,
        )

        rows = [
            _result_row(
                "first",
                {0, 2},
                baseline_hard=[0],
                baseline_standard=[1, 2],
                baseline_priority=[],
                final_hard=[0, 2],
                final_standard=[1],
                final_priority=[],
                promoted=[2],
                sources={
                    "statistical": [0, 2, 3],
                    "structural": [2],
                    "change_point": [2],
                    "reconstruction": [1],
                },
            ),
            _result_row(
                "second",
                {1},
                baseline_hard=[],
                baseline_standard=[0, 3],
                baseline_priority=[1],
                final_hard=[],
                final_standard=[0],
                final_priority=[1, 3],
                quality_flags=[3],
                quality_blocked=[3],
                sources={
                    "statistical": [3],
                    "structural": [0, 3],
                    "change_point": [3],
                    "reconstruction": [],
                },
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_results(root / "results.csv", rows)
            (root / "frozen_predictions.jsonl").write_text("{}\n", encoding="utf-8")
            (root / "prediction_manifest.json").write_text("{}\n", encoding="utf-8")
            (root / "lane_transitions.csv").write_text("dataset\n", encoding="utf-8")
            (root / "source_agreement_diagnostics.csv").write_text(
                "dataset\n", encoding="utf-8"
            )
            database = root / "input.db"
            database.write_bytes(b"fixture database")
            summary = {
                "datasets": len(rows),
                "retrospective": True,
                "selected_from_prior_test_review": True,
                "deployment_eligible": False,
                **reaggregate_results(rows),
            }
            (root / "summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )

            generated = generate_report(
                root,
                database_path=database,
                truth_loader={"first": {0, 2}, "second": {1}}.__getitem__,
            )

            self.assertTrue(generated["summary_agreement"]["exact"])
            diagnostics = generated["source_diagnostics"]
            self.assertEqual(diagnostics["sources"]["statistical"]["raw_candidates"], 4)
            self.assertEqual(diagnostics["sources"]["statistical"]["tp"], 2)
            self.assertEqual(diagnostics["sources"]["statistical"]["overlap"]["outside"], 1)
            self.assertEqual(diagnostics["agreement_count_distribution"], {"0": 0, "1": 2, "2": 0, "3": 2, "4": 0})
            self.assertEqual(diagnostics["promoted"]["tp"], 1)
            self.assertEqual(diagnostics["quality_blocked"]["fp"], 1)
            self.assertEqual(diagnostics["hard_confidence_confirmed"]["tp"], 1)
            self.assertEqual(diagnostics["workload_change"]["standard_review"]["delta"], -2)
            self.assertEqual(diagnostics["workload_change"]["priority_review"]["delta"], 1)
            self.assertTrue((root / "independent_reaggregation.json").is_file())
            self.assertTrue((root / "RESULT_REPORT.md").is_file())
            self.assertIn(
                "Aggregate source agreement on baseline Standard candidates across all 2 result rows",
                (root / "RESULT_REPORT.md").read_text(encoding="utf-8"),
            )
            manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
            self.assertEqual(set(manifest["outputs"]), {
                "RESULT_REPORT.md",
                "independent_reaggregation.json",
            })
            self.assertTrue({
                "reporting.py", "config.py", "prediction.py", "routing.py",
                "evaluation.py", "inputs.py", "database.py",
            } <= {Path(path).name for path in manifest["code"]})
            self.assertIsInstance(manifest["git_commit_sha"], str)

    def test_rejects_summary_metric_mismatch(self) -> None:
        from experiments.exp263_264_full_missing_aware_integration.reporting import (
            require_exact_summary_agreement,
        )

        lane = {
            "datasets": 1, "alerts": 1, "tp": 1, "fp": 0,
            "micro_precision": 1.0, "macro_mean_f1": 1.0,
            "zero_f1_datasets": 0,
        }
        actual = {"datasets": 1, **{name: dict(lane) for name in (
            "baseline_hard", "autonomous_hard", "standard_review", "priority_review", "review",
        )}}
        expected = {name: dict(value) if type(value) is dict else value for name, value in actual.items()}
        expected["autonomous_hard"]["fp"] = 1

        with self.assertRaisesRegex(
            ValueError, "summary mismatch: lane=autonomous_hard field=fp"
        ):
            require_exact_summary_agreement(actual, expected)

    def test_rejects_impossible_promoted_indices_before_loading_truth(self) -> None:
        from experiments.exp263_264_full_missing_aware_integration.reporting import (
            source_diagnostics,
        )

        row = _result_row(
            "impossible_promotion", {2}, baseline_hard=[], baseline_standard=[1, 2],
            baseline_priority=[], final_hard=[1], final_standard=[2], final_priority=[],
            promoted=[1], sources={
                "statistical": [2], "structural": [2], "change_point": [2],
                "reconstruction": [],
            },
        )
        with self.assertRaisesRegex(ValueError, "routing mismatch: .*promoted_indices"):
            source_diagnostics([row], truth_loader=lambda _: self.fail("truth must not load"))

    def test_rejects_stale_quality_blocked_indices_before_loading_truth(self) -> None:
        from experiments.exp263_264_full_missing_aware_integration.reporting import (
            source_diagnostics,
        )

        row = _result_row(
            "stale_quality", {2}, baseline_hard=[], baseline_standard=[1, 2],
            baseline_priority=[], final_hard=[2], final_standard=[1], final_priority=[],
            promoted=[2], quality_blocked=[1], sources={
                "statistical": [2], "structural": [2], "change_point": [2],
                "reconstruction": [],
            },
        )
        with self.assertRaisesRegex(ValueError, "routing mismatch: .*quality_blocked_indices"):
            source_diagnostics([row], truth_loader=lambda _: self.fail("truth must not load"))


if __name__ == "__main__":
    unittest.main()
