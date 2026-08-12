from __future__ import annotations

import sys
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np

from experiments.paper_public_baselines.evaluation import (
    ProposalAuthority,
    evaluate_frozen_scores,
    evaluate_rows,
    matched_budget_for_dataset,
    write_evaluation_bundle,
)
from experiments.paper_public_baselines.config import METHOD_IDS
from experiments.paper_public_baselines.runner import (
    ScoreRunConfig,
    freeze_score_root,
    run_methods,
)
from experiments.paper_public_baselines.tests.test_runner import (
    CountingScorer,
    _build_database,
)


class EvaluationBoundaryTests(unittest.TestCase):
    def test_labels_are_not_opened_when_manifest_verification_fails(self) -> None:
        loader = Mock()
        proposal = Mock()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "manifest"):
                evaluate_frozen_scores(
                    Path(directory),
                    Path(directory) / "exp448",
                    label_loader=loader,
                    proposal_loader=proposal,
                )

        loader.assert_not_called()
        proposal.assert_not_called()

    def test_evaluation_module_does_not_eagerly_import_label_capability(self) -> None:
        self.assertNotIn(
            "experiments.paper_public_baselines.label_access", sys.modules
        )

    def test_matched_budget_uses_count_not_proposal_coordinates(self) -> None:
        proposal = frozenset({7, 9})
        baseline_scores = np.asarray([0.1, 0.9, 0.8])

        selected = matched_budget_for_dataset(
            baseline_scores, (0, 1, 2), proposal
        )

        self.assertEqual(selected, (1, 2))

    def test_confusion_and_macro_micro_are_exact(self) -> None:
        rows = evaluate_rows(
            predictions={"a": (1,), "b": (0,)},
            labels={"a": (0, 1), "b": (0, 1)},
        )

        self.assertEqual(rows["micro"]["tp"], 1)
        self.assertEqual(rows["micro"]["fp"], 1)
        self.assertEqual(rows["micro"]["fn"], 1)
        self.assertAlmostEqual(rows["micro"]["f1"], 0.5)
        self.assertAlmostEqual(rows["macro"]["f1"], 0.5)

    def test_continuous_metrics_are_computed_only_when_both_classes_exist(self) -> None:
        rows = evaluate_rows(
            predictions={"a": (1,)},
            labels={"a": (0, 1)},
            scores={"a": np.asarray([0.1, 0.9])},
        )

        self.assertEqual(rows["continuous_dataset_count"], 1)
        self.assertEqual(rows["micro"]["average_precision"], 1.0)
        self.assertEqual(rows["micro"]["auroc"], 1.0)

    def test_micro_rank_metrics_include_scored_single_class_datasets(self) -> None:
        rows = evaluate_rows(
            predictions={"mixed": (), "normal_only": ()},
            labels={"mixed": (0, 1), "normal_only": (0, 0)},
            scores={
                "mixed": np.asarray([0.1, 0.9]),
                "normal_only": np.asarray([1.0, 0.95]),
            },
        )

        self.assertAlmostEqual(rows["micro"]["average_precision"], 1.0 / 3.0)
        self.assertAlmostEqual(rows["micro"]["auroc"], 1.0 / 3.0)
        self.assertEqual(rows["macro"]["average_precision"], 1.0)
        self.assertEqual(rows["continuous_dataset_count"], 1)

    def test_complete_injected_evaluation_separates_proposal_and_six_baselines(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "toy.sqlite"
            _build_database(database)
            run_root = root / "run"
            config = ScoreRunConfig(
                db_path=database,
                run_root=run_root,
                datasets=("toy",),
                methods=METHOD_IDS,
                workers=1,
                database_sha256=hashlib.sha256(database.read_bytes()).hexdigest(),
            )
            run_methods(config, scorer=CountingScorer())
            freeze_score_root(run_root)
            authority = ProposalAuthority(
                dataset_order=("toy",),
                alerts_by_dataset={"toy": frozenset({1})},
                expected_confusion=(1, 0, 0, 1),
                author_ablation=(),
                reviewed_commit="a" * 40,
                artifact_map_sha256="b" * 64,
            )

            bundle = evaluate_frozen_scores(
                run_root,
                root / "exp448",
                label_loader=lambda order: {"toy": (0, 1)},
                proposal_loader=lambda path: authority,
            )
            artifact_sha = write_evaluation_bundle(run_root, bundle)

        self.assertEqual(len(bundle.summaries), 50)
        self.assertEqual(len(bundle.bootstrap), 48)
        self.assertTrue(
            any(row["is_external_baseline"] is False for row in bundle.summaries)
        )
        self.assertEqual(len(artifact_sha), 64)


if __name__ == "__main__":
    unittest.main()
