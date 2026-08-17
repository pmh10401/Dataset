from __future__ import annotations

from dataclasses import replace
import tempfile
import unittest
from pathlib import Path

from experiments.paper_public_baselines.evaluation import EvaluationBundle
from experiments.paper_public_baselines.official_comparison_antigravity import (
    EXPECTED_METHOD_ORDER_ANTIGRAVITY,
    combine_official_evaluations_antigravity,
    verify_official_comparison_antigravity,
    write_official_comparison_antigravity,
)


CLASSICAL = EXPECTED_METHOD_ORDER_ANTIGRAVITY[:4]


def _summary(method: str, protocol: str, alpha: float | None) -> dict[str, object]:
    return {
        "method": method,
        "is_external_baseline": method != "proposed_equipment_anomaly_detection",
        "protocol": protocol,
        "alpha": alpha,
        "support": "full_population",
        "dataset_count": 1117,
        "completed_datasets": 1117,
        "abstained_datasets": 0,
        "coverage": 1.0,
        "alerts": 1,
        "tp": 1,
        "fp": 0,
        "fn": 0,
        "tn": 1,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "false_positive_rate": 0.0,
        "average_precision": None if method.startswith("proposed") else 0.75,
        "auroc": None if method.startswith("proposed") else 0.8,
        "macro_precision": 1.0,
        "macro_recall": 1.0,
        "macro_f1": 1.0,
        "macro_false_positive_rate": 0.0,
        "macro_average_precision": None if method.startswith("proposed") else 0.7,
        "macro_auroc": None if method.startswith("proposed") else 0.78,
        "continuous_dataset_count": 0 if method.startswith("proposed") else 1117,
    }


def _bundle(methods: tuple[str, ...], *, database: str = "a" * 64) -> EvaluationBundle:
    summaries: list[dict[str, object]] = [
        _summary("proposed_equipment_anomaly_detection", "proposal_frozen", None)
    ]
    dataset_rows: list[dict[str, object]] = []
    for method in methods:
        for protocol, alpha in (
            ("conformal", 0.005),
            ("conformal", 0.01),
            ("conformal", 0.02),
            ("matched_alert_budget", None),
        ):
            summaries.append(_summary(method, protocol, alpha))
        dataset_rows.append(
            {
                "method": method,
                "is_external_baseline": True,
                "protocol": "matched_alert_budget",
                "alpha": None,
                "support": "full_population",
                "status": "complete",
                "dataset": "toy",
                "tp": 1,
                "fp": 0,
                "fn": 0,
                "tn": 1,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "false_positive_rate": 0.0,
                "alert_count": 1,
                "average_precision": 0.75,
                "auroc": 0.8,
            }
        )
    bootstrap = tuple(
        {
            "method": method,
            "protocol": "matched_alert_budget",
            "alpha": None,
            "support": "full_population",
            "metric": "macro_f1_difference_baseline_minus_proposal",
            "point_estimate": -0.1,
            "ci_lower": -0.2,
            "ci_upper": -0.05,
            "draws": 2000,
            "draw_sha256": "b" * 64,
        }
        for method in methods
    )
    return EvaluationBundle(
        metadata={
            "schema": "paper-public-baselines.evaluation.v1",
            "score_manifest_sha256": "c" * 64,
            "database_sha256": database,
            "dataset_count": 1117,
            "common_support_count": 1117,
            "method_count": len(methods),
            "planned_deep_methods_complete": all(
                method in methods for method in ("deep_svdd_1d", "drocc_1d")
            ),
            "proposal_reviewed_commit": "d" * 40,
            "proposal_artifact_map_sha256": "e" * 64,
            "bootstrap_seed": 20260812,
            "bootstrap_samples": 2000,
            "micro_rank_score": "one_minus_train_oof_upper_tail_p_value",
            "macro_rank_score": "within_dataset_raw_anomaly_score",
            "evaluation_status": "retrospective_comparison_complete_no_promotion",
        },
        summaries=tuple(summaries),
        dataset_metrics=tuple(dataset_rows),
        bootstrap=bootstrap,
        author_ablation=(
            {
                "paper_name": "제안기술",
                "internal_provenance": "hard_plus_stable_26",
                "tp": 1,
                "fp": 0,
                "fn": 0,
                "tn": 1,
                "f1": 1.0,
                "macro_f1": 1.0,
            },
        ),
    )


class OfficialComparisonTests(unittest.TestCase):
    def test_combiner_requires_exact_six_unique_public_methods(self) -> None:
        comparison = combine_official_evaluations_antigravity(
            _bundle(CLASSICAL),
            _bundle(("deep_svdd_1d",)),
            _bundle(("drocc_1d",)),
            drocc_parity_status="user_confirmed_no_difference",
        )

        self.assertEqual(
            comparison.public_methods, EXPECTED_METHOD_ORDER_ANTIGRAVITY
        )
        self.assertEqual(comparison.dataset_count, 1117)
        self.assertEqual(len(comparison.summaries), 25)

    def test_combiner_rejects_database_identity_drift(self) -> None:
        drifted = _bundle(("deep_svdd_1d",), database="0" * 64)
        with self.assertRaisesRegex(ValueError, "database/proposal identity"):
            combine_official_evaluations_antigravity(
                _bundle(CLASSICAL),
                drifted,
                _bundle(("drocc_1d",)),
                drocc_parity_status="user_confirmed_no_difference",
            )

    def test_writer_is_create_only_and_scorer_free_verifier_reloads(self) -> None:
        comparison = combine_official_evaluations_antigravity(
            _bundle(CLASSICAL),
            _bundle(("deep_svdd_1d",)),
            _bundle(("drocc_1d",)),
            drocc_parity_status="user_confirmed_no_difference",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "comparison"
            first = write_official_comparison_antigravity(root, comparison)
            second = write_official_comparison_antigravity(root, comparison)
            reloaded = verify_official_comparison_antigravity(root)

        self.assertEqual(first, second)
        self.assertEqual(reloaded.public_methods, EXPECTED_METHOD_ORDER_ANTIGRAVITY)


if __name__ == "__main__":
    unittest.main()
