from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.paper_public_baselines.evaluation import EvaluationBundle
from experiments.paper_public_baselines.reporting import (
    public_comparison_section,
    render_report,
    write_report_artifacts,
)


def toy_bundle() -> EvaluationBundle:
    base = {
        "support": "full_population",
        "dataset_count": 2,
        "completed_datasets": 2,
        "abstained_datasets": 0,
        "coverage": 1.0,
        "alerts": 2,
        "tp": 1,
        "fp": 1,
        "fn": 1,
        "tn": 17,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
        "false_positive_rate": 1 / 18,
        "average_precision": 0.6,
        "auroc": 0.7,
        "macro_precision": 0.5,
        "macro_recall": 0.5,
        "macro_f1": 0.5,
        "macro_false_positive_rate": 1 / 18,
        "macro_average_precision": 0.6,
        "macro_auroc": 0.7,
        "continuous_dataset_count": 2,
    }
    summaries = (
        {
            **base,
            "method": "proposed_equipment_anomaly_detection",
            "is_external_baseline": False,
            "protocol": "proposal_frozen",
            "alpha": None,
            "average_precision": None,
            "auroc": None,
        },
        {
            **base,
            "method": "knn_distance",
            "is_external_baseline": True,
            "protocol": "matched_alert_budget",
            "alpha": None,
        },
    )
    return EvaluationBundle(
        metadata={
            "schema": "paper-public-baselines.evaluation.v1",
            "dataset_count": 2,
            "common_support_count": 2,
            "evaluation_status": "retrospective_comparison_complete_no_promotion",
        },
        summaries=summaries,
        dataset_metrics=(),
        bootstrap=(
            {
                "method": "knn_distance",
                "protocol": "matched_alert_budget",
                "alpha": None,
                "support": "full_population",
                "estimand": "dataset_macro_f1_baseline_minus_proposal",
                "point_estimate": -0.1,
                "lower": -0.2,
                "upper": 0.0,
                "draw_count": 2000,
                "seed": 20260812,
                "draw_sha256": "a" * 64,
            },
        ),
        author_ablation=(
            {
                "paper_name": "안정성 선별 적용 제안기술",
                "internal_provenance": "hard_plus_stable_26",
                "tp": 1,
                "fp": 0,
                "fn": 1,
                "tn": 18,
                "f1": 2 / 3,
                "macro_f1": 0.5,
            },
        ),
    )


class ReportingTests(unittest.TestCase):
    def test_report_separates_public_baselines_and_author_ablation(self) -> None:
        text = render_report(toy_bundle())

        self.assertIn("공개 기법 비교", text)
        self.assertIn("제안기술 내부 ablation", text)
        public = public_comparison_section(text)
        self.assertNotIn("Exp404/416", public)
        self.assertNotIn("hard_plus_stable_26", public)

    def test_report_never_claims_field_validation(self) -> None:
        text = render_report(toy_bundle())

        for forbidden in ("양산 검증 완료", "현장 효과 입증", "고장 예방 보장"):
            self.assertNotIn(forbidden, text)

    def test_report_artifacts_are_byte_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = write_report_artifacts(root, toy_bundle())
            second = write_report_artifacts(root, toy_bundle(), verify_only=True)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)


if __name__ == "__main__":
    unittest.main()
