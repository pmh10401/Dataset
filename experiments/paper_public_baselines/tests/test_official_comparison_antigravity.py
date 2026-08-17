from __future__ import annotations

from dataclasses import replace
import tempfile
import unittest
from pathlib import Path

from experiments.paper_public_baselines.evaluation import EvaluationBundle
from experiments.paper_public_baselines.official_comparison_antigravity import (
    EXPECTED_METHOD_ORDER_ANTIGRAVITY,
    Exp449ContinuousProposalAntigravity,
    _validate_exp449_continuous_shard_antigravity,
    attach_exp449_continuous_metrics_antigravity,
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
    def test_exp449_continuous_shard_preserves_alert_boundary(self) -> None:
        shard = {
            "schema": "exp449-proposal-shard-antigravity.v1",
            "catalog_sha256": "a" * 64,
            "backend_record": {"selected_backend": "cpu"},
            "scientific_payload": {
                "dataset": "alpha",
                "test_identity": {
                    "dataset": "alpha",
                    "split": "TEST",
                    "count": 4,
                    "indices": [0, 1, 2, 3],
                },
                "policies": [
                    {
                        "policy_id": policy,
                        "order": order,
                        "alerts": [1, 3] if policy == "hard_plus_stable" else [],
                        "scores": [0.1, 0.8, 0.4, 0.9]
                        if policy == "hard_plus_stable"
                        else [0.1, 0.2, 0.3, 0.4],
                        "score_mask": [False, True, False, True]
                        if policy == "hard_plus_stable"
                        else [False, False, False, False],
                    }
                    for order, policy in enumerate(
                        (
                            "hard_only",
                            "hard_plus_source",
                            "hard_plus_stable",
                            "hard_plus_unstable",
                        )
                    )
                ],
            },
        }

        alerts, scores, policies = _validate_exp449_continuous_shard_antigravity(
            shard,
            dataset="alpha",
            catalog_sha256="a" * 64,
            backend_record={"selected_backend": "cpu"},
        )

        self.assertEqual(alerts, frozenset({1, 3}))
        self.assertEqual(scores.tolist(), [0.1, 0.8, 0.4, 0.9])
        self.assertEqual(policies["hard_plus_stable"], frozenset({1, 3}))

    def test_attaches_proposal_ap_and_auroc_without_changing_alerts(self) -> None:
        authority = Exp449ContinuousProposalAntigravity(
            dataset_order=("alpha",),
            alerts_by_dataset={"alpha": frozenset({1, 3})},
            scores_by_dataset={"alpha": (0.1, 0.8, 0.4, 0.9)},
            policy_alerts_by_dataset={
                "alpha": {
                    "hard_only": frozenset(),
                    "hard_plus_source": frozenset(),
                    "hard_plus_stable": frozenset({1, 3}),
                    "hard_plus_unstable": frozenset(),
                }
            },
            manifest_sha256="b" * 64,
            context_sha256="c" * 64,
            catalog_sha256="d" * 64,
            test_row_count=4,
        )
        base = EvaluationBundle(
            metadata={"schema": "paper-public-baselines.evaluation.v1"},
            summaries=(
                {
                    **_summary(
                        "proposed_equipment_anomaly_detection",
                        "proposal_frozen",
                        None,
                    ),
                    "dataset_count": 1,
                },
            ),
            dataset_metrics=(
                {
                    "method": "proposed_equipment_anomaly_detection",
                    "is_external_baseline": False,
                    "protocol": "proposal_frozen",
                    "alpha": None,
                    "support": "full_population",
                    "status": "complete",
                    "dataset": "alpha",
                    "tp": 2,
                    "fp": 0,
                    "fn": 0,
                    "tn": 2,
                    "precision": 1.0,
                    "recall": 1.0,
                    "f1": 1.0,
                    "false_positive_rate": 0.0,
                    "alert_count": 2,
                    "average_precision": None,
                    "auroc": None,
                },
            ),
            bootstrap=(),
            author_ablation=(),
        )

        updated = attach_exp449_continuous_metrics_antigravity(
            base, authority, {"alpha": (0, 1, 0, 1)}
        )

        proposal = updated.summaries[0]
        self.assertEqual(proposal["alerts"], 2)
        self.assertEqual(proposal["average_precision"], 1.0)
        self.assertEqual(proposal["auroc"], 1.0)
        self.assertTrue(updated.metadata["proposal_continuous_score_available"])
        self.assertEqual(
            updated.metadata["proposal_score_semantics"],
            "route_preserving_composite_anomaly_score",
        )

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

    def test_combiner_accepts_exp449_continuous_metrics_when_required(self) -> None:
        bundles = []
        for methods in (CLASSICAL, ("deep_svdd_1d",), ("drocc_1d",)):
            bundle = _bundle(methods)
            summaries = []
            for row in bundle.summaries:
                changed = dict(row)
                if changed["method"] == "proposed_equipment_anomaly_detection":
                    changed.update(
                        average_precision=0.9,
                        auroc=0.95,
                        macro_average_precision=0.85,
                        macro_auroc=0.9,
                        continuous_dataset_count=1000,
                    )
                summaries.append(changed)
            bundles.append(
                replace(
                    bundle,
                    metadata={
                        **bundle.metadata,
                        "schema": "paper-public-baselines.evaluation-exp449-continuous-antigravity.v1",
                        "proposal_continuous_score_available": True,
                        "proposal_score_semantics": "route_preserving_composite_anomaly_score",
                    },
                    summaries=tuple(summaries),
                )
            )

        comparison = combine_official_evaluations_antigravity(
            *bundles,
            drocc_parity_status="user_confirmed_no_difference",
            require_proposal_continuous_antigravity=True,
        )

        self.assertTrue(comparison.metadata["proposal_continuous_score_available"])
        self.assertEqual(comparison.summaries[0]["average_precision"], 0.9)


if __name__ == "__main__":
    unittest.main()
