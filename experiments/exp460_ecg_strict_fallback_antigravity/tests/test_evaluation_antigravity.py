from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
import unittest


def _clone_antigravity(value: object) -> object:
    cloned = object.__new__(type(value))
    for field in fields(type(value)):
        object.__setattr__(cloned, field.name, getattr(value, field.name))
    return cloned


def _authority_with_mutations_antigravity(
    *,
    passes: tuple[bool, ...] | None = None,
    period_change_aucs: tuple[float, ...] | None = None,
    baseline_confusion: tuple[int, int, int, int] | None = None,
    candidate_confusion: tuple[int, int, int, int] | None = None,
) -> object:
    from experiments.exp460_ecg_strict_fallback_antigravity.predecessor_antigravity import (
        capture_exp459_fallback_authority_antigravity,
    )

    authority = _clone_antigravity(capture_exp459_fallback_authority_antigravity())
    if passes is None:
        passes = tuple(record.passed for record in authority.fold_records)
    if period_change_aucs is None:
        period_change_aucs = tuple(record.period_change_auc for record in authority.fold_records)
    if len(passes) != 10 or len(period_change_aucs) != 10:
        raise AssertionError("expected ten record values")

    records = []
    for original, passed, auc in zip(
        authority.fold_records,
        passes,
        period_change_aucs,
        strict=True,
    ):
        record = _clone_antigravity(original)
        object.__setattr__(record, "passed", passed)
        object.__setattr__(record, "period_change_auc", auc)
        records.append(record)
    record_tuple = tuple(records)
    object.__setattr__(authority, "fold_records", record_tuple)

    summaries = []
    for index, summary in enumerate(authority.seed_summaries):
        start = index * 5
        stop = start + 5
        summary_clone = _clone_antigravity(summary)
        summary_records = record_tuple[start:stop]
        object.__setattr__(summary_clone, "fold_records", summary_records)
        object.__setattr__(
            summary_clone,
            "passed",
            all(record.passed for record in summary_records),
        )
        summaries.append(summary_clone)
    object.__setattr__(authority, "seed_summaries", tuple(summaries))

    decision = _clone_antigravity(authority.decision)
    decision_passed = all(passes)
    object.__setattr__(decision, "passed", decision_passed)
    object.__setattr__(decision, "status", "stable" if decision_passed else "unstable")
    object.__setattr__(authority, "decision", decision)

    if baseline_confusion is not None:
        confusion = _clone_antigravity(authority.baseline_confusion)
        for name, value in zip(("tp", "fp", "fn", "tn"), baseline_confusion, strict=True):
            object.__setattr__(confusion, name, value)
        object.__setattr__(authority, "baseline_confusion", confusion)

    if candidate_confusion is not None:
        confusion = _clone_antigravity(authority.candidate_confusion)
        for name, value in zip(("tp", "fp", "fn", "tn"), candidate_confusion, strict=True):
            object.__setattr__(confusion, name, value)
        object.__setattr__(authority, "candidate_confusion", confusion)
    return authority


class FallbackEvaluationAntigravityTest(unittest.TestCase):
    def test_majority_and_strict_route_thresholds_antigravity(self) -> None:
        """Break caught: majority or strict comparisons route the wrong side of the pass threshold."""
        from experiments.exp460_ecg_strict_fallback_antigravity.evaluation_antigravity import (
            compare_fallback_rules_antigravity,
        )

        table = (
            ("ten_of_ten", (True, True, True, True, True, True, True, True, True, True), "candidate", "candidate"),
            ("nine_of_ten", (False, True, True, True, True, True, True, True, True, True), "candidate", "baseline"),
            ("eight_of_ten", (False, False, True, True, True, True, True, True, True, True), "candidate", "baseline"),
            ("seven_of_ten", (False, False, False, True, True, True, True, True, True, True), "baseline", "baseline"),
            ("zero_of_ten", (False, False, False, False, False, False, False, False, False, False), "baseline", "baseline"),
        )
        for label, passes, expected_majority, expected_strict in table:
            with self.subTest(case=label):
                result = compare_fallback_rules_antigravity(
                    _authority_with_mutations_antigravity(passes=passes)
                )
                self.assertEqual(result.rule_results[2].route, expected_majority)
                self.assertEqual(result.rule_results[3].route, expected_strict)

    def test_compare_recomputes_current_metrics_and_status_antigravity(self) -> None:
        """Break caught: evaluation drifts from the frozen four-row current-authority table."""
        from experiments.exp460_ecg_strict_fallback_antigravity.evaluation_antigravity import (
            compare_fallback_rules_antigravity,
        )
        from experiments.exp460_ecg_strict_fallback_antigravity.predecessor_antigravity import (
            capture_exp459_fallback_authority_antigravity,
        )

        result = compare_fallback_rules_antigravity(
            capture_exp459_fallback_authority_antigravity()
        )
        self.assertEqual(
            [
                (
                    row.rule,
                    row.route,
                    row.tp,
                    row.fp,
                    row.fn,
                    row.tn,
                    row.f1,
                )
                for row in result.rule_results
            ],
            [
                ("always_baseline", "baseline", 609, 66, 837, 70788, 0.5742574257425742),
                ("always_candidate", "candidate", 617, 66, 829, 70788, 0.5796148426491311),
                ("majority_8_of_10", "candidate", 617, 66, 829, 70788, 0.5796148426491311),
                ("strict_10_of_10", "baseline", 609, 66, 837, 70788, 0.5742574257425742),
            ],
        )
        self.assertEqual(result.strict_route, "baseline")
        self.assertEqual(
            result.strict_status,
            "fallback_triggered_performance_not_improved",
        )
        self.assertFalse(result.joint_success)
        self.assertFalse(result.all_records_passed)
        self.assertEqual(result.passed_record_count, 8)
        self.assertEqual(result.worst_period_change_auc, 0.8004124541019063)

    def test_compare_recomputes_f1_from_confusions_antigravity(self) -> None:
        """Break caught: F1 values are trusted instead of recomputed from the confusion matrix."""
        from experiments.exp460_ecg_strict_fallback_antigravity.evaluation_antigravity import (
            compare_fallback_rules_antigravity,
        )

        result = compare_fallback_rules_antigravity(
            _authority_with_mutations_antigravity(
                passes=(True, True, True, True, True, True, True, True, True, True),
                period_change_aucs=(
                    0.91,
                    0.92,
                    0.93,
                    0.94,
                    0.95,
                    0.96,
                    0.97,
                    0.98,
                    0.99,
                    0.951,
                ),
            )
        )
        self.assertEqual(result.rule_results[0].f1, 1218 / 2121)
        self.assertEqual(result.rule_results[1].f1, 1234 / 2129)
        self.assertEqual(result.rule_results[3].f1, 1234 / 2129)
        self.assertEqual(result.strict_route, "candidate")
        self.assertEqual(result.strict_status, "candidate_retained_joint_success")
        self.assertTrue(result.joint_success)

    def test_result_contracts_are_exact_immutable_and_reject_forgery_antigravity(self) -> None:
        """Break caught: evaluation result objects accept reordered rows, bad metrics, or forged status."""
        from experiments.exp460_ecg_strict_fallback_antigravity.contracts_antigravity import (
            FallbackEvaluationAntigravity,
            FallbackRuleComparisonAntigravity,
            validate_fallback_evaluation_antigravity,
        )
        from experiments.exp460_ecg_strict_fallback_antigravity.evaluation_antigravity import (
            compare_fallback_rules_antigravity,
        )

        authority = _authority_with_mutations_antigravity(
            passes=(True, True, True, True, True, True, True, True, True, True),
            period_change_aucs=(
                0.91,
                0.92,
                0.93,
                0.94,
                0.95,
                0.96,
                0.97,
                0.98,
                0.99,
                0.951,
            ),
        )
        result = compare_fallback_rules_antigravity(authority)
        self.assertEqual(
            tuple(field.name for field in fields(FallbackRuleComparisonAntigravity)),
            ("rule", "route", "tp", "fp", "fn", "tn", "f1"),
        )
        self.assertEqual(
            tuple(field.name for field in fields(FallbackEvaluationAntigravity)),
            (
                "rule_results",
                "strict_route",
                "strict_status",
                "joint_success",
                "all_records_passed",
                "passed_record_count",
                "worst_period_change_auc",
            ),
        )
        with self.assertRaises(FrozenInstanceError):
            result.strict_status = result.strict_status  # type: ignore[misc]
        self.assertIs(
            type(validate_fallback_evaluation_antigravity(result, authority)),
            FallbackEvaluationAntigravity,
        )

        forged_rows = result.rule_results[1:] + result.rule_results[:1]
        with self.assertRaises(ValueError):
            validate_fallback_evaluation_antigravity(
                FallbackEvaluationAntigravity(
                    rule_results=forged_rows,
                    strict_route=result.strict_route,
                    strict_status=result.strict_status,
                    joint_success=result.joint_success,
                    all_records_passed=result.all_records_passed,
                    passed_record_count=result.passed_record_count,
                    worst_period_change_auc=result.worst_period_change_auc,
                ),
                authority,
            )

        with self.assertRaises(ValueError):
            validate_fallback_evaluation_antigravity(
                FallbackEvaluationAntigravity(
                    rule_results=result.rule_results,
                    strict_route=result.strict_route,
                    strict_status="fallback_triggered_performance_not_improved",
                    joint_success=True,
                    all_records_passed=result.all_records_passed,
                    passed_record_count=result.passed_record_count,
                    worst_period_change_auc=result.worst_period_change_auc,
                ),
                authority,
            )

        with self.assertRaises(ValueError):
            FallbackRuleComparisonAntigravity(
                rule="always_baseline",
                route="baseline",
                tp=True,
                fp=66,
                fn=837,
                tn=70788,
                f1=0.5742574257425742,
            )

        with self.assertRaises(ValueError):
            FallbackEvaluationAntigravity(
                rule_results=result.rule_results,
                strict_route="candidate",
                strict_status="candidate_retained_joint_success",
                joint_success=True,
                all_records_passed=result.all_records_passed,
                passed_record_count=result.passed_record_count,
                worst_period_change_auc=float("nan"),
            )

    def test_compare_rejects_malformed_authority_antigravity(self) -> None:
        """Break caught: evaluation accepts malformed authority instead of failing fast."""
        from experiments.exp460_ecg_strict_fallback_antigravity.evaluation_antigravity import (
            compare_fallback_rules_antigravity,
        )

        malformed = _authority_with_mutations_antigravity()
        object.__setattr__(malformed, "fold_records", malformed.fold_records[:-1])
        with self.assertRaises(ValueError):
            compare_fallback_rules_antigravity(malformed)

        malformed = _authority_with_mutations_antigravity()
        record = _clone_antigravity(malformed.fold_records[0])
        object.__setattr__(record, "passed", 0)
        object.__setattr__(
            malformed,
            "fold_records",
            (record,) + malformed.fold_records[1:],
        )
        with self.assertRaises(ValueError):
            compare_fallback_rules_antigravity(malformed)

        with self.assertRaises(ValueError):
            compare_fallback_rules_antigravity(
                _authority_with_mutations_antigravity(
                    passes=(True, True, True, True, True, True, True, True, True, True),
                    period_change_aucs=(
                        0.91,
                        0.92,
                        0.93,
                        0.94,
                        0.95,
                        0.96,
                        0.97,
                        0.98,
                        0.99,
                        0.951,
                    ),
                    baseline_confusion=(609, 70, 900, 70788),
                )
            )

        with self.assertRaises(ValueError):
            compare_fallback_rules_antigravity(
                _authority_with_mutations_antigravity(
                    baseline_confusion=(610, 66, 837, 70788),
                )
            )

        with self.assertRaises(ValueError):
            compare_fallback_rules_antigravity(
                _authority_with_mutations_antigravity(
                    candidate_confusion=(617, 66, 830, 70788),
                )
            )

        with self.assertRaises(ValueError):
            compare_fallback_rules_antigravity(object())


if __name__ == "__main__":
    unittest.main()
