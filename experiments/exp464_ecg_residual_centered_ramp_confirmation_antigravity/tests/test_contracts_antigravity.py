from __future__ import annotations

from dataclasses import asdict
from collections import OrderedDict
from dataclasses import FrozenInstanceError, fields, replace
from types import MappingProxyType
import unittest


class ContractsAntigravityTest(unittest.TestCase):
    def _summary_antigravity(
        self,
        method: str,
        *,
        fold_count: int = 10,
        passed_count: int,
        failed_count: int,
        min_gradual_drift_auc: float,
        min_period_change_auc: float,
        min_pooled_auc: float,
        max_normal_wilson_upper: float = 0.013439124039399194,
        all_scores_finite: bool = True,
    ):
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.contracts_antigravity import (
            Exp463MethodSummaryAntigravity,
        )

        return Exp463MethodSummaryAntigravity(
            method=method,
            fold_count=fold_count,
            passed_count=passed_count,
            failed_count=failed_count,
            min_gradual_drift_auc=min_gradual_drift_auc,
            min_period_change_auc=min_period_change_auc,
            min_pooled_auc=min_pooled_auc,
            max_normal_wilson_upper=max_normal_wilson_upper,
            all_scores_finite=all_scores_finite,
        )

    def test_contract_inventory_is_immutable_and_keeps_c0_c1_c2_c7_order_antigravity(self) -> None:
        """Break caught: authority contracts lose exact field order, immutability, or control order."""
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.contracts_antigravity import (
            Exp463AuthorityAntigravity,
            Exp463MethodSummaryAntigravity,
            validate_exp463_authority_antigravity,
        )
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.predecessor_antigravity import (
            capture_exp463_authority_antigravity,
        )

        self.assertEqual(
            tuple(field.name for field in fields(Exp463MethodSummaryAntigravity)),
            (
                "method",
                "fold_count",
                "passed_count",
                "failed_count",
                "min_gradual_drift_auc",
                "min_period_change_auc",
                "min_pooled_auc",
                "max_normal_wilson_upper",
                "all_scores_finite",
            ),
        )
        self.assertEqual(
            tuple(field.name for field in fields(Exp463AuthorityAntigravity)),
            (
                "predecessor_experiment",
                "terminal_closeout_commit",
                "task5_report_sha256",
                "status",
                "reviewed_cli_commit",
                "private_run_count",
                "result_row_count",
                "result_failed_row_count",
                "result_summary_count",
                "result_leaf_count",
                "method_fold_metrics_sha256",
                "artifact_map_sha256",
                "official_evaluate_count",
                "official_inspect_count",
                "label_query_count",
                "control_summaries",
                "control_row_digests",
            ),
        )

        authority = capture_exp463_authority_antigravity()
        self.assertIsNone(validate_exp463_authority_antigravity(authority))
        self.assertEqual(tuple(authority.control_summaries), ("C0", "C1", "C2", "C7"))
        self.assertIsInstance(authority.control_summaries, MappingProxyType)
        with self.assertRaises(TypeError):
            authority.control_summaries["C0"] = authority.control_summaries["C0"]  # type: ignore[index]
        with self.assertRaises(FrozenInstanceError):
            authority.status = "different"  # type: ignore[misc]

    def test_summary_type_has_real_runtime_identity_and_exact_type_boundary_antigravity(self) -> None:
        """Break caught: summary contract is an alias or accepts lookalike runtime types."""
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity import contracts_antigravity
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.config_antigravity import (
            Exp463MethodSummarySnapshotAntigravity,
        )
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.contracts_antigravity import (
            Exp463AuthorityAntigravity,
            Exp463MethodSummaryAntigravity,
        )

        summary = self._summary_antigravity(
            "C0",
            passed_count=9,
            failed_count=1,
            min_gradual_drift_auc=0.8402306222021025,
            min_period_change_auc=0.907392736783864,
            min_pooled_auc=0.9212266108344651,
        )
        self.assertEqual(Exp463MethodSummaryAntigravity.__name__, "Exp463MethodSummaryAntigravity")
        self.assertEqual(
            Exp463MethodSummaryAntigravity.__module__,
            "experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.contracts_antigravity",
        )
        self.assertIs(contracts_antigravity.Exp463MethodSummaryAntigravity, Exp463MethodSummaryAntigravity)
        self.assertTrue(repr(summary).startswith("Exp463MethodSummaryAntigravity("))
        self.assertEqual(asdict(summary)["method"], "C0")
        self.assertNotEqual(type(summary).__name__, Exp463MethodSummarySnapshotAntigravity.__name__)

        class SummarySubclassAntigravity(Exp463MethodSummaryAntigravity):
            pass

        subclass_summary = SummarySubclassAntigravity(
            method="C0",
            fold_count=10,
            passed_count=9,
            failed_count=1,
            min_gradual_drift_auc=0.8402306222021025,
            min_period_change_auc=0.907392736783864,
            min_pooled_auc=0.9212266108344651,
            max_normal_wilson_upper=0.013439124039399194,
            all_scores_finite=True,
        )
        snapshot_summary = Exp463MethodSummarySnapshotAntigravity(
            method="C0",
            fold_count=10,
            passed_count=9,
            failed_count=1,
            min_gradual_drift_auc=0.8402306222021025,
            min_period_change_auc=0.907392736783864,
            min_pooled_auc=0.9212266108344651,
            max_normal_wilson_upper=0.013439124039399194,
            all_scores_finite=True,
        )
        for forged in (subclass_summary, snapshot_summary):
            with self.subTest(forged_type=type(forged).__name__), self.assertRaises(ValueError):
                Exp463AuthorityAntigravity(
                    predecessor_experiment="Exp463",
                    terminal_closeout_commit="147225831e9c13e9df5043fa88757bb03e01b79a",
                    task5_report_sha256="e43abf66e43921941dece398514858be91e39d121c1173b26b352b67ba78e8e5",
                    status="stopped_before_official_evaluation_qualification_failed",
                    reviewed_cli_commit="923cc4ad6dd6737f84f4ea2c4f621eb652ac54b4",
                    private_run_count=2,
                    result_row_count=70,
                    result_failed_row_count=30,
                    result_summary_count=7,
                    result_leaf_count=7,
                    method_fold_metrics_sha256="84f610e437c59cb6dce4bd73e2ae8ec09c056d7a41178c3f7b790c0f01803fa6",
                    artifact_map_sha256="25fb4ab237ae8f2601c2bc03b05f4e6a2d0cb9889876a83fa0f2c4ee7c4f0ea5",
                    official_evaluate_count=0,
                    official_inspect_count=0,
                    label_query_count=0,
                    control_summaries=OrderedDict(
                        (
                            ("C0", forged),
                            ("C1", self._summary_antigravity("C1", passed_count=7, failed_count=3, min_gradual_drift_auc=0.83450279161008, min_period_change_auc=0.8984897640963734, min_pooled_auc=0.9128093405764297)),
                            ("C2", self._summary_antigravity("C2", passed_count=9, failed_count=1, min_gradual_drift_auc=0.8399225391076908, min_period_change_auc=0.9044753785020874, min_pooled_auc=0.9193733967104271)),
                            ("C7", self._summary_antigravity("C7", passed_count=3, failed_count=7, min_gradual_drift_auc=0.7953196519289775, min_period_change_auc=0.8960376741612595, min_pooled_auc=0.9010220185101353)),
                        )
                    ),
                    control_row_digests=OrderedDict(
                        (f"C0:20260721:{fold}", "0" * 64) for fold in range(5)
                    ),
                )

    def test_authority_constructor_copies_mapping_inputs_before_freezing_antigravity(self) -> None:
        """Break caught: authority keeps a mutable alias to caller-owned control mappings."""
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.contracts_antigravity import (
            Exp463AuthorityAntigravity,
            Exp463MethodSummaryAntigravity,
        )
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.predecessor_antigravity import (
            capture_exp463_authority_antigravity,
        )

        summaries = OrderedDict(
            (
                (
                    "C0",
                    Exp463MethodSummaryAntigravity(
                        method="C0",
                        fold_count=10,
                        passed_count=9,
                        failed_count=1,
                        min_gradual_drift_auc=0.8402306222021025,
                        min_period_change_auc=0.907392736783864,
                        min_pooled_auc=0.9212266108344651,
                        max_normal_wilson_upper=0.013439124039399194,
                        all_scores_finite=True,
                    ),
                ),
                (
                    "C1",
                    Exp463MethodSummaryAntigravity(
                        method="C1",
                        fold_count=10,
                        passed_count=7,
                        failed_count=3,
                        min_gradual_drift_auc=0.83450279161008,
                        min_period_change_auc=0.8984897640963734,
                        min_pooled_auc=0.9128093405764297,
                        max_normal_wilson_upper=0.013439124039399194,
                        all_scores_finite=True,
                    ),
                ),
                (
                    "C2",
                    Exp463MethodSummaryAntigravity(
                        method="C2",
                        fold_count=10,
                        passed_count=9,
                        failed_count=1,
                        min_gradual_drift_auc=0.8399225391076908,
                        min_period_change_auc=0.9044753785020874,
                        min_pooled_auc=0.9193733967104271,
                        max_normal_wilson_upper=0.013439124039399194,
                        all_scores_finite=True,
                    ),
                ),
                (
                    "C7",
                    Exp463MethodSummaryAntigravity(
                        method="C7",
                        fold_count=10,
                        passed_count=3,
                        failed_count=7,
                        min_gradual_drift_auc=0.7953196519289775,
                        min_period_change_auc=0.8960376741612595,
                        min_pooled_auc=0.9010220185101353,
                        max_normal_wilson_upper=0.013439124039399194,
                        all_scores_finite=True,
                    ),
                ),
            )
        )
        frozen_authority = capture_exp463_authority_antigravity()

        authority = Exp463AuthorityAntigravity(
            predecessor_experiment="Exp463",
            terminal_closeout_commit="147225831e9c13e9df5043fa88757bb03e01b79a",
            task5_report_sha256="e43abf66e43921941dece398514858be91e39d121c1173b26b352b67ba78e8e5",
            status="stopped_before_official_evaluation_qualification_failed",
            reviewed_cli_commit="923cc4ad6dd6737f84f4ea2c4f621eb652ac54b4",
            private_run_count=2,
            result_row_count=70,
            result_failed_row_count=30,
            result_summary_count=7,
            result_leaf_count=7,
            method_fold_metrics_sha256="84f610e437c59cb6dce4bd73e2ae8ec09c056d7a41178c3f7b790c0f01803fa6",
            artifact_map_sha256="25fb4ab237ae8f2601c2bc03b05f4e6a2d0cb9889876a83fa0f2c4ee7c4f0ea5",
            official_evaluate_count=0,
            official_inspect_count=0,
            label_query_count=0,
            control_summaries=summaries,
            control_row_digests=OrderedDict(frozen_authority.control_row_digests),
        )
        summaries["C0"] = summaries["C1"]
        self.assertEqual(authority.control_summaries["C0"].passed_count, 9)
        self.assertEqual(tuple(authority.control_summaries), ("C0", "C1", "C2", "C7"))

    def test_validator_rejects_forged_terminal_fact_and_each_control_field_antigravity(self) -> None:
        """Break caught: validator misses a forged terminal fact or altered C0/C1/C2/C7 field."""
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.contracts_antigravity import (
            validate_exp463_authority_antigravity,
        )
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.predecessor_antigravity import (
            capture_exp463_authority_antigravity,
        )

        authority = capture_exp463_authority_antigravity()
        terminal_mutations = (
            ("predecessor_experiment", "Exp462"),
            ("terminal_closeout_commit", "147225831e9c13e9df5043fa88757bb03e01b790"),
            ("task5_report_sha256", "0" * 64),
            ("status", "qualification_failed_blocked_before_routing_research"),
            ("reviewed_cli_commit", "023cc4ad6dd6737f84f4ea2c4f621eb652ac54b4"),
            ("private_run_count", 3),
            ("result_row_count", 71),
            ("result_failed_row_count", 29),
            ("result_summary_count", 6),
            ("result_leaf_count", 8),
            ("method_fold_metrics_sha256", "8" * 64),
            ("artifact_map_sha256", "1" * 64),
            ("official_evaluate_count", 1),
            ("official_inspect_count", 1),
            ("label_query_count", 1),
        )
        for field_name, value in terminal_mutations:
            with self.subTest(field_name=field_name), self.assertRaises(ValueError):
                validate_exp463_authority_antigravity(replace(authority, **{field_name: value}))

        for method in ("C0", "C1", "C2", "C7"):
            summary = authority.control_summaries[method]
            summary_mutations = (
                {"method": "C1" if method != "C1" else "C0"},
                {"fold_count": summary.fold_count + 1, "failed_count": summary.failed_count + 1},
                {
                    "passed_count": summary.passed_count - 1,
                    "failed_count": summary.failed_count + 1,
                },
                {"passed_count": summary.passed_count + 1, "failed_count": summary.failed_count - 1},
                {"min_gradual_drift_auc": summary.min_gradual_drift_auc + 0.001},
                {"min_period_change_auc": summary.min_period_change_auc + 0.001},
                {"min_pooled_auc": summary.min_pooled_auc + 0.001},
                {"max_normal_wilson_upper": summary.max_normal_wilson_upper + 0.001},
                {"all_scores_finite": False if summary.all_scores_finite else True},
            )
            for changes in summary_mutations:
                forged = replace(summary, **changes)
                forged_map = OrderedDict(authority.control_summaries)
                forged_map[method] = forged
                with self.subTest(method=method, changes=changes), self.assertRaises(ValueError):
                    validate_exp463_authority_antigravity(
                        replace(authority, control_summaries=forged_map)
                    )

        forged_digests = OrderedDict(authority.control_row_digests)
        first_key = next(iter(forged_digests))
        forged_digests[first_key] = "f" * 64
        with self.assertRaises(ValueError):
            validate_exp463_authority_antigravity(
                replace(authority, control_row_digests=forged_digests)
            )


if __name__ == "__main__":
    unittest.main()
