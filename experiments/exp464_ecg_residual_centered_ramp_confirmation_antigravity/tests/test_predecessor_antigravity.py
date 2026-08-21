from __future__ import annotations

from copy import deepcopy
import inspect
from pathlib import Path
from unittest.mock import patch
import unittest


class PredecessorAntigravityTest(unittest.TestCase):
    def _load_versioned_evidence_antigravity(self):
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity import (
            predecessor_antigravity,
        )

        state, catalog, champion = predecessor_antigravity._exp463_ledger_rows_antigravity()
        report_text = predecessor_antigravity._report_bytes_antigravity().decode("utf-8")
        implementation_report_text = predecessor_antigravity._IMPLEMENTATION_REPORT_ANTIGRAVITY.read_text(
            encoding="utf-8"
        )
        return predecessor_antigravity, deepcopy(state), deepcopy(catalog), deepcopy(champion), report_text, implementation_report_text

    def _capture_with_evidence_antigravity(
        self,
        state: dict,
        catalog: dict,
        champion: dict,
        report_text: str,
        implementation_report_text: str,
    ):
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity import (
            predecessor_antigravity,
        )

        original_read_text = Path.read_text

        def patched_read_text(path_self: Path, *args, **kwargs):
            if path_self == predecessor_antigravity._IMPLEMENTATION_REPORT_ANTIGRAVITY:
                return implementation_report_text
            return original_read_text(path_self, *args, **kwargs)

        with (
            patch.object(
                predecessor_antigravity,
                "_exp463_ledger_rows_antigravity",
                return_value=(state, catalog, champion),
            ),
            patch.object(
                predecessor_antigravity,
                "_report_bytes_antigravity",
                return_value=report_text.encode("utf-8"),
            ),
            patch("pathlib.Path.read_text", new=patched_read_text),
        ):
            return predecessor_antigravity.capture_exp463_authority_antigravity()

    def test_capture_returns_exact_frozen_exp463_authority_antigravity(self) -> None:
        """Break caught: Exp464 consumes non-terminal or altered Exp463 private TRAIN authority."""
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.predecessor_antigravity import (
            capture_exp463_authority_antigravity,
        )

        authority = capture_exp463_authority_antigravity()
        self.assertEqual(authority.predecessor_experiment, "Exp463")
        self.assertEqual(
            authority.terminal_closeout_commit,
            "147225831e9c13e9df5043fa88757bb03e01b79a",
        )
        self.assertEqual(
            authority.task5_report_sha256,
            "e43abf66e43921941dece398514858be91e39d121c1173b26b352b67ba78e8e5",
        )
        self.assertEqual(
            authority.status,
            "stopped_before_official_evaluation_qualification_failed",
        )
        self.assertEqual(
            authority.reviewed_cli_commit,
            "923cc4ad6dd6737f84f4ea2c4f621eb652ac54b4",
        )
        self.assertEqual(authority.private_run_count, 2)
        self.assertEqual(authority.result_row_count, 70)
        self.assertEqual(authority.result_failed_row_count, 30)
        self.assertEqual(authority.result_summary_count, 7)
        self.assertEqual(authority.result_leaf_count, 7)
        self.assertEqual(
            authority.artifact_map_sha256,
            "25fb4ab237ae8f2601c2bc03b05f4e6a2d0cb9889876a83fa0f2c4ee7c4f0ea5",
        )
        self.assertEqual(authority.official_evaluate_count, 0)
        self.assertEqual(authority.official_inspect_count, 0)
        self.assertEqual(authority.label_query_count, 0)
        self.assertEqual(
            tuple(
                (method, summary.passed_count, summary.min_gradual_drift_auc, summary.min_period_change_auc)
                for method, summary in authority.control_summaries.items()
            ),
            (
                ("C0", 9, 0.8402306222021025, 0.907392736783864),
                ("C1", 7, 0.83450279161008, 0.8984897640963734),
                ("C2", 9, 0.8399225391076908, 0.9044753785020874),
                ("C7", 3, 0.7953196519289775, 0.8960376741612595),
            ),
        )

    def test_capture_rejects_consistent_forged_terminal_row_fields_antigravity(self) -> None:
        """Break caught: mutually consistent forged terminal ledger evidence must be rejected field-by-field."""
        (
            _module,
            state,
            catalog,
            champion,
            report_text,
            implementation_report_text,
        ) = self._load_versioned_evidence_antigravity()

        shared_field_mutations = (
            (("status",), "qualification_failed_blocked_before_routing_research"),
            (("implementation_status",), "task_5_review_ship_qualification_failed_task_6_allowed"),
            (("official_evaluate_started",), True),
            (("official_inspect_started",), True),
            (("official_evaluate_invocations",), 1),
            (("official_inspect_invocations",), 1),
            (("result_root_created",), True),
            (("official_result",), {"fake": True}),
            (("prediction_sha256",), "1" * 64),
            (("evaluation_sha256",), "2" * 64),
            (("decision_sha256",), "3" * 64),
            (("artifact_map_sha256",), "4" * 64),
            (("dashboard_snapshot_sha256",), "5" * 64),
            (("test_labels_opened",), True),
            (("label_query_count",), 1),
            (("official_root_present",), True),
            (("review_only",), True),
        )
        for path, value in shared_field_mutations:
            forged_state = deepcopy(state)
            forged_catalog = deepcopy(catalog)
            forged_champion = deepcopy(champion)
            for row in (forged_state, forged_catalog, forged_champion):
                target = row
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
            with self.subTest(path=path, value=value), self.assertRaises(ValueError):
                self._capture_with_evidence_antigravity(
                    forged_state,
                    forged_catalog,
                    forged_champion,
                    report_text,
                    implementation_report_text,
                )

    def test_capture_rejects_consistent_forged_private_run_evidence_antigravity(self) -> None:
        """Break caught: capture must reject forged exact-equality booleans and non-two-successful private runs."""
        (
            _module,
            state,
            catalog,
            champion,
            report_text,
            implementation_report_text,
        ) = self._load_versioned_evidence_antigravity()

        private_run_mutations = (
            (("task5_private_train_result", "exact_equality", "rows"), False),
            (("task5_private_train_result", "exact_equality", "failed_rows"), False),
            (("task5_private_train_result", "exact_equality", "summaries"), False),
            (("task5_private_train_result", "exact_equality", "decision"), False),
            (("task5_private_train_result", "exact_equality", "dashboard"), False),
            (("task5_private_train_result", "exact_equality", "leaf_bytes"), False),
            (("task5_private_train_result", "exact_equality", "leaf_hashes"), False),
            (("task5_private_train_result", "exact_equality", "artifact_map_hash"), False),
            (("task5_private_train_result", "row_count"), 71),
            (("task5_private_train_result", "failed_row_count"), 29),
            (("task5_private_train_result", "summary_count"), 6),
            (("task5_private_train_result", "leaf_count"), 8),
            (("task5_private_train_result", "artifact_map_sha256"), "6" * 64),
            (("task5_private_train_result", "label_query_count"), 1),
            (("task5_private_train_result", "official_root_exists_after"), True),
            (("task5_private_train_result", "status"), "stopped_before_official_evaluation_qualification_failed"),
        )
        for path, value in private_run_mutations:
            forged_state = deepcopy(state)
            forged_catalog = deepcopy(catalog)
            forged_champion = deepcopy(champion)
            for row in (forged_state, forged_catalog, forged_champion):
                target = row
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
            with self.subTest(path=path, value=value), self.assertRaises(ValueError):
                self._capture_with_evidence_antigravity(
                    forged_state,
                    forged_catalog,
                    forged_champion,
                    report_text,
                    implementation_report_text,
                )

        run_mutations = (
            ("single_run", [deepcopy(state["task5_private_train_result"]["runs"][0])]),
            (
                "run1_evaluate_nonzero",
                [
                    {**deepcopy(state["task5_private_train_result"]["runs"][0]), "evaluate_exit": 1},
                    deepcopy(state["task5_private_train_result"]["runs"][1]),
                ],
            ),
            (
                "run2_inspect_nonzero",
                [
                    deepcopy(state["task5_private_train_result"]["runs"][0]),
                    {**deepcopy(state["task5_private_train_result"]["runs"][1]), "inspect_exit": 1},
                ],
            ),
        )
        for name, runs in run_mutations:
            forged_state = deepcopy(state)
            forged_catalog = deepcopy(catalog)
            forged_champion = deepcopy(champion)
            for row in (forged_state, forged_catalog, forged_champion):
                row["task5_private_train_result"]["runs"] = deepcopy(runs)
            with self.subTest(name=name), self.assertRaises(ValueError):
                self._capture_with_evidence_antigravity(
                    forged_state,
                    forged_catalog,
                    forged_champion,
                    report_text,
                    implementation_report_text,
                )

    def test_capture_rejects_consistent_forged_report_commit_and_summary_evidence_antigravity(self) -> None:
        """Break caught: capture must reject mutually consistent forged tracked report evidence."""
        (
            module,
            state,
            catalog,
            champion,
            report_text,
            implementation_report_text,
        ) = self._load_versioned_evidence_antigravity()

        forged_commit = "247225831e9c13e9df5043fa88757bb03e01b79a"
        with patch.object(
            module,
            "_tracked_terminal_closeout_commit_antigravity",
            return_value=forged_commit,
        ), self.assertRaises(ValueError):
            self._capture_with_evidence_antigravity(
                deepcopy(state),
                deepcopy(catalog),
                deepcopy(champion),
                report_text,
                implementation_report_text,
            )

        report_mutations = (
            ("9/10", "8/10"),
            ("0.8402306222021025", "0.8412306222021025"),
            ("0.907392736783864", "0.908392736783864"),
            ("0.9212266108344651", "0.9222266108344651"),
        )
        for original, new in report_mutations:
            text = report_text.replace(original, new, 1)
            with self.subTest(original=original, new=new), self.assertRaises(ValueError):
                self._capture_with_evidence_antigravity(
                    deepcopy(state),
                    deepcopy(catalog),
                    deepcopy(champion),
                    text,
                    implementation_report_text,
                )

    def test_capture_stays_source_free_and_uses_only_versioned_exp463_evidence_antigravity(self) -> None:
        """Break caught: predecessor capture opens result roots or bypasses the tracked report and ledgers."""
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity import (
            predecessor_antigravity,
        )

        source = inspect.getsource(predecessor_antigravity)
        for required in (
            "RESEARCH_STATE.json",
            "EXPERIMENT_CATALOG.json",
            "CHAMPION_LEDGER.json",
            "task-5-report-antigravity.html",
            "docs/EXP463_IMPLEMENTATION_REPORT_antigravity.html",
        ):
            with self.subTest(required=required):
                self.assertIn(required, source)
        for forbidden in (
            "sqlite3",
            "load_normal2_train_antigravity",
            "/private/tmp/exp463_task5_private_run1_antigravity",
            "/Users/mac/Documents/Dataset/exp463_ecg_prefix_ramp_gated_qualification_antigravity_20260821",
            "TEST",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
