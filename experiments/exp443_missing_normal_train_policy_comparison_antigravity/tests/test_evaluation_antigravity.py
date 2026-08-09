"""RED/GREEN tests for the separate Exp443 post-freeze evaluator."""
from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256
import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from experiments.exp443_missing_normal_train_policy_comparison_antigravity.bootstrap_antigravity import (
    BootstrapResultAntigravity,
)
from experiments.exp443_missing_normal_train_policy_comparison_antigravity.config_antigravity import FROZEN_CONFIG
from experiments.exp443_missing_normal_train_policy_comparison_antigravity.reporting_antigravity import (
    FrozenPublicationAntigravity,
)
from experiments.exp443_missing_normal_train_policy_comparison_antigravity import evaluation_antigravity


POLICIES = tuple(FROZEN_CONFIG.policies)


def _publication(root: Path) -> FrozenPublicationAntigravity:
    return FrozenPublicationAntigravity(
        str(root / "attempt_0000_antigravity"),
        str(root / "attempt_0000_antigravity" / "publication_antigravity"),
        "a" * 64,
        "b" * 64,
        tuple((policy, str(index + 1) * 64) for index, policy in enumerate(POLICIES)),
        tuple((policy, str(index + 4) * 64) for index, policy in enumerate(POLICIES)),
        0,
    )


def _positive_bootstrap(population: int) -> BootstrapResultAntigravity:
    return BootstrapResultAntigravity(
        micro_delta_ci=(0.01, 0.05),
        macro_f1_delta_ci=(0.02, 0.08),
        seed=443,
        samples=10,
        population=population,
        numpy_version="2.3.5",
        sample_indices_sha256="c" * 64,
    )


class _Cursor:
    def __init__(self, rows) -> None:
        self._rows = rows

    def fetchall(self):
        return list(self._rows)


class _Connection:
    def __init__(self, datasets: tuple[str, ...], events: list[str]) -> None:
        self.datasets = set(datasets)
        self.events = events
        self.label_queries: list[tuple[str, str]] = []

    def execute(self, sql: str, parameters=()):
        normalized = " ".join(sql.split())
        if normalized == "PRAGMA quick_check":
            self.events.append("quick_check")
            return _Cursor([("ok",)])
        self.assert_label_sql(normalized)
        dataset = parameters[0]
        if dataset not in self.datasets:
            raise AssertionError("unexpected dataset")
        self.events.append(f"label:{dataset}")
        self.label_queries.append((normalized, dataset))
        return _Cursor([(0, 0)])

    @staticmethod
    def assert_label_sql(sql: str) -> None:
        expected = " ".join(evaluation_antigravity.LABEL_SQL_ANTIGRAVITY.split())
        if sql != expected:
            raise AssertionError(f"unexpected SQL: {sql}")


class EvaluationAntigravityTests(unittest.TestCase):
    def test_small_policy_evaluation_defines_rescue_regression_strong_loss_and_transitions(self) -> None:
        labels = {
            "FamilyA_normal_0": (1, 0),
            "FamilyB_normal_0": (1, 0),
        }
        baseline = {
            "FamilyA_normal_0": (),
            "FamilyB_normal_0": (0,),
        }
        policy = {
            "FamilyA_normal_0": (0,),
            "FamilyB_normal_0": (0, 1),
        }
        result = evaluation_antigravity._compute_policy_evaluation_for_test_antigravity(
            POLICIES[1], baseline, policy, labels, set(labels),
            bootstrap_runner=lambda rows: _positive_bootstrap(len(rows)),
            maximum_fp=0,
            expected_unaffected_count=0,
            pre_freeze_label_access_count=0,
        )
        by_dataset = {row.dataset: row for row in result.dataset_rows}
        rescued = by_dataset["FamilyA_normal_0"]
        regressed = by_dataset["FamilyB_normal_0"]
        self.assertTrue(rescued.rescued)
        self.assertEqual((rescued.added_tp, rescued.added_fp), (1, 0))
        self.assertTrue(regressed.regressed)
        self.assertTrue(regressed.strong_regression)
        self.assertEqual((regressed.added_tp, regressed.added_fp), (0, 1))
        self.assertEqual(result.rescued_datasets, ("FamilyA_normal_0",))
        self.assertEqual(result.regressed_datasets, ("FamilyB_normal_0",))
        self.assertEqual(result.strong_regressed_datasets, ("FamilyB_normal_0",))
        self.assertEqual({row.family for row in result.family_rows}, {"FamilyA", "FamilyB"})
        self.assertEqual(result.status, "no_retrospective_improvement_signal")

    def test_strong_regression_threshold_is_inclusive_at_point_eight(self) -> None:
        self.assertTrue(evaluation_antigravity._is_strong_regression_antigravity(0.8, 0.79))
        self.assertFalse(evaluation_antigravity._is_strong_regression_antigravity(0.799999999999, 0.7))
        self.assertFalse(evaluation_antigravity._is_strong_regression_antigravity(0.8, 0.8))

    def test_label_loader_inspects_twice_before_one_connection_and_queries_all_datasets_once(self) -> None:
        datasets = tuple(f"dataset_{index:04d}" for index in range(FROZEN_CONFIG.dataset_count))
        rows = tuple({"dataset": dataset, "n_test": 1, "policy": POLICIES[0]} for dataset in datasets)
        publication_rows = MappingProxyType({policy: rows for policy in POLICIES})
        events: list[str] = []
        connection = _Connection(datasets, events)

        @contextmanager
        def open_connection():
            events.append("open")
            yield connection
            events.append("close")

        with tempfile.TemporaryDirectory(prefix="exp443_task10_labels_antigravity_") as temporary:
            publication = _publication(Path(temporary))

            def inspect(_attempt):
                events.append("inspect")
                return publication

            source = SimpleNamespace(label_access_count=0, datasets=datasets)
            with patch.object(evaluation_antigravity, "inspect_publication_antigravity", side_effect=inspect), \
                 patch.object(evaluation_antigravity, "_publication_rows_antigravity", return_value=publication_rows), \
                 patch.object(evaluation_antigravity, "_open_label_connection_antigravity", side_effect=lambda: open_connection()), \
                 patch.object(evaluation_antigravity, "verify_frozen_sources_antigravity", return_value=source):
                frozen = evaluation_antigravity.load_frozen_test_labels_antigravity(publication)

        self.assertEqual(events[:3], ["inspect", "inspect", "open"])
        self.assertLess(events.index("close"), len(events) - 1)
        self.assertEqual(events[-1], "inspect")
        self.assertEqual(len(connection.label_queries), FROZEN_CONFIG.dataset_count)
        self.assertEqual(frozen.label_query_calls, FROZEN_CONFIG.dataset_count)
        self.assertEqual(frozen.descriptor_bound_connection_count, 1)
        self.assertEqual(frozen.pre_freeze_label_access_count, 0)

    def test_label_loader_rejects_noncontiguous_or_nonbinary_rows(self) -> None:
        for rows in ([(1, 0)], [(0, 2)]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                evaluation_antigravity._normalize_label_rows_antigravity(rows, expected_count=1, dataset="bad_normal_0")

    def test_success_gate_requires_every_preregistered_condition_without_selecting_winner(self) -> None:
        gates = evaluation_antigravity._success_gates_antigravity(
            micro_delta=0.01,
            macro_f1_delta=0.02,
            bootstrap=_positive_bootstrap(1117),
            policy_fp=303,
            rescued_count=1,
            strong_regression_count=0,
            unaffected_identical_count=1098,
            pre_freeze_label_access_count=0,
            error_count=0,
            nonfinite_count=0,
        )
        self.assertTrue(all(gates.values()))
        failed = evaluation_antigravity._success_gates_antigravity(
            micro_delta=0.01,
            macro_f1_delta=0.02,
            bootstrap=_positive_bootstrap(1117),
            policy_fp=304,
            rescued_count=1,
            strong_regression_count=0,
            unaffected_identical_count=1098,
            pre_freeze_label_access_count=0,
            error_count=0,
            nonfinite_count=0,
        )
        self.assertFalse(failed["false_alarm_gate"])

    def test_evaluation_inventory_is_exact_and_artifact_map_excludes_itself(self) -> None:
        self.assertEqual(
            evaluation_antigravity.EVALUATION_NAMES_ANTIGRAVITY,
            (
                "evaluation_antigravity.json",
                "dataset_metrics_antigravity.jsonl",
                "coordinate_transitions_antigravity.jsonl",
                "rescued_datasets_antigravity.jsonl",
                "regressed_datasets_antigravity.jsonl",
                "family_diagnostics_antigravity.jsonl",
                "confusion_antigravity.json",
                "bootstrap_antigravity.json",
                "decision_antigravity.json",
                "artifact_sha256_antigravity.json",
            ),
        )
        payloads = {name: b"{}\n" for name in evaluation_antigravity.EVALUATION_NAMES_ANTIGRAVITY[:-1]}
        artifact = evaluation_antigravity._artifact_map_bytes_antigravity(payloads)
        parsed = json.loads(artifact)
        self.assertEqual(parsed["self_excluded"], "artifact_sha256_antigravity.json")
        self.assertEqual(set(parsed["files"]), set(payloads))
        self.assertEqual(parsed["files"]["decision_antigravity.json"], sha256(b"{}\n").hexdigest())

    def test_public_compute_rejects_nonreproduced_official_baseline(self) -> None:
        datasets = tuple(f"Family_normal_{index:04d}" for index in range(FROZEN_CONFIG.dataset_count))
        publication_rows = MappingProxyType({
            policy: tuple(
                {"dataset": dataset, "n_test": 1, "policy": policy, "final_hard": ()}
                for dataset in datasets
            )
            for policy in POLICIES
        })
        labels = evaluation_antigravity.FrozenLabelsAntigravity(
            datasets=datasets,
            labels_by_dataset={dataset: (0,) for dataset in datasets},
            publication_manifest_sha256="a" * 64,
            prediction_sha256=tuple((policy, str(index + 1) * 64) for index, policy in enumerate(POLICIES)),
            database_sha256=FROZEN_CONFIG.database_sha256,
            label_identity_sha256="d" * 64,
            descriptor_bound_connection_count=1,
            label_query_calls=FROZEN_CONFIG.dataset_count,
            label_rows=FROZEN_CONFIG.dataset_count,
        )
        source = SimpleNamespace(
            datasets=datasets,
            affected_datasets=datasets[: FROZEN_CONFIG.affected_dataset_count],
            hard_coordinates_exp404=frozenset(),
            hard_coordinates_exp291=frozenset(),
            label_access_count=0,
        )
        with tempfile.TemporaryDirectory(prefix="exp443_task10_baseline_antigravity_") as temporary:
            publication = _publication(Path(temporary))
            with patch.object(evaluation_antigravity, "_publication_rows_antigravity", return_value=publication_rows), \
                 patch.object(evaluation_antigravity, "verify_frozen_sources_antigravity", return_value=source), \
                 self.assertRaisesRegex(ValueError, "baseline reproduction"):
                evaluation_antigravity.compute_policy_evaluation_antigravity(publication, labels)

    def test_full_evaluation_payload_replays_from_rows_without_labels(self) -> None:
        datasets = tuple(f"Family_normal_{index:04d}" for index in range(FROZEN_CONFIG.dataset_count))
        labels = {dataset: (0,) for dataset in datasets}
        coordinates = {dataset: () for dataset in datasets}
        affected = set(datasets[: FROZEN_CONFIG.affected_dataset_count])
        policy_results = tuple(
            evaluation_antigravity._compute_policy_evaluation_for_test_antigravity(
                policy,
                coordinates,
                coordinates,
                labels,
                affected,
                bootstrap_runner=evaluation_antigravity.paired_dataset_bootstrap_antigravity,
                maximum_fp=FROZEN_CONFIG.baseline_fp,
                expected_unaffected_count=FROZEN_CONFIG.dataset_count - FROZEN_CONFIG.affected_dataset_count,
                pre_freeze_label_access_count=0,
            )
            for policy in POLICIES
        )
        result = evaluation_antigravity.EvaluationResultAntigravity(
            publication_manifest_sha256="a" * 64,
            prediction_sha256=tuple((policy, str(index + 1) * 64) for index, policy in enumerate(POLICIES)),
            database_sha256=FROZEN_CONFIG.database_sha256,
            label_identity_sha256="d" * 64,
            label_rows=FROZEN_CONFIG.dataset_count,
            label_query_calls=FROZEN_CONFIG.dataset_count,
            policy_results=policy_results,
        )
        payloads = evaluation_antigravity._evaluation_payloads_antigravity(result)
        evaluation_payload = json.loads(payloads["evaluation_antigravity.json"])
        for summary in evaluation_payload["policy_summaries"]:
            self.assertEqual(summary["affected_diagnostic"]["dataset_count"], 19)
            self.assertEqual(summary["unaffected_diagnostic"]["dataset_count"], 1098)
            self.assertEqual(summary["pre_freeze_label_access_count"], 0)
            self.assertEqual(summary["error_count"], 0)
            self.assertEqual(summary["nonfinite_count"], 0)
        replay = evaluation_antigravity._evaluation_result_from_payloads_antigravity(
            payloads, _enforce_official_antigravity=False,
        )
        self.assertEqual(replay, result)
        self.assertEqual(
            {policy.policy: policy.status for policy in replay.policy_results},
            {policy: "no_retrospective_improvement_signal" for policy in POLICIES},
        )

    def test_create_only_evaluation_replay_does_not_reopen_labels_or_overwrite(self) -> None:
        datasets = tuple(f"Family_normal_{index:04d}" for index in range(FROZEN_CONFIG.dataset_count))
        labels_by_dataset = {dataset: (0,) for dataset in datasets}
        coordinates = {dataset: () for dataset in datasets}
        affected = set(datasets[: FROZEN_CONFIG.affected_dataset_count])
        policy_results = tuple(
            evaluation_antigravity._compute_policy_evaluation_for_test_antigravity(
                policy,
                coordinates,
                coordinates,
                labels_by_dataset,
                affected,
                bootstrap_runner=lambda rows: _positive_bootstrap(len(rows)),
                maximum_fp=FROZEN_CONFIG.baseline_fp,
                expected_unaffected_count=FROZEN_CONFIG.dataset_count - FROZEN_CONFIG.affected_dataset_count,
                pre_freeze_label_access_count=0,
            )
            for policy in POLICIES
        )
        result = evaluation_antigravity.EvaluationResultAntigravity(
            publication_manifest_sha256="a" * 64,
            prediction_sha256=tuple((policy, str(index + 1) * 64) for index, policy in enumerate(POLICIES)),
            database_sha256=FROZEN_CONFIG.database_sha256,
            label_identity_sha256="d" * 64,
            label_rows=FROZEN_CONFIG.dataset_count,
            label_query_calls=FROZEN_CONFIG.dataset_count,
            policy_results=policy_results,
        )
        with tempfile.TemporaryDirectory(prefix="exp443_task10_create_only_antigravity_") as temporary:
            root = Path(temporary)
            publication = _publication(root)
            destination = root / "evaluation_antigravity"
            with patch.object(evaluation_antigravity, "inspect_publication_antigravity", return_value=publication), \
                 patch.object(evaluation_antigravity, "load_frozen_test_labels_antigravity", return_value=object()) as load_labels, \
                 patch.object(evaluation_antigravity, "compute_policy_evaluation_antigravity", return_value=result) as compute, \
                 patch.object(evaluation_antigravity, "_evaluation_result_from_payloads_antigravity", return_value=result):
                first = evaluation_antigravity.evaluate_publication_antigravity(publication.attempt_path, destination)
                before = {path.name: path.read_bytes() for path in destination.iterdir()}
                load_labels.reset_mock()
                compute.reset_mock()
                second = evaluation_antigravity.evaluate_publication_antigravity(publication.attempt_path, destination)
                after = {path.name: path.read_bytes() for path in destination.iterdir()}
                load_labels.assert_not_called()
                compute.assert_not_called()
            self.assertEqual(first, second)
            self.assertEqual(before, after)
            self.assertEqual(set(before), set(evaluation_antigravity.EVALUATION_NAMES_ANTIGRAVITY))


if __name__ == "__main__":
    unittest.main()
