from __future__ import annotations

from contextlib import ExitStack
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.exp466_ecg200_phase_registration_margin_antigravity.artifacts_antigravity import (
    publish_result_antigravity,
)
from experiments.exp466_ecg200_phase_registration_margin_antigravity.replay_antigravity import (
    inspect_result_antigravity,
)
from experiments.exp466_ecg200_phase_registration_margin_antigravity.tests.test_artifacts_antigravity import (
    LEAVES_ANTIGRAVITY,
    canonical_test_bytes_antigravity,
    jsonl_rows_antigravity,
    refresh_hashes_antigravity,
    task5_inputs_antigravity,
    write_jsonl_antigravity,
)
from experiments.exp466_ecg200_phase_registration_margin_antigravity.tests.test_qualification_antigravity import (
    normal_confirmation_records_antigravity,
)


def changed_number_antigravity(value: float) -> float:
    return value - 0.125 if value > 0.5 else value + 0.125


def set_path_antigravity(
    value: dict[str, object],
    path: tuple[str | int, ...],
    replacement,
) -> None:
    target = value
    for key in path[:-1]:
        target = target[key]  # type: ignore[index,assignment]
    final = path[-1]
    target[final] = replacement(target[final])  # type: ignore[index]


def mutate_metric_antigravity(
    root: Path,
    method: str,
    path: tuple[str | int, ...],
    replacement=changed_number_antigravity,
) -> None:
    metrics_path = root / "method_fold_metrics_antigravity.jsonl"
    rows = jsonl_rows_antigravity(metrics_path)
    row = next(
        item
        for item in rows
        if item["seed"] == 20260721
        and item["fold_index"] == 0
        and item["method"] == method
    )
    set_path_antigravity(row, path, replacement)
    write_jsonl_antigravity(metrics_path, rows)
    refresh_hashes_antigravity(root)


def assert_current_hashes_antigravity(test: unittest.TestCase, root: Path) -> None:
    artifact_map = json.loads(
        (root / "artifact_sha256_antigravity.json").read_text(encoding="utf-8"),
    )
    test.assertEqual(
        tuple(sorted(artifact_map)),
        tuple(
            name
            for name in LEAVES_ANTIGRAVITY
            if name != "artifact_sha256_antigravity.json"
        ),
    )
    for name, digest in artifact_map.items():
        test.assertEqual(digest, sha256((root / name).read_bytes()).hexdigest())


class ReplayAntigravityTest(unittest.TestCase):
    def test_temporary_publish_inspect_is_stable_and_scorer_data_free_antigravity(
        self,
    ) -> None:
        """Break caught: inspection reloads data/rescores or returns unstable evidence."""
        records, qualification, decision = task5_inputs_antigravity()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "result_antigravity"
            publish_result_antigravity(root, records, qualification, decision)

            from experiments.exp466_ecg200_phase_registration_margin_antigravity import (
                phase_registration_antigravity,
                predecessor_antigravity,
                qualification_antigravity,
                scoring_antigravity,
                sources_antigravity,
            )

            targets = (
                (sources_antigravity, "load_ecg200_normal_train_antigravity"),
                (predecessor_antigravity, "score_c0_fold_antigravity"),
                (phase_registration_antigravity, "score_phase_registration_antigravity"),
                (scoring_antigravity, "score_fold_methods_antigravity"),
                (qualification_antigravity, "qualify_antigravity"),
            )
            with ExitStack() as stack:
                for module, name in targets:
                    stack.enter_context(
                        patch.object(
                            module,
                            name,
                            side_effect=AssertionError(f"inspect called forbidden {name}"),
                        ),
                    )
                first = inspect_result_antigravity(root)
                second = inspect_result_antigravity(root)

            self.assertEqual(first, second)
            self.assertEqual(first.qualification, qualification)
            self.assertEqual(first.decision, decision)
            self.assertIs(type(first.dashboard), dict)
            self.assertEqual(first.dashboard_html, first.dashboard["html_antigravity"])
            self.assertEqual(
                tuple(name for name, _digest in first.leaf_sha256),
                LEAVES_ANTIGRAVITY,
            )
            self.assertEqual(len(first.artifact_map_sha256), 64)

    def test_nonzero_normal_confirmation_publish_inspects_as_failed_result_antigravity(
        self,
    ) -> None:
        """Break caught: failed normal-confirmation evidence cannot round-trip exactly."""
        from experiments.exp466_ecg200_phase_registration_margin_antigravity.evaluation_antigravity import (
            evaluate_qualification_antigravity,
        )
        from experiments.exp466_ecg200_phase_registration_margin_antigravity.qualification_antigravity import (
            qualify_antigravity,
        )

        records = normal_confirmation_records_antigravity()
        qualification = qualify_antigravity(records)
        decision = evaluate_qualification_antigravity(qualification)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "failed_result_antigravity"
            publish_result_antigravity(root, records, qualification, decision)
            inspected = inspect_result_antigravity(root)
        self.assertEqual(inspected.qualification, qualification)
        self.assertEqual(inspected.decision, decision)
        self.assertEqual(
            dict(inspected.qualification.normal_confirmations),
            {20260721: 1, 20261721: 1},
        )

    def test_recanonicalized_rehashed_semantic_mutations_are_rejected_antigravity(
        self,
    ) -> None:
        """Break caught: inspect trusts stored derived values after hashes are refreshed."""
        mutations = (
            ("margin", "P9", ("query_margins", "period_change", 0), changed_number_antigravity),
            ("fit max", "P9", ("fit_normal_max_margin",), changed_number_antigravity),
            (
                "confirmation",
                "P9",
                ("confirmed", "period_change", 0),
                lambda value: not value,
            ),
            ("P9 rank", "P9", ("primary_scores", "period_change", 0), changed_number_antigravity),
            ("C9 score", "C9", ("primary_scores", "period_change", 0), changed_number_antigravity),
            ("AUC", "C9", ("per_type_auc", 3), changed_number_antigravity),
            ("mean", "C9", ("mean_auc",), changed_number_antigravity),
            ("C0 delta", "C9", ("c0_deltas", "period_change"), changed_number_antigravity),
        )
        records, qualification, decision = task5_inputs_antigravity()
        for label, method, path, replacement in mutations:
            with self.subTest(field=label), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "result_antigravity"
                publish_result_antigravity(root, records, qualification, decision)
                mutate_metric_antigravity(root, method, path, replacement)
                assert_current_hashes_antigravity(self, root)
                with self.assertRaises(ValueError):
                    inspect_result_antigravity(root)

        document_mutations = (
            (
                "decision",
                "decision_antigravity.json",
                lambda value: value.__setitem__(
                    "primary_passed_count",
                    value["primary_passed_count"] - 1,
                ),
            ),
            (
                "dashboard",
                "dashboard_snapshot_antigravity.json",
                lambda value: value.__setitem__(
                    "주기_AUC_개선_fold_수",
                    value["주기_AUC_개선_fold_수"] + 1,
                ),
            ),
        )
        for label, name, mutate in document_mutations:
            with self.subTest(field=label), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "result_antigravity"
                publish_result_antigravity(root, records, qualification, decision)
                path = root / name
                value = json.loads(path.read_text(encoding="utf-8"))
                mutate(value)
                path.write_bytes(canonical_test_bytes_antigravity(value))
                refresh_hashes_antigravity(root)
                assert_current_hashes_antigravity(self, root)
                with self.assertRaises(ValueError):
                    inspect_result_antigravity(root)

    def test_stale_artifact_map_is_rejected_separately_antigravity(self) -> None:
        """Break caught: inspect accepts a stale digest before semantic replay."""
        records, qualification, decision = task5_inputs_antigravity()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "result_antigravity"
            publish_result_antigravity(root, records, qualification, decision)
            map_path = root / "artifact_sha256_antigravity.json"
            artifact_map = json.loads(map_path.read_text(encoding="utf-8"))
            changed_name = "context_antigravity.json"
            artifact_map[changed_name] = "0" * 64
            map_path.write_bytes(canonical_test_bytes_antigravity(artifact_map))
            self.assertNotEqual(
                artifact_map[changed_name],
                sha256((root / changed_name).read_bytes()).hexdigest(),
            )
            with self.assertRaises(ValueError):
                inspect_result_antigravity(root)

    def test_swapped_blocked_indices_with_preserved_oof_are_rejected_antigravity(
        self,
    ) -> None:
        """Break caught: inspection accepts a different blocked-fold assignment."""
        records, qualification, decision = task5_inputs_antigravity()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "result_antigravity"
            publish_result_antigravity(root, records, qualification, decision)
            metrics_path = root / "method_fold_metrics_antigravity.jsonl"
            rows = jsonl_rows_antigravity(metrics_path)
            for method in ("C0", "P9", "C9"):
                first = next(
                    row
                    for row in rows
                    if row["seed"] == 20260721
                    and row["fold_index"] == 0
                    and row["method"] == method
                )
                second = next(
                    row
                    for row in rows
                    if row["seed"] == 20260721
                    and row["fold_index"] == 1
                    and row["method"] == method
                )
                for field in ("input_fit_indices", "input_query_indices"):
                    first[field], second[field] = second[field], first[field]
            write_jsonl_antigravity(metrics_path, rows)
            refresh_hashes_antigravity(root)
            assert_current_hashes_antigravity(self, root)
            with self.assertRaisesRegex(ValueError, "blocked fold"):
                inspect_result_antigravity(root)

    def test_rehashed_context_summary_and_failed_leaf_mutations_are_rejected_antigravity(
        self,
    ) -> None:
        """Break caught: inspection trusts a derived non-metric semantic leaf."""
        records, qualification, decision = task5_inputs_antigravity()
        context_mutations = (
            (
                "gate",
                lambda value: value["gates"].__setitem__("per_type_auc_min", 0.86),
            ),
            (
                "count",
                lambda value: value["normal_trial_counts"].__setitem__("20260721", 68),
            ),
        )
        for label, mutate in context_mutations:
            with self.subTest(leaf=f"context {label}"), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "result_antigravity"
                publish_result_antigravity(root, records, qualification, decision)
                path = root / "context_antigravity.json"
                value = json.loads(path.read_text(encoding="utf-8"))
                mutate(value)
                path.write_bytes(canonical_test_bytes_antigravity(value))
                refresh_hashes_antigravity(root)
                assert_current_hashes_antigravity(self, root)
                with self.assertRaises(ValueError):
                    inspect_result_antigravity(root)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "summary_result_antigravity"
            publish_result_antigravity(root, records, qualification, decision)
            path = root / "method_summary_antigravity.jsonl"
            rows = jsonl_rows_antigravity(path)
            rows[0]["fold_count"] = 9
            write_jsonl_antigravity(path, rows)
            refresh_hashes_antigravity(root)
            assert_current_hashes_antigravity(self, root)
            with self.assertRaises(ValueError):
                inspect_result_antigravity(root)

        from experiments.exp466_ecg200_phase_registration_margin_antigravity.evaluation_antigravity import (
            evaluate_qualification_antigravity,
        )
        from experiments.exp466_ecg200_phase_registration_margin_antigravity.qualification_antigravity import (
            qualify_antigravity,
        )

        failed_records = normal_confirmation_records_antigravity()
        failed_qualification = qualify_antigravity(failed_records)
        failed_decision = evaluate_qualification_antigravity(failed_qualification)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "failed_leaf_result_antigravity"
            publish_result_antigravity(
                root,
                failed_records,
                failed_qualification,
                failed_decision,
            )
            path = root / "failed_folds_antigravity.jsonl"
            rows = jsonl_rows_antigravity(path)
            target = next(
                row
                for row in rows
                if "normal_confirmation_present" in row["failure_reasons"]
            )
            target["failure_reasons"].remove("normal_confirmation_present")
            write_jsonl_antigravity(path, rows)
            refresh_hashes_antigravity(root)
            assert_current_hashes_antigravity(self, root)
            with self.assertRaises(ValueError):
                inspect_result_antigravity(root)


if __name__ == "__main__":
    unittest.main()
