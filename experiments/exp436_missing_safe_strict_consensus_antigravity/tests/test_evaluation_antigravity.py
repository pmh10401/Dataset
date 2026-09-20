"""Post-freeze Exp436 evaluation boundary behaviors.

These tests deliberately use temporary publication/source trees and a real
SQLite label boundary.  The public command has fixed paths; fixtures exercise
only private test seams so no production path, loader, config, or bootstrap
override is exposed.
"""
from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256
import inspect
import json
import os
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.exp436_missing_safe_strict_consensus_antigravity import evaluation_antigravity as module
from experiments.exp436_missing_safe_strict_consensus_antigravity.config_antigravity import FROZEN_CONFIG
from experiments.exp436_missing_safe_strict_consensus_antigravity.execution_profile_antigravity import (
    EXP436_PROFILE,
    EXP437_PROFILE,
)
from experiments.exp436_missing_safe_strict_consensus_antigravity.reporting_antigravity import (
    PUBLICATION_DIRECTORY,
    FrozenPublication,
)


class EvaluationAntigravityTests(unittest.TestCase):
    def _comparison_rows(self):
        return (
            module.DatasetComparison("A", 4, module.ConfusionCounts(0, 0, 1, 3), module.ConfusionCounts(1, 0, 0, 3)),
            module.DatasetComparison("B", 4, module.ConfusionCounts(1, 0, 0, 3), module.ConfusionCounts(1, 0, 0, 3)),
            module.DatasetComparison("C", 8, module.ConfusionCounts(4, 0, 0, 4), module.ConfusionCounts(4, 1, 0, 3)),
        )

    def _evaluation(self):
        rows = self._comparison_rows()
        baseline = module._aggregate_counts(rows, combined=False)
        combined = module._aggregate_counts(rows, combined=True)
        bootstrap = module._paired_dataset_bootstrap_for_test(rows, repetitions=8, seed=123)
        return module.Exp436Evaluation(
            dataset_count=3, candidate_count=2, added_fp=1, dataset_rows=rows,
            baseline_confusion=baseline, combined_confusion=combined,
            micro_f1_delta=combined.f1 - baseline.f1,
            macro_f1_delta=sum(row.combined_f1 - row.baseline_f1 for row in rows) / 3,
            micro_f1_ci=bootstrap.micro_f1_ci, macro_f1_ci=bootstrap.macro_f1_ci,
            rescued_datasets=("A",), regressed_datasets=("C",),
            strong_baseline_regressions=("C",), prediction_sha256="a" * 64,
            manifest_sha256="b" * 64, bootstrap=bootstrap,
        )

    def _source(self, root: Path):
        output = root / "output_antigravity"
        publication = output / PUBLICATION_DIRECTORY
        checkpoint = root / "checkpoint_antigravity"
        output.mkdir(); publication.mkdir(); checkpoint.mkdir()
        (publication / "placeholder_antigravity.json").write_bytes(b"{}\n")
        return output, checkpoint, FrozenPublication(publication, {})

    def _tree(self, root: Path):
        return {str(path.relative_to(root)): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}

    def test_hand_derived_confusion_rescue_regression_and_strong_regression(self):
        rescued, stable, regression = self._comparison_rows()
        self.assertEqual((rescued.baseline.f1, rescued.combined.f1), (0.0, 1.0))
        self.assertTrue(rescued.rescued)
        self.assertFalse(stable.regressed)
        self.assertTrue(regression.regressed)
        self.assertTrue(regression.strong_baseline)

    def test_zero_denominator_f1_is_zero_for_micro_and_dataset_macro(self):
        zero = module.ConfusionCounts(0, 0, 0, 7)
        comparison = module.DatasetComparison("zero", 7, zero, zero)
        self.assertEqual(zero.f1, 0.0)
        self.assertEqual(comparison.baseline_f1, 0.0)
        self.assertEqual(comparison.combined_f1, 0.0)

    def test_combined_prediction_cannot_lose_a_baseline_true_positive(self):
        with self.assertRaisesRegex(ValueError, "monotonic"):
            module.DatasetComparison("bad", 4, module.ConfusionCounts(1, 0, 0, 3), module.ConfusionCounts(0, 1, 1, 2))

    def test_aggregate_counts_are_hand_derived(self):
        baseline = module._aggregate_counts(self._comparison_rows(), combined=False)
        combined = module._aggregate_counts(self._comparison_rows(), combined=True)
        self.assertEqual(baseline.to_payload(), {"tp": 5, "fp": 0, "fn": 1, "tn": 10})
        self.assertEqual(combined.to_payload(), {"tp": 6, "fp": 1, "fn": 0, "tn": 9})

    def test_paired_bootstrap_is_deterministic_and_uses_frozen_defaults(self):
        first = module._paired_dataset_bootstrap(self._comparison_rows())
        second = module._paired_dataset_bootstrap(self._comparison_rows())
        self.assertEqual(first.to_payload(), second.to_payload())
        self.assertEqual((first.repetitions, first.seed), (20_000, 43_520_260_806))

    def test_bootstrap_quantiles_are_linear(self):
        rows = self._comparison_rows()
        result = module._paired_dataset_bootstrap_for_test(rows, repetitions=19, seed=43520260806)
        self.assertEqual(result.micro_f1_delta_ci.to_payload(), {"lower": result.micro_f1_delta_ci.lower, "upper": result.micro_f1_delta_ci.upper})

    def test_every_success_gate_fails_at_its_boundary(self):
        common = dict(dataset_count=1117, candidate_count=100, rescued_datasets=tuple(f"R{i}" for i in range(10)), added_fp=30, micro_f1_delta=0.1, macro_f1_delta=0.1, micro_f1_ci=(0.01, 0.2), macro_f1_ci=(0.01, 0.2), prediction_label_access_count=0)
        cases = {
            "dataset_count_gate": {"dataset_count": 1116},
            "candidate_budget_gate": {"candidate_count": 101},
            "rescued_dataset_gate": {"rescued_datasets": tuple(f"R{i}" for i in range(9))},
            "added_false_positive_gate": {"added_fp": 31},
            "micro_f1_point_gate": {"micro_f1_delta": 0.0},
            "macro_f1_point_gate": {"macro_f1_delta": 0.0},
            "micro_f1_ci_gate": {"micro_f1_ci": (0.0, 0.2)},
            "macro_f1_ci_gate": {"macro_f1_ci": (0.0, 0.2)},
            "prediction_label_access_gate": {"prediction_label_access_count": 1},
        }
        for name, change in cases.items():
            with self.subTest(name=name):
                decision = module.decide_exp436_success(**(common | change))
                self.assertFalse(decision.success)
                self.assertEqual(decision.reasons, (name,))

    def test_success_accepts_every_exact_gate_boundary(self):
        decision = module.decide_exp436_success(dataset_count=1117, candidate_count=100, rescued_datasets=tuple(f"R{i}" for i in range(10)), added_fp=30, micro_f1_delta=0.0001, macro_f1_delta=0.0001, micro_f1_ci=(0.0001, 0.1), macro_f1_ci=(0.0001, 0.1), prediction_label_access_count=0)
        self.assertTrue(decision.success)

    def test_exp404_parser_rejects_duplicate_coordinates(self):
        row = b'{"dataset":"A","candidate_index":2,"contains_test_labels":false,"contains_evaluation_metrics":false}\n'
        with self.assertRaisesRegex(ValueError, "duplicate"):
            module._parse_exp404_hard_payload(row + row)

    def test_exp291_parser_rejects_unsorted_or_duplicate_indices(self):
        raw = b'{"dataset":"A","automatic_hard_indices":[2,2],"contains_test_labels":false}\n'
        with self.assertRaisesRegex(ValueError, "invalid"):
            module._parse_hard_payload(raw)

    def test_frozen_exp404_hash_coordinates_and_exp291_identity_are_exact(self):
        hard = module._load_frozen_hard_coordinates()
        self.assertEqual(len(hard), 1117)
        self.assertEqual(sum(bool(indices) for indices in hard.values()), 888)
        self.assertEqual(sum(len(indices) for indices in hard.values()), 2218)

    def test_pinned_reader_uses_the_opened_inode_bytes(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "hard_antigravity.jsonl"
            original = b"frozen"
            path.write_bytes(original)
            payload, digest = module._read_pinned_bytes(path)
            self.assertEqual((payload, digest), (original, sha256(original).hexdigest()))

    def test_label_payload_requires_sorted_noncontiguous_actual_test_identities(self):
        identities, labels, _ = module._label_payload({"instance_indices": [4, 19, 27], "labels": [0, 1, 0]}, dataset="Demo")
        self.assertEqual(identities, (4, 19, 27))
        self.assertEqual(labels, (0, 1, 0))

    def test_counts_use_actual_test_identities_not_range_test_size(self):
        labels = {4: 0, 19: 1, 27: 0}
        self.assertEqual(module._counts({19}, labels).to_payload(), {"tp": 1, "fp": 0, "fn": 0, "tn": 2})
        with self.assertRaisesRegex(ValueError, "identity"):
            module._counts({1}, labels)

    def test_hard_and_exp436_candidates_must_be_disjoint(self):
        with TemporaryDirectory() as temporary:
            output, checkpoint, publication = self._source(Path(temporary))
            manifest = {"prediction_sha256": "a" * 64, "database_sha256": "b" * 64, "label_access_count": 0, "code_commit": "c" * 40}
            summary = ({"dataset": "Demo", "test_size": 3},)
            prediction = ({"dataset": "Demo", "instance_index": 19},)
            with patch.object(module, "_inspect_publication", return_value=publication), patch.object(module, "_validate_verified_publication", return_value=(manifest, summary, prediction)), patch.object(module, "_snapshot_frozen_tree", return_value={".": ("directory", None)}), patch.object(module, "_load_labels", side_effect=AssertionError("labels must remain closed")):
                with self.assertRaisesRegex(ValueError, "overlaps"):
                    module._evaluate_frozen_prediction_for_test(publication, hard_coordinates={"Demo": (19,)}, label_loader=lambda _: None, checkpoint=type("Store", (), {"root": checkpoint})(), expected_datasets=("Demo",))

    def test_out_of_identity_coordinate_is_rejected_before_confusion(self):
        with self.assertRaisesRegex(ValueError, "identity"):
            module._counts({9}, {4: 0, 19: 1})

    def test_prelabel_reverify_blocks_label_loader_when_source_changed(self):
        with TemporaryDirectory() as temporary:
            output, checkpoint, publication = self._source(Path(temporary))
            manifest = {"prediction_sha256": "a" * 64, "database_sha256": "b" * 64, "label_access_count": 0, "code_commit": "c" * 40}
            summary = ({"dataset": "Demo", "test_size": 3},)
            events = []
            def replay(*_args, **_kwargs):
                events.append("replay")
                if len(events) == 2:
                    raise ValueError("source changed before labels")
                return publication
            with patch.object(module, "_inspect_publication", side_effect=replay), patch.object(module, "_validate_verified_publication", return_value=(manifest, summary, ())), patch.object(module, "_snapshot_frozen_tree", return_value={".": ("directory", None)}):
                with self.assertRaisesRegex(ValueError, "source changed before labels"):
                    module._evaluate_frozen_prediction_for_test(publication, hard_coordinates={"Demo": ()}, label_loader=lambda _: (_ for _ in ()).throw(AssertionError("label access before reverify")), checkpoint=type("Store", (), {"root": checkpoint})(), expected_datasets=("Demo",), evaluation_root=Path(temporary) / "evaluation_antigravity")
            self.assertEqual(events, ["replay", "replay"])

    def test_label_sql_uses_one_descriptor_bound_connection_and_only_identity_and_label(self):
        with TemporaryDirectory() as temporary:
            database = Path(temporary) / "labels_antigravity.db"
            connection = sqlite3.connect(database)
            connection.executescript("CREATE TABLE datasets (id INTEGER PRIMARY KEY, name TEXT UNIQUE); CREATE TABLE instances (dataset_id INTEGER, split TEXT, instance_index INTEGER, label TEXT, values_blob BLOB, labels_blob BLOB); INSERT INTO datasets VALUES (1, 'Demo'); INSERT INTO instances VALUES (1, 'TEST', 4, '0', X'00', X'01'); INSERT INTO instances VALUES (1, 'TEST', 19, '1', X'00', X'01');")
            connection.commit(); connection.close()
            traces = []
            @contextmanager
            def pinned(path, digest):
                opened = sqlite3.connect(path); opened.set_trace_callback(traces.append)
                try: yield opened
                finally: opened.close()
            with patch.object(module, "verified_read_only_connection", pinned):
                payload = module._load_test_labels("Demo", database_path=database, expected_sha256="d" * 64)
            self.assertEqual(payload["instance_indices"], [4, 19])
            sql = " ".join(traces).lower()
            self.assertIn("instance_index", sql); self.assertIn("label", sql)
            self.assertNotIn("values_blob", sql); self.assertNotIn("labels_blob", sql)

    def test_evaluation_artifacts_have_exact_eight_file_inventory_and_self_excluded_hash(self):
        with TemporaryDirectory() as temporary, patch.object(module, "_exclusive_rename", os.rename):
            root = Path(temporary) / "evaluation_antigravity"
            module._write_evaluation_artifacts(root, self._evaluation())
            self.assertEqual({item.name for item in root.iterdir()}, module._EVALUATION_FILES)
            hashes = json.loads((root / module.EVALUATION_HASH_FILE).read_bytes())
            self.assertNotIn(module.EVALUATION_HASH_FILE, hashes["artifact_sha256"])
            for name, digest in hashes["artifact_sha256"].items():
                self.assertEqual(sha256((root / name).read_bytes()).hexdigest(), digest)

    def test_evaluation_stage_reopens_and_verifies_before_returning(self):
        with TemporaryDirectory() as temporary, patch.object(module, "_exclusive_rename", os.rename):
            root = Path(temporary) / "evaluation_antigravity"
            with patch.object(module, "_verify_evaluation_artifacts", wraps=module._verify_evaluation_artifacts) as verify:
                module._write_evaluation_artifacts(root, self._evaluation())
            self.assertEqual(verify.call_count, 1)
            self.assertTrue(root.is_dir())

    def test_evaluation_destination_race_never_clobbers_existing_tree(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "evaluation_antigravity"
            root.mkdir(); (root / "keep_antigravity.txt").write_text("keep")
            with self.assertRaisesRegex(ValueError, "inventory|replay"):
                module._write_evaluation_artifacts(root, self._evaluation())
            self.assertEqual((root / "keep_antigravity.txt").read_text(), "keep")

    def test_existing_evaluation_is_replay_only(self):
        with TemporaryDirectory() as temporary, patch.object(module, "_exclusive_rename", os.rename):
            root = Path(temporary) / "evaluation_antigravity"
            module._write_evaluation_artifacts(root, self._evaluation())
            before = self._tree(root)
            replayed = module._write_evaluation_artifacts(root, self._evaluation())
            self.assertEqual(replayed.to_payload(), self._evaluation().to_payload())
            self.assertEqual(self._tree(root), before)

    def test_source_mutation_after_labels_prevents_evaluation_publish(self):
        with TemporaryDirectory() as temporary:
            output, checkpoint, publication = self._source(Path(temporary))
            source = module._snapshot_frozen_tree(output)
            (output / "changed_antigravity.txt").write_text("mutated")
            with patch.object(module, "_inspect_publication", return_value=publication):
                with self.assertRaisesRegex(ValueError, "changed"):
                    module._verify_frozen_source(output, type("Store", (), {"root": checkpoint})(), expected_total=1, snapshot=source)

    def test_no_evaluation_root_is_created_before_postlabel_verification(self):
        with TemporaryDirectory() as temporary:
            output, checkpoint, _ = self._source(Path(temporary))
            target = Path(temporary) / "evaluation_antigravity"
            with self.assertRaisesRegex(ValueError, "source|checkpoint"):
                module._verify_frozen_source(output, type("Store", (), {"root": checkpoint})(), expected_total=1, snapshot={})
            self.assertFalse(target.exists())

    def test_postlabel_source_revalidation_happens_before_any_evaluation_staging_write(self):
        """A source change after labels must fail before mkdir/create-only staging."""
        with TemporaryDirectory() as temporary:
            output, checkpoint, publication = self._source(Path(temporary))
            manifest = {"prediction_sha256": "a" * 64, "database_sha256": FROZEN_CONFIG.database_sha256,
                        "label_access_count": 0, "reviewed_commit": "c" * 40}
            verifications = iter((None, ValueError("source changed after labels")))
            def verify(*_args, **_kwargs):
                outcome = next(verifications)
                if outcome is not None:
                    raise outcome
            verified = SimpleNamespace(manifest_sha256="b" * 64, publication=publication.publication)
            with patch.object(module, "_inspect_publication", return_value=verified), \
                    patch.object(module, "_validate_verified_publication", return_value=(manifest, ({"dataset": "Demo", "test_size": 1},), ())), \
                    patch.object(module, "_snapshot_frozen_tree", return_value={".": ("directory", None)}), \
                    patch.object(module, "_verify_frozen_source", side_effect=verify), \
                    patch.object(module, "_write_new", side_effect=AssertionError("evaluation staging began")):
                with self.assertRaisesRegex(ValueError, "source changed after labels"):
                    module._evaluate_frozen_prediction_for_test(
                        publication, hard_coordinates={"Demo": ()},
                        label_loader=lambda _dataset: {"instance_indices": [0], "labels": [0]},
                        checkpoint=type("Store", (), {"root": checkpoint})(),
                        expected_datasets=("Demo",), evaluation_root=Path(temporary) / "evaluation_antigravity",
                    )

    def test_public_exports_are_only_the_two_production_apis(self):
        self.assertEqual(module.__all__, ["evaluate_exp436", "decide_exp436_success"])
        self.assertEqual(
            tuple(inspect.signature(module.evaluate_exp436).parameters),
            ("output_root", "evaluation_root", "profile"),
        )
        self.assertEqual(
            inspect.signature(module.evaluate_exp436).parameters["profile"].kind,
            inspect.Parameter.KEYWORD_ONLY,
        )

    def test_exp437_evaluation_rejects_exp436_fixed_paths_before_labels(self) -> None:
        with patch.object(module, "_load_test_labels") as legacy_labels, \
                patch.object(module, "verified_read_only_connection") as connection, \
                patch.object(module, "_load_test_labels_from_connection") as connection_labels:
            with self.assertRaisesRegex(ValueError, "fixed paths"):
                module.evaluate_exp436(
                    EXP436_PROFILE.output,
                    EXP436_PROFILE.evaluation,
                    profile=EXP437_PROFILE,
                )
        legacy_labels.assert_not_called()
        connection.assert_not_called()
        connection_labels.assert_not_called()

    def test_public_evaluation_keeps_only_a_keyword_profile_extension(self) -> None:
        signature = inspect.signature(module.evaluate_exp436)
        self.assertEqual(
            tuple(signature.parameters),
            ("output_root", "evaluation_root", "profile"),
        )
        self.assertEqual(
            signature.parameters["profile"].kind,
            inspect.Parameter.KEYWORD_ONLY,
        )


if __name__ == "__main__":
    unittest.main()
