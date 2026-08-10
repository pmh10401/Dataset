"""RED/GREEN coverage for Exp444's post-freeze TEST-label boundary."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from experiments.exp444_train_only_stability_precision_gate_antigravity.evaluation_antigravity import (
    evaluate_frozen_publication_antigravity,
    inspect_evaluation_antigravity,
)
from experiments.exp444_train_only_stability_precision_gate_antigravity import evaluation_antigravity as evaluation


@dataclass(frozen=True)
class _CandidateAntigravity:
    dataset: str
    instance_index: int

    @property
    def coordinate(self) -> tuple[str, int]:
        return (self.dataset, self.instance_index)


@dataclass(frozen=True)
class _PublicationAntigravity:
    publication_root: Path
    manifest: dict[str, object]
    selection: object


class _LabelsAntigravity:
    def __init__(self, rows: dict[str, tuple[tuple[int, int], ...]], events: list[str], *, fail_dataset: str | None = None, fail_close: bool = False) -> None:
        self.rows, self.events, self.closed = rows, events, False
        self.fail_dataset, self.fail_close = fail_dataset, fail_close

    def labels_for_dataset_antigravity(self, dataset: str) -> tuple[tuple[int, int], ...]:
        self.events.append(f"query:{dataset}")
        if dataset == self.fail_dataset:
            raise RuntimeError(f"query {dataset} failed")
        return self.rows[dataset]

    def close_antigravity(self) -> None:
        self.closed = True
        self.events.append("close_labels")
        if self.fail_close:
            raise RuntimeError("close labels failed")


class _DependenciesAntigravity:
    """Complete test boundary: inspection, pinned authority, labels, and publication."""
    dataset_count = 2
    label_count = 4

    def __init__(self, root: Path, *, tamper_second: bool = False, tamper_inspection: int | None = None, fail_open: bool = False, fail_after_labels: bool = False, fail_query: str | None = None, fail_close: bool = False) -> None:
        self.root, self.events = root, []
        self.tamper_second, self.tamper_inspection, self.fail_open, self.fail_after_labels = tamper_second, tamper_inspection, fail_open, fail_after_labels
        self.fail_query, self.fail_close = fail_query, fail_close
        candidates = (_CandidateAntigravity("A", 1), _CandidateAntigravity("B", 0))
        self.selection = type("SelectionAntigravity", (), {"source": candidates, "stable": (candidates[0],), "unstable": (candidates[1],)})()
        leaf = {"dataset_summary_antigravity.jsonl": "a" * 64, "source_audit_antigravity.json": "b" * 64}
        self.publication = _PublicationAntigravity(root / "publication_antigravity", {"prediction_sha256": "c" * 64, "leaf_sha256": leaf}, self.selection)

    def inspect_publication_antigravity(self) -> _PublicationAntigravity:
        self.events.append("inspect_publication")
        if self.tamper_second and self.events.count("inspect_publication") == 2:
            self.publication.manifest["prediction_sha256"] = "d" * 64
        if self.tamper_inspection == self.events.count("inspect_publication"):
            self.publication.manifest["prediction_sha256"] = "d" * 64
        return self.publication

    def publication_identity_antigravity(self, publication: _PublicationAntigravity) -> dict[str, object]:
        self.events.append("identity")
        return {
            "root_device": 1, "root_inode": 2,
            "manifest_sha256": sha256(repr(sorted(publication.manifest.items())).encode()).hexdigest(),
            "prediction_sha256": publication.manifest["prediction_sha256"],
            "leaf_sha256": dict(publication.manifest["leaf_sha256"]),
            "dataset_summary_sha256": publication.manifest["leaf_sha256"]["dataset_summary_antigravity.jsonl"],
            "source_audit_sha256": publication.manifest["leaf_sha256"]["source_audit_antigravity.json"],
            "dataset_order": ["A", "B"],
            "source_order": [(x.dataset, x.instance_index) for x in publication.selection.source],
            "stable": [(x.dataset, x.instance_index) for x in publication.selection.stable],
            "unstable": [(x.dataset, x.instance_index) for x in publication.selection.unstable],
        }

    def hard_predictions_antigravity(self, _identity: dict[str, object]) -> dict[str, set[int]]:
        self.events.append("hard")
        return {"A": {0}, "B": set()}

    def open_labels_antigravity(self, _identity: dict[str, object]) -> _LabelsAntigravity:
        self.events.append("open_labels")
        if self.fail_open:
            raise RuntimeError("open failed")
        return _LabelsAntigravity({"A": ((0, 0), (1, 1)), "B": ((0, 0), (1, 0))}, self.events, fail_dataset=self.fail_query, fail_close=self.fail_close)

    def evaluation_root_antigravity(self) -> Path:
        return self.root / "evaluation_antigravity"

    def after_labels_antigravity(self) -> None:
        if self.fail_after_labels:
            raise RuntimeError("forced post-label failure")


class FrozenEvaluationAntigravityTests(unittest.TestCase):
    def _single_audit_payload_antigravity(self, root: Path) -> dict[str, object]:
        audits = list(root.glob("evaluation_audit_*_antigravity.json"))
        self.assertEqual(len(audits), 1)
        return json.loads(audits[0].read_text())

    def _evaluated_payloads_antigravity(self, root: Path) -> tuple[_DependenciesAntigravity, dict[str, bytes]]:
        dependencies = _DependenciesAntigravity(root)
        evaluate_frozen_publication_antigravity(dependencies=dependencies)
        evaluation_root = dependencies.evaluation_root_antigravity()
        return dependencies, {name: (evaluation_root / name).read_bytes() for name in evaluation.EVALUATION_NAMES_ANTIGRAVITY}

    def _refresh_artifact_map_antigravity(self, payloads: dict[str, bytes]) -> dict[str, bytes]:
        refreshed = dict(payloads)
        refreshed["artifact_sha256_antigravity.json"] = evaluation.canonical_json_bytes_antigravity({
            "schema": "exp444.evaluation-artifact-sha256.v1",
            "self_excluded": "artifact_sha256_antigravity.json",
            "files": {name: sha256(refreshed[name]).hexdigest() for name in evaluation.EVALUATION_NAMES_ANTIGRAVITY[:-1]},
        })
        return refreshed

    def test_production_population_baseline_and_lane_conservation_gates_fail_closed_antigravity(self) -> None:
        """Wrong frozen population, baseline, or lane deltas must fail before payload creation."""
        identity = {"source_order": [["A", 0]], "stable": [["A", 0]], "unstable": []}
        rows = ({"dataset": "A", "baseline": [1915, 303, 2003, 191679], "frozen_source": [1916, 303, 2002, 191679], "stable": [1916, 303, 2002, 191679], "unstable": [1915, 303, 2003, 191679]},)
        transitions = ({"dataset": "A", "instance_index": 0, "lane": "stable_review", "label": 1, "baseline_predicted": False, "frozen_source_predicted": True, "stable_predicted": True, "unstable_predicted": False},)
        with self.assertRaisesRegex(ValueError, "dataset|query|rows"):
            evaluation._validate_prepublication_antigravity(identity, rows, transitions, attempts=1, completions=1, label_rows=1, production=True)
        wrong = dict(rows[0]); wrong["baseline"] = [1914, 303, 2003, 191680]
        with self.assertRaisesRegex(ValueError, "baseline"):
            evaluation._validate_prepublication_antigravity(identity, (wrong,), transitions, attempts=1117, completions=1117, label_rows=195900, production=True)
        bad_transition = dict(transitions[0]); bad_transition["label"] = 0
        with self.assertRaisesRegex(ValueError, "conservation"):
            evaluation._validate_prepublication_antigravity(identity, rows, (bad_transition,), attempts=1, completions=1, label_rows=195900, production=False)

    def test_verified_label_session_hash_binds_and_authorizes_only_frozen_query_antigravity(self) -> None:
        """Replacing the descriptor helper or widening its SQL boundary must fail."""
        with tempfile.TemporaryDirectory(prefix="exp444_verified_labels_antigravity_") as directory:
            database = Path(directory) / "labels_antigravity.sqlite"
            connection = sqlite3.connect(database)
            connection.executescript("CREATE TABLE datasets(id INTEGER PRIMARY KEY,name TEXT); CREATE TABLE instances(dataset_id INTEGER,instance_index INTEGER,split TEXT,label INTEGER);")
            connection.execute("INSERT INTO datasets VALUES(1,'A')")
            connection.executemany("INSERT INTO instances VALUES(1,?, 'TEST',?)", ((0, 0), (1, 1)))
            connection.commit(); connection.close()
            digest = sha256(database.read_bytes()).hexdigest()
            with evaluation._open_verified_labels_antigravity(database, digest, ("A",)) as labels:
                self.assertFalse(hasattr(labels, "_connection"))
                self.assertFalse(hasattr(labels, "__dict__"))
                self.assertEqual(labels.labels_for_dataset_antigravity("A"), ((0, 0), (1, 1)))
                self.assertFalse(hasattr(labels, "execute_for_test_antigravity"))
                with self.assertRaisesRegex(ValueError, "order"):
                    labels.labels_for_dataset_antigravity("A")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                with evaluation._open_verified_labels_antigravity(database, "0" * 64, ("A",)):
                    pass
    def test_import_firewall_keeps_sqlite_unloaded_until_post_freeze_branch_antigravity(self) -> None:
        """Moving the real label dependency to module scope must fail this isolated import."""
        completed = subprocess.run(
            [sys.executable, "-B", "-c", "import sys; import experiments.exp444_train_only_stability_precision_gate_antigravity.evaluation_antigravity; assert 'sqlite3' not in sys.modules"],
            cwd=Path(__file__).parents[3], capture_output=True, text=True, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_evaluator_reads_labels_only_after_double_inspection_antigravity(self) -> None:
        """Removing either inspection or moving label open earlier must fail."""
        with tempfile.TemporaryDirectory(prefix="exp444_evaluation_antigravity_") as directory:
            dependencies = _DependenciesAntigravity(Path(directory))
            result = evaluate_frozen_publication_antigravity(dependencies=dependencies)

            self.assertEqual(dependencies.events[:5], ["inspect_publication", "identity", "hard", "inspect_publication", "identity"])
            self.assertEqual(dependencies.events[5], "open_labels")
            self.assertEqual(result.status, "descriptive_complete_no_promotion")
            self.assertFalse(result.scientific_winner_selected)
            self.assertEqual((result.query_attempts, result.query_completions, result.rows), (2, 2, 4))
            self.assertEqual(result.baseline_confusion, (0, 1, 1, 2))
            self.assertEqual(result.stable_confusion, (1, 1, 0, 2))

    def test_publication_change_blocks_labels_and_existing_replay_never_opens_them_antigravity(self) -> None:
        """A changed second Task8 identity cannot cross the label boundary or clobber replay."""
        with tempfile.TemporaryDirectory(prefix="exp444_evaluation_tamper_antigravity_") as directory:
            root = Path(directory)
            tampered = _DependenciesAntigravity(root, tamper_second=True)
            with self.assertRaisesRegex(ValueError, "identity|publication"):
                evaluate_frozen_publication_antigravity(dependencies=tampered)
            self.assertNotIn("open_labels", tampered.events)

            dependencies = _DependenciesAntigravity(root)
            first = evaluate_frozen_publication_antigravity(dependencies=dependencies)
            replay = inspect_evaluation_antigravity(dependencies=dependencies)
            self.assertEqual(first.evaluation_sha256, replay.evaluation_sha256)
            self.assertEqual(dependencies.events.count("open_labels"), 1)

    def test_replay_reconstructs_derived_leaves_even_when_artifact_map_is_regenerated_antigravity(self) -> None:
        """Changing a derived diagnostic and its map hash must not make replay accept it."""
        with tempfile.TemporaryDirectory(prefix="exp444_evaluation_replay_antigravity_") as directory:
            dependencies = _DependenciesAntigravity(Path(directory))
            evaluate_frozen_publication_antigravity(dependencies=dependencies)
            root = dependencies.evaluation_root_antigravity()
            leaf = root / "rescued_datasets_antigravity.jsonl"
            leaf.write_text('{"dataset":"B"}\n')
            artifact = __import__("json").loads((root / "artifact_sha256_antigravity.json").read_text())
            artifact["files"][leaf.name] = sha256(leaf.read_bytes()).hexdigest()
            (root / "artifact_sha256_antigravity.json").write_bytes(evaluation.canonical_json_bytes_antigravity(artifact))
            with self.assertRaisesRegex(ValueError, "derived|replay|evaluation"):
                inspect_evaluation_antigravity(dependencies=dependencies)

    def test_post_label_failure_writes_a_unique_audit_sibling_antigravity(self) -> None:
        """Post-label exceptions preserve a new audit record without mutating a commit."""
        with tempfile.TemporaryDirectory(prefix="exp444_evaluation_audit_antigravity_") as directory:
            dependencies = _DependenciesAntigravity(Path(directory), fail_after_labels=True)
            with self.assertRaisesRegex(RuntimeError, "forced"):
                evaluate_frozen_publication_antigravity(dependencies=dependencies)
            audits = list(Path(directory).glob("evaluation_audit_*_antigravity.json"))
            self.assertEqual(len(audits), 1)
            self.assertIn('"query_attempts":2', audits[0].read_text())

    def test_each_post_label_identity_checkpoint_blocks_publication_antigravity(self) -> None:
        """Removing any close/compute/pre-publish reinspection would commit after a Task8 swap."""
        for inspection in (3, 4, 5):
            with self.subTest(inspection=inspection), tempfile.TemporaryDirectory(prefix="exp444_identity_checkpoint_antigravity_") as directory:
                dependencies = _DependenciesAntigravity(Path(directory), tamper_inspection=inspection)
                with self.assertRaisesRegex(ValueError, "Task8 publication changed"):
                    evaluate_frozen_publication_antigravity(dependencies=dependencies)
                self.assertFalse(dependencies.evaluation_root_antigravity().exists())

    def test_open_failure_starts_audit_before_first_label_query_antigravity(self) -> None:
        """Deferring audit until query one would lose a failed label-bound open."""
        with tempfile.TemporaryDirectory(prefix="exp444_open_audit_antigravity_") as directory:
            dependencies = _DependenciesAntigravity(Path(directory), fail_open=True)
            with self.assertRaisesRegex(RuntimeError, "open"):
                evaluate_frozen_publication_antigravity(dependencies=dependencies)
            audits = list(Path(directory).glob("evaluation_audit_*_antigravity.json"))
            self.assertEqual(len(audits), 1)
            self.assertIn('"commit_state":"none"', audits[0].read_text())
            self.assertIn('"query_attempts":0', audits[0].read_text())

    def test_partial_writer_failure_records_commit_write_partial_audit_antigravity(self) -> None:
        """A writer exception after destination creation must retain truthful partial evidence."""
        with tempfile.TemporaryDirectory(prefix="exp444_partial_write_antigravity_") as directory:
            dependencies = _DependenciesAntigravity(Path(directory))
            def fail_writer(root: Path, _payloads: object) -> None:
                root.mkdir(mode=0o700)
                (root / "partial_antigravity.json").write_text("partial")
                raise RuntimeError("writer failed")
            with mock.patch.object(evaluation, "_publish_create_only_antigravity", side_effect=fail_writer), self.assertRaisesRegex(RuntimeError, "writer"):
                evaluate_frozen_publication_antigravity(dependencies=dependencies)
            audit = next(Path(directory).glob("evaluation_audit_*_antigravity.json")).read_text()
            self.assertIn('"phase":"commit_write"', audit)
            self.assertIn('"commit_state":"partial"', audit)

    def test_second_label_query_failure_audits_the_completed_first_dataset_antigravity(self) -> None:
        """Removing query accounting or close-on-query-error would lose truthful evidence."""
        with tempfile.TemporaryDirectory(prefix="exp444_second_query_audit_antigravity_") as directory:
            dependencies = _DependenciesAntigravity(Path(directory), fail_query="B")
            with self.assertRaisesRegex(RuntimeError, "query B failed"):
                evaluate_frozen_publication_antigravity(dependencies=dependencies)
            audit = self._single_audit_payload_antigravity(Path(directory))
            self.assertEqual({name: audit[name] for name in ("phase", "query_attempts", "query_completions", "rows", "session_state", "commit_state")}, {
                "phase": "labels_open", "query_attempts": 2, "query_completions": 1, "rows": 2,
                "session_state": "closed", "commit_state": "none",
            })

    def test_label_session_close_failure_audits_all_successful_queries_antigravity(self) -> None:
        """Suppressing a close failure or resetting counters would falsify the audit sibling."""
        with tempfile.TemporaryDirectory(prefix="exp444_close_audit_antigravity_") as directory:
            dependencies = _DependenciesAntigravity(Path(directory), fail_close=True)
            with self.assertRaisesRegex(RuntimeError, "close labels failed"):
                evaluate_frozen_publication_antigravity(dependencies=dependencies)
            audit = self._single_audit_payload_antigravity(Path(directory))
            self.assertEqual({name: audit[name] for name in ("phase", "query_attempts", "query_completions", "rows", "session_state", "commit_state")}, {
                "phase": "labels_open", "query_attempts": 2, "query_completions": 2, "rows": 4,
                "session_state": "close_failed", "commit_state": "none",
            })

    def test_complete_writer_then_failure_preserves_complete_evaluation_audit_antigravity(self) -> None:
        """Downgrading a complete destination to partial after a late writer error is incorrect."""
        with tempfile.TemporaryDirectory(prefix="exp444_complete_write_audit_antigravity_") as directory:
            dependencies = _DependenciesAntigravity(Path(directory))
            written: dict[str, bytes] = {}
            real_writer = evaluation._publish_create_only_antigravity

            def write_all_then_fail(root: Path, payloads: object) -> None:
                self.assertIsInstance(payloads, dict)
                written.update(payloads)
                real_writer(root, payloads)
                raise RuntimeError("writer failed after complete evaluation")

            with mock.patch.object(evaluation, "_publish_create_only_antigravity", side_effect=write_all_then_fail), self.assertRaisesRegex(RuntimeError, "complete evaluation"):
                evaluate_frozen_publication_antigravity(dependencies=dependencies)
            root = dependencies.evaluation_root_antigravity()
            self.assertEqual({path.name for path in root.iterdir()}, set(evaluation.EVALUATION_NAMES_ANTIGRAVITY))
            self.assertEqual({name: (root / name).read_bytes() for name in evaluation.EVALUATION_NAMES_ANTIGRAVITY}, written)
            artifact = json.loads((root / "artifact_sha256_antigravity.json").read_text())
            self.assertEqual({name: sha256((root / name).read_bytes()).hexdigest() for name in evaluation.EVALUATION_NAMES_ANTIGRAVITY[:-1]}, artifact["files"])
            audit = self._single_audit_payload_antigravity(Path(directory))
            self.assertEqual({name: audit[name] for name in ("phase", "commit_state")}, {"phase": "commit_write", "commit_state": "complete"})

    def test_complete_evaluation_reentry_replays_without_labels_or_audit_antigravity(self) -> None:
        """Taking the label branch on an existing complete evaluation would violate replay isolation."""
        with tempfile.TemporaryDirectory(prefix="exp444_complete_reentry_antigravity_") as directory:
            dependencies, before = self._evaluated_payloads_antigravity(Path(directory))
            first = inspect_evaluation_antigravity(dependencies=dependencies)
            with mock.patch.object(dependencies, "open_labels_antigravity", side_effect=AssertionError("labels must stay closed")):
                second = evaluate_frozen_publication_antigravity(dependencies=dependencies)
            root = dependencies.evaluation_root_antigravity()
            self.assertEqual(second.evaluation_sha256, first.evaluation_sha256)
            self.assertEqual({name: (root / name).read_bytes() for name in evaluation.EVALUATION_NAMES_ANTIGRAVITY}, before)
            self.assertEqual(list(Path(directory).glob("evaluation_audit_*_antigravity.json")), [])

    def test_replay_rejects_semantic_mutations_after_artifact_map_refresh_antigravity(self) -> None:
        """Schema/order/transition gates must reject mutations even with a current artifact map."""
        with tempfile.TemporaryDirectory(prefix="exp444_semantic_replay_antigravity_") as directory:
            _, payloads = self._evaluated_payloads_antigravity(Path(directory))

            with self.subTest(mutation="nonofficial two-row payload under production replay"):
                with self.assertRaisesRegex(ValueError, "dataset|query|rows"):
                    evaluation._result_from_payloads_antigravity(payloads, production=True)

            def mutate_jsonl(name: str, transform: object) -> dict[str, bytes]:
                changed = dict(payloads)
                rows = [json.loads(line) for line in changed[name].splitlines()]
                transform(rows)
                changed[name] = evaluation.canonical_jsonl_bytes_antigravity(rows)
                return self._refresh_artifact_map_antigravity(changed)

            with self.subTest(mutation="metric extra key"):
                changed = mutate_jsonl("dataset_metrics_antigravity.jsonl", lambda rows: rows[0].update({"extra": 1}))
                with self.assertRaisesRegex(ValueError, "metrics"):
                    evaluation._result_from_payloads_antigravity(changed)
            with self.subTest(mutation="metric row reorder"):
                changed = mutate_jsonl("dataset_metrics_antigravity.jsonl", lambda rows: rows.reverse())
                with self.assertRaisesRegex(ValueError, "metrics"):
                    evaluation._result_from_payloads_antigravity(changed)
            with self.subTest(mutation="transition extra key"):
                changed = mutate_jsonl("candidate_transitions_antigravity.jsonl", lambda rows: rows[0].update({"extra": 1}))
                with self.assertRaisesRegex(ValueError, "transitions"):
                    evaluation._result_from_payloads_antigravity(changed)
            with self.subTest(mutation="transition reorder"):
                changed = mutate_jsonl("candidate_transitions_antigravity.jsonl", lambda rows: rows.reverse())
                with self.assertRaisesRegex(ValueError, "transitions"):
                    evaluation._result_from_payloads_antigravity(changed)
            with self.subTest(mutation="frozen source prediction false"):
                changed = mutate_jsonl("candidate_transitions_antigravity.jsonl", lambda rows: rows[0].update({"frozen_source_predicted": False}))
                with self.assertRaisesRegex(ValueError, "transitions"):
                    evaluation._result_from_payloads_antigravity(changed)
