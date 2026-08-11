"""TDD coverage for the Exp447 single-use post-freeze evaluator."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import tempfile
import unittest
import json
from unittest.mock import patch
from pathlib import Path

from experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity.config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity.contracts_antigravity import PublicationIdentityAntigravity


def _identity_antigravity() -> PublicationIdentityAntigravity:
    datasets = tuple(f"dataset_{index:04d}_antigravity" for index in range(1117))
    source = tuple((datasets[11 + index], 0) for index in range(12)) + tuple((datasets[11], index) for index in range(1, 4)) + tuple((datasets[30 + index], 0) for index in range(15))
    return PublicationIdentityAntigravity(
        root_device=1, root_inode=2,
        leaf_sha256={"leaf_antigravity.json": "a" * 64},
        manifest_sha256="b" * 64, prediction_sha256="c" * 64,
        artifact_map_sha256="d" * 64, dataset_order=datasets,
        source_coordinates=source, stable_coordinates=source[:26],
        unstable_coordinates=source[26:], label_access_count=0,
    )


def _labels_antigravity(identity: PublicationIdentityAntigravity) -> dict[str, tuple[tuple[int, int], ...]]:
    lengths = [176 if index < 425 else 175 for index in range(1117)]
    rows = {dataset: [(index, 0) for index in range(length)] for dataset, length in zip(identity.dataset_order, lengths, strict=True)}
    flat = [(dataset, index) for dataset in identity.dataset_order for index, _ in rows[dataset]]
    for dataset, index in flat[:3918]:
        rows[dataset][index] = (index, 1)
    return {dataset: tuple(value) for dataset, value in rows.items()}


@dataclass
class _HardAntigravity:
    sha256: str
    coordinates: frozenset[tuple[str, int]]


class _SessionAntigravity:
    def __init__(self, labels: dict[str, tuple[tuple[int, int], ...]], events: list[str], *, close_error: Exception | None = None, query_error: Exception | None = None) -> None:
        self._labels = labels
        self._events = events
        self._close_error = close_error
        self._query_error = query_error

    def labels_for_dataset_antigravity(self, dataset: str) -> tuple[tuple[int, int], ...]:
        self._events.append(f"labels:{dataset}")
        if self._query_error is not None:
            raise self._query_error
        return self._labels[dataset]

    def close_antigravity(self) -> None:
        self._events.append("close_labels")
        if self._close_error is not None:
            raise self._close_error


class _DependenciesAntigravity:
    def __init__(self, root: Path, *, second_identity: PublicationIdentityAntigravity | None = None, after_identities: list[PublicationIdentityAntigravity] | None = None, close_error: Exception | None = None, query_error: Exception | None = None) -> None:
        self.events: list[str] = []
        self.root = root
        self.identity = _identity_antigravity()
        self.second_identity = self.identity if second_identity is None else second_identity
        self.labels = _labels_antigravity(self.identity)
        # 1,915 positive and 303 negative hard coordinates: frozen baseline.
        flat = [(dataset, index) for dataset, rows in self.labels.items() for index, label in rows if label == 1]
        negative = [(dataset, index) for dataset, rows in self.labels.items() for index, label in rows if label == 0]
        self.hard = _HardAntigravity("e" * 64, frozenset(flat[:1915] + negative[:303]))
        self.close_error = close_error
        self.query_error = query_error
        self.after_identities = list(after_identities or [])

    def capture_exp446(self) -> PublicationIdentityAntigravity:
        self.events.append("capture_exp446")
        return self.identity if self.events.count("capture_exp446") == 1 else self.second_identity

    def capture_exp404(self, identity: PublicationIdentityAntigravity) -> _HardAntigravity:
        self.events.append("capture_exp404")
        self.assert_identity(identity)
        return self.hard

    def derive_hard(self, hard: _HardAntigravity, identity: PublicationIdentityAntigravity) -> frozenset[tuple[str, int]]:
        self.events.append("derive_hard")
        self.assert_identity(identity)
        return hard.coordinates

    def compare_exp446(self, first: PublicationIdentityAntigravity, second: PublicationIdentityAntigravity) -> None:
        self.events.append("compare_exp446_identity")
        if first != second:
            raise ValueError("Exp446 inspection drift")

    def open_labels(self, first: PublicationIdentityAntigravity, second: PublicationIdentityAntigravity) -> _SessionAntigravity:
        self.events.append("open_labels")
        self.compare_exp446(first, second)
        return _SessionAntigravity(self.labels, self.events, close_error=self.close_error, query_error=self.query_error)

    def capture_after(self) -> PublicationIdentityAntigravity:
        self.events.append("capture_exp446_post")
        return self.after_identities.pop(0) if self.after_identities else self.identity

    def assert_identity(self, identity: PublicationIdentityAntigravity) -> None:
        if identity != self.identity:
            raise ValueError("identity differs")

    def bootstrap(self, baseline: object, comparator: object, name: str, *, draws: object = None) -> dict[str, object]:
        self.events.append(f"bootstrap:{name}")
        from experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity.bootstrap_antigravity import paired_bootstrap_antigravity
        return paired_bootstrap_antigravity(baseline, comparator, name, draws=draws).to_payload_antigravity()


class EvaluationAntigravityTests(unittest.TestCase):
    def test_evaluator_orders_double_inspection_before_labels_and_conserves_outputs_antigravity(self) -> None:
        from experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity.evaluation_antigravity import _evaluate_with_dependencies_antigravity
        with tempfile.TemporaryDirectory() as directory:
            dependencies = _DependenciesAntigravity(Path(directory) / "evaluation_root_antigravity")
            result = _evaluate_with_dependencies_antigravity("a" * 40, dependencies)
        self.assertEqual(dependencies.events[:5], [
            "capture_exp446", "capture_exp404", "derive_hard", "capture_exp446", "compare_exp446_identity",
        ])
        self.assertEqual(dependencies.events[5], "open_labels")
        self.assertEqual(result.metrics["hard_baseline"].confusion.to_payload_antigravity(), {"tp": 1915, "fp": 303, "fn": 2003, "tn": 191679})
        self.assertEqual(result.metrics["hard_plus_source_30"].confusion.to_payload_antigravity(), {"tp": 1930, "fp": 318, "fn": 1988, "tn": 191664})
        self.assertEqual(result.label_rows, 195900)
        self.assertEqual(result.dataset_count, 1117)
        self.assertEqual(len(result.bootstrap), 3)
        self.assertEqual(result.evaluation_status, "descriptive_complete_no_promotion")
        self.assertEqual(result.transition_counts["source"], 30)
        self.assertEqual(result.transition_counts["stable"], 26)
        self.assertEqual(result.transition_counts["unstable"], 4)
        self.assertEqual(result.transition_counts["source"], result.transition_counts["stable"] + result.transition_counts["unstable"])

    def test_second_inspection_drift_never_opens_labels_or_creates_root_antigravity(self) -> None:
        from experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity.evaluation_antigravity import _evaluate_with_dependencies_antigravity
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "evaluation_root_antigravity"
            second = _identity_antigravity()
            object.__setattr__(second, "root_inode", 3)
            dependencies = _DependenciesAntigravity(root, second_identity=second)
            with self.assertRaisesRegex(ValueError, "drift"):
                _evaluate_with_dependencies_antigravity("a" * 40, dependencies)
            self.assertNotIn("open_labels", dependencies.events)
            self.assertFalse(root.exists())

    def test_label_close_failure_preserves_audit_without_destination_root_antigravity(self) -> None:
        from experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity.evaluation_antigravity import _evaluate_with_dependencies_antigravity
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "evaluation_root_antigravity"
            dependencies = _DependenciesAntigravity(root, close_error=OSError("close broken"))
            with self.assertRaisesRegex(OSError, "close broken"):
                _evaluate_with_dependencies_antigravity("a" * 40, dependencies)
            audits = list(Path(directory).glob("evaluation_audit_*_antigravity.json"))
            self.assertEqual(len(audits), 1)
            self.assertFalse(root.exists())

    def test_audit_counts_attempt_before_failed_label_query_and_closes_session_antigravity(self) -> None:
        from experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity.evaluation_antigravity import _evaluate_with_dependencies_antigravity
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "evaluation_root_antigravity"
            dependencies = _DependenciesAntigravity(root, query_error=OSError("query broken"))
            with self.assertRaisesRegex(OSError, "query broken"):
                _evaluate_with_dependencies_antigravity("a" * 40, dependencies)
            audit = json.loads(next(Path(directory).glob("evaluation_audit_*_antigravity.json")).read_text(encoding="utf-8"))
            self.assertEqual(audit["query_attempts"], 1)
            self.assertEqual(audit["query_completions"], 0)
            self.assertTrue(audit["label_session_closed"])

    def test_post_label_inspection_failure_matrix_preserves_audit_antigravity(self) -> None:
        from experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity.evaluation_antigravity import _evaluate_with_dependencies_antigravity
        for phase, position in (("post_close", 0), ("post_compute", 1), ("pre_commit", 2)):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "evaluation_root_antigravity"
                drift = _identity_antigravity(); object.__setattr__(drift, "root_inode", 9)
                after = [_identity_antigravity(), _identity_antigravity(), _identity_antigravity()]
                after[position] = drift
                dependencies = _DependenciesAntigravity(root, after_identities=after)
                with self.assertRaisesRegex(ValueError, "drift"):
                    _evaluate_with_dependencies_antigravity("a" * 40, dependencies)
                audit = json.loads(next(Path(directory).glob("evaluation_audit_*_antigravity.json")).read_text(encoding="utf-8"))
                self.assertTrue(audit["label_session_closed"])
                self.assertEqual(audit["destination_state"], "absent")

    def test_create_only_and_audit_failure_matrix_antigravity(self) -> None:
        from experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity import evaluation_antigravity as module
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "evaluation_root_antigravity"
            dependencies = _DependenciesAntigravity(root)
            original = module.write_create_only_antigravity
            calls = 0
            def fail_mid_write(path: Path, payload: bytes) -> None:
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise OSError("partial write")
                original(path, payload)
            with patch.object(module, "write_create_only_antigravity", side_effect=fail_mid_write), self.assertRaisesRegex(OSError, "partial write"):
                module._evaluate_with_dependencies_antigravity("a" * 40, dependencies)
            self.assertTrue(root.is_dir())
            self.assertNotEqual({path.name for path in root.iterdir()}, set(module._LEAVES_ANTIGRAVITY))
            audit = json.loads(next(Path(directory).glob("evaluation_audit_*_antigravity.json")).read_text(encoding="utf-8"))
            self.assertEqual(audit["destination_state"], "partial")
            self.assertTrue(audit["label_session_closed"])
            with self.assertRaises(ValueError):
                module._evaluate_with_dependencies_antigravity("a" * 40, dependencies)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "evaluation_root_antigravity"
            dependencies = _DependenciesAntigravity(root, query_error=OSError("query original"))
            with patch.object(module, "_write_audit_antigravity", side_effect=OSError("audit broken")), self.assertRaisesRegex(OSError, "query original"):
                module._evaluate_with_dependencies_antigravity("a" * 40, dependencies)
            self.assertFalse(root.exists())

    def test_existing_and_symlink_destination_are_rejected_before_labels_antigravity(self) -> None:
        from experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity import evaluation_antigravity as module
        for kind in ("partial", "symlink"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "evaluation_root_antigravity"
                if kind == "partial":
                    root.mkdir(); (root / "partial_antigravity.json").write_text("{}\n", encoding="utf-8")
                else:
                    target = Path(directory) / "target_antigravity"; target.mkdir(); root.symlink_to(target, target_is_directory=True)
                dependencies = _DependenciesAntigravity(root)
                with self.assertRaises(ValueError):
                    module._evaluate_with_dependencies_antigravity("a" * 40, dependencies)
                self.assertNotIn("open_labels", dependencies.events)

    def test_inspect_rejects_semantic_leaf_tamper_even_when_artifact_hash_is_recomputed_antigravity(self) -> None:
        from experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity.evaluation_antigravity import _inspect_root_with_dependencies_antigravity
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "evaluation_root_antigravity"
            root.mkdir()
            (root / "evaluation_antigravity.json").write_text('{"schema":"exp447","dataset_count":1117}\n', encoding="utf-8")
            digest = sha256((root / "evaluation_antigravity.json").read_bytes()).hexdigest()
            (root / "artifact_sha256_antigravity.json").write_text('{"evaluation_antigravity.json":"' + digest + '"}\n', encoding="utf-8")
            dependencies = _DependenciesAntigravity(root)
            with self.assertRaises(ValueError):
                _inspect_root_with_dependencies_antigravity(dependencies)

    def test_inspect_rejects_real_confusion_tamper_after_artifact_map_rehash_antigravity(self) -> None:
        from experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity.evaluation_antigravity import _evaluate_with_dependencies_antigravity, _inspect_root_with_dependencies_antigravity
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "evaluation_root_antigravity"
            dependencies = _DependenciesAntigravity(root)
            _evaluate_with_dependencies_antigravity("a" * 40, dependencies)
            _inspect_root_with_dependencies_antigravity(dependencies)
            confusion_path = root / "confusion_antigravity.json"
            confusion = json.loads(confusion_path.read_text(encoding="utf-8"))
            confusion["methods"][1]["tp"] += 1
            confusion_path.write_text(json.dumps(confusion, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
            artifact = {path.name: sha256(path.read_bytes()).hexdigest() for path in root.iterdir() if path.name != "artifact_sha256_antigravity.json"}
            (root / "artifact_sha256_antigravity.json").write_text(json.dumps(artifact, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                _inspect_root_with_dependencies_antigravity(dependencies)

    def test_semantic_decision_bootstrap_transition_tampers_survive_rehash_but_fail_replay_antigravity(self) -> None:
        from experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity.evaluation_antigravity import _evaluate_with_dependencies_antigravity, _inspect_root_with_dependencies_antigravity
        def mutate_decision(root: Path) -> None:
            path = root / "decision_antigravity.json"; payload = json.loads(path.read_text(encoding="utf-8")); payload["operational_alert_eligible"] = True; path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
        def mutate_bootstrap(root: Path) -> None:
            path = root / "bootstrap_antigravity.json"; payload = json.loads(path.read_text(encoding="utf-8")); payload["comparisons"][0]["micro_delta"] += 0.25; path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
        def mutate_transition(root: Path) -> None:
            path = root / "candidate_transitions_antigravity.jsonl"; rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]; rows[0]["transition"] = "added_false_alarm"; path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
        for name, mutate in (("decision", mutate_decision), ("bootstrap", mutate_bootstrap), ("transition", mutate_transition)):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "evaluation_root_antigravity"; dependencies = _DependenciesAntigravity(root)
                _evaluate_with_dependencies_antigravity("a" * 40, dependencies); mutate(root)
                artifact = {path.name: sha256(path.read_bytes()).hexdigest() for path in root.iterdir() if path.name != "artifact_sha256_antigravity.json"}
                (root / "artifact_sha256_antigravity.json").write_text(json.dumps(artifact, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
                with self.assertRaises(ValueError): _inspect_root_with_dependencies_antigravity(dependencies)

    def test_inspect_rejects_exact_schema_type_index_and_order_tampers_antigravity(self) -> None:
        """A rehashed leaf cannot change a value's type, row index, or row order."""
        from experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity.evaluation_antigravity import _evaluate_with_dependencies_antigravity, _inspect_root_with_dependencies_antigravity

        def rewrite(root: Path, name: str, payload: object, *, jsonl: bool = False) -> None:
            path = root / name
            if jsonl:
                path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in payload), encoding="utf-8")
            else:
                path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")

        def rehash(root: Path) -> None:
            artifact = {path.name: sha256(path.read_bytes()).hexdigest() for path in root.iterdir() if path.name != "artifact_sha256_antigravity.json"}
            rewrite(root, "artifact_sha256_antigravity.json", artifact)

        def bool_row_index(root: Path) -> None:
            path = root / "dataset_metrics_antigravity.jsonl"; rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            rows[0]["dataset_order"] = True; rewrite(root, path.name, rows, jsonl=True)

        def bool_second_row_index(root: Path) -> None:
            path = root / "dataset_metrics_antigravity.jsonl"; rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            rows[1]["dataset_order"] = True; rewrite(root, path.name, rows, jsonl=True)

        def swapped_dataset_rows(root: Path) -> None:
            path = root / "dataset_metrics_antigravity.jsonl"; rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            rows[0], rows[1] = rows[1], rows[0]; rewrite(root, path.name, rows, jsonl=True)

        def bool_evaluation_count(root: Path) -> None:
            path = root / "evaluation_antigravity.json"; payload = json.loads(path.read_text(encoding="utf-8")); payload["label_rows"] = True; rewrite(root, path.name, payload)

        def extra_decision_key(root: Path) -> None:
            path = root / "decision_antigravity.json"; payload = json.loads(path.read_text(encoding="utf-8")); payload["unexpected"] = None; rewrite(root, path.name, payload)

        def integer_zero_f1(root: Path) -> None:
            path = root / "dataset_metrics_antigravity.jsonl"; rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            for row in rows:
                for method in ("hard_baseline", "hard_plus_source_30", "hard_plus_stable_26", "hard_plus_unstable_4"):
                    if row[method]["f1"] == 0.0:
                        row[method]["f1"] = 0; rewrite(root, path.name, rows, jsonl=True); return
            self.fail("fixture did not provide an exact zero F1 row")

        def bool_diagnostic_increment(root: Path) -> None:
            path = root / "rescued_datasets_antigravity.jsonl"; rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            row = next(item for item in rows if item["tp_increase"] == 1)
            row["tp_increase"] = True; rewrite(root, path.name, rows, jsonl=True)

        for name, mutate in (("bool_row_index", bool_row_index), ("bool_second_row_index", bool_second_row_index), ("swapped_dataset_rows", swapped_dataset_rows), ("bool_evaluation_count", bool_evaluation_count), ("extra_decision_key", extra_decision_key), ("integer_zero_f1", integer_zero_f1), ("bool_diagnostic_increment", bool_diagnostic_increment)):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "evaluation_root_antigravity"; dependencies = _DependenciesAntigravity(root)
                _evaluate_with_dependencies_antigravity("a" * 40, dependencies); mutate(root); rehash(root)
                with self.assertRaises(ValueError): _inspect_root_with_dependencies_antigravity(dependencies)

    def test_label_type_index_and_population_failures_are_audited_antigravity(self) -> None:
        """The one label session rejects malformed TEST rows before publication."""
        from experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity.evaluation_antigravity import _evaluate_with_dependencies_antigravity
        for name, mutate in (
            ("bool_index", lambda labels, first: labels.__setitem__(first, ((True, 1),) + labels[first][1:])),
            ("noncontiguous_index", lambda labels, first: labels.__setitem__(first, ((1, 1),) + labels[first][1:])),
            ("wrong_population", lambda labels, first: labels.__setitem__(first, labels[first][:-1])),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "evaluation_root_antigravity"; dependencies = _DependenciesAntigravity(root)
                first = dependencies.identity.dataset_order[0]; mutate(dependencies.labels, first)
                with self.assertRaises(ValueError): _evaluate_with_dependencies_antigravity("a" * 40, dependencies)
                audit = json.loads(next(Path(directory).glob("evaluation_audit_*_antigravity.json")).read_text(encoding="utf-8"))
                self.assertTrue(audit["label_session_closed"])
                self.assertFalse(root.exists())

    def test_descriptor_reader_rejects_namespace_replacement_and_symlink_races_antigravity(self) -> None:
        """The final root identity check binds descriptor reads to their namespace."""
        from experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity import evaluation_antigravity as evaluator
        from experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity import replay_antigravity as module
        for mode in ("replacement", "symlink"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "evaluation_root_antigravity"; dependencies = _DependenciesAntigravity(root)
                evaluator._evaluate_with_dependencies_antigravity("a" * 40, dependencies)
                original_stat = module.os.stat
                replacement_done = False

                def race(path: object, *args: object, **kwargs: object):
                    nonlocal replacement_done
                    if Path(path) == root and not replacement_done:
                        replacement_done = True
                        old = Path(directory) / "old_evaluation_root_antigravity"; root.rename(old)
                        if mode == "replacement":
                            root.mkdir()
                        else:
                            root.symlink_to(old, target_is_directory=True)
                    return original_stat(path, *args, **kwargs)

                with patch.object(module.os, "stat", side_effect=race), self.assertRaises(ValueError):
                    module._pinned_evaluation_leaves_antigravity(root)

    def test_transition_replay_rejects_reordered_typed_and_cross_dataset_tampers_antigravity(self) -> None:
        """Rehashed transition rows remain a canonical, per-dataset evidence sequence."""
        from experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity.evaluation_antigravity import _evaluate_with_dependencies_antigravity, _inspect_root_with_dependencies_antigravity

        def rehash(root: Path) -> None:
            artifact = {path.name: sha256(path.read_bytes()).hexdigest() for path in root.iterdir() if path.name != "artifact_sha256_antigravity.json"}
            (root / "artifact_sha256_antigravity.json").write_text(json.dumps(artifact, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")

        def rows(root: Path) -> list[dict[str, object]]:
            return [json.loads(line) for line in (root / "candidate_transitions_antigravity.jsonl").read_text(encoding="utf-8").splitlines()]

        def write(root: Path, value: list[dict[str, object]]) -> None:
            (root / "candidate_transitions_antigravity.jsonl").write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in value), encoding="utf-8")

        def reordered(root: Path) -> None:
            value = rows(root); value[0], value[1] = value[1], value[0]; write(root, value)

        def bool_index(root: Path) -> None:
            value = rows(root); value[0]["candidate_index"] = False; write(root, value)

        def wrong_index(root: Path) -> None:
            value = rows(root); value[0]["candidate_index"] = 1; write(root, value)

        def cross_dataset_outcomes(root: Path) -> None:
            value = rows(root)
            rescued = next(row for row in value if row["transition"] == "rescued")
            false_alarm = next(row for row in value if row["transition"] == "added_false_alarm" and row["dataset"] != rescued["dataset"])
            rescued["transition"], false_alarm["transition"] = false_alarm["transition"], rescued["transition"]
            write(root, value)

        for name, mutate in (("reordered", reordered), ("bool_index", bool_index), ("wrong_index", wrong_index), ("cross_dataset_outcomes", cross_dataset_outcomes)):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "evaluation_root_antigravity"; dependencies = _DependenciesAntigravity(root)
                _evaluate_with_dependencies_antigravity("a" * 40, dependencies); mutate(root); rehash(root)
                with self.assertRaises(ValueError): _inspect_root_with_dependencies_antigravity(dependencies)

    def test_audit_binds_full_frozen_exp446_identity_antigravity(self) -> None:
        """A post-label failure audit identifies the exact immutable predecessor."""
        from experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity.evaluation_antigravity import _evaluate_with_dependencies_antigravity
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "evaluation_root_antigravity"; dependencies = _DependenciesAntigravity(root, query_error=OSError("query identity"))
            with self.assertRaisesRegex(OSError, "query identity"):
                _evaluate_with_dependencies_antigravity("a" * 40, dependencies)
            audit = json.loads(next(Path(directory).glob("evaluation_audit_*_antigravity.json")).read_text(encoding="utf-8"))
            self.assertEqual(set(audit), {
                "schema", "phase", "query_attempts", "query_completions", "label_rows", "label_session_closed",
                "destination_state", "error_type", "error", "close_error_type", "close_error", "exp446_identity",
            })
            self.assertEqual(audit["exp446_identity"], {
                "root_device": dependencies.identity.root_device, "root_inode": dependencies.identity.root_inode,
                "manifest_sha256": dependencies.identity.manifest_sha256,
                "artifact_map_sha256": dependencies.identity.artifact_map_sha256,
                "prediction_sha256": dependencies.identity.prediction_sha256,
                "leaf_sha256": dict(dependencies.identity.leaf_sha256),
            })

    def test_evaluate_and_source_free_inspect_have_fixed_zero_diagnostics_antigravity(self) -> None:
        """All four diagnostic counts survive immutable replay, including zeros."""
        from experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity.evaluation_antigravity import _evaluate_with_dependencies_antigravity, _inspect_root_with_dependencies_antigravity
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "evaluation_root_antigravity"; dependencies = _DependenciesAntigravity(root)
            evaluated = _evaluate_with_dependencies_antigravity("a" * 40, dependencies)
            inspected = _inspect_root_with_dependencies_antigravity(dependencies)
            expected = {"rescued", "false_alarm", "regressed", "strong_regression"}
            self.assertEqual(set(evaluated.diagnostic_counts), expected)
            self.assertEqual(set(inspected.diagnostic_counts), expected)
            self.assertTrue(all(type(value) is int and value >= 0 for value in evaluated.diagnostic_counts.values()))
            self.assertEqual(evaluated, inspected)


if __name__ == "__main__":
    unittest.main()
