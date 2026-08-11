"""TDD coverage for the Exp447 single-use post-freeze evaluator."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import tempfile
import unittest
import json
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
    def __init__(self, root: Path, *, second_identity: PublicationIdentityAntigravity | None = None, close_error: Exception | None = None, query_error: Exception | None = None) -> None:
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
        return self.identity

    def assert_identity(self, identity: PublicationIdentityAntigravity) -> None:
        if identity != self.identity:
            raise ValueError("identity differs")

    def bootstrap(self, baseline: object, comparator: object, name: str, *, draws: object = None) -> dict[str, object]:
        self.events.append(f"bootstrap:{name}")
        return {"comparator_name": name, "micro_delta": 0.0, "macro_delta": 0.0, "micro_interval": [0.0, 0.0], "macro_interval": [0.0, 0.0]}


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


if __name__ == "__main__":
    unittest.main()
