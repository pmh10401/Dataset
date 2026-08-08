"""TDD coverage for the Exp441 post-freeze TEST-label read boundary."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from unittest.mock import patch
import ast
import builtins
import importlib
import importlib.abc
import importlib.util
import inspect
import math
import os
import sys
import tempfile
import types
import unittest

import numpy as np

from experiments.exp441_hard_empty_single_shot_recovery_antigravity.artifacts_antigravity import (
    canonical_json_bytes_antigravity,
)
from experiments.exp441_hard_empty_single_shot_recovery_antigravity.contracts_antigravity import (
    BootstrapResultAntigravity,
    CandidateEvidenceAntigravity,
    DatasetEvaluationFactsAntigravity,
    EvaluationFactsAntigravity,
    FrozenPublicationAntigravity,
    FrozenSourceSnapshotAntigravity,
    RankedCandidateAntigravity,
    RetrospectiveEvaluationAntigravity,
    SelectionResultAntigravity,
)
from experiments.exp441_hard_empty_single_shot_recovery_antigravity.selector_antigravity import (
    select_hard_empty_single_shot_antigravity,
)

evaluation: object


_LABEL_SQL = """\
SELECT i.instance_index, i.label
FROM instances AS i
JOIN datasets AS d ON d.id = i.dataset_id
WHERE d.name = ? AND i.split = 'TEST'
ORDER BY i.instance_index
"""


def _candidate(dataset: str, index: int, ratio: float) -> CandidateEvidenceAntigravity:
    return CandidateEvidenceAntigravity(dataset, index, 32, (1, 2), 2, ratio, 0.5)


def _snapshot_and_selection() -> tuple[FrozenSourceSnapshotAntigravity, object]:
    """A real 1,117-context production-shape contract, not a helper under test."""
    datasets = tuple(f"D{index:04d}" for index in range(1117))
    hard_datasets = datasets[:321]
    hard = frozenset((dataset, 7 + (index % 2)) for index, dataset in enumerate(
        hard_datasets + hard_datasets[:125]
    ))
    candidates = []
    for dataset, index in sorted(hard):
        candidates.append(_candidate(dataset, index, 1.0))
    for index, dataset in enumerate(hard_datasets):
        candidates.append(_candidate(dataset, 20 + index, 1.5))
    eligible_datasets = datasets[321:370]
    for index in range(73):
        candidates.append(_candidate(eligible_datasets[index % len(eligible_datasets)], 100 + index, 2.0 + index))
    rows = {dataset: 1_000 for dataset in datasets}
    digests = {dataset: f"{index + 1:064x}" for index, dataset in enumerate(datasets)}
    snapshot = FrozenSourceSnapshotAntigravity(
        datasets, rows, digests, hard, tuple(candidates),
        "a" * 64, "b" * 64, "c" * 64, "d" * 64, "e" * 64, "f" * 64,
        "0" * 64, "1" * 64, "2" * 64,
    )
    return snapshot, select_hard_empty_single_shot_antigravity(snapshot)


def _publication(snapshot: FrozenSourceSnapshotAntigravity, selection: object) -> FrozenPublicationAntigravity:
    audit = sha256(canonical_json_bytes_antigravity(evaluation._snapshot_payload_antigravity(snapshot))).hexdigest()
    return FrozenPublicationAntigravity("3" * 64, "4" * 64, "5" * 64, audit, selection)  # type: ignore[arg-type]


class _Cursor:
    def __init__(self, rows: list[tuple[object, object]]) -> None:
        self.rows = rows

    def fetchall(self) -> list[tuple[object, object]]:
        return list(self.rows)


class _Connection:
    def __init__(self, rows: dict[str, list[tuple[object, object]]]) -> None:
        self.rows = rows
        self.calls: list[tuple[str, object]] = []
        self.quick_check_rows: list[tuple[object, object]] = [("ok",)]

    def execute(self, sql: str, params: object = ()) -> _Cursor:
        self.calls.append((sql, params))
        if sql == "PRAGMA quick_check":
            return _Cursor(self.quick_check_rows)
        return _Cursor(self.rows[params[0]])  # type: ignore[index]


class _InputsResolutionSentinel(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """A real import hook whose loaded helper explodes if preflight leaks through."""

    module_name = "experiments.exp417_causal_multihorizon_shadow.inputs"

    def __init__(self) -> None:
        self.resolutions = 0
        self.connection_entries = 0

    def find_spec(self, fullname: str, path: object = None, target: object = None) -> object:
        if fullname == self.module_name:
            self.resolutions += 1
            return importlib.util.spec_from_loader(fullname, self)
        return None

    def create_module(self, spec: object) -> object:
        return None

    def exec_module(self, module: object) -> None:
        @contextmanager
        def explode(*args: object, **kwargs: object):
            self.connection_entries += 1
            raise AssertionError("Exp417 connection opened before preflight completed")
            yield None
        module.verified_read_only_connection = explode  # type: ignore[attr-defined]


class Exp441FrozenContextLabelsAntigravityTests(unittest.TestCase):
    def test_task11_public_evaluation_has_no_arguments(self) -> None:
        """Task 11 is a fixed-root route: callers cannot supply TEST-result controls."""
        self.assertEqual(tuple(inspect.signature(evaluation.evaluate_exp441_antigravity).parameters), ())

    def setUp(self) -> None:
        global evaluation
        evaluation = importlib.import_module(
            "experiments.exp441_hard_empty_single_shot_recovery_antigravity.evaluation_antigravity"
        )
        self.snapshot, self.selection = _snapshot_and_selection()
        self.publication = _publication(self.snapshot, self.selection)
        self.rows = {dataset: [(3, "0"), (7, 0), (8, "1"), (9, 1)] + [(index, 0) for index in range(10, 1006)]
                     for dataset in self.snapshot.context_datasets}
        self.connection = _Connection(self.rows)

    def _run(self, *, publication_sequence: list[object] | None = None,
             source_sequence: list[object] | None = None,
             selector_sequence: list[object] | None = None,
             audit_values: list[str] | None = None) -> object:
        publications = iter(publication_sequence or [self.publication] * 3)
        sources = iter(source_sequence or [self.snapshot] * 2)
        selections = iter(selector_sequence or [self.selection] * 2)
        audits = iter(audit_values or [self.publication.source_audit_sha256] * 3)
        self.audit_calls = 0
        def fake_audit(snapshot: object) -> str:
            self.audit_calls += 1
            return next(audits)
        self.connection_entries = 0
        @contextmanager
        def fake_connection(path: object, digest: object):
            self.connection_entries += 1
            self.assertEqual(str(path), evaluation.FROZEN_CONFIG.database_path)
            self.assertEqual(digest, evaluation.FROZEN_CONFIG.database_sha256)
            yield self.connection
        inputs = types.ModuleType("experiments.exp417_causal_multihorizon_shadow.inputs")
        inputs.verified_read_only_connection = fake_connection
        with patch.object(evaluation, "inspect_exp441_publication_antigravity", side_effect=lambda: next(publications)), \
             patch.object(evaluation, "verify_frozen_sources_antigravity", side_effect=lambda: next(sources)), \
             patch.object(evaluation, "select_hard_empty_single_shot_antigravity", side_effect=lambda snapshot: next(selections)), \
             patch.object(evaluation, "_source_audit_sha256_antigravity", side_effect=fake_audit), \
             patch.dict(sys.modules, {inputs.__name__: inputs}):
            return evaluation.load_frozen_context_labels_antigravity()

    def test_loads_exact_context_queries_and_preserves_noncontiguous_identities(self) -> None:
        result = self._run()
        self.assertEqual(result.context_datasets, self.snapshot.context_datasets)
        self.assertEqual(result.labels_by_dataset["D0000"][:4], ((3, 0), (7, 0), (8, 1), (9, 1)))
        self.assertEqual((result.descriptor_bound_connection_count, result.label_query_calls, result.out_of_scope_label_rows), (1, 1117, 0))
        self.assertEqual(self.connection_entries, 1)
        self.assertEqual(self.connection.calls[0], ("PRAGMA quick_check", ()))
        self.assertEqual(self.connection.calls[1:], [(_LABEL_SQL, (dataset,)) for dataset in self.snapshot.context_datasets])
        self.assertNotIn(("CornellWhaleChallenge",), [params for _, params in self.connection.calls])
        self.assertNotIn(("Wafer_normal_1",), [params for _, params in self.connection.calls])

    def _assert_preflight_rejected(self, message: str, publications: list[object], sources: list[object],
                                   selections: list[object] | None = None, *, validator_error: bool = False) -> None:
        sentinel = _InputsResolutionSentinel()
        module_name = sentinel.module_name
        prior = sys.modules.pop(module_name, None)
        sys.meta_path.insert(0, sentinel)
        publication_iter, source_iter = iter(publications), iter(sources)
        selection_iter = iter(selections or [self.selection])
        try:
            validator = patch.object(evaluation, "validate_production_selection_antigravity", side_effect=ValueError("production count")) if validator_error else nullcontext()
            with patch.object(evaluation, "inspect_exp441_publication_antigravity", side_effect=lambda: next(publication_iter)), \
                 patch.object(evaluation, "verify_frozen_sources_antigravity", side_effect=lambda: next(source_iter)), \
                 patch.object(evaluation, "select_hard_empty_single_shot_antigravity", side_effect=lambda snapshot: next(selection_iter)), \
                 validator:
                with self.assertRaisesRegex(ValueError, message):
                    evaluation.load_frozen_context_labels_antigravity()
            self.assertEqual(sentinel.resolutions, 0)
            self.assertEqual(sentinel.connection_entries, 0)
            self.assertEqual(self.connection.calls, [])
        finally:
            sys.meta_path.remove(sentinel)
            sys.modules.pop(module_name, None)
            if prior is not None:
                sys.modules[module_name] = prior

    def test_every_preflight_family_blocks_exp417_resolution_and_connection(self) -> None:
        self._assert_preflight_rejected("publication", [ValueError("publication")], [self.snapshot])
        self._assert_preflight_rejected("sources", [self.publication], [ValueError("sources")])
        bad_audit = FrozenPublicationAntigravity("3" * 64, "4" * 64, "5" * 64, "6" * 64, self.selection)
        self._assert_preflight_rejected("source-audit", [bad_audit], [self.snapshot])
        self._assert_preflight_rejected("production count", [self.publication], [self.snapshot], validator_error=True)
        _, mismatch_selection = _snapshot_and_selection()
        object.__setattr__(mismatch_selection, "selected", tuple(reversed(mismatch_selection.selected)))
        mismatch_publication = _publication(self.snapshot, mismatch_selection)
        self._assert_preflight_rejected("publication selection", [mismatch_publication], [self.snapshot], [self.selection])
        snapshot, selection = _snapshot_and_selection()
        publication = _publication(snapshot, selection)
        object.__setattr__(snapshot, "test_row_count_by_dataset", dict(snapshot.test_row_count_by_dataset))
        self._assert_preflight_rejected("row-count", [publication], [snapshot], [selection])
        snapshot, selection = _snapshot_and_selection()
        publication = _publication(snapshot, selection)
        object.__setattr__(snapshot, "hard_coordinates", frozenset({("D0000", 1000)}))
        self._assert_preflight_rejected("coordinate", [publication], [snapshot], [selection])
        # Break caught: removing the preflight `index < 0` guard admits an invalid Hard coordinate.
        snapshot, selection = _snapshot_and_selection()
        publication = _publication(snapshot, selection)
        object.__setattr__(snapshot, "hard_coordinates", frozenset({("D0000", -1)}))
        self._assert_preflight_rejected("coordinate", [publication], [snapshot], [selection])
        snapshot, selection = _snapshot_and_selection()
        publication = _publication(snapshot, selection)
        object.__setattr__(selection.selected[0].candidate, "instance_index", 1000)
        self._assert_preflight_rejected("coordinate", [publication], [snapshot], [selection])
        drift = FrozenPublicationAntigravity("7" * 64, "4" * 64, "5" * 64, self.publication.source_audit_sha256, self.selection)
        self._assert_preflight_rejected("publication changed", [self.publication, drift], [self.snapshot], [self.selection])

    def test_module_import_never_resolves_exp417(self) -> None:
        module_name = "experiments.exp441_hard_empty_single_shot_recovery_antigravity.evaluation_antigravity"
        inputs_name = _InputsResolutionSentinel.module_name
        sentinel = _InputsResolutionSentinel()
        prior_evaluation, prior_inputs = sys.modules.pop(module_name, None), sys.modules.pop(inputs_name, None)
        sys.meta_path.insert(0, sentinel)
        try:
            importlib.import_module(module_name)
            self.assertEqual(sentinel.resolutions, 0)
        finally:
            sys.meta_path.remove(sentinel)
            sys.modules.pop(module_name, None)
            if prior_evaluation is not None:
                sys.modules[module_name] = prior_evaluation
            if prior_inputs is not None:
                sys.modules[inputs_name] = prior_inputs

    def test_post_connection_drift_fails_closed(self) -> None:
        changed = FrozenPublicationAntigravity("7" * 64, "4" * 64, "5" * 64, self.publication.source_audit_sha256, self.selection)
        with self.assertRaises(ValueError):
            self._run(publication_sequence=[self.publication, self.publication, changed])
        self.assertEqual(len(self.connection.calls), 1118)

    def test_post_close_source_selector_and_source_audit_drift_fail_closed(self) -> None:
        changed_source, _ = _snapshot_and_selection()
        object.__setattr__(changed_source, "state_sha256", "9" * 64)
        with self.subTest("source"), self.assertRaisesRegex(ValueError, "authority"):
            self._run(source_sequence=[self.snapshot, changed_source])
        changed_selector_source, changed_selector = _snapshot_and_selection()
        object.__setattr__(changed_selector, "selected", tuple(reversed(changed_selector.selected)))
        with self.subTest("selector"), self.assertRaisesRegex(ValueError, "source binding"):
            self._run(source_sequence=[self.snapshot, changed_selector_source], selector_sequence=[self.selection, changed_selector])
        # The third audit invocation is line 183 after post-close source/selector replay.
        normal = self.publication.source_audit_sha256
        with self.subTest("source audit"), self.assertRaisesRegex(ValueError, "post-connection source binding changed"):
            self._run(audit_values=[normal, normal, "8" * 64])
        self.assertEqual(self.audit_calls, 3)

    def test_rejects_quick_check_schema_order_counts_and_labels(self) -> None:
        self.connection.quick_check_rows = [("not ok",)]
        with self.assertRaisesRegex(ValueError, "quick_check"):
            self._run()
        self.connection.quick_check_rows = [("ok",)]
        cases = (
            ([(3, "0")], "count"), ([(3, "0", 0)] + [(index, 0) for index in range(10, 1009)], "schema"),
            ([(9, 1), (8, 0)] + [(index, 0) for index in range(10, 1008)], "index"),
            ([(3, "0"), (3, 1)] + [(index, 0) for index in range(10, 1008)], "index"),
            ([(True, 0)] + [(index, 0) for index in range(10, 1008)], "index"),
            # Break caught: removing the loaded-row `index < 0` guard accepts an invalid TEST identity.
            ([(-1, 0)] + [(index, 0) for index in range(10, 1009)], "index"),
            ([(3, True), (9, 1)] + [(index, 0) for index in range(10, 1008)], "label"),
            ([(3, "2"), (9, 1)] + [(index, 0) for index in range(10, 1008)], "label"),
        )
        for rows, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.rows["D0000"] = rows
                self._run()
            self.rows["D0000"] = [(3, "0"), (7, 0), (8, "1"), (9, 1)] + [(index, 0) for index in range(10, 1006)]

    def test_requires_hard_and_selected_actual_identity(self) -> None:
        self.rows["D0000"] = [(3, "0"), (8, 0), (9, 1)] + [(index, 0) for index in range(10, 1006)] + [(1008, 0)]
        with self.assertRaisesRegex(ValueError, "coordinate"):
            self._run()

    def test_result_detaches_mutable_mapping_and_rows(self) -> None:
        source = {"D": [(3, 0)]}
        result = evaluation.FrozenContextLabelsAntigravity(("D",), source, "a" * 64, "b" * 64, self.selection, 1, 1, 0)
        source["D"][0] = (9, 1)
        self.assertEqual(result.labels_by_dataset["D"], ((3, 0),))
        self.assertIsInstance(result.labels_by_dataset, MappingProxyType)
        with self.assertRaises(TypeError):
            result.labels_by_dataset["D"] = ((9, 1),)  # type: ignore[index]

    def test_module_has_no_top_level_label_capable_imports(self) -> None:
        source = evaluation.__loader__.get_source(evaluation.__name__) or ""
        tree = ast.parse(source)
        imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        imported = "\n".join(ast.unparse(node) for node in imports)
        self.assertNotIn("exp417", imported)
        self.assertNotIn("sqlite3", imported)
        self.assertNotIn("evaluation", imported)
        self.assertIn("write_create_only_antigravity", imported)


class Exp441RetrospectiveEvaluationAntigravityTests(unittest.TestCase):
    def setUp(self) -> None:
        global evaluation
        evaluation = importlib.import_module(
            "experiments.exp441_hard_empty_single_shot_recovery_antigravity.evaluation_antigravity"
        )

    def _complete_result(self, datasets: tuple[str, ...] | None = None) -> RetrospectiveEvaluationAntigravity:
        """Small-value, full-population synthetic evidence for Task 11 byte replay only."""
        names = datasets or tuple(f"T{index:04d}" for index in range(1117))
        facts = tuple(DatasetEvaluationFactsAntigravity(
            name, (0, 0, 0, 1), (0, 0, 0, 1), 0.0, 0.0, 0.0, 0, 0,
        ) for name in names)
        rows = tuple((0, 0, 0, 1, 0, 0, 0, 1, 0.0) for _ in facts)
        aggregate = EvaluationFactsAntigravity(
            (0, 0, 0, 1117), (0, 0, 0, 1117), 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0,
        )
        return RetrospectiveEvaluationAntigravity(
            facts, (), (), rows, aggregate, BootstrapResultAntigravity((0.0, 0.0), (0.0, 0.0)),
            "engineering_reproduction_failed", False,
        )

    def test_task11_pure_replay_validates_closed_hash_graph_and_calls_bootstrap_once(self) -> None:
        """A byte change in any member must fail before a forged frozen result can return."""
        snapshot, selection = _snapshot_and_selection()
        publication = _publication(snapshot, selection)
        payloads = evaluation._result_artifacts_antigravity(self._complete_result(snapshot.context_datasets), publication)
        bootstrap_calls: list[tuple[int, int]] = []
        real_bootstrap = evaluation.paired_dataset_bootstrap_antigravity
        def counted(rows: object) -> BootstrapResultAntigravity:
            bootstrap_calls.append(np.asarray(rows).shape)
            return real_bootstrap(rows)
        with patch.object(evaluation, "paired_dataset_bootstrap_antigravity", side_effect=counted):
            replayed = evaluation._replay_evaluation_payloads_antigravity(payloads)
        self.assertEqual(bootstrap_calls, [(1117, 9)])
        self.assertEqual(replayed.evaluation_sha256, sha256(payloads["artifact_sha256_antigravity.json"]).hexdigest())
        self.assertEqual(replayed.publication_sha256, publication.publication_sha256)
        for name in payloads:
            tampered = dict(payloads)
            tampered[name] = payloads[name] + b" "
            with self.subTest(name=name), self.assertRaises(ValueError):
                evaluation._replay_evaluation_payloads_antigravity(tampered)

    def test_task11_governance_is_exact_for_all_three_publication_documents(self) -> None:
        """Missing, truthy and nested governance cannot be smuggled into a replay document."""
        snapshot, selection = _snapshot_and_selection()
        payloads = evaluation._result_artifacts_antigravity(self._complete_result(), _publication(snapshot, selection))
        families = (
            ("evaluation_antigravity.json", evaluation._EVALUATION_NON_GOVERNANCE_KEYS_ANTIGRAVITY, "evaluation"),
            ("bootstrap_antigravity.json", evaluation._BOOTSTRAP_NON_GOVERNANCE_KEYS_ANTIGRAVITY, "bootstrap"),
            ("decision_antigravity.json", evaluation._DECISION_NON_GOVERNANCE_KEYS_ANTIGRAVITY, "decision"),
        )
        for name, keys, label in families:
            parsed = evaluation.parse_canonical_json_bytes_antigravity(payloads[name])
            for mutation in ("missing", "truthy", "nested"):
                altered = dict(parsed)
                if mutation == "missing":
                    del altered["review_only"]
                elif mutation == "truthy":
                    altered["review_only"] = 1
                else:
                    altered["governance"] = {"review_only": True}
                with self.subTest(name=name, mutation=mutation), self.assertRaises(ValueError):
                    evaluation._strict_payload_antigravity(altered, keys, label, governance=True)

    def test_task11_replay_rejects_hash_valid_confusion_and_builtin_type_forgery(self) -> None:
        """Hash binding alone cannot hide a confusion mismatch or a JSON bool/float coercion."""
        snapshot, selection = _snapshot_and_selection()
        payloads = dict(evaluation._result_artifacts_antigravity(self._complete_result(snapshot.context_datasets), _publication(snapshot, selection)))
        confusion = evaluation.parse_canonical_json_bytes_antigravity(payloads["confusion_antigravity.json"])
        confusion["candidate_tp"] = 1
        payloads["confusion_antigravity.json"] = evaluation.canonical_json_bytes_antigravity(confusion)
        evaluation_payload = evaluation.parse_canonical_json_bytes_antigravity(payloads["evaluation_antigravity.json"])
        evaluation_payload["confusion_sha256"] = sha256(payloads["confusion_antigravity.json"]).hexdigest()
        payloads["evaluation_antigravity.json"] = evaluation.canonical_json_bytes_antigravity(evaluation_payload)
        decision = evaluation.parse_canonical_json_bytes_antigravity(payloads["decision_antigravity.json"])
        decision["confusion_sha256"] = sha256(payloads["confusion_antigravity.json"]).hexdigest()
        decision["evaluation_sha256"] = sha256(payloads["evaluation_antigravity.json"]).hexdigest()
        payloads["decision_antigravity.json"] = evaluation.canonical_json_bytes_antigravity(decision)
        artifact = evaluation.parse_canonical_json_bytes_antigravity(payloads["artifact_sha256_antigravity.json"])
        artifact["files"] = {name: sha256(payloads[name]).hexdigest() for name in artifact["files"]}
        payloads["artifact_sha256_antigravity.json"] = evaluation.canonical_json_bytes_antigravity(artifact)
        with self.assertRaisesRegex(ValueError, "aggregate"):
            evaluation._replay_evaluation_payloads_antigravity(payloads)
        with self.assertRaises(ValueError):
            evaluation._strict_int_antigravity(1117.0, 1117, "dataset count")
        with self.assertRaises(ValueError):
            evaluation._finite_float_antigravity(0, "float")

    def test_task11_failure_validator_rejects_wrong_flags_and_existing_invalid_destination_audits(self) -> None:
        """Existing invalid evidence is never repaired, and its audit is itself exact-schema."""
        failure = evaluation.canonical_json_bytes_antigravity({
            "schema": "exp441.evaluation-failure.v1", "status": "engineering_reproduction_failed",
            "phase": "synthetic", "error_type": "ValueError", "error": "synthetic",
            "labels_opened": False, "destination_committed": True, **evaluation._GOVERNANCE_ANTIGRAVITY,
        })
        self.assertEqual(evaluation._validate_failure_payload_antigravity(failure)["phase"], "synthetic")
        broken = evaluation.parse_canonical_json_bytes_antigravity(failure); broken["labels_opened"] = 0
        with self.assertRaises(ValueError):
            evaluation._validate_failure_payload_antigravity(evaluation.canonical_json_bytes_antigravity(broken))
        with tempfile.TemporaryDirectory() as temporary:
            parent, destination = Path(temporary), Path(temporary) / "evaluation_antigravity"
            os.mkdir(destination, 0o700)
            with patch.object(evaluation, "_EVALUATION_DESTINATION_ANTIGRAVITY", destination):
                with self.assertRaises(ValueError):
                    evaluation.evaluate_exp441_antigravity()
            audit = list(parent.glob("failed_evaluation_audit_*_antigravity"))
            self.assertEqual(len(audit), 1)
            evidence = (audit[0] / "failure_antigravity.json").read_bytes()
            self.assertIs(evaluation._validate_failure_payload_antigravity(evidence)["destination_committed"], True)

    def test_task11_every_governance_leaf_rejects_every_document_mutation_family(self) -> None:
        """All 9 flat governance leaves are exact booleans/strings, never aliases or nested controls."""
        snapshot, selection = _snapshot_and_selection()
        payloads = evaluation._result_artifacts_antigravity(self._complete_result(snapshot.context_datasets), _publication(snapshot, selection))
        documents = (
            ("evaluation_antigravity.json", evaluation._EVALUATION_NON_GOVERNANCE_KEYS_ANTIGRAVITY, "evaluation"),
            ("bootstrap_antigravity.json", evaluation._BOOTSTRAP_NON_GOVERNANCE_KEYS_ANTIGRAVITY, "bootstrap"),
            ("decision_antigravity.json", evaluation._DECISION_NON_GOVERNANCE_KEYS_ANTIGRAVITY, "decision"),
        )
        for filename, keys, label in documents:
            original = evaluation.parse_canonical_json_bytes_antigravity(payloads[filename])
            for key, expected in evaluation._GOVERNANCE_ANTIGRAVITY.items():
                mutations = {
                    "missing": lambda value: value.pop(key),
                    "nested": lambda value: value.__setitem__(key, {"value": expected}),
                    "truthy_type": lambda value: value.__setitem__(key, 1 if type(expected) is bool else True),
                    "wrong_value": lambda value: value.__setitem__(key, (not expected) if type(expected) is bool else expected + "_wrong"),
                    "extra_alias": lambda value: value.__setitem__(key + "_alias", expected),
                }
                for mutation, apply in mutations.items():
                    altered = dict(original)
                    apply(altered)
                    with self.subTest(document=label, key=key, mutation=mutation), self.assertRaises(ValueError):
                        evaluation._strict_payload_antigravity(altered, keys, label, governance=True)
        failure = evaluation.canonical_json_bytes_antigravity({
            "schema": "exp441.evaluation-failure.v1", "status": "engineering_reproduction_failed",
            "phase": "synthetic", "error_type": "ValueError", "error": "synthetic",
            "labels_opened": False, "destination_committed": False, **evaluation._GOVERNANCE_ANTIGRAVITY,
        })
        original_failure = evaluation.parse_canonical_json_bytes_antigravity(failure)
        for key, expected in evaluation._GOVERNANCE_ANTIGRAVITY.items():
            for mutation in ("missing", "nested", "truthy_type", "wrong_value", "extra_alias"):
                altered = dict(original_failure)
                if mutation == "missing": del altered[key]
                elif mutation == "nested": altered[key] = {"value": expected}
                elif mutation == "truthy_type": altered[key] = 1 if type(expected) is bool else True
                elif mutation == "wrong_value": altered[key] = (not expected) if type(expected) is bool else expected + "_wrong"
                else: altered[key + "_alias"] = expected
                with self.subTest(document="failure", key=key, mutation=mutation), self.assertRaises(ValueError):
                    evaluation._validate_failure_payload_antigravity(evaluation.canonical_json_bytes_antigravity(altered))

    def test_task11_exact_inventory_rejects_missing_extra_link_fifo_and_mode_members(self) -> None:
        """Pinned replay admits only eight 0600 regular files in one 0700 directory."""
        snapshot, selection = _snapshot_and_selection()
        payloads = evaluation._result_artifacts_antigravity(self._complete_result(snapshot.context_datasets), _publication(snapshot, selection))
        for mutation in ("missing", "extra", "symlink", "fifo", "mode"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "evaluation_antigravity"; os.mkdir(root, 0o700)
                for name, payload in payloads.items(): evaluation.write_create_only_antigravity(root / name, payload)
                target = root / "evaluation_antigravity.json"
                if mutation == "missing": target.unlink()
                elif mutation == "extra": evaluation.write_create_only_antigravity(root / "extra_antigravity.json", b"x")
                elif mutation == "symlink":
                    target.unlink(); target.symlink_to(root / "bootstrap_antigravity.json")
                elif mutation == "fifo":
                    target.unlink(); os.mkfifo(target, 0o600)
                else: os.chmod(target, 0o644)
                with self.assertRaises(ValueError):
                    with evaluation.pin_exact_regular_inventory_antigravity(root, evaluation._EVALUATION_FILENAMES_ANTIGRAVITY):
                        pass

    def test_task11_failure_preservation_keeps_primary_error_and_adds_secondary_note(self) -> None:
        """A failure to write failure evidence must never replace the original engineering error."""
        with tempfile.TemporaryDirectory() as temporary:
            primary = ValueError("primary synthetic failure")
            with patch.object(evaluation, "write_create_only_antigravity", side_effect=OSError("secondary write failure")):
                evaluation._write_failure_antigravity(Path(temporary), None, primary, "synthetic", False, True)
            self.assertEqual(str(primary), "primary synthetic failure")
            self.assertTrue(any("failure preservation failed: OSError: secondary write failure" in note
                                for note in getattr(primary, "__notes__", ())))

    def test_task11_create_high_level_phases_follow_the_preregistered_order(self) -> None:
        """The route cannot label/compute/write/rename before its prescribed predecessor gate."""
        snapshot, _selection = _snapshot_and_selection(); zero = SelectionResultAntigravity((), (), (), (), (), frozenset())
        publication = FrozenPublicationAntigravity("3" * 64, "4" * 64, "5" * 64, "6" * 64, zero)
        labels = evaluation.FrozenContextLabelsAntigravity(snapshot.context_datasets, {d: () for d in snapshot.context_datasets},
            publication.publication_sha256, publication.source_audit_sha256, zero, 1, 1117, 0)
        preflight = (publication, snapshot, zero, snapshot.context_datasets, MappingProxyType(dict(snapshot.test_row_count_by_dataset)))
        events: list[str] = []; real_write=evaluation.write_create_only_antigravity; real_pin=evaluation.pin_exact_regular_inventory_antigravity
        real_replay=evaluation._replay_evaluation_payloads_antigravity; real_rename=evaluation.exclusive_rename_antigravity
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "evaluation_antigravity"
            def preflight_call() -> object: events.append("preflight"); return preflight
            def label_call() -> object: events.append("labels"); return labels
            def compute_call(value: object) -> object: events.append("compute"); return self._complete_result(snapshot.context_datasets)
            def write_call(path: Path, payload: bytes) -> None: events.append("write"); real_write(path, payload)
            def pin_call(*args: object, **kwargs: object) -> object: events.append("pin"); return real_pin(*args, **kwargs)
            def replay_call(payloads: object) -> object: events.append("replay"); return real_replay(payloads)
            def rename_call(source: Path, target: Path) -> None: events.append("rename"); real_rename(source, target)
            with patch.object(evaluation, "_EVALUATION_DESTINATION_ANTIGRAVITY", destination), \
                 patch.object(evaluation, "_preflight_antigravity", side_effect=preflight_call), \
                 patch.object(evaluation, "load_frozen_context_labels_antigravity", side_effect=label_call), \
                 patch.object(evaluation, "compute_retrospective_evaluation_antigravity", side_effect=compute_call), \
                 patch.object(evaluation, "write_create_only_antigravity", side_effect=write_call), \
                 patch.object(evaluation, "pin_exact_regular_inventory_antigravity", side_effect=pin_call), \
                 patch.object(evaluation, "_replay_evaluation_payloads_antigravity", side_effect=replay_call), \
                 patch.object(evaluation, "exclusive_rename_antigravity", side_effect=rename_call):
                evaluation.evaluate_exp441_antigravity()
        compressed = [event for index, event in enumerate(events) if index == 0 or event != events[index - 1]]
        self.assertEqual(compressed, ["preflight", "labels", "compute", "preflight", "write", "pin", "replay", "preflight", "rename", "pin", "replay", "preflight"])

    def test_task11_existing_destination_replays_without_task9_or_exp417_access(self) -> None:
        """An existing valid eight-file destination is inspected, never repaired or relabelled."""
        snapshot, _selection = _snapshot_and_selection()
        zero_selection = SelectionResultAntigravity((), (), (), (), (), frozenset())
        publication = FrozenPublicationAntigravity("3" * 64, "4" * 64, "5" * 64, "6" * 64, zero_selection)
        payloads = evaluation._result_artifacts_antigravity(self._complete_result(snapshot.context_datasets), publication)
        with tempfile.TemporaryDirectory() as temporary:
            parent, destination = Path(temporary), Path(temporary) / "evaluation_antigravity"
            os.mkdir(destination, 0o700)
            for name, payload in payloads.items():
                evaluation.write_create_only_antigravity(destination / name, payload)
            before = {name: (destination / name).read_bytes() for name in payloads}
            preflight = (publication, snapshot, zero_selection, snapshot.context_datasets,
                         MappingProxyType(dict(snapshot.test_row_count_by_dataset)))
            def explode(*args: object, **kwargs: object) -> object:
                raise AssertionError("existing evaluation must not open Task9/Exp417")
            with patch.object(evaluation, "_EVALUATION_DESTINATION_ANTIGRAVITY", destination), \
                 patch.object(evaluation, "_preflight_antigravity", return_value=preflight), \
                 patch.object(evaluation, "load_frozen_context_labels_antigravity", side_effect=explode):
                replayed = evaluation.evaluate_exp441_antigravity()
            self.assertEqual(replayed.evaluation_sha256, sha256(payloads["artifact_sha256_antigravity.json"]).hexdigest())
            self.assertEqual({name: (destination / name).read_bytes() for name in payloads}, before)

    def test_task11_existing_pin_rejects_member_replacement_during_preflight(self) -> None:
        """The existing-destination return retains its first descriptor through later authority checks."""
        snapshot, _selection = _snapshot_and_selection()
        zero = SelectionResultAntigravity((), (), (), (), (), frozenset())
        publication = FrozenPublicationAntigravity("3" * 64, "4" * 64, "5" * 64, "6" * 64, zero)
        payloads = evaluation._result_artifacts_antigravity(self._complete_result(snapshot.context_datasets), publication)
        preflight = (publication, snapshot, zero, snapshot.context_datasets, MappingProxyType(dict(snapshot.test_row_count_by_dataset)))
        with tempfile.TemporaryDirectory() as temporary:
            parent, destination = Path(temporary), Path(temporary) / "evaluation_antigravity"
            os.mkdir(destination, 0o700)
            for name, payload in payloads.items(): evaluation.write_create_only_antigravity(destination / name, payload)
            target = destination / "evaluation_antigravity.json"
            def mutate() -> object:
                replacement = destination / "replacement_antigravity.json"
                evaluation.write_create_only_antigravity(replacement, target.read_bytes())
                os.replace(replacement, target)
                return preflight
            with patch.object(evaluation, "_EVALUATION_DESTINATION_ANTIGRAVITY", destination), \
                 patch.object(evaluation, "_preflight_antigravity", side_effect=mutate):
                with self.assertRaisesRegex(ValueError, "pinned inventory"):
                    evaluation.evaluate_exp441_antigravity()
            self.assertTrue(target.exists())
            self.assertEqual(len(list(parent.glob("failed_evaluation_audit_*_antigravity"))), 1)

    def test_task11_pre_rename_failure_preserves_staging_and_never_creates_destination(self) -> None:
        """A pre-label engineering fault preserves only a failed private staging sibling."""
        with tempfile.TemporaryDirectory() as temporary:
            parent, destination = Path(temporary), Path(temporary) / "evaluation_antigravity"
            with patch.object(evaluation, "_EVALUATION_DESTINATION_ANTIGRAVITY", destination), \
                 patch.object(evaluation, "_preflight_antigravity", side_effect=ValueError("synthetic preflight")):
                with self.assertRaisesRegex(ValueError, "synthetic preflight"):
                    evaluation.evaluate_exp441_antigravity()
            self.assertFalse(destination.exists())
            failed = list(parent.glob("failed_evaluation_staging_*_antigravity"))
            self.assertEqual(len(failed), 1)
            failure = evaluation.parse_canonical_json_bytes_antigravity((failed[0] / "failure_antigravity.json").read_bytes())
            self.assertEqual((failure["phase"], failure["labels_opened"], failure["destination_committed"]), ("preflight", False, False))
            self.assertEqual(set(failure), {
                "schema", "status", "phase", "error_type", "error", "labels_opened", "destination_committed",
                *evaluation._GOVERNANCE_KEYS_ANTIGRAVITY,
            })
            self.assertEqual({key: failure[key] for key in evaluation._GOVERNANCE_KEYS_ANTIGRAVITY}, dict(evaluation._GOVERNANCE_ANTIGRAVITY))

    def test_task11_destination_race_is_never_adopted_as_a_concurrent_winner(self) -> None:
        """A directory that appears before RENAME_EXCL is an engineering failure, never a replay input."""
        snapshot, _selection = _snapshot_and_selection()
        zero_selection = SelectionResultAntigravity((), (), (), (), (), frozenset())
        publication = FrozenPublicationAntigravity("3" * 64, "4" * 64, "5" * 64, "6" * 64, zero_selection)
        labels = evaluation.FrozenContextLabelsAntigravity(snapshot.context_datasets,
            {dataset: () for dataset in snapshot.context_datasets}, publication.publication_sha256,
            publication.source_audit_sha256, zero_selection, 1, 1117, 0)
        preflight = (publication, snapshot, zero_selection, snapshot.context_datasets,
                     MappingProxyType(dict(snapshot.test_row_count_by_dataset)))
        with tempfile.TemporaryDirectory() as temporary:
            parent, destination = Path(temporary), Path(temporary) / "evaluation_antigravity"
            calls = 0
            def racing_preflight() -> object:
                nonlocal calls
                calls += 1
                if calls == 3:
                    os.mkdir(destination, 0o700)
                return preflight
            with patch.object(evaluation, "_EVALUATION_DESTINATION_ANTIGRAVITY", destination), \
                 patch.object(evaluation, "_preflight_antigravity", side_effect=racing_preflight), \
                 patch.object(evaluation, "load_frozen_context_labels_antigravity", return_value=labels), \
                 patch.object(evaluation, "compute_retrospective_evaluation_antigravity", return_value=self._complete_result(snapshot.context_datasets)):
                with self.assertRaisesRegex(FileExistsError, "appeared"):
                    evaluation.evaluate_exp441_antigravity()
            self.assertEqual(list(destination.iterdir()), [])
            self.assertEqual(len(list(parent.glob("failed_evaluation_staging_*_antigravity"))), 1)

    def test_task11_rejects_reordered_or_alternate_label_authority_before_compute(self) -> None:
        """Task 10 cannot receive labels for a reordered context or an alternate same-sized selection."""
        snapshot, selection = _snapshot_and_selection()
        publication = _publication(snapshot, selection)
        alternate = _snapshot_and_selection()[1]
        object.__setattr__(alternate, "selected", tuple(reversed(alternate.selected)))
        original = evaluation.FrozenContextLabelsAntigravity(snapshot.context_datasets,
            {dataset: () for dataset in snapshot.context_datasets}, publication.publication_sha256,
            publication.source_audit_sha256, selection, 1, 1117, 0)
        cases = (
            evaluation.FrozenContextLabelsAntigravity(tuple(reversed(snapshot.context_datasets)),
                {dataset: () for dataset in snapshot.context_datasets}, publication.publication_sha256,
                publication.source_audit_sha256, selection, 1, 1117, 0),
            evaluation.FrozenContextLabelsAntigravity(snapshot.context_datasets,
                {dataset: () for dataset in snapshot.context_datasets}, publication.publication_sha256,
                publication.source_audit_sha256, alternate, 1, 1117, 0),
        )
        preflight = (publication, snapshot, selection, snapshot.context_datasets,
                     MappingProxyType(dict(snapshot.test_row_count_by_dataset)))
        for altered in cases:
            with self.subTest(selection=altered.selection != original.selection), tempfile.TemporaryDirectory() as temporary:
                destination = Path(temporary) / "evaluation_antigravity"
                with patch.object(evaluation, "_EVALUATION_DESTINATION_ANTIGRAVITY", destination), \
                     patch.object(evaluation, "_preflight_antigravity", return_value=preflight), \
                     patch.object(evaluation, "load_frozen_context_labels_antigravity", return_value=altered), \
                     patch.object(evaluation, "compute_retrospective_evaluation_antigravity", side_effect=AssertionError("compute called")):
                    with self.assertRaisesRegex(ValueError, "frozen labels"):
                        evaluation.evaluate_exp441_antigravity()
                self.assertFalse(destination.exists())
                self.assertEqual(len(list(Path(temporary).glob("failed_evaluation_staging_*_antigravity"))), 1)

    def test_task11_post_rename_failure_keeps_destination_and_writes_separate_audit(self) -> None:
        """No post-rename fault may repair or add to the immutable eight-file destination."""
        snapshot, _selection = _snapshot_and_selection()
        zero_selection = SelectionResultAntigravity((), (), (), (), (), frozenset())
        publication = FrozenPublicationAntigravity("3" * 64, "4" * 64, "5" * 64, "6" * 64, zero_selection)
        labels = evaluation.FrozenContextLabelsAntigravity(snapshot.context_datasets,
            {dataset: () for dataset in snapshot.context_datasets}, publication.publication_sha256,
            publication.source_audit_sha256, zero_selection, 1, 1117, 0)
        preflight = (publication, snapshot, zero_selection, snapshot.context_datasets,
                     MappingProxyType(dict(snapshot.test_row_count_by_dataset)))
        with tempfile.TemporaryDirectory() as temporary:
            parent, destination = Path(temporary), Path(temporary) / "evaluation_antigravity"
            def fail_after_commit(pinned: object) -> object:
                raise OSError("synthetic post-rename replay")
            with patch.object(evaluation, "_EVALUATION_DESTINATION_ANTIGRAVITY", destination), \
                 patch.object(evaluation, "_preflight_antigravity", return_value=preflight), \
                 patch.object(evaluation, "load_frozen_context_labels_antigravity", return_value=labels), \
                 patch.object(evaluation, "compute_retrospective_evaluation_antigravity", return_value=self._complete_result(snapshot.context_datasets)), \
                 patch.object(evaluation, "_replay_pinned_evaluation_antigravity", side_effect=fail_after_commit):
                with self.assertRaisesRegex(OSError, "post-rename"):
                    evaluation.evaluate_exp441_antigravity()
            self.assertEqual(set(path.name for path in destination.iterdir()), set(evaluation._EVALUATION_FILENAMES_ANTIGRAVITY))
            audits = list(parent.glob("failed_evaluation_audit_*_antigravity"))
            self.assertEqual(len(audits), 1)
            failure = evaluation.parse_canonical_json_bytes_antigravity((audits[0] / "failure_antigravity.json").read_bytes())
            self.assertIs(failure["destination_committed"], True)

    def _sparse_snapshot(self) -> object:
        """Hand-derived sparse identities: B/C/A order is intentionally not lexical."""
        b = _candidate("B", 5, 3.0)
        c = _candidate("C", 7, 2.0)
        winners = (
            RankedCandidateAntigravity(b, 1, True, 1),
            RankedCandidateAntigravity(c, 2, True, 2),
        )
        selection = SelectionResultAntigravity(
            (b, c), (b, c), (b, c), winners, winners, frozenset({("A", 10)}),
        )
        return evaluation.FrozenContextLabelsAntigravity(
            ("B", "C", "A"), {
                "B": ((5, 0), (9, 1)),
                "C": ((7, 1), (11, 0)),
                "A": ((3, 1), (4, 0), (8, 1), (10, 1), (12, 0)),
            }, "a" * 64, "b" * 64, selection, 1, 3, 0,
        )

    def test_sparse_facts_preserve_actual_identity_context_order_and_rescue(self) -> None:
        """Using ordinal positions, sorting context, or a candidate label inversion changes these literals."""
        sparse = evaluation._compute_sparse_retrospective_antigravity(self._sparse_snapshot())
        facts, rescued, regressed, rows = sparse
        self.assertEqual(tuple(item.dataset for item in facts), ("B", "C", "A"))
        self.assertEqual(tuple(item.baseline_confusion for item in facts), (
            (0, 0, 1, 1), (0, 0, 1, 1), (1, 0, 2, 2),
        ))
        self.assertEqual(tuple(item.union_confusion for item in facts), (
            (0, 1, 1, 0), (1, 0, 0, 1), (1, 0, 2, 2),
        ))
        self.assertEqual((facts[0].candidate_tp, facts[0].candidate_fp), (0, 1))
        self.assertEqual((facts[1].candidate_tp, facts[1].candidate_fp), (1, 0))
        self.assertEqual(rescued, ("C",))
        self.assertEqual(regressed, ())
        self.assertEqual(rows[0], (0, 0, 1, 1, 0, 1, 1, 0, 0.0))
        self.assertEqual(rows[1], (0, 0, 1, 1, 1, 0, 0, 1, 1.0))
        self.assertEqual(rows[2], (1, 0, 2, 2, 1, 0, 2, 2, 0.0))

    def test_sparse_identity_absence_and_regression_strong_loss_boundaries_fail_or_classify_strictly(self) -> None:
        """Dropping an actual coordinate, weakening <, or excluding baseline F1 == .8 is a bug."""
        missing = self._sparse_snapshot()
        object.__setattr__(missing, "labels_by_dataset", MappingProxyType({
            "B": ((9, 1),), "C": ((7, 1), (11, 0)),
            "A": ((3, 1), (4, 0), (8, 1), (10, 1), (12, 0)),
        }))
        with self.assertRaisesRegex(ValueError, "absent"):
            evaluation._compute_sparse_retrospective_antigravity(missing)
        missing_hard = self._sparse_snapshot()
        object.__setattr__(missing_hard, "labels_by_dataset", MappingProxyType({
            "B": ((5, 0), (9, 1)), "C": ((7, 1), (11, 0)),
            "A": ((3, 1), (4, 0), (8, 1), (12, 0)),
        }))
        with self.assertRaisesRegex(ValueError, "absent"):
            evaluation._compute_sparse_retrospective_antigravity(missing_hard)
        # These direct per-dataset facts isolate strict regression and the .8 strong-loss threshold.
        exact = evaluation._strong_loss_antigravity(0.8, 0.7)
        just_below = evaluation._strong_loss_antigravity(0.7999999999999999, 0.7)
        self.assertEqual(exact, 0.10000000000000009)
        self.assertEqual(just_below, 0.0)
        self.assertTrue(evaluation._is_regression_antigravity(0.7, 0.8))
        self.assertFalse(evaluation._is_regression_antigravity(0.8, 0.8))

    def test_status_constants_tolerance_and_engineering_precedence_are_frozen(self) -> None:
        """A rounded value, reversed precedence, or an altered frozen decimal must change status."""
        facts = EvaluationFactsAntigravity(
            (1915, 303, 2003, 191679), (1930, 318, 1988, 191664), 15, 15, 15, 0,
            0.0, 0.6241851368970013, 0.6260136230943887,
            0.0018284861973874023, 0.012444046553267682,
        )
        bootstrap = BootstrapResultAntigravity(
            (0.00005373870255999115, 0.003814109586073529),
            (0.006535362578334826, 0.019158460161145925),
        )
        self.assertEqual(facts.micro_delta.hex(), "0x1.df53a1ad05800p-10")
        self.assertEqual(facts.macro_delta.hex(), "0x1.97c43a7cf3ab8p-7")
        self.assertEqual(tuple(value.hex() for value in bootstrap.micro_ci), (
            "0x1.c2cafc29c4668p-15", "0x1.f3ec47deffdf7p-9"))
        self.assertEqual(tuple(value.hex() for value in bootstrap.macro_ci), (
            "0x1.ac4d308a9c7cep-8", "0x1.39e467f55da52p-6"))
        self.assertEqual(evaluation._retrospective_status_antigravity(facts, bootstrap), "retrospective_reproduction_complete")
        self.assertTrue(evaluation._is_tolerant_reproduction_value_antigravity(0.0, 1e-15))
        self.assertFalse(evaluation._is_tolerant_reproduction_value_antigravity(0.0, math.nextafter(1e-15, math.inf)))
        at_tolerance = BootstrapResultAntigravity(
            (math.nextafter(bootstrap.micro_ci[0], math.inf), bootstrap.micro_ci[1]), bootstrap.macro_ci,
        )
        self.assertEqual(evaluation._retrospective_status_antigravity(facts, at_tolerance), "retrospective_reproduction_complete")
        over_tolerance = BootstrapResultAntigravity(
            (bootstrap.micro_ci[0] + 2e-15, bootstrap.micro_ci[1]), bootstrap.macro_ci,
        )
        self.assertEqual(evaluation._retrospective_status_antigravity(facts, over_tolerance), "retrospective_reproduction_discrepancy")
        engineering_and_discrepant = EvaluationFactsAntigravity(
            facts.baseline_confusion, facts.union_confusion, 15, 15, 15, 0, 0.0,
            facts.baseline_f1 + 1e-16, facts.union_f1, facts.micro_delta + 1.1e-15, facts.macro_delta,
        )
        self.assertEqual(evaluation._retrospective_status_antigravity(engineering_and_discrepant, bootstrap), "engineering_reproduction_failed")
        exact_mismatches = (
            replace(facts, baseline_confusion=(1916, 303, 2003, 191679)),
            replace(facts, union_confusion=(1930, 319, 1988, 191664)),
            replace(facts, candidate_tp=14), replace(facts, candidate_fp=14),
            replace(facts, rescued_datasets=14), replace(facts, regressed_datasets=1),
            replace(facts, maximum_strong_loss=1.0),
            replace(facts, baseline_f1=math.nextafter(facts.baseline_f1, math.inf)),
            replace(facts, union_f1=math.nextafter(facts.union_f1, math.inf)),
        )
        for altered in exact_mismatches:
            with self.subTest(exact=altered):
                self.assertEqual(evaluation._retrospective_status_antigravity(altered, bootstrap), "engineering_reproduction_failed")
        tolerant_mismatches = (
            (replace(facts, micro_delta=facts.micro_delta + 2e-15), bootstrap),
            (replace(facts, macro_delta=facts.macro_delta + 2e-15), bootstrap),
            (facts, replace(bootstrap, micro_ci=(bootstrap.micro_ci[0] + 2e-15, bootstrap.micro_ci[1]))),
            (facts, replace(bootstrap, micro_ci=(bootstrap.micro_ci[0], bootstrap.micro_ci[1] + 2e-15))),
            (facts, replace(bootstrap, macro_ci=(bootstrap.macro_ci[0] + 2e-15, bootstrap.macro_ci[1]))),
            (facts, replace(bootstrap, macro_ci=(bootstrap.macro_ci[0], bootstrap.macro_ci[1] + 2e-15))),
        )
        for altered_facts, altered_bootstrap in tolerant_mismatches:
            with self.subTest(tolerant=(altered_facts, altered_bootstrap)):
                self.assertEqual(evaluation._retrospective_status_antigravity(altered_facts, altered_bootstrap), "retrospective_reproduction_discrepancy")
        inside_tolerance = (
            (replace(facts, micro_delta=math.nextafter(facts.micro_delta, math.inf)), bootstrap),
            (replace(facts, macro_delta=math.nextafter(facts.macro_delta, math.inf)), bootstrap),
            (facts, replace(bootstrap, micro_ci=(math.nextafter(bootstrap.micro_ci[0], math.inf), bootstrap.micro_ci[1]))),
            (facts, replace(bootstrap, micro_ci=(bootstrap.micro_ci[0], math.nextafter(bootstrap.micro_ci[1], math.inf)))),
            (facts, replace(bootstrap, macro_ci=(math.nextafter(bootstrap.macro_ci[0], math.inf), bootstrap.macro_ci[1]))),
            (facts, replace(bootstrap, macro_ci=(bootstrap.macro_ci[0], math.nextafter(bootstrap.macro_ci[1], math.inf)))),
        )
        for altered_facts, altered_bootstrap in inside_tolerance:
            with self.subTest(inside=(altered_facts, altered_bootstrap)):
                self.assertEqual(evaluation._retrospective_status_antigravity(altered_facts, altered_bootstrap), "retrospective_reproduction_complete")
        for attribute in ("micro_delta", "macro_delta"):
            for nonfinite in (float("nan"), float("inf")):
                invalid = replace(facts)
                object.__setattr__(invalid, attribute, nonfinite)
                with self.subTest(attribute=attribute, nonfinite=nonfinite):
                    self.assertEqual(evaluation._retrospective_status_antigravity(invalid, bootstrap), "engineering_reproduction_failed")
        for attribute in ("micro_ci", "macro_ci"):
            for nonfinite in (float("nan"), float("inf")):
                invalid = replace(bootstrap)
                values = list(getattr(invalid, attribute)); values[0] = nonfinite
                object.__setattr__(invalid, attribute, tuple(values))
                with self.subTest(attribute=attribute, nonfinite=nonfinite):
                    self.assertEqual(evaluation._retrospective_status_antigravity(facts, invalid), "engineering_reproduction_failed")

    def test_each_tolerant_field_uses_the_last_accepted_representable_float(self) -> None:
        """Every final accepted float has one adjacent +inf value that the real classifier rejects."""
        facts = EvaluationFactsAntigravity(
            (1915, 303, 2003, 191679), (1930, 318, 1988, 191664), 15, 15, 15, 0,
            0.0, 0.6241851368970013, 0.6260136230943887,
            0.0018284861973874023, 0.012444046553267682,
        )
        bootstrap = BootstrapResultAntigravity(
            (0.00005373870255999115, 0.003814109586073529),
            (0.006535362578334826, 0.019158460161145925),
        )
        def replace_field(position: int, value: float) -> tuple[EvaluationFactsAntigravity, BootstrapResultAntigravity]:
            if position == 0:
                return replace(facts, micro_delta=value), bootstrap
            if position == 1:
                return replace(facts, macro_delta=value), bootstrap
            if position < 4:
                values = list(bootstrap.micro_ci); values[position - 2] = value
                return facts, replace(bootstrap, micro_ci=tuple(values))
            values = list(bootstrap.macro_ci); values[position - 4] = value
            return facts, replace(bootstrap, macro_ci=tuple(values))
        expected = (facts.micro_delta, facts.macro_delta, *bootstrap.micro_ci, *bootstrap.macro_ci)
        for position, value in enumerate(expected):
            accepted = value
            while math.isclose(math.nextafter(accepted, math.inf), value, rel_tol=0.0, abs_tol=1e-15):
                accepted = math.nextafter(accepted, math.inf)
            rejected = math.nextafter(accepted, math.inf)
            accepted_facts, accepted_bootstrap = replace_field(position, accepted)
            rejected_facts, rejected_bootstrap = replace_field(position, rejected)
            with self.subTest(position=position):
                self.assertEqual(math.nextafter(accepted, math.inf), rejected)
                self.assertTrue(math.isclose(accepted, value, rel_tol=0.0, abs_tol=1e-15))
                self.assertFalse(math.isclose(rejected, value, rel_tol=0.0, abs_tol=1e-15))
                self.assertEqual(evaluation._retrospective_status_antigravity(accepted_facts, accepted_bootstrap), "retrospective_reproduction_complete")
                self.assertEqual(evaluation._retrospective_status_antigravity(rejected_facts, rejected_bootstrap), "retrospective_reproduction_discrepancy")

    def test_pure_compute_and_helper_direct_call_allowlists_are_exact(self) -> None:
        """Any unreviewed direct call in the public pure calculation chain is a boundary regression."""
        source = evaluation.__loader__.get_source(evaluation.__name__) or ""
        tree = ast.parse(source)
        functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
        def calls(name: str) -> set[str]:
            return {ast.unparse(node.func) for node in ast.walk(functions[name]) if isinstance(node, ast.Call)}
        self.assertEqual(calls("compute_retrospective_evaluation_antigravity"), {
            "EvaluationFactsAntigravity", "RetrospectiveEvaluationAntigravity", "ValueError",
            "_compute_sparse_retrospective_antigravity", "_retrospective_status_antigravity",
            "_strong_loss_antigravity", "float", "isinstance", "len", "max", "np.asarray",
            "np.asarray([fact.delta for fact in dataset_facts], dtype=np.float64).mean",
            "paired_dataset_bootstrap_antigravity", "project_f1_antigravity", "range", "sum", "tuple",
            "validate_production_selection_antigravity",
        })
        self.assertEqual(calls("_compute_sparse_retrospective_antigravity"), {
            "DatasetEvaluationFactsAntigravity", "ValueError", "_confusion_for_predictions_antigravity",
            "_is_regression_antigravity", "any", "facts.append", "float", "frozenset",
            "hard_by_dataset.get", "hard_by_dataset.setdefault", "hard_by_dataset.setdefault(dataset, set()).add",
            "isinstance", "len", "project_f1_antigravity", "regressed.append", "rescued.append",
            "rows.append", "selected_by_dataset.get", "selected_by_dataset.setdefault",
            "selected_by_dataset.setdefault(ranked.dataset, set()).add", "set",
            "set(hard_by_dataset).difference", "set(selected_by_dataset).difference",
            "snapshot.labels_by_dataset.get", "sum", "tuple", "type",
        })

    def test_public_calculation_has_one_snapshot_argument_and_never_reopens_task9_boundaries(self) -> None:
        """Adding a retuning parameter or calling a loader/source route would violate the frozen split."""
        source, selection = _snapshot_and_selection()
        required: dict[str, set[int]] = {dataset: set() for dataset in source.context_datasets}
        for dataset, index in selection.hard_coordinates:
            required[dataset].add(index)
        for ranked in selection.selected:
            required[ranked.dataset].add(ranked.instance_index)
        snapshot = evaluation.FrozenContextLabelsAntigravity(
            source.context_datasets,
            {dataset: tuple((index, 0) for index in sorted(indices)) for dataset, indices in required.items()},
            "a" * 64, "b" * 64, selection, 1, 1117, 0,
        )
        bootstrap_calls: list[tuple[int, int]] = []
        def fake_bootstrap(rows: object) -> BootstrapResultAntigravity:
            values = np.asarray(rows)
            bootstrap_calls.append(values.shape)
            return BootstrapResultAntigravity((0.0, 0.0), (0.0, 0.0))
        def explode(*args: object, **kwargs: object) -> object:
            raise AssertionError("pure retrospective calculation crossed a forbidden boundary")
        before_selection = selection.to_payload()
        sentinel = _InputsResolutionSentinel()
        inputs_name = sentinel.module_name
        prior_inputs = sys.modules.pop(inputs_name, None)
        sys.meta_path.insert(0, sentinel)
        try:
            with patch.object(evaluation, "paired_dataset_bootstrap_antigravity", side_effect=fake_bootstrap), \
             patch.object(evaluation, "load_frozen_context_labels_antigravity", side_effect=explode), \
             patch.object(evaluation, "_load_labels_antigravity", side_effect=explode), \
             patch.object(evaluation, "_preflight_antigravity", side_effect=explode), \
             patch.object(evaluation, "inspect_exp441_publication_antigravity", side_effect=explode), \
             patch.object(evaluation, "verify_frozen_sources_antigravity", side_effect=explode), \
             patch.object(evaluation, "select_hard_empty_single_shot_antigravity", side_effect=explode), \
             patch.object(builtins, "open", side_effect=explode), \
             patch.object(Path, "open", side_effect=explode), \
             patch.object(os, "open", side_effect=explode), \
             patch.object(os, "stat", side_effect=explode):
                result = evaluation.compute_retrospective_evaluation_antigravity(snapshot)
        finally:
            sys.meta_path.remove(sentinel)
            sys.modules.pop(inputs_name, None)
            if prior_inputs is not None:
                sys.modules[inputs_name] = prior_inputs
        self.assertEqual(bootstrap_calls, [(1117, 9)])
        self.assertEqual(result.status, "engineering_reproduction_failed")
        self.assertIs(result.scientific_winner_selected, False)
        self.assertEqual(selection.to_payload(), before_selection)
        self.assertEqual(sentinel.resolutions, 0)
        self.assertEqual(tuple(inspect.signature(evaluation.compute_retrospective_evaluation_antigravity).parameters), ("snapshot",))

    def test_radically_changed_labels_never_retune_frozen_selection_or_coordinates(self) -> None:
        """TEST labels may change retrospective facts but cannot change frozen selected identities."""
        source, selection = _snapshot_and_selection()
        required: dict[str, tuple[int, ...]] = {dataset: () for dataset in source.context_datasets}
        mutable: dict[str, list[int]] = {dataset: [] for dataset in source.context_datasets}
        for dataset, index in selection.hard_coordinates:
            mutable[dataset].append(index)
        for ranked in selection.selected:
            mutable[ranked.dataset].append(ranked.instance_index)
        required = {dataset: tuple(sorted(set(indices))) for dataset, indices in mutable.items()}
        zero = evaluation.FrozenContextLabelsAntigravity(
            source.context_datasets,
            {dataset: tuple((index, 0) for index in indices) for dataset, indices in required.items()},
            "a" * 64, "b" * 64, selection, 1, 1117, 0,
        )
        one = evaluation.FrozenContextLabelsAntigravity(
            source.context_datasets,
            {dataset: tuple((index, 1) for index in indices) for dataset, indices in required.items()},
            "a" * 64, "b" * 64, selection, 1, 1117, 0,
        )
        calls: list[tuple[int, int]] = []
        def bootstrap(rows: object) -> BootstrapResultAntigravity:
            calls.append(np.asarray(rows).shape)
            return BootstrapResultAntigravity((0.0, 0.0), (0.0, 0.0))
        frozen_selection = selection.to_payload()
        with patch.object(evaluation, "paired_dataset_bootstrap_antigravity", side_effect=bootstrap):
            zero_result = evaluation.compute_retrospective_evaluation_antigravity(zero)
            one_result = evaluation.compute_retrospective_evaluation_antigravity(one)
        self.assertEqual(calls, [(1117, 9), (1117, 9)])
        self.assertEqual(zero.selection.to_payload() if hasattr(zero, "selection") else None, frozen_selection)
        self.assertEqual(one.selection.to_payload() if hasattr(one, "selection") else None, frozen_selection)
        self.assertEqual(selection.to_payload(), frozen_selection)
        self.assertNotEqual(zero_result.facts.candidate_tp, one_result.facts.candidate_tp)
