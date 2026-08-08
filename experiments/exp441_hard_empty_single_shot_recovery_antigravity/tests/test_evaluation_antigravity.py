"""TDD coverage for the Exp441 post-freeze TEST-label read boundary."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import replace
from hashlib import sha256
from types import MappingProxyType
from unittest.mock import patch
import ast
import importlib
import importlib.abc
import importlib.util
import inspect
import math
import sys
import types
import unittest

import numpy as np

from experiments.exp441_hard_empty_single_shot_recovery_antigravity.artifacts_antigravity import (
    canonical_json_bytes_antigravity,
)
from experiments.exp441_hard_empty_single_shot_recovery_antigravity.contracts_antigravity import (
    BootstrapResultAntigravity,
    CandidateEvidenceAntigravity,
    EvaluationFactsAntigravity,
    FrozenPublicationAntigravity,
    FrozenSourceSnapshotAntigravity,
    RankedCandidateAntigravity,
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

    def test_module_has_no_top_level_label_capable_or_write_imports(self) -> None:
        source = evaluation.__loader__.get_source(evaluation.__name__) or ""
        tree = ast.parse(source)
        imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        imported = "\n".join(ast.unparse(node) for node in imports)
        self.assertNotIn("exp417", imported)
        self.assertNotIn("sqlite3", imported)
        self.assertNotIn("evaluation", imported)
        self.assertNotIn("write_", source)


class Exp441RetrospectiveEvaluationAntigravityTests(unittest.TestCase):
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
        invalid = replace(facts)
        object.__setattr__(invalid, "micro_delta", float("nan"))
        self.assertEqual(evaluation._retrospective_status_antigravity(invalid, bootstrap), "engineering_reproduction_failed")
        invalid = replace(facts)
        object.__setattr__(invalid, "macro_delta", float("inf"))
        self.assertEqual(evaluation._retrospective_status_antigravity(invalid, bootstrap), "engineering_reproduction_failed")

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
        with patch.object(evaluation, "paired_dataset_bootstrap_antigravity", side_effect=fake_bootstrap), \
             patch.object(evaluation, "load_frozen_context_labels_antigravity", side_effect=explode), \
             patch.object(evaluation, "inspect_exp441_publication_antigravity", side_effect=explode), \
             patch.object(evaluation, "verify_frozen_sources_antigravity", side_effect=explode), \
             patch.object(evaluation, "select_hard_empty_single_shot_antigravity", side_effect=explode):
            result = evaluation.compute_retrospective_evaluation_antigravity(snapshot)
        self.assertEqual(bootstrap_calls, [(1117, 9)])
        self.assertEqual(result.status, "engineering_reproduction_failed")
        self.assertIs(result.scientific_winner_selected, False)
        self.assertEqual(tuple(inspect.signature(evaluation.compute_retrospective_evaluation_antigravity).parameters), ("snapshot",))
