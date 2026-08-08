"""TDD coverage for the Exp441 post-freeze TEST-label read boundary."""
from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256
from types import MappingProxyType
from unittest.mock import patch
import ast
import importlib
import sys
import types
import unittest

from experiments.exp441_hard_empty_single_shot_recovery_antigravity.artifacts_antigravity import (
    canonical_json_bytes_antigravity,
)
from experiments.exp441_hard_empty_single_shot_recovery_antigravity.contracts_antigravity import (
    CandidateEvidenceAntigravity,
    FrozenPublicationAntigravity,
    FrozenSourceSnapshotAntigravity,
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

    def execute(self, sql: str, params: object = ()) -> _Cursor:
        self.calls.append((sql, params))
        if sql == "PRAGMA quick_check":
            return _Cursor([("ok",)])
        return _Cursor(self.rows[params[0]])  # type: ignore[index]


class Exp441FrozenContextLabelsAntigravityTests(unittest.TestCase):
    def setUp(self) -> None:
        global evaluation
        evaluation = importlib.import_module(
            "experiments.exp441_hard_empty_single_shot_recovery_antigravity.evaluation_antigravity"
        )
        self.snapshot, self.selection = _snapshot_and_selection()
        self.publication = _publication(self.snapshot, self.selection)
        self.rows = {dataset: [(3, "0"), (7, 0), (8, 0), (9, 1)] + [(index, 0) for index in range(10, 1006)]
                     for dataset in self.snapshot.context_datasets}
        self.connection = _Connection(self.rows)

    def _run(self, *, publication_sequence: list[object] | None = None,
             source_sequence: list[object] | None = None) -> object:
        publications = iter(publication_sequence or [self.publication] * 3)
        sources = iter(source_sequence or [self.snapshot] * 2)
        @contextmanager
        def fake_connection(path: object, digest: object):
            self.assertEqual(str(path), evaluation.FROZEN_CONFIG.database_path)
            self.assertEqual(digest, evaluation.FROZEN_CONFIG.database_sha256)
            yield self.connection
        inputs = types.ModuleType("experiments.exp417_causal_multihorizon_shadow.inputs")
        inputs.verified_read_only_connection = fake_connection
        with patch.object(evaluation, "inspect_exp441_publication_antigravity", side_effect=lambda: next(publications)), \
             patch.object(evaluation, "verify_frozen_sources_antigravity", side_effect=lambda: next(sources)), \
             patch.dict(sys.modules, {inputs.__name__: inputs}):
            return evaluation.load_frozen_context_labels_antigravity()

    def test_loads_exact_context_queries_and_preserves_noncontiguous_identities(self) -> None:
        result = self._run()
        self.assertEqual(result.context_datasets, self.snapshot.context_datasets)
        self.assertEqual(result.labels_by_dataset["D0000"][:4], ((3, 0), (7, 0), (8, 0), (9, 1)))
        self.assertEqual((result.descriptor_bound_connection_count, result.label_query_calls, result.out_of_scope_label_rows), (1, 1117, 0))
        self.assertEqual(self.connection.calls[0], ("PRAGMA quick_check", ()))
        self.assertEqual(self.connection.calls[1:], [(_LABEL_SQL, (dataset,)) for dataset in self.snapshot.context_datasets])
        self.assertNotIn(("CornellWhaleChallenge",), [params for _, params in self.connection.calls])
        self.assertNotIn(("Wafer_normal_1",), [params for _, params in self.connection.calls])

    def test_preflight_failures_do_not_resolve_connection_helper(self) -> None:
        failures = (
            ("publication", [ValueError("publication")], [self.snapshot]),
            ("sources", [self.publication], [ValueError("sources")]),
            ("source audit", [FrozenPublicationAntigravity("3" * 64, "4" * 64, "5" * 64, "6" * 64, self.selection)], [self.snapshot]),
            ("publication drift", [self.publication, FrozenPublicationAntigravity("7" * 64, "4" * 64, "5" * 64, self.publication.source_audit_sha256, self.selection)], [self.snapshot]),
        )
        for name, publications, sources in failures:
            with self.subTest(name=name), self.assertRaises(Exception):
                self._run(publication_sequence=publications, source_sequence=sources)
            self.assertEqual(self.connection.calls, [])

    def test_post_connection_drift_fails_closed(self) -> None:
        changed = FrozenPublicationAntigravity("7" * 64, "4" * 64, "5" * 64, self.publication.source_audit_sha256, self.selection)
        with self.assertRaises(ValueError):
            self._run(publication_sequence=[self.publication, self.publication, changed])
        self.assertEqual(len(self.connection.calls), 1118)

    def test_rejects_bad_quick_check_indices_schema_counts_and_labels(self) -> None:
        cases = (
            ([(3, "0")], "count"), ([(3, "0"), (3, 1)] + [(index, 0) for index in range(10, 1008)], "index"),
            ([(True, 0)] + [(index, 0) for index in range(10, 1008)], "index"),
            ([(3, True), (9, 1)] + [(index, 0) for index in range(10, 1008)], "label"),
            ([(3, "2"), (9, 1)] + [(index, 0) for index in range(10, 1008)], "label"),
        )
        for rows, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.rows["D0000"] = rows
                self._run()
            self.rows["D0000"] = [(3, "0"), (7, 0), (8, 0), (9, 1)] + [(index, 0) for index in range(10, 1006)]

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
