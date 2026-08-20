"""Result-bound checkpoint and bounded recovery contracts for Exp450."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from paper.equipment_anomaly_detection.contracts_paper import RawDataset
from paper.equipment_anomaly_detection.hard_route_paper import (
    HardRouteResult,
    RouteLanes,
    SourceExpansion,
)
from paper.equipment_anomaly_detection.sources_paper.hard_sources_paper import (
    EXPECTED_HARD_SOURCES,
    HardSourceBundle,
    _completed_paper,
)

from experiments.exp450_resampling_image_ablation_antigravity.artifacts_antigravity import (
    append_dataset_condition_failure_antigravity,
    canonical_result_bytes_antigravity,
    read_dataset_condition_success_antigravity,
    verify_dataset_condition_inventory_antigravity,
    write_dataset_condition_success_antigravity,
)
from experiments.exp450_resampling_image_ablation_antigravity.contracts_antigravity import (
    Exp450BackendDecision,
    Exp450Condition,
    Exp450DatasetConditionResult,
    Exp450RunContext,
    canonical_json_bytes,
)


CONDITION_A = Exp450Condition("A", "linear", "linear")
CONDITION_B = Exp450Condition("B", "polyphase_pchip", "linear")
CONDITION_C = Exp450Condition("C", "linear", "area")
CONDITION_D = Exp450Condition("D", "polyphase_pchip", "area")
BACKEND = Exp450BackendDecision("cpu", "test", False, False, 1, 1)


def _raw(dataset: str) -> RawDataset:
    train = tuple(
        np.asarray([index, index + 0.25, index + 0.5], dtype="<f4") for index in range(5)
    )
    test = tuple(
        np.asarray([index + 0.75, index + 1.0, index + 1.25], dtype="<f4") for index in range(2)
    )
    return RawDataset(
        dataset, 3, tuple(range(len(train))), tuple(row.tobytes() for row in train), train,
        tuple(range(len(test))), tuple(row.tobytes() for row in test), test,
    )


def _catalog(raws: tuple[RawDataset, ...]) -> dict[str, object]:
    entries = []
    for number, raw in enumerate(raws):
        def identity(split: str, indices: tuple[int, ...], blobs: tuple[bytes, ...]) -> dict[str, object]:
            bare = {
                "dataset": raw.dataset, "dataset_id": number, "series_length": raw.metadata_length,
                "split": split, "indices": list(indices),
                "values_blob_sha256": [sha256(blob).hexdigest() for blob in blobs],
            }
            return {**bare, "row_digest": sha256(canonical_json_bytes(bare)).hexdigest(), "count": len(blobs)}
        entries.append({
            "dataset": raw.dataset, "dataset_id": number, "series_length": raw.metadata_length,
            "train": identity("TRAIN", raw.train_indices, raw.train_blobs),
            "test": identity("TEST", raw.test_indices, raw.test_blobs),
        })
    bare = {"schema": "exp450-fixture-catalog-antigravity.v1", "mode": "smoke", "database": {"fixture": "test"}, "datasets": entries}
    return {**bare, "catalog_sha256": sha256(canonical_json_bytes(bare)).hexdigest()}


def _context(root: Path, catalog: dict[str, object]) -> Exp450RunContext:
    return Exp450RunContext(
        "smoke", root, "a" * 64, catalog["catalog_sha256"],
        {"scoring_antigravity.py": "b" * 64}, BACKEND, "c" * 64,
    )


def _result(raw: RawDataset, condition: Exp450Condition) -> Exp450DatasetConditionResult:
    source = {
        name: _completed_paper(
            name, np.asarray([0., 1., 2., 3., 4.], dtype=np.float64),
            np.asarray([1.5, 4.5], dtype=np.float64),
        ) for name in EXPECTED_HARD_SOURCES
    }
    lanes = RouteLanes(frozenset(), frozenset(), frozenset())
    route = HardRouteResult(
        lanes, frozenset(), frozenset(), frozenset(), lanes,
        SourceExpansion(frozenset(), frozenset()), lanes, frozenset(), lanes,
        frozenset(), frozenset(), len(raw.test_rows),
    )
    return Exp450DatasetConditionResult(raw.dataset, condition.condition_id, HardSourceBundle(EXPECTED_HARD_SOURCES, source), route)


class Exp450ArtifactRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory(prefix="exp450_artifacts_antigravity_")
        self.root = Path(self.temporary.name) / "run_antigravity"
        self.raw = _raw("first_antigravity")
        self.catalog = _catalog((self.raw,))
        self.context = _context(self.root, self.catalog)
        self.result = _result(self.raw, CONDITION_A)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_dataset_condition_checkpoint_round_trips_typed_result(self) -> None:
        """Dropping any typed source or route science must break checkpoint round-trip."""
        path = write_dataset_condition_success_antigravity(self.root, self.context, self.catalog, self.raw, self.result)
        reread = read_dataset_condition_success_antigravity(self.root, self.context, self.catalog, self.raw, CONDITION_A)
        self.assertEqual(canonical_result_bytes_antigravity(reread), canonical_result_bytes_antigravity(self.result))
        self.assertIn("antigravity", path.name)

    def test_tampered_checkpoint_is_rejected(self) -> None:
        """Changing a checkpoint coordinate without re-binding it must never reuse it."""
        path = write_dataset_condition_success_antigravity(self.root, self.context, self.catalog, self.raw, self.result)
        path.write_bytes(path.read_bytes().replace(b'"condition_id":"A"', b'"condition_id":"B"'))
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            read_dataset_condition_success_antigravity(self.root, self.context, self.catalog, self.raw, CONDITION_A)

    def test_checkpoint_rejects_a_root_that_is_not_its_bound_context_root(self) -> None:
        """Writing through a different root would make a context-bound result movable."""
        with self.assertRaisesRegex(ValueError, "root"):
            write_dataset_condition_success_antigravity(
                self.root.parent / "other_run_antigravity", self.context,
                self.catalog, self.raw, self.result,
            )

    def test_context_rejects_non_sha256_identity_fields(self) -> None:
        """A merely 64-character non-hex identity would not bind a real closure."""
        with self.assertRaisesRegex(ValueError, "digest"):
            Exp450RunContext(
                "smoke", self.root, "z" * 64, self.catalog["catalog_sha256"],
                {"scoring_antigravity.py": "b" * 64}, BACKEND, "c" * 64,
            )

    def test_inventory_reconstructs_second_wave_from_failed_and_never_attempted(self) -> None:
        """Treating failure records as success would omit exactly the needed retry work."""
        second = _raw("second_antigravity")
        catalog = _catalog((self.raw, second))
        context = _context(self.root, catalog)
        write_dataset_condition_success_antigravity(self.root, context, catalog, self.raw, self.result)
        append_dataset_condition_failure_antigravity(self.root, context, catalog, self.raw, CONDITION_B, 1, RuntimeError("once"))
        inventory = verify_dataset_condition_inventory_antigravity(
            self.root, context, catalog, {self.raw.dataset: self.raw, second.dataset: second},
        )
        self.assertEqual(inventory.completed, ((self.raw.dataset, "A"),))
        self.assertEqual(
            inventory.wave_two,
            (
                (self.raw.dataset, "B"), (self.raw.dataset, "C"), (self.raw.dataset, "D"),
                (second.dataset, "A"), (second.dataset, "B"), (second.dataset, "C"), (second.dataset, "D"),
            ),
        )
        self.assertEqual(inventory.failures[0].attempt, 1)
        self.assertEqual(inventory.failures[0].wave, 1)

    def test_inventory_never_schedules_a_third_attempt_after_a_wave_two_failure(self) -> None:
        """Requeueing a wave-two failure would silently exceed the bounded recovery rule."""
        append_dataset_condition_failure_antigravity(
            self.root, self.context, self.catalog, self.raw, CONDITION_A, 1, RuntimeError("first"),
        )
        append_dataset_condition_failure_antigravity(
            self.root, self.context, self.catalog, self.raw, CONDITION_A, 2, RuntimeError("second"),
        )
        inventory = verify_dataset_condition_inventory_antigravity(
            self.root, self.context, self.catalog, {self.raw.dataset: self.raw},
        )
        self.assertNotIn((self.raw.dataset, "A"), inventory.wave_two)

    def test_failure_ledger_accepts_exactly_one_wave_one_then_wave_two(self) -> None:
        """The only retryable history for one coordinate is the ordered 1→2 pair."""
        append_dataset_condition_failure_antigravity(
            self.root, self.context, self.catalog, self.raw, CONDITION_A, 1, RuntimeError("first"),
        )
        append_dataset_condition_failure_antigravity(
            self.root, self.context, self.catalog, self.raw, CONDITION_A, 2, RuntimeError("second"),
        )
        inventory = verify_dataset_condition_inventory_antigravity(
            self.root, self.context, self.catalog, {self.raw.dataset: self.raw},
        )
        self.assertEqual(
            tuple((failure.condition_id, failure.wave, failure.attempt) for failure in inventory.failures),
            (("A", 1, 1), ("A", 2, 2)),
        )

    def test_failure_ledger_rejects_a_duplicate_wave_two_append(self) -> None:
        """A third recorded attempt for one coordinate must fail before it reaches disk."""
        append_dataset_condition_failure_antigravity(
            self.root, self.context, self.catalog, self.raw, CONDITION_A, 1, RuntimeError("first"),
        )
        append_dataset_condition_failure_antigravity(
            self.root, self.context, self.catalog, self.raw, CONDITION_A, 2, RuntimeError("second"),
        )
        with self.assertRaisesRegex(ValueError, "failure ledger"):
            append_dataset_condition_failure_antigravity(
                self.root, self.context, self.catalog, self.raw, CONDITION_A, 2, RuntimeError("third"),
            )

    def test_failure_ledger_rejects_a_duplicate_wave_one_append(self) -> None:
        """Repeating wave one would give one coordinate two first attempts."""
        append_dataset_condition_failure_antigravity(
            self.root, self.context, self.catalog, self.raw, CONDITION_A, 1, RuntimeError("first"),
        )
        with self.assertRaisesRegex(ValueError, "failure ledger"):
            append_dataset_condition_failure_antigravity(
                self.root, self.context, self.catalog, self.raw, CONDITION_A, 1, RuntimeError("duplicate"),
            )

    def test_inventory_rejects_a_canonical_out_of_order_failure_ledger(self) -> None:
        """A valid checksum cannot authorize a wave-one record after wave two began."""
        second = _raw("second_antigravity")
        catalog = _catalog((self.raw, second))
        context = _context(self.root, catalog)
        append_dataset_condition_failure_antigravity(
            self.root, context, catalog, self.raw, CONDITION_A, 1, RuntimeError("first"),
        )
        append_dataset_condition_failure_antigravity(
            self.root, context, catalog, self.raw, CONDITION_A, 2, RuntimeError("second"),
        )
        entry = catalog["datasets"][1]
        bare = {
            "schema": "exp450-dataset-condition-failure-antigravity.v1",
            "context_sha256": context.sha256,
            "catalog_sha256": context.catalog_sha256,
            "dataset": second.dataset,
            "condition_id": "A",
            "raw_identity": entry,
            "backend": context.backend.artifact_record(),
            "wave": 1,
            "attempt": 3,
            "exception_type": "RuntimeError",
            "exception_message": "late first wave",
        }
        path = self.root / "checkpoints_antigravity" / "failures_antigravity" / "attempt_000003_antigravity.json"
        path.write_bytes(canonical_json_bytes({**bare, "record_sha256": sha256(canonical_json_bytes(bare)).hexdigest()}))
        with self.assertRaisesRegex(ValueError, "failure ledger"):
            verify_dataset_condition_inventory_antigravity(
                self.root, context, catalog, {self.raw.dataset: self.raw, second.dataset: second},
            )

    def test_interrupted_and_resumed_successes_match_after_root_context_normalization(self) -> None:
        """Resumption must reproduce the same typed success envelopes after root binding alone differs."""
        first, second = _raw("first_antigravity"), _raw("second_antigravity")
        catalog = _catalog((first, second))
        with TemporaryDirectory(prefix="exp450_uninterrupted_antigravity_") as uninterrupted_dir, TemporaryDirectory(prefix="exp450_resumed_antigravity_") as resumed_dir:
            uninterrupted_root = Path(uninterrupted_dir) / "run_antigravity"
            resumed_root = Path(resumed_dir) / "run_antigravity"
            uninterrupted_context = _context(uninterrupted_root, catalog)
            resumed_context = _context(resumed_root, catalog)
            conditions = (CONDITION_A, CONDITION_B, CONDITION_C, CONDITION_D)
            by_condition = {condition.condition_id: condition for condition in conditions}
            for raw in (first, second):
                for condition in conditions:
                    write_dataset_condition_success_antigravity(uninterrupted_root, uninterrupted_context, catalog, raw, _result(raw, condition))
            write_dataset_condition_success_antigravity(resumed_root, resumed_context, catalog, first, _result(first, CONDITION_A))
            append_dataset_condition_failure_antigravity(resumed_root, resumed_context, catalog, first, CONDITION_B, 1, RuntimeError("once"))
            resumed_inventory = verify_dataset_condition_inventory_antigravity(resumed_root, resumed_context, catalog, {first.dataset: first, second.dataset: second})
            for dataset, condition_id in resumed_inventory.wave_two:
                raw = {first.dataset: first, second.dataset: second}[dataset]
                condition = by_condition[condition_id]
                write_dataset_condition_success_antigravity(resumed_root, resumed_context, catalog, raw, _result(raw, condition))
            def normalized(root: Path) -> list[bytes]:
                values = []
                for path in sorted(root.glob("checkpoints_antigravity/*_antigravity/*_antigravity/dataset_condition_success_antigravity.json")):
                    value = json.loads(path.read_text(encoding="utf-8"))
                    value["context_sha256"] = "0" * 64
                    values.append(canonical_json_bytes(value))
                return values
            self.assertEqual(normalized(uninterrupted_root), normalized(resumed_root))


if __name__ == "__main__":
    unittest.main()
