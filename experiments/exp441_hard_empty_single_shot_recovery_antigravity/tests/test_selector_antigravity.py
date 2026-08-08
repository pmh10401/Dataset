"""RED/GREEN behavior tests for the pure Exp441 Hard-empty selector."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import random
import unittest

from experiments.exp441_hard_empty_single_shot_recovery_antigravity.contracts_antigravity import (
    CandidateEvidenceAntigravity,
    FrozenSourceSnapshotAntigravity,
    validate_production_selection_antigravity,
)
from experiments.exp441_hard_empty_single_shot_recovery_antigravity.sources_antigravity import (
    hard_coordinate_identity_sha256_antigravity,
    verify_frozen_sources_antigravity,
)
from experiments.exp441_hard_empty_single_shot_recovery_antigravity.artifacts_antigravity import (
    canonical_json_bytes_antigravity,
)
from experiments.exp441_hard_empty_single_shot_recovery_antigravity.selector_antigravity import (
    select_hard_empty_single_shot_antigravity,
    selected_coordinate_identity_sha256_antigravity,
)


def _candidate(
    dataset: str,
    index: int,
    *,
    ratio: float = 1.0,
    rank: float = 0.5,
    agreement: int = 2,
) -> CandidateEvidenceAntigravity:
    return CandidateEvidenceAntigravity(
        dataset=dataset,
        instance_index=index,
        raw_length=32,
        depths=tuple(range(1, agreement + 1)),
        agreement_count=agreement,
        second_excess_ratio=ratio,
        second_empirical_rank=rank,
    )


def _snapshot(
    candidates: tuple[CandidateEvidenceAntigravity, ...],
    hard: tuple[tuple[str, int], ...] = (),
) -> FrozenSourceSnapshotAntigravity:
    datasets = tuple(sorted({item.dataset for item in candidates} | {dataset for dataset, _ in hard}))
    rows = {dataset: max(
        [32]
        + [item.instance_index + 1 for item in candidates if item.dataset == dataset]
        + [index + 1 for name, index in hard if name == dataset]
    ) for dataset in datasets}
    digests = {dataset: f"{index + 1:064x}" for index, dataset in enumerate(datasets)}
    return FrozenSourceSnapshotAntigravity(
        context_datasets=datasets,
        test_row_count_by_dataset=rows,
        record_digest_by_dataset=digests,
        hard_coordinates=frozenset(hard),
        provisional_candidates=candidates,
        context_sha256="a" * 64,
        state_sha256="b" * 64,
        retained_inventory_sha256="c" * 64,
        publication_prediction_sha256="d" * 64,
        publication_manifest_sha256="e" * 64,
        publication_artifact_map_sha256="f" * 64,
        hard_prediction_sha256="0" * 64,
        hard_manifest_sha256="1" * 64,
        hard_coordinate_identity_sha256=hard_coordinate_identity_sha256_antigravity(hard),
        label_access_count=0,
    )


def _fixture_candidates() -> tuple[CandidateEvidenceAntigravity, ...]:
    return (
        _candidate("A", 7, ratio=2.0, rank=0.2, agreement=2),
        _candidate("A", 8, ratio=2.0, rank=0.2, agreement=3),
        _candidate("B", 1, ratio=3.0, rank=0.5),
        _candidate("C", 3, ratio=3.0, rank=0.5),
        _candidate("a", 2, ratio=3.0, rank=0.5),
    )


def select_fixture(dataset: str) -> tuple:
    """Return selected rows for the local tie fixture, filtered by dataset."""
    result = select_hard_empty_single_shot_antigravity(_snapshot(_fixture_candidates()))
    return tuple(item for item in result.selected if item.dataset == dataset)


class HardEmptySelectorAntigravityTests(unittest.TestCase):
    def test_only_hard_empty_dataset_can_contribute(self) -> None:
        # Break caught: using candidate-level Hard exclusion while admitting an active Hard dataset.
        candidates = (
            _candidate("A", 7, ratio=1.0),
            _candidate("B", 3, ratio=9.0),
            _candidate("C", 9, ratio=8.0),
        )
        result = select_hard_empty_single_shot_antigravity(
            _snapshot(candidates, hard=(("B", 1), ("C", 9)))
        )
        self.assertEqual(tuple(item.dataset for item in result.eligible_candidates), ("A",))
        self.assertEqual(tuple((item.dataset, item.instance_index) for item in result.selected), (("A", 7),))

    def test_agreement_count_is_eligibility_only_not_a_ranking_key(self) -> None:
        # Break caught: sorting by agreement count would wrongly prefer the 3-depth candidate at index 8.
        result = select_hard_empty_single_shot_antigravity(_snapshot(_fixture_candidates()))
        self.assertEqual(tuple(item.instance_index for item in select_fixture("A")), (7,))
        local_a = next(item for item in result.winners if item.dataset == "A")
        self.assertEqual(local_a.candidate.agreement_count, 2)
        self.assertEqual(local_a.candidate.second_excess_ratio, 2.0)

    def test_local_and_global_ties_use_the_frozen_order(self) -> None:
        # Break caught: changing ratio/rank/index or Python case-sensitive dataset tie-break changes this payload.
        result = select_hard_empty_single_shot_antigravity(_snapshot(_fixture_candidates()))
        expected_global_top_30 = (("B", 1), ("C", 3), ("a", 2), ("A", 7))
        self.assertEqual(
            tuple((item.dataset, item.instance_index) for item in result.selected),
            expected_global_top_30,
        )
        self.assertEqual(tuple(item.global_rank for item in result.winners), (1, 2, 3, 4))
        self.assertEqual(tuple(item.selection_rank for item in result.selected), (1, 2, 3, 4))

    def test_input_shuffle_is_deterministic_and_payload_bytes_are_stable(self) -> None:
        # Break caught: leaking worker/input completion order into any population or rank list.
        baseline = select_hard_empty_single_shot_antigravity(_snapshot(_fixture_candidates()))
        expected = canonical_json_bytes_antigravity(baseline.to_payload())
        for seed in range(20):
            shuffled = list(_fixture_candidates())
            random.Random(seed).shuffle(shuffled)
            candidate_snapshot = _snapshot(tuple(shuffled))
            result = select_hard_empty_single_shot_antigravity(candidate_snapshot)
            self.assertEqual(canonical_json_bytes_antigravity(result.to_payload()), expected, seed)

    def test_sparse_fixture_is_not_padded_to_the_production_budget(self) -> None:
        # Break caught: manufacturing absent winners to reach the fixed global budget.
        result = select_hard_empty_single_shot_antigravity(_snapshot((_candidate("A", 7),)))
        self.assertEqual(len(result.winners), 1)
        self.assertEqual(len(result.selected), 1)
        with self.assertRaises(ValueError):
            validate_production_selection_antigravity(result)

    def test_selector_rejects_duplicate_input_coordinates(self) -> None:
        # Break caught: accepting duplicate coordinates that inflate provisional population counts.
        snapshot = _snapshot((_candidate("A", 7),))
        object.__setattr__(snapshot, "provisional_candidates", (_candidate("A", 7), _candidate("A", 7)))
        with self.assertRaisesRegex(ValueError, "duplicated"):
            select_hard_empty_single_shot_antigravity(snapshot)

    def test_selected_and_hard_coordinate_identities_are_distinct_encodings(self) -> None:
        # Break caught: reusing Exp416's newline-free array identity for Exp441's selected JSONL.
        coordinates = (_candidate("A", 7), _candidate("B", 2))
        selected_payload = b'{"dataset":"A","instance_index":7}\n{"dataset":"B","instance_index":2}\n'
        hard_payload = b'[["A",7],["B",2]]'
        self.assertEqual(
            selected_coordinate_identity_sha256_antigravity(coordinates),
            sha256(selected_payload).hexdigest(),
        )
        self.assertEqual(
            hard_coordinate_identity_sha256_antigravity((candidate.coordinate for candidate in coordinates)),
            sha256(hard_payload).hexdigest(),
        )
        self.assertNotEqual(
            selected_coordinate_identity_sha256_antigravity(coordinates),
            hard_coordinate_identity_sha256_antigravity((candidate.coordinate for candidate in coordinates)),
        )
        with self.assertRaises(ValueError):
            hard_coordinate_identity_sha256_antigravity(json.loads(selected_payload))
        with self.assertRaises(ValueError):
            selected_coordinate_identity_sha256_antigravity(json.loads(hard_payload))

    def test_real_replay_is_label_free_and_matches_frozen_selection_identity(self) -> None:
        # Break caught: opening SQLite/labels or replaying a changed source produces different structure or identity.
        snapshot = verify_frozen_sources_antigravity()
        result = select_hard_empty_single_shot_antigravity(snapshot)
        self.assertEqual(
            (
                len(snapshot.context_datasets),
                len(snapshot.provisional_candidates),
                len(result.outside_hard_candidates),
                len(result.eligible_candidates),
                len(result.winners),
                len(result.selected),
            ),
            (1117, 840, 394, 73, 49, 30),
        )
        self.assertEqual(snapshot.hard_coordinate_identity_sha256,
                         "858e24e6aa182e78985f571778be02bcc87332639958fa887515c24074654322")
        self.assertEqual(hard_coordinate_identity_sha256_antigravity(snapshot.hard_coordinates),
                         "858e24e6aa182e78985f571778be02bcc87332639958fa887515c24074654322")
        self.assertEqual(selected_coordinate_identity_sha256_antigravity(result.selected),
                         "db0b9ed00ed5b9313fef10306d04bb539136c5b28b97ebcd325662d9302a3048")
        self.assertEqual(snapshot.label_access_count, 0)
        selected_payload = b"".join(
            json.dumps(
                {"dataset": item.dataset, "instance_index": item.instance_index},
                ensure_ascii=False, separators=(",", ":"),
            ).encode("utf-8") + b"\n"
            for item in sorted(result.selected, key=lambda value: value.coordinate)
        )
        hard_payload = json.dumps(
            [[dataset, index] for dataset, index in sorted(snapshot.hard_coordinates)],
            ensure_ascii=False, separators=(",", ":"),
        ).encode("utf-8")
        with self.assertRaises(ValueError):
            hard_coordinate_identity_sha256_antigravity(json.loads(selected_payload))
        with self.assertRaises(ValueError):
            selected_coordinate_identity_sha256_antigravity(json.loads(hard_payload))
        selector_source = Path(__import__(
            "experiments.exp441_hard_empty_single_shot_recovery_antigravity.selector_antigravity",
            fromlist=["__file__"],
        ).__file__).read_text()
        self.assertNotIn("sqlite3", selector_source)
        self.assertNotIn("evaluation_antigravity", selector_source)


if __name__ == "__main__":
    unittest.main()
