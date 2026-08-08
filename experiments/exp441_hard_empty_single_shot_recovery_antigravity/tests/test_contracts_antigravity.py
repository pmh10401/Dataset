"""Immutable value-object behavior for Exp441."""
from __future__ import annotations

import math
from types import MappingProxyType
import unittest

from experiments.exp441_hard_empty_single_shot_recovery_antigravity.contracts_antigravity import (
    CandidateEvidenceAntigravity,
    ParsedRecordAntigravity,
    RankedCandidateAntigravity,
    ReadinessBindingAntigravity,
    SelectionResultAntigravity,
    BackendAssessmentAntigravity,
    validate_production_selection_antigravity,
)


def _candidate(dataset: str = "A", index: int = 7) -> CandidateEvidenceAntigravity:
    return CandidateEvidenceAntigravity(
        dataset=dataset, instance_index=index, raw_length=32,
        depths=(1, 2), agreement_count=2,
        second_excess_ratio=1.5, second_empirical_rank=0.1,
    )


class Exp441ContractsAntigravityTests(unittest.TestCase):
    def test_candidate_roundtrip_and_frozen_rank_keys(self) -> None:
        """A ranking-key mutation or payload drift must be observable at the contract boundary."""
        candidate = _candidate()
        self.assertEqual(candidate.dataset_key, (-1.5, 0.1, 7))
        self.assertEqual(candidate.global_key, (-1.5, 0.1, "A", 7))
        self.assertEqual(CandidateEvidenceAntigravity.from_payload(candidate.to_payload()), candidate)

    def test_candidate_rejects_invalid_evidence_and_unknown_payload_key(self) -> None:
        """A nonfinite score, weak agreement, or ambiguous depth evidence cannot enter selection."""
        invalid = [
            {"agreement_count": 1},
            {"depths": (2, 1)},
            {"depths": (1, 1)},
            {"second_excess_ratio": math.inf},
            {"second_empirical_rank": math.nan},
        ]
        for replacement in invalid:
            raw = _candidate().to_payload()
            raw.update(replacement)
            with self.subTest(replacement=replacement):
                with self.assertRaises(ValueError):
                    CandidateEvidenceAntigravity.from_payload(raw)
        raw = _candidate().to_payload()
        raw["unexpected"] = 1
        with self.assertRaises(ValueError):
            CandidateEvidenceAntigravity.from_payload(raw)

    def test_records_and_selection_copy_mutable_caller_containers(self) -> None:
        """Post-construction caller mutation must not alter frozen source or selection evidence."""
        candidates = [_candidate()]
        record = ParsedRecordAntigravity("A", 32, candidates)
        candidates.append(_candidate("A", 8))
        self.assertEqual(record.candidates, (_candidate(),))
        ranked = RankedCandidateAntigravity(_candidate(), global_rank=1, selected=True, selection_rank=1)
        selected = [ranked]
        result = SelectionResultAntigravity(
            provisional_candidates=(_candidate(),), outside_hard_candidates=(_candidate(),),
            eligible_candidates=(_candidate(),), winners=(ranked,), selected=selected,
            hard_coordinates=frozenset(),
        )
        selected.clear()
        self.assertEqual(result.selected, (ranked,))
        self.assertIsInstance(MappingProxyType({}), type(MappingProxyType({})))

    def test_selection_rejects_hard_active_duplicate_or_over_budget_candidates(self) -> None:
        """A selector bug that leaks a Hard-active, duplicate, or 31st coordinate must fail closed."""
        hard_candidate = _candidate("A", 7)
        ranked = RankedCandidateAntigravity(hard_candidate, 1, True, 1)
        with self.assertRaises(ValueError):
            SelectionResultAntigravity(
                (hard_candidate,), (), (), (ranked,), (ranked,), frozenset({("A", 1)}),
            )
        base = _candidate("B", 1)
        duplicate = RankedCandidateAntigravity(base, 1, True, 1)
        with self.assertRaises(ValueError):
            SelectionResultAntigravity((base,), (base,), (base,), (duplicate,), (duplicate, duplicate), frozenset())
        ranked_many = tuple(
            RankedCandidateAntigravity(_candidate(f"D{number}", number), number + 1, True, number + 1)
            for number in range(31)
        )
        candidates_many = tuple(item.candidate for item in ranked_many)
        with self.assertRaises(ValueError):
            SelectionResultAntigravity(candidates_many, candidates_many, candidates_many, ranked_many, ranked_many, frozenset())

    def test_production_validator_requires_exact_structure(self) -> None:
        """A sparse fixture must not be mistaken for the fixed 840/394/73/49/30 run."""
        item = _candidate()
        ranked = RankedCandidateAntigravity(item, 1, True, 1)
        sparse = SelectionResultAntigravity((item,), (item,), (item,), (ranked,), (ranked,), frozenset())
        with self.assertRaises(ValueError):
            validate_production_selection_antigravity(sparse)

    def test_readiness_binding_uses_a_clean_git_commit_identity(self) -> None:
        """Rejecting a valid 40-character HEAD would block the fixed readiness lifecycle."""
        backend = BackendAssessmentAntigravity(
            "cpu", 11, False, "not_run_inapplicable", "inapplicable_no_tensor_stage",
            "inapplicable_no_tensor_stage", "inapplicable_no_tensor_stage", {},
        )
        binding = ReadinessBindingAntigravity("a" * 40, "b" * 64, backend, "c" * 64)
        self.assertEqual(binding.reviewed_commit, "a" * 40)
