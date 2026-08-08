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
    terminal_attempt_authority_sha256_from_bytes_antigravity,
    validate_production_selection_antigravity,
)


def _candidate(dataset: str = "A", index: int = 7) -> CandidateEvidenceAntigravity:
    return CandidateEvidenceAntigravity(
        dataset=dataset, instance_index=index, raw_length=32,
        depths=(1, 2), agreement_count=2,
        second_excess_ratio=1.5, second_empirical_rank=0.1,
    )


def _ranked(candidates: tuple[CandidateEvidenceAntigravity, ...], selected_count: int | None = None) -> tuple[RankedCandidateAntigravity, ...]:
    selected_count = len(candidates) if selected_count is None else selected_count
    return tuple(
        RankedCandidateAntigravity(item, index, index <= selected_count, index if index <= selected_count else None)
        for index, item in enumerate(candidates, start=1)
    )


class Exp441ContractsAntigravityTests(unittest.TestCase):
    def test_terminal_authority_identity_is_ordered_complete_and_byte_sensitive(self) -> None:
        """Every one of the six fixed terminal files participates in one deterministic identity."""
        authority = {
            "source_snapshot_antigravity.json": b"source",
            "selection_antigravity.json": b"selection",
            "progress_antigravity.json": b"progress",
            "pid_antigravity": b"pid",
            "events_antigravity.jsonl": b"events\n",
            "text_antigravity.log": b"text\n",
        }
        identity = terminal_attempt_authority_sha256_from_bytes_antigravity(authority)
        self.assertRegex(identity, r"^[0-9a-f]{64}$")
        self.assertEqual(identity, terminal_attempt_authority_sha256_from_bytes_antigravity(dict(authority)))
        changed = dict(authority); changed["pid_antigravity"] = b"different-pid"
        self.assertNotEqual(identity, terminal_attempt_authority_sha256_from_bytes_antigravity(changed))
        with self.assertRaisesRegex(ValueError, "authority"):
            terminal_attempt_authority_sha256_from_bytes_antigravity({"pid_antigravity": b"pid"})

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

    def test_selection_rejects_duplicate_coordinates_in_each_candidate_population(self) -> None:
        """Set comparison must not hide a duplicate that inflates a production population count."""
        item = _candidate()
        ranked = _ranked((item,))[0]
        fixtures = (
            ((item, item), (item,), (item,), (ranked,), (ranked,)),
            ((item,), (item, item), (item,), (ranked,), (ranked,)),
            ((item,), (item,), (item, item), (ranked,), (ranked,)),
        )
        for populations in fixtures:
            with self.subTest(populations=populations[:3]):
                with self.assertRaises(ValueError):
                    SelectionResultAntigravity(*populations, frozenset())

    def test_selection_requires_exact_local_winners_and_global_order(self) -> None:
        """A worse local candidate or globally misordered winner list cannot be ranked as selected."""
        best_a = CandidateEvidenceAntigravity("A", 1, 32, (1, 2), 2, 2.0, 0.1)
        worse_a = CandidateEvidenceAntigravity("A", 2, 32, (1, 2), 2, 1.0, 0.1)
        best_b = CandidateEvidenceAntigravity("B", 1, 32, (1, 2), 2, 3.0, 0.1)
        provisional = (best_a, worse_a, best_b)
        with self.assertRaises(ValueError):
            SelectionResultAntigravity(
                provisional, provisional, provisional, _ranked((worse_a, best_b)), _ranked((worse_a, best_b)), frozenset(),
            )
        with self.assertRaises(ValueError):
            SelectionResultAntigravity(
                provisional, provisional, provisional, _ranked((best_a, best_b)), _ranked((best_a, best_b)), frozenset(),
            )

    def test_backend_environment_is_recursively_detached_and_roundtrips(self) -> None:
        """Mutating a nested caller mapping or list must not mutate backend readiness evidence."""
        environment = {"runtime": {"device": "cpu"}, "versions": ["3.14", {"mlx": None}]}
        backend = BackendAssessmentAntigravity(
            "cpu", 11, False, "not_run_inapplicable", "inapplicable_no_tensor_stage",
            "inapplicable_no_tensor_stage", "inapplicable_no_tensor_stage", environment,
        )
        environment["runtime"]["device"] = "changed"
        environment["versions"][1]["mlx"] = "changed"
        self.assertEqual(backend.environment["runtime"]["device"], "cpu")
        self.assertIsInstance(backend.environment, type(MappingProxyType({})))
        self.assertEqual(BackendAssessmentAntigravity.from_payload(backend.to_payload()), backend)

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
