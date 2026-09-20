from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
import unittest


def _clone_antigravity(value: object) -> object:
    cloned = object.__new__(type(value))
    for field in fields(type(value)):
        object.__setattr__(cloned, field.name, getattr(value, field.name))
    return cloned


class ContractsAntigravityTest(unittest.TestCase):
    def test_contract_fields_and_immutability_antigravity(self) -> None:
        from experiments.exp460_ecg_strict_fallback_antigravity.contracts_antigravity import (
            ConfusionAntigravity,
            Exp459DecisionAntigravity,
            Exp459FallbackAuthorityAntigravity,
            Exp459FoldRecordAntigravity,
            Exp459SeedSummaryAntigravity,
        )
        from experiments.exp460_ecg_strict_fallback_antigravity.predecessor_antigravity import (
            capture_exp459_fallback_authority_antigravity,
        )

        self.assertEqual(
            tuple(field.name for field in fields(ConfusionAntigravity)),
            ("tp", "fp", "fn", "tn"),
        )
        self.assertEqual(
            tuple(field.name for field in fields(Exp459FoldRecordAntigravity)),
            ("seed", "fold_index", "period_change_auc", "passed"),
        )
        self.assertEqual(
            tuple(field.name for field in fields(Exp459SeedSummaryAntigravity)),
            ("seed", "fold_records", "passed"),
        )
        self.assertEqual(
            tuple(field.name for field in fields(Exp459DecisionAntigravity)),
            ("status", "passed"),
        )
        self.assertEqual(
            tuple(field.name for field in fields(Exp459FallbackAuthorityAntigravity)),
            (
                "fold_records",
                "seed_summaries",
                "decision",
                "baseline_confusion",
                "candidate_confusion",
                "frozen_authority_digest",
                "official_artifact_map_sha256",
                "decision_sha256",
                "label_query_count",
            ),
        )

        authority = capture_exp459_fallback_authority_antigravity()
        with self.assertRaises(FrozenInstanceError):
            authority.decision = authority.decision  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            authority.fold_records[0].passed = False  # type: ignore[misc]

    def test_validator_rejects_bool_as_int_container_and_semantic_drift_antigravity(
        self,
    ) -> None:
        from experiments.exp460_ecg_strict_fallback_antigravity.contracts_antigravity import (
            validate_exp459_fallback_authority_antigravity,
        )
        from experiments.exp460_ecg_strict_fallback_antigravity.predecessor_antigravity import (
            capture_exp459_fallback_authority_antigravity,
        )

        authority = capture_exp459_fallback_authority_antigravity()
        self.assertIs(
            type(validate_exp459_fallback_authority_antigravity(authority)),
            type(authority),
        )

        def mutate_record_auc_antigravity() -> object:
            mutated = _clone_antigravity(authority)
            records = list(mutated.fold_records)
            target = _clone_antigravity(records[3])
            object.__setattr__(target, "period_change_auc", 0.8500000000000001)
            records[3] = target
            object.__setattr__(mutated, "fold_records", tuple(records))
            return mutated

        def mutate_record_pass_antigravity() -> object:
            mutated = _clone_antigravity(authority)
            records = list(mutated.fold_records)
            target = _clone_antigravity(records[4])
            object.__setattr__(target, "passed", False)
            records[4] = target
            object.__setattr__(mutated, "fold_records", tuple(records))
            return mutated

        def mutate_summary_antigravity() -> object:
            mutated = _clone_antigravity(authority)
            summaries = list(mutated.seed_summaries)
            target = _clone_antigravity(summaries[0])
            object.__setattr__(target, "passed", True)
            summaries[0] = target
            object.__setattr__(mutated, "seed_summaries", tuple(summaries))
            return mutated

        def mutate_decision_antigravity() -> object:
            mutated = _clone_antigravity(authority)
            decision = _clone_antigravity(mutated.decision)
            object.__setattr__(decision, "status", "stable")
            object.__setattr__(mutated, "decision", decision)
            return mutated

        def mutate_confusion_antigravity() -> object:
            mutated = _clone_antigravity(authority)
            confusion = _clone_antigravity(mutated.baseline_confusion)
            object.__setattr__(confusion, "tp", 610)
            object.__setattr__(mutated, "baseline_confusion", confusion)
            return mutated

        def mutate_hash_antigravity() -> object:
            mutated = _clone_antigravity(authority)
            object.__setattr__(mutated, "decision_sha256", "0" * 64)
            return mutated

        mutations = (
            lambda: (
                lambda mutated: (
                    object.__setattr__(mutated, "label_query_count", True),
                    mutated,
                )[-1]
            )(_clone_antigravity(authority)),
            lambda: (
                lambda mutated: (
                    object.__setattr__(mutated, "fold_records", list(mutated.fold_records)),
                    mutated,
                )[-1]
            )(_clone_antigravity(authority)),
            lambda: (
                lambda mutated: (
                    object.__setattr__(mutated, "fold_records", mutated.fold_records[1:]),
                    mutated,
                )[-1]
            )(_clone_antigravity(authority)),
            lambda: (
                lambda mutated: (
                    object.__setattr__(
                        mutated,
                        "fold_records",
                        mutated.fold_records[1:] + mutated.fold_records[:1],
                    ),
                    mutated,
                )[-1]
            )(_clone_antigravity(authority)),
            lambda: (
                lambda mutated: (
                    object.__setattr__(
                        mutated,
                        "fold_records",
                        mutated.fold_records[:-1] + (mutated.fold_records[-2],),
                    ),
                    mutated,
                )[-1]
            )(_clone_antigravity(authority)),
            mutate_record_auc_antigravity,
            mutate_record_pass_antigravity,
            mutate_summary_antigravity,
            mutate_decision_antigravity,
            mutate_confusion_antigravity,
            mutate_hash_antigravity,
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                validate_exp459_fallback_authority_antigravity(mutate())


if __name__ == "__main__":
    unittest.main()
