from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from types import MappingProxyType
import unittest


def _clone_antigravity(value: object) -> object:
    cloned = object.__new__(type(value))
    for field in fields(type(value)):
        object.__setattr__(cloned, field.name, getattr(value, field.name))
    return cloned


class ContractsAntigravityTest(unittest.TestCase):
    def test_contract_fields_and_immutability_antigravity(self) -> None:
        from experiments.exp461_ecg_period_signature_qualification_antigravity.contracts_antigravity import (
            Exp460FailedFoldAntigravity,
            Exp460PeriodAuthorityAntigravity,
        )
        from experiments.exp461_ecg_period_signature_qualification_antigravity.predecessor_antigravity import (
            capture_exp460_period_authority_antigravity,
        )

        self.assertEqual(
            tuple(field.name for field in fields(Exp460FailedFoldAntigravity)),
            ("seed", "fold_index", "period_change_auc"),
        )
        self.assertEqual(
            tuple(field.name for field in fields(Exp460PeriodAuthorityAntigravity)),
            (
                "predecessor_experiment",
                "truth_commit",
                "run_root",
                "reviewed_commit",
                "artifact_map_sha256",
                "dashboard_snapshot_sha256",
                "decision_sha256",
                "status",
                "strict_route",
                "joint_success",
                "passed_record_count",
                "failed_fold_count",
                "worst_period_change_auc",
                "failed_folds",
                "label_query_count",
                "leaf_sha256",
            ),
        )

        authority = capture_exp460_period_authority_antigravity()
        self.assertIsInstance(authority.leaf_sha256, MappingProxyType)
        with self.assertRaises(TypeError):
            authority.leaf_sha256["decision_antigravity.json"] = "0" * 64  # type: ignore[index]
        with self.assertRaises(FrozenInstanceError):
            authority.status = "stable"  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            authority.failed_folds[0].period_change_auc = 0.85  # type: ignore[misc]

    def test_validator_rejects_snapshot_drift_and_mutable_containers_antigravity(self) -> None:
        from experiments.exp461_ecg_period_signature_qualification_antigravity.contracts_antigravity import (
            validate_exp460_period_authority_antigravity,
        )
        from experiments.exp461_ecg_period_signature_qualification_antigravity.predecessor_antigravity import (
            capture_exp460_period_authority_antigravity,
        )

        authority = capture_exp460_period_authority_antigravity()
        self.assertIs(
            type(validate_exp460_period_authority_antigravity(authority)),
            type(authority),
        )

        def mutate_failed_auc_antigravity() -> object:
            mutated = _clone_antigravity(authority)
            rows = list(mutated.failed_folds)
            row = _clone_antigravity(rows[1])
            object.__setattr__(row, "period_change_auc", 0.8004124541019064)
            rows[1] = row
            object.__setattr__(mutated, "failed_folds", tuple(rows))
            return mutated

        def mutate_wrong_route_antigravity() -> object:
            mutated = _clone_antigravity(authority)
            object.__setattr__(mutated, "strict_route", "candidate")
            return mutated

        def mutate_wrong_hash_antigravity() -> object:
            mutated = _clone_antigravity(authority)
            mapping = dict(mutated.leaf_sha256)
            mapping["dashboard_snapshot_antigravity.json"] = "0" * 64
            object.__setattr__(mutated, "leaf_sha256", mapping)
            return mutated

        def mutate_wrong_count_antigravity() -> object:
            mutated = _clone_antigravity(authority)
            object.__setattr__(mutated, "passed_record_count", 9)
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
                    object.__setattr__(mutated, "failed_folds", list(mutated.failed_folds)),
                    mutated,
                )[-1]
            )(_clone_antigravity(authority)),
            lambda: (
                lambda mutated: (
                    object.__setattr__(
                        mutated,
                        "failed_folds",
                        tuple(reversed(mutated.failed_folds)),
                    ),
                    mutated,
                )[-1]
            )(_clone_antigravity(authority)),
            mutate_failed_auc_antigravity,
            mutate_wrong_route_antigravity,
            mutate_wrong_hash_antigravity,
            mutate_wrong_count_antigravity,
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                validate_exp460_period_authority_antigravity(mutate())


if __name__ == "__main__":
    unittest.main()
