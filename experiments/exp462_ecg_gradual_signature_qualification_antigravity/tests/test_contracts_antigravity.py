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
    def test_authority_contract_fields_and_immutability_antigravity(self) -> None:
        """Break caught: predecessor authority stops being strict and immutable."""
        from experiments.exp462_ecg_gradual_signature_qualification_antigravity.contracts_antigravity import (
            Exp461GradualAuthorityAntigravity,
            Exp461HistoricalPeriodFoldAntigravity,
        )
        from experiments.exp462_ecg_gradual_signature_qualification_antigravity.predecessor_antigravity import (
            capture_exp461_gradual_authority_antigravity,
        )

        self.assertEqual(
            tuple(field.name for field in fields(Exp461HistoricalPeriodFoldAntigravity)),
            ("seed", "fold_index", "period_change_auc"),
        )
        self.assertEqual(
            tuple(field.name for field in fields(Exp461GradualAuthorityAntigravity)),
            (
                "predecessor_experiment",
                "terminal_commit",
                "artifact_map_sha256",
                "decision_sha256",
                "dashboard_snapshot_sha256",
                "summary_sha256",
                "status",
                "primary_candidate",
                "passed_record_count",
                "required_record_count",
                "failed_gradual_drift_auc",
                "period_min_auc",
                "historical_period_folds",
                "label_query_count",
                "private_sha256",
            ),
        )
        authority = capture_exp461_gradual_authority_antigravity()
        self.assertIsInstance(authority.private_sha256, MappingProxyType)
        with self.assertRaises(TypeError):
            authority.private_sha256["decision_antigravity.json"] = "0" * 64  # type: ignore[index]
        with self.assertRaises(FrozenInstanceError):
            authority.status = "passed"  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            authority.historical_period_folds[0].period_change_auc = 0.1  # type: ignore[misc]

    def test_validator_rejects_snapshot_drift_and_mutable_containers_antigravity(self) -> None:
        """Break caught: copied Exp461 authority accepts changed values or mutable hashes."""
        from experiments.exp462_ecg_gradual_signature_qualification_antigravity.contracts_antigravity import (
            validate_exp461_gradual_authority_antigravity,
        )
        from experiments.exp462_ecg_gradual_signature_qualification_antigravity.predecessor_antigravity import (
            capture_exp461_gradual_authority_antigravity,
        )

        authority = capture_exp461_gradual_authority_antigravity()
        self.assertIs(type(validate_exp461_gradual_authority_antigravity(authority)), type(authority))
        mutations = []
        mutated = _clone_antigravity(authority)
        object.__setattr__(mutated, "terminal_commit", "3a306ff3e276dae13b28b9733db30e9934da7eb2")
        mutations.append(mutated)
        mutated = _clone_antigravity(authority)
        object.__setattr__(mutated, "label_query_count", True)
        mutations.append(mutated)
        mutated = _clone_antigravity(authority)
        object.__setattr__(mutated, "historical_period_folds", list(mutated.historical_period_folds))
        mutations.append(mutated)
        mutated = _clone_antigravity(authority)
        object.__setattr__(mutated, "historical_period_folds", tuple(reversed(mutated.historical_period_folds)))
        mutations.append(mutated)
        mutated = _clone_antigravity(authority)
        mapping = dict(mutated.private_sha256)
        mapping["summary_antigravity.json"] = "0" * 64
        object.__setattr__(mutated, "private_sha256", mapping)
        mutations.append(mutated)
        for value in mutations:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_exp461_gradual_authority_antigravity(value)


if __name__ == "__main__":
    unittest.main()
