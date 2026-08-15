"""Exp449 authorized Smoke A lifecycle tests using temporary typed authorities only."""

from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.exp449_reproducible_rebaseline_antigravity import operations_antigravity as operations
from experiments.exp449_reproducible_rebaseline_antigravity.contracts_antigravity import Exp449Context, Exp449TerminalStatus
from experiments.exp449_reproducible_rebaseline_antigravity.execution_authority_antigravity import VerifiedExp449SmokeAAuthorityAntigravity


def _context_antigravity() -> Exp449Context:
    context = object.__new__(Exp449Context)
    for name, value in {
        "experiment_id": "Exp449", "mode": "smoke",
        "run_root": operations.FROZEN_EXP449_CONFIG.smoke_a_scientific_root,
        "database_sha256": "a" * 64, "catalog_sha256": "b" * 64,
        "code_closure": {"fixture_antigravity.py": "c" * 64},
        "lineage_sha256": "d" * 64, "codemap_sha256": "e" * 64,
        "interpreter": "typed-fixture-python", "environment": {"fixture": "typed"},
        "backend_record": {"selected_backend": "cpu"}, "seeds_and_constants": {"seed": 449},
        "historical_parity_status": "not_used",
    }.items():
        object.__setattr__(context, name, value)
    return context


def _verified_antigravity() -> VerifiedExp449SmokeAAuthorityAntigravity:
    verified = object.__new__(VerifiedExp449SmokeAAuthorityAntigravity)
    grant = SimpleNamespace(
        target="smoke-a", scientific_root=operations.FROZEN_EXP449_CONFIG.smoke_a_scientific_root,
        operations_root=operations.FROZEN_EXP449_CONFIG.smoke_a_operations_root,
        claim_path=operations.SMOKE_A_CLAIM_PATH_ANTIGRAVITY,
    )
    for name, value in {
        "approved_authority_commit": "a" * 40, "implementation_commit": "b" * 40,
        "grant": grant, "grant_sha256": "c" * 64, "grant_bytes": b"fixture",
        "digest_bytes": b"fixture", "grant_stat_identity": (1, 2, 3, 4),
        "digest_stat_identity": (5, 6, 7, 8),
    }.items():
        object.__setattr__(verified, name, value)
    return verified


class _LeaseAntigravity(AbstractContextManager):
    capability = object()

    def __exit__(self, *_args: object) -> None:
        return None


class Exp449ExecutionLifecycleTests(unittest.TestCase):
    """Mutation-order tests keep production roots, grants, DBs, and labels absent."""

    def setUp(self) -> None:
        context = _context_antigravity()
        self.config = operations.Exp449WaveConfig(
            context=context,
            catalog={"catalog_sha256": "b" * 64, "datasets": [{"dataset": name} for name in operations.FROZEN_EXP449_CONFIG.smoke_datasets]},
            raw_datasets={name: object() for name in operations.FROZEN_EXP449_CONFIG.smoke_datasets},
            acceleration_manifest={"fixture": "typed"},
        )
        self.verified = _verified_antigravity()
        self.complete = Exp449TerminalStatus("smoke", "dataset_local_complete", 23, 0, 0, 1)
        self.incomplete_one = Exp449TerminalStatus("smoke", "proposal_score_incomplete", 22, 1, 0, 1)
        self.incomplete_two = Exp449TerminalStatus("smoke", "proposal_score_incomplete", 22, 1, 0, 2)

    def test_fresh_complete_freezes_verifies_then_publishes_terminal(self) -> None:
        """Publishing complete before scorer-free verification would authorize an unverified freeze."""
        calls: list[str] = []
        score = object()
        with (
            patch.object(operations, "_authorized_smoke_a_wave_config_antigravity", return_value=self.config),
            patch.object(operations, "acquire_smoke_a_execution_lease_antigravity", return_value=_LeaseAntigravity()),
            patch.object(operations, "run_exp449_wave_antigravity", return_value=operations.Exp449AuthorizedWaveResultAntigravity(self.complete, score)),
            patch.object(operations, "freeze_authorized_exp449_proposal_antigravity", side_effect=lambda *_a, **_k: calls.append("freeze")),
            patch("experiments.exp449_reproducible_rebaseline_antigravity.verifier_antigravity.verify_exp449_proposal_antigravity", side_effect=lambda *_a, **_k: calls.append("verify")),
            patch.object(operations, "_snapshot_antigravity", return_value=object()),
            patch.object(operations, "_publish_terminal_complete_antigravity", side_effect=lambda *_a, **_k: calls.append("terminal") or self.complete),
            patch.object(operations, "refresh_derived_status_antigravity"),
        ):
            status = operations.execute_smoke_a_fresh_antigravity(self.verified)
        self.assertEqual(status, self.complete)
        self.assertEqual(calls, ["freeze", "verify", "terminal"])

    def test_fresh_second_wave_incomplete_never_calls_freeze_or_terminal_complete(self) -> None:
        """Running Task6/7 or freeze after the final failed wave would consume incomplete science."""
        waves = [
            operations.Exp449AuthorizedWaveResultAntigravity(self.incomplete_one, None),
            operations.Exp449AuthorizedWaveResultAntigravity(self.incomplete_two, None),
        ]
        with (
            patch.object(operations, "_authorized_smoke_a_wave_config_antigravity", return_value=self.config),
            patch.object(operations, "acquire_smoke_a_execution_lease_antigravity", return_value=_LeaseAntigravity()),
            patch.object(operations, "run_exp449_wave_antigravity", side_effect=waves),
            patch.object(operations, "freeze_authorized_exp449_proposal_antigravity", side_effect=AssertionError("freeze")),
            patch.object(operations, "_publish_terminal_complete_antigravity", side_effect=AssertionError("terminal complete")),
        ):
            status = operations.execute_smoke_a_fresh_antigravity(self.verified)
        self.assertEqual(status, self.incomplete_two)

    def test_resume_after_verified_freeze_is_verifier_only_terminal_recovery(self) -> None:
        """A frozen complete root must recover its terminal without another dataset scorer call."""
        with (
            patch.object(operations, "_authorized_smoke_a_wave_config_antigravity", return_value=self.config),
            patch.object(operations, "_acquire_smoke_a_resume_lease_antigravity", return_value=_LeaseAntigravity()),
            patch.object(operations, "_existing_terminal_status_antigravity", return_value=None),
            patch.object(operations, "_freeze_exists_antigravity", return_value=True),
            patch.object(operations, "_snapshot_antigravity", return_value=object()),
            patch.object(operations, "run_exp449_wave_antigravity", side_effect=AssertionError("dataset scorer")),
            patch.object(operations, "_publish_terminal_complete_antigravity", return_value=self.complete),
        ):
            status = operations.execute_smoke_a_resume_antigravity(self.verified)
        self.assertEqual(status, self.complete)


if __name__ == "__main__":
    unittest.main()
