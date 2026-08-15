"""Exp449 authorized Smoke A lifecycle tests using temporary typed authorities only."""

from __future__ import annotations

from contextlib import AbstractContextManager, ExitStack
from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from experiments.exp449_reproducible_rebaseline_antigravity import operations_antigravity as operations
from experiments.exp449_reproducible_rebaseline_antigravity.contracts_antigravity import Exp449Context, Exp449TerminalStatus
from experiments.exp449_reproducible_rebaseline_antigravity.execution_authority_antigravity import VerifiedExp449SmokeAAuthorityAntigravity
from experiments.exp449_reproducible_rebaseline_antigravity import contracts_antigravity as contracts
from experiments.exp449_reproducible_rebaseline_antigravity import execution_authority_antigravity as authority
from experiments.exp449_reproducible_rebaseline_antigravity.tests.test_manifest_verifier_antigravity import (
    _TypedProposalFixtureAntigravity,
    _context_antigravity as _typed_context_antigravity,
)
from experiments.exp449_reproducible_rebaseline_antigravity.tests.test_readiness_antigravity import _catalog_antigravity


def _assert_science_equal_antigravity(test: unittest.TestCase, actual: object, expected: object) -> None:
    """Compare every ProposalScore scientific field, excluding wall-clock timing only."""
    if isinstance(actual, np.ndarray) or isinstance(expected, np.ndarray):
        test.assertIsInstance(actual, np.ndarray); test.assertIsInstance(expected, np.ndarray)
        test.assertEqual(actual.dtype, expected.dtype); test.assertEqual(actual.shape, expected.shape)
        test.assertEqual(actual.tobytes(), expected.tobytes()); return
    if is_dataclass(actual) or is_dataclass(expected):
        test.assertTrue(is_dataclass(actual) and is_dataclass(expected)); test.assertEqual(type(actual), type(expected))
        for field in fields(actual):
            if type(actual).__name__ == "ProposalScore" and field.name == "stage_seconds":
                continue
            _assert_science_equal_antigravity(test, getattr(actual, field.name), getattr(expected, field.name))
        return
    if isinstance(actual, Mapping) or isinstance(expected, Mapping):
        test.assertTrue(isinstance(actual, Mapping) and isinstance(expected, Mapping))
        test.assertEqual(tuple(actual), tuple(expected))
        for key in actual: _assert_science_equal_antigravity(test, actual[key], expected[key])
        return
    if isinstance(actual, (tuple, list)) or isinstance(expected, (tuple, list)):
        test.assertEqual(type(actual), type(expected)); test.assertEqual(len(actual), len(expected))
        for left, right in zip(actual, expected, strict=True): _assert_science_equal_antigravity(test, left, right)
        return
    if isinstance(actual, (set, frozenset)) or isinstance(expected, (set, frozenset)):
        test.assertEqual(actual, expected); return
    test.assertEqual(actual, expected)


def _freeze_bytes_antigravity(root: Path) -> tuple[bytes, dict[str, bytes]]:
    freeze = root / "stage_artifacts_antigravity" / "proposal_freeze_antigravity"
    outer = json.loads((freeze / "proposal_manifest_antigravity.json").read_text())
    outer["context_sha256"] = "0" * 64
    for descriptor in outer["task45_checkpoints"]:
        record_path = root / descriptor["relative_path"]
        record = json.loads(record_path.read_text()); record["context_sha256"] = "0" * 64
        if "assessment_sha256" in record.get("backend", {}): record["backend"]["assessment_sha256"] = "0" * 64
        if "record_sha256" in record:
            bare = {key: value for key, value in record.items() if key != "record_sha256"}
            record["record_sha256"] = sha256(json.dumps(bare, sort_keys=True, separators=(",", ":")).encode() + b"\n").hexdigest()
        data = json.dumps(record, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        descriptor["size"] = len(data); descriptor["sha256"] = sha256(data).hexdigest()
    normalized = json.dumps(outer, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    return normalized, {str(path.relative_to(freeze)): path.read_bytes() for path in sorted((freeze / "shards_antigravity").rglob("*_proposal_shard_antigravity.json"))}


class _TemporaryAuthorizedSmokeAAntigravity:
    """Real typed paper recovery in temporary roots; only authority paths and heavy scoring are replaced."""

    def __init__(self) -> None:
        self.directory = TemporaryDirectory(dir=Path.cwd())
        root = Path(self.directory.name)
        self.scientific_root, self.operations_root, self.claim_path = (
            root / "science_antigravity", root / "operations_antigravity", root / "claim_antigravity.lock",
        )
        self.stack = ExitStack(); self.heavy_calls = 0

    def __enter__(self) -> "_TemporaryAuthorizedSmokeAAntigravity":
        from paper.equipment_anomaly_detection.acceleration_paper import build_acceleration_manifest
        from paper.equipment_anomaly_detection.contracts_paper import FrozenTask8BackendDecision
        from paper.equipment_anomaly_detection import proposal_score_paper
        from experiments.exp449_reproducible_rebaseline_antigravity.artifacts_antigravity import context_sha256_antigravity
        from experiments.exp449_reproducible_rebaseline_antigravity.execution_authority_antigravity import (
            Exp449SmokeAGrantAntigravity, VerifiedExp449SmokeAAuthorityAntigravity, canonical_grant_bytes_antigravity,
        )
        temporary_config = SimpleNamespace(**{
            field.name: getattr(operations.FROZEN_EXP449_CONFIG, field.name)
            for field in fields(operations.FROZEN_EXP449_CONFIG)
        })
        temporary_config.smoke_a_scientific_root = self.scientific_root
        temporary_config.smoke_a_operations_root = self.operations_root
        self.stack.enter_context(patch.object(operations, "FROZEN_EXP449_CONFIG", temporary_config))
        self.stack.enter_context(patch.object(operations, "SMOKE_A_CLAIM_PATH_ANTIGRAVITY", self.claim_path))
        self.stack.enter_context(patch.object(authority, "FROZEN_EXP449_CONFIG", temporary_config))
        self.stack.enter_context(patch.object(authority, "SMOKE_A_CLAIM_PATH_ANTIGRAVITY", self.claim_path))
        self.fixture = _TypedProposalFixtureAntigravity(self.scientific_root)
        context_acceleration = build_acceleration_manifest("f" * 64, logical_cpu_count=1)
        backend = FrozenTask8BackendDecision.from_acceleration_manifest(context_acceleration)
        context = _typed_context_antigravity(self.scientific_root, self.fixture.catalog, backend)
        acceleration = build_acceleration_manifest(context_sha256_antigravity(context), logical_cpu_count=1)
        self.config = operations.Exp449WaveConfig(
            context, self.fixture.catalog, self.fixture.score.raw, context_acceleration,
            task45_acceleration_manifest=acceleration,
        )
        grant = Exp449SmokeAGrantAntigravity(
            schema="exp449-smoke-a-execution-grant-antigravity.v1", experiment_id="Exp449", target="smoke-a",
            policy_id=temporary_config.policy_id, implementation_commit="1" * 40,
            scientific_root=self.scientific_root, operations_root=self.operations_root, claim_path=self.claim_path,
            nonce="0123456789abcdef0123456789abcdef", maximum_waves=2, label_access=False,
            logical_cpu_count=authority.EXP449_LOGICAL_CPU_COUNT_ANTIGRAVITY,
            dataset_workers=authority.EXP449_DATASET_WORKERS_ANTIGRAVITY,
            database_path="/fixture/temporary-exp449-antigravity.sqlite", database_size=17, database_sha256="a" * 64,
            smoke_catalog_sha256=self.fixture.catalog["catalog_sha256"], smoke_datasets=temporary_config.smoke_datasets,
            context_sha256=context_sha256_antigravity(context), scientific_code_closure={"fixture.py": "b" * 64},
            authority_code_closure={"authority.py": "c" * 64}, lineage_lock_sha256="d" * 64,
            codemap_lock_sha256="e" * 64, acceleration_assessment_sha256="f" * 64, canonical_backend="cpu",
            package_inventory={"fixture": "1"}, seeds_and_constants={"seed": 449}, acceleration={"selected_backend": "cpu"},
            design_facts=authority._DESIGN_FACTS_ANTIGRAVITY, grant_created_at="2026-08-15T00:00:00Z",
        )
        grant_bytes = canonical_grant_bytes_antigravity(grant)
        self.verified = VerifiedExp449SmokeAAuthorityAntigravity(
            "2" * 40, "1" * 40, grant, sha256(grant_bytes).hexdigest(), grant_bytes, b"fixture digest", (1, 2, 3, 4), (5, 6, 7, 8),
        )
        def score_dataset(values: object):
            self.heavy_calls += 1
            return self.fixture.score.sources[values.dataset], self.fixture.score.routes[values.dataset]
        self.stack.enter_context(patch.object(operations, "readiness_for_authorized_implementation_antigravity", return_value={}))
        self.stack.enter_context(patch.object(operations, "readiness_for_authorized_resume_antigravity", return_value={}))
        self.stack.enter_context(patch.object(proposal_score_paper, "verify_exp84_environment_paper"))
        self.stack.enter_context(patch.object(proposal_score_paper, "verify_pycatch22_environment_paper"))
        self.stack.enter_context(patch.object(proposal_score_paper, "_score_dataset_task4_task5", side_effect=score_dataset))
        return self

    def __exit__(self, *_args: object) -> None:
        self.stack.close(); self.directory.cleanup()

    @property
    def paths(self) -> tuple[Path, Path, Path]:
        return self.scientific_root, self.operations_root, self.claim_path

    def acquire_fresh(self):
        return operations.acquire_smoke_a_execution_lease_antigravity(self.verified, self.config.context, _test_paths_antigravity=self.paths)

    def acquire_resume(self):
        return operations._acquire_smoke_a_resume_lease_antigravity(self.verified, self.config, _test_paths_antigravity=self.paths)


class _ReadinessBoundResumeFixtureAntigravity:
    """Temporary claim/receipt whose unchanged grant is derived from real resume facts."""
    def __enter__(self):
        from paper.equipment_anomaly_detection.acceleration_paper import build_acceleration_manifest
        from paper.equipment_anomaly_detection.contracts_paper import FrozenTask8BackendDecision
        from experiments.exp449_reproducible_rebaseline_antigravity.execution_authority_antigravity import Exp449SmokeAGrantAntigravity, VerifiedExp449SmokeAAuthorityAntigravity, canonical_grant_bytes_antigravity
        self.directory = TemporaryDirectory(dir=Path.cwd()); root = Path(self.directory.name)
        self.roots = tuple(root / f"root_{index}_antigravity" for index in range(6)); self.claim = root / "claim_antigravity.lock"
        self.stack = ExitStack(); base = operations.FROZEN_EXP449_CONFIG
        config = SimpleNamespace(**{field.name: getattr(base, field.name) for field in fields(base)})
        config.smoke_a_scientific_root, config.smoke_a_operations_root = self.roots[:2]
        config.smoke_b_scientific_root, config.smoke_b_operations_root = self.roots[2:4]
        config.official_scientific_root, config.official_operations_root = self.roots[4:]
        for module in (operations, authority, contracts): self.stack.enter_context(patch.object(module, "FROZEN_EXP449_CONFIG", config))
        self.stack.enter_context(patch.object(contracts, "_SMOKE_ROOTS", frozenset({self.roots[0].resolve(), self.roots[2].resolve()})))
        self.stack.enter_context(patch.object(contracts, "_OFFICIAL_ROOTS", frozenset({self.roots[4].resolve()})))
        self.stack.enter_context(patch.object(operations, "SMOKE_A_CLAIM_PATH_ANTIGRAVITY", self.claim)); self.stack.enter_context(patch.object(authority, "SMOKE_A_CLAIM_PATH_ANTIGRAVITY", self.claim))
        self.catalog = _catalog_antigravity("smoke"); implementation, approved = "1" * 40, "2" * 40
        package = {"schema": "fixture", "algorithm": "fixture", "records": [{"project": "fixture", "version": "1"}], "sha256": "f" * 64}
        self.providers = (
            patch.object(operations, "_require_reviewed_head_antigravity", return_value=implementation),
            patch.object(operations, "_require_capable_interpreter_antigravity", return_value={"executable": "fixture-python", "python_version": "3.14.6"}),
            patch.object(operations, "_readiness_catalog_builder_antigravity", side_effect=lambda mode: self.catalog if mode == "smoke" else _catalog_antigravity("full")),
            patch.object(operations, "_dependency_facts_antigravity", return_value={"pycatch22": "0.5.0", "aeon": "1.5.0", "hydra": "constructed", "torch": "2.13.0"}),
            patch.object(operations, "_package_inventory_facts_antigravity", return_value=package),
            patch.object(operations, "_fixed_roots_antigravity", return_value=self.roots),
        )
        for item in self.providers: self.stack.enter_context(item)
        facts = operations._readiness_facts_antigravity(implementation, logical_cpu_count=12)
        smoke = facts["contexts"]["smoke_a"]; acceleration = build_acceleration_manifest(smoke["acceleration_assessment_sha256"], logical_cpu_count=12)
        context = Exp449Context.from_verified_inputs(run_root=self.roots[0], mode="smoke", catalog=self.catalog, acceleration=FrozenTask8BackendDecision.from_acceleration_manifest(acceleration), database_sha256=facts["database"]["sha256"], code_closure=facts["code_closure"], lineage_sha256=facts["lineage_sha256"], codemap_sha256=facts["codemap_sha256"], interpreter=facts["interpreter"]["executable"], environment={"interpreter": facts["interpreter"], "dependencies": facts["dependencies"], "package_inventory": facts["package_inventory"]}, seeds_and_constants=facts["seeds_and_constants"])
        grant = Exp449SmokeAGrantAntigravity(schema="exp449-smoke-a-execution-grant-antigravity.v1", experiment_id="Exp449", target="smoke-a", policy_id=config.policy_id, implementation_commit=implementation, scientific_root=self.roots[0], operations_root=self.roots[1], claim_path=self.claim, nonce="0123456789abcdef0123456789abcdef", maximum_waves=2, label_access=False, logical_cpu_count=12, dataset_workers=11, database_path=facts["database"]["path"], database_size=facts["database"]["observed_size"], database_sha256=facts["database"]["sha256"], smoke_catalog_sha256=smoke["catalog_sha256"], smoke_datasets=config.smoke_datasets, context_sha256=smoke["context_sha256"], scientific_code_closure=facts["code_closure"], authority_code_closure={"experiments/exp449_reproducible_rebaseline_antigravity/execution_authority_antigravity.py": operations._sha256_file_antigravity(operations._REPOSITORY_ROOT_ANTIGRAVITY / "experiments/exp449_reproducible_rebaseline_antigravity/execution_authority_antigravity.py")}, lineage_lock_sha256=facts["lineage_lock_paper_sha256"], codemap_lock_sha256=facts["codemap_lock_sha256"], acceleration_assessment_sha256=smoke["acceleration_assessment_sha256"], canonical_backend="cpu", package_inventory=facts["package_inventory"], seeds_and_constants=facts["seeds_and_constants"], acceleration=facts["acceleration"], design_facts=authority._DESIGN_FACTS_ANTIGRAVITY, grant_created_at="2026-08-15T00:00:00Z")
        data = canonical_grant_bytes_antigravity(grant); self.verified = VerifiedExp449SmokeAAuthorityAntigravity(approved, implementation, grant, sha256(data).hexdigest(), data, b"digest", (1,2,3,4), (5,6,7,8)); self.context = context; self.config = operations.Exp449WaveConfig(context, self.catalog, {name: object() for name in config.smoke_datasets}, acceleration)
        self.stack.enter_context(patch.object(operations, "verify_smoke_a_authorization_commit_antigravity", return_value=self.verified))
        with operations.acquire_smoke_a_execution_lease_antigravity(self.verified, context, _test_paths_antigravity=(self.roots[0], self.roots[1], self.claim)): pass
        self.roots[0].mkdir(); return self
    def __exit__(self, *_): self.stack.close(); self.directory.cleanup()
    @property
    def paths(self): return self.roots[0], self.roots[1], self.claim


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

    def test_resume_rejects_a_complete_terminal_without_a_verified_freeze(self) -> None:
        """Returning a complete terminal without a freeze would trust an unverified publication."""
        with (
            patch.object(operations, "_authorized_smoke_a_wave_config_antigravity", return_value=self.config),
            patch.object(operations, "_acquire_smoke_a_resume_lease_antigravity", return_value=_LeaseAntigravity()),
            patch.object(operations, "_existing_terminal_status_antigravity", return_value=self.complete),
            patch.object(operations, "_freeze_exists_antigravity", return_value=False),
        ):
            with self.assertRaisesRegex(ValueError, "complete terminal lacks a verified freeze"):
                operations.execute_smoke_a_resume_antigravity(self.verified)

    def test_real_temporary_direct_wave_reuses_success_checkpoints_without_rescoring(self) -> None:
        """A completed Task4/5 namespace must reconstruct Task6/7 from typed checkpoints without heavy scoring."""
        with _TemporaryAuthorizedSmokeAAntigravity() as fixture:
            with fixture.acquire_fresh() as lease:
                first = operations.run_exp449_wave_antigravity(
                    fixture.config, _execution_capability=lease.capability, _execution_lease=lease,
                )
                self.assertEqual(first.status.status, "dataset_local_complete")
                self.assertIsNotNone(first.score)
                self.assertEqual(fixture.heavy_calls, len(fixture.config.raw_datasets))
            before = fixture.heavy_calls
            with fixture.acquire_resume() as lease:
                recovered = operations.run_exp449_wave_antigravity(
                    fixture.config, _execution_capability=lease.capability, _execution_lease=lease,
                )
            self.assertEqual(recovered.status.status, "dataset_local_complete")
            self.assertIsNotNone(recovered.score)
            _assert_science_equal_antigravity(self, recovered.score, first.score)
            self.assertEqual(fixture.heavy_calls, before)

    def test_real_task45_recovery_freezes_with_the_separate_bridge_manifest(self) -> None:
        """The bridge manifest binds checkpoints while the outer backend remains freeze-authoritative."""
        with _TemporaryAuthorizedSmokeAAntigravity() as fixture:
            with fixture.acquire_fresh() as lease:
                result = operations.run_exp449_wave_antigravity(
                    fixture.config, _execution_capability=lease.capability, _execution_lease=lease,
                )
                status = operations._finalize_authorized_complete_antigravity(fixture.config, result, lease)
            self.assertEqual(status.status, "dataset_local_complete")

    def test_real_direct_fresh_and_resume_are_scientifically_byte_identical(self) -> None:
        """Lifecycle authorization changes only root-bound authority, never ProposalScore science or shard bytes."""
        scores: list[object] = []; artifacts: list[tuple[bytes, dict[str, bytes]]] = []
        with _TemporaryAuthorizedSmokeAAntigravity() as direct:
            with direct.acquire_fresh() as lease:
                result = operations.run_exp449_wave_antigravity(direct.config, _execution_capability=lease.capability, _execution_lease=lease)
                scores.append(result.score); operations._finalize_authorized_complete_antigravity(direct.config, result, lease)
            artifacts.append(_freeze_bytes_antigravity(direct.scientific_root))
        with _TemporaryAuthorizedSmokeAAntigravity() as fresh:
            original_acquire = operations.acquire_smoke_a_execution_lease_antigravity
            original_freeze = operations.freeze_authorized_exp449_proposal_antigravity; captured: list[object] = []
            def acquire(verified: object, context: object): return original_acquire(verified, context, _test_paths_antigravity=fresh.paths)
            def freeze(score: object, **kwargs: object): captured.append(score); return original_freeze(score, **kwargs)
            with patch.object(operations, "_authorized_smoke_a_wave_config_antigravity", return_value=fresh.config), patch.object(operations, "acquire_smoke_a_execution_lease_antigravity", side_effect=acquire), patch.object(operations, "freeze_authorized_exp449_proposal_antigravity", side_effect=freeze):
                self.assertEqual(operations.execute_smoke_a_fresh_antigravity(fresh.verified).status, "dataset_local_complete")
            scores.append(captured.pop()); artifacts.append(_freeze_bytes_antigravity(fresh.scientific_root))
        with _TemporaryAuthorizedSmokeAAntigravity() as resumed:
            with resumed.acquire_fresh() as lease:
                result = operations.run_exp449_wave_antigravity(resumed.config, _execution_capability=lease.capability, _execution_lease=lease)
                scores.append(result.score)
                operations.freeze_authorized_exp449_proposal_antigravity(result.score, config=resumed.config, _execution_capability=lease.capability)
            before = resumed.heavy_calls; original_resume = operations._acquire_smoke_a_resume_lease_antigravity
            def acquire_resume(verified: object, config: object): return original_resume(verified, config, _test_paths_antigravity=resumed.paths)
            with patch.object(operations, "_authorized_smoke_a_wave_config_antigravity", return_value=resumed.config), patch.object(operations, "_acquire_smoke_a_resume_lease_antigravity", side_effect=acquire_resume):
                self.assertEqual(operations.execute_smoke_a_resume_antigravity(resumed.verified).status, "dataset_local_complete")
            self.assertEqual(resumed.heavy_calls, before); artifacts.append(_freeze_bytes_antigravity(resumed.scientific_root))
        for score in scores[1:]: _assert_science_equal_antigravity(self, score, scores[0])
        for artifact in artifacts[1:]: self.assertEqual(artifact, artifacts[0])

    def test_real_resume_acquisition_rejects_authority_and_claim_tampering(self) -> None:
        """A resumed lease must rebind its exact consumed claim to the verified context authority."""
        with _TemporaryAuthorizedSmokeAAntigravity() as fixture:
            with fixture.acquire_fresh() as lease:
                operations.run_exp449_wave_antigravity(
                    fixture.config, _execution_capability=lease.capability, _execution_lease=lease,
                )
            for field, changed in (
                ("database_sha256", "b" * 64),
                ("scientific_code_closure", {"fixture.py": "c" * 64}),
                ("package_inventory", {"fixture": "2"}),
                ("acceleration", {"selected_backend": "cpu", "worker_fixture": 2}),
            ):
                with self.subTest(field=field):
                    grant = replace(fixture.verified.grant, **{field: changed})
                    altered = replace(fixture.verified, grant=grant, grant_sha256="e" * 64)
                    with self.assertRaisesRegex(ValueError, "resume claim differs"):
                        operations._acquire_smoke_a_resume_lease_antigravity(altered, fixture.config, _test_paths_antigravity=fixture.paths)
            original = fixture.claim_path.read_bytes(); fixture.claim_path.write_bytes(original + b" ")
            with self.assertRaisesRegex(ValueError, "(resume claim differs|persisted authority record is noncanonical)"):
                fixture.acquire_resume()

    def test_task45_bridge_rejects_worker_decision_drift(self) -> None:
        """The distinct assessment binding cannot change CPU selection, fallback, or worker policy."""
        from paper.equipment_anomaly_detection.acceleration_paper import build_acceleration_manifest
        with _TemporaryAuthorizedSmokeAAntigravity() as fixture:
            drift = build_acceleration_manifest("a" * 64, logical_cpu_count=2)
            with self.assertRaisesRegex(ValueError, "bridge decision differs"):
                operations.Exp449WaveConfig(
                    fixture.config.context, fixture.config.catalog, fixture.config.raw_datasets,
                    fixture.config.acceleration_manifest, task45_acceleration_manifest=drift,
                )

    def test_verified_checkpoint_and_attempt_tampering_block_snapshot_before_resume_scoring(self) -> None:
        """Resume snapshot verification must reject persisted Task4/5 semantic evidence before composition."""
        from paper.equipment_anomaly_detection.artifacts_paper import append_task4_task5_attempt_record
        with _TemporaryAuthorizedSmokeAAntigravity() as fixture:
            with fixture.acquire_fresh() as lease:
                result = operations.run_exp449_wave_antigravity(fixture.config, _execution_capability=lease.capability, _execution_lease=lease)
            success = next((fixture.scientific_root / "stage_artifacts_paper" / "proposal_work" / "task4_task5_success").rglob("task4_task5_success_antigravity.json"))
            success.write_bytes(success.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "canonical|success"):
                operations._snapshot_antigravity(fixture.config)
        with _TemporaryAuthorizedSmokeAAntigravity() as fixture:
            with fixture.acquire_fresh() as lease:
                result = operations.run_exp449_wave_antigravity(fixture.config, _execution_capability=lease.capability, _execution_lease=lease)
            _context, task45 = operations._authority_antigravity(fixture.config)
            append_task4_task5_attempt_record(fixture.scientific_root, fixture.config.catalog, task45, next(iter(result.score.raw.values())), 1, RuntimeError("fixture"))
            attempt = next((fixture.scientific_root / "stage_artifacts_paper" / "proposal_work" / "task4_task5_attempts").iterdir())
            attempt.write_bytes(attempt.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "canonical|attempt"):
                operations._snapshot_antigravity(fixture.config)

    def test_real_resume_readiness_and_lease_reject_current_fact_and_receipt_drift(self) -> None:
        """Unchanged grant/claim evidence must reject drift before scorer or freeze entry."""
        with _ReadinessBoundResumeFixtureAntigravity() as fixture:
            self.assertEqual(operations.readiness_for_authorized_resume_antigravity(fixture.verified)["reviewed_commit"], fixture.verified.implementation_commit)
            with patch.object(operations, "_dependency_facts_antigravity", return_value={"pycatch22": "0.5.0", "aeon": "1.5.0", "hydra": "constructed", "torch": "drift"}):
                with self.assertRaisesRegex(ValueError, "dependency facts"):
                    operations.readiness_for_authorized_resume_antigravity(fixture.verified)
            with patch.object(operations, "_package_inventory_facts_antigravity", return_value={"schema": "fixture", "algorithm": "fixture", "records": [{"project": "fixture", "version": "drift"}], "sha256": "f" * 64}):
                with self.assertRaisesRegex(ValueError, "readiness facts differ"):
                    operations.readiness_for_authorized_resume_antigravity(fixture.verified)
            closure = operations._closure_facts_antigravity(); closure["code_closure"] = {**closure["code_closure"], "fixture_drift.py": "a" * 64}
            with patch.object(operations, "_closure_facts_antigravity", return_value=closure):
                with self.assertRaisesRegex(ValueError, "readiness facts differ|context facts differ"):
                    operations.readiness_for_authorized_resume_antigravity(fixture.verified)
            altered_catalog = dict(fixture.catalog); altered_catalog["database"] = {**altered_catalog["database"], "sha256": "0" * 64}
            with patch.object(operations, "_readiness_catalog_builder_antigravity", side_effect=lambda mode: altered_catalog if mode == "smoke" else _catalog_antigravity("full")):
                with self.assertRaisesRegex(ValueError, "catalog database"):
                    operations.readiness_for_authorized_resume_antigravity(fixture.verified)
            original_contexts = operations._readiness_contexts_antigravity
            def worker_drift(**kwargs: object):
                contexts, acceleration = original_contexts(**kwargs)
                changed = dict(acceleration); changed["workers"] = {"logical_cpu_count": 13, "dataset_workers": 12, "inner_numerical_threads": 1, "outer_execution_mode": "parallel_task4_task5"}
                return contexts, changed
            with patch.object(operations, "_readiness_contexts_antigravity", side_effect=worker_drift):
                with self.assertRaisesRegex(ValueError, "readiness facts differ"):
                    operations.readiness_for_authorized_resume_antigravity(fixture.verified)
            receipt = fixture.roots[1] / "execution_authority_consumed_antigravity.json"; receipt.write_bytes(receipt.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "noncanonical|receipt differs"):
                operations._acquire_smoke_a_resume_lease_antigravity(fixture.verified, fixture.config, _test_paths_antigravity=fixture.paths)


if __name__ == "__main__":
    unittest.main()
