"""Exp449 in-memory Task4/5 completion gate tests."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
import json
from itertools import count
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from experiments.exp449_reproducible_rebaseline_antigravity import runner_antigravity as runner
from experiments.exp449_reproducible_rebaseline_antigravity import operations_antigravity as operations
from experiments.exp449_reproducible_rebaseline_antigravity.config_antigravity import FROZEN_EXP449_CONFIG
from experiments.exp449_reproducible_rebaseline_antigravity.contracts_antigravity import Exp449Context
from paper.equipment_anomaly_detection import proposal_score_paper
from paper.equipment_anomaly_detection.contracts_paper import LowFrequencyResult, RawDataset
from paper.equipment_anomaly_detection.hard_route_paper import HardRouteResult, RouteLanes
from paper.equipment_anomaly_detection.routing_paper import PolicyBundle
from paper.equipment_anomaly_detection.sources_paper.hard_sources_paper import HardSourceBundle
from paper.equipment_anomaly_detection.sources_paper.stability_paper import StabilityPartition


def _raw_dataset(dataset: str) -> RawDataset:
    train = tuple(np.asarray((float(index), float(index + 1)), dtype=np.float64) for index in range(4))
    test = (np.asarray((1.0, 2.0), dtype=np.float64),)
    return RawDataset(dataset, 2, (0, 1, 2, 3), (b"0", b"1", b"2", b"3"), train, (0,), (b"test",), test)


def _route() -> HardRouteResult:
    route = object.__new__(HardRouteResult)
    object.__setattr__(route, "exp253", RouteLanes(frozenset(), frozenset(), frozenset()))
    return route


def _typed_stage_result(cls: type[object], **attributes: object) -> object:
    result = object.__new__(cls)
    for name, value in attributes.items():
        object.__setattr__(result, name, value)
    return result


class Exp449DatasetLocalBijectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.raw = {"Alpha": _raw_dataset("Alpha"), "Beta": _raw_dataset("Beta")}
        self.sources = {dataset: HardSourceBundle((), {}) for dataset in self.raw}
        self.routes = {dataset: _route() for dataset in self.raw}
        self.catalog = {"datasets": [{"dataset": "Alpha"}, {"dataset": "Beta"}]}

    def test_missing_dataset_is_rejected(self) -> None:
        """Removing one local Task4/5 return must close the global-scoring gate."""
        with self.assertRaisesRegex(ValueError, "catalog bijection"):
            runner.verify_dataset_local_bijection_antigravity(
                self.catalog, {"Alpha": self.raw["Alpha"]}, {"Alpha": self.sources["Alpha"]},
                {"Alpha": self.routes["Alpha"]}, mode="smoke",
            )

    def test_duplicate_catalog_dataset_is_rejected(self) -> None:
        """A duplicate catalog entry must never collapse into a mapping silently."""
        with self.assertRaisesRegex(ValueError, "catalog bijection"):
            runner.verify_dataset_local_bijection_antigravity(
                {"datasets": [{"dataset": "Alpha"}, {"dataset": "Alpha"}]},
                self.raw, self.sources, self.routes, mode="smoke",
            )

    def test_extra_dataset_is_rejected(self) -> None:
        """An unlisted local result must not enter population-global scoring."""
        raw = {**self.raw, "Gamma": _raw_dataset("Gamma")}
        sources = {**self.sources, "Gamma": HardSourceBundle((), {})}
        routes = {**self.routes, "Gamma": _route()}
        with self.assertRaisesRegex(ValueError, "catalog bijection"):
            runner.verify_dataset_local_bijection_antigravity(self.catalog, raw, sources, routes, mode="smoke")

    def test_catalog_order_drift_is_rejected(self) -> None:
        """A catalog whose order drifts from canonical lexical order is invalid."""
        with self.assertRaisesRegex(ValueError, "catalog bijection"):
            runner.verify_dataset_local_bijection_antigravity(
                {"datasets": [{"dataset": "Beta"}, {"dataset": "Alpha"}]},
                self.raw, self.sources, self.routes, mode="smoke",
            )

    def test_source_route_identity_mismatch_is_rejected(self) -> None:
        """Task4 and Task5 must each cover the exact same ordered catalog."""
        with self.assertRaisesRegex(ValueError, "catalog bijection"):
            runner.verify_dataset_local_bijection_antigravity(
                self.catalog, self.raw, {"Alpha": self.sources["Alpha"]}, self.routes, mode="smoke",
            )

    def test_full_catalog_requires_exactly_1117_datasets(self) -> None:
        """The full mode cannot accidentally run Task6 over a partial population."""
        with self.assertRaisesRegex(ValueError, "full catalog count"):
            runner.verify_dataset_local_bijection_antigravity(
                self.catalog, self.raw, self.sources, self.routes, mode="full",
            )

    def test_complete_smoke_catalog_returns_its_exact_order(self) -> None:
        """The gate returns the successful catalog order for downstream composition."""
        self.assertEqual(
            runner.verify_dataset_local_bijection_antigravity(
                self.catalog, self.raw, self.sources, self.routes, mode="smoke",
            ),
            ("Alpha", "Beta"),
        )


class Exp449CompositionGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.raw = {"Alpha": _raw_dataset("Alpha"), "Beta": _raw_dataset("Beta")}
        self.catalog = {"datasets": [{"dataset": "Alpha"}, {"dataset": "Beta"}]}
        self.acceleration = {
            "selected_backend": "cpu",
            "fallback_reason": "MLX parity benchmark not yet approved",
            "assessment_sha256": "a" * 64,
            "workers": {
                "logical_cpu_count": 2,
                "dataset_workers": 1,
                "inner_numerical_threads": 1,
                "outer_execution_mode": "parallel_task4_task5",
            },
        }
        self.sources = {dataset: HardSourceBundle((), {}) for dataset in self.raw}
        self.routes = {dataset: _route() for dataset in self.raw}
        self.low = _typed_stage_result(LowFrequencyResult, datasets=("Alpha", "Beta"))
        self.stability = _typed_stage_result(StabilityPartition)
        self.policies = _typed_stage_result(PolicyBundle)

    def _collector(self, raw: object, backend: object, **kwargs: object) -> tuple[dict[str, HardSourceBundle], dict[str, HardRouteResult]]:
        return self.sources, self.routes

    def _stage_patches(self):
        return (
            patch.object(runner, "score_low_frequency", return_value=self.low),
            patch.object(runner, "classify_stability", return_value=self.stability),
            patch.object(runner, "build_hard_evidence_surface", return_value=object()),
            patch.object(runner, "build_policies", return_value=self.policies),
        )

    def _assert_proposal_score_semantic_parity(self, actual, expected) -> None:
        self.assertEqual(tuple(actual.raw), tuple(expected.raw))
        self.assertEqual(tuple(actual.sources), tuple(expected.sources))
        self.assertEqual(tuple(actual.routes), tuple(expected.routes))
        for dataset in actual.raw:
            self.assertIs(actual.raw[dataset], expected.raw[dataset])
            self.assertIs(actual.sources[dataset], expected.sources[dataset])
            self.assertIs(actual.routes[dataset], expected.routes[dataset])
        self.assertIs(actual.low_frequency, expected.low_frequency)
        self.assertIs(actual.stability, expected.stability)
        self.assertIs(actual.policies, expected.policies)
        self.assertEqual(actual.backend, expected.backend)
        self.assertEqual(dict(actual.stage_seconds), dict(expected.stage_seconds))

    def test_missing_success_blocks_task6_and_task7(self) -> None:
        """A local omission fails before either population-global scientific task starts."""
        def missing_collector(raw: object, backend: object, **kwargs: object):
            return {"Alpha": self.sources["Alpha"]}, {"Alpha": self.routes["Alpha"]}

        with patch.object(runner, "score_low_frequency", side_effect=AssertionError("Task6 called")), \
             patch.object(runner, "classify_stability", side_effect=AssertionError("Task7 called")):
            with self.assertRaisesRegex(ValueError, "catalog bijection"):
                runner.compose_exp449_proposal_score_antigravity(
                    self.raw, self.acceleration, root=Path("/tmp/exp449_fixture_antigravity"), context=object(),
                    catalog=self.catalog, collector=missing_collector,
                )

    def test_collector_receives_the_exact_verified_recovery_inputs(self) -> None:
        """Task4 can reuse only the root, context, catalog, and assessment its caller bound."""
        received: dict[str, object] = {}

        def collector(raw: object, backend: object, **kwargs: object):
            received.update(kwargs)
            return self.sources, self.routes

        root = Path("/tmp/exp449_fixture_antigravity")
        context = object()
        with ExitStack() as stack:
            stack.enter_context(patch.object(runner, "_label_firewall"))
            for stage_patch in self._stage_patches():
                stack.enter_context(stage_patch)
            runner.compose_exp449_proposal_score_antigravity(
                self.raw, self.acceleration, root=root, context=context,
                catalog=self.catalog, collector=collector,
            )
        self.assertIs(received["root"], root)
        self.assertIs(received["context"], context)
        self.assertIs(received["catalog"], self.catalog)
        self.assertIs(received["acceleration_manifest"], self.acceleration)

    def test_collector_progress_is_forwarded_only_through_the_private_operations_seam(self) -> None:
        """Dropping a result-neutral Task4/5 event would make operations progress opaque."""
        received: list[dict[str, object]] = []

        def collector(raw: object, backend: object, **kwargs: object):
            progress = kwargs["progress"]
            progress({
                "stage": "task4_task5", "phase": "start", "completed": 0, "total": 2,
                "elapsed_s": 0.0,
            })
            return self.sources, self.routes

        with ExitStack() as stack:
            stack.enter_context(patch.object(runner, "_label_firewall"))
            for stage_patch in self._stage_patches():
                stack.enter_context(stage_patch)
            runner.compose_exp449_proposal_score_antigravity(
                self.raw, self.acceleration, root=Path("/tmp/exp449_fixture_antigravity"), context=object(),
                catalog=self.catalog, collector=collector, _progress_callback=received.append,
            )
        self.assertEqual(received, [{
            "stage": "task4_task5", "phase": "start", "completed": 0, "total": 2,
            "elapsed_s": 0.0,
        }])

    def test_complete_fixture_preserves_existing_proposal_score_semantics(self) -> None:
        """A complete catalog retains every in-memory ProposalScore semantic field."""
        paper_low = self.low
        paper_stability = self.stability
        paper_policies = self.policies
        timeline = count(100.0)
        with ExitStack() as stack:
            stack.enter_context(patch.object(runner, "_label_firewall"))
            stack.enter_context(patch.object(runner, "monotonic", side_effect=lambda: float(next(timeline))))
            for stage_patch in self._stage_patches():
                stack.enter_context(stage_patch)
            exp449 = runner.compose_exp449_proposal_score_antigravity(
                self.raw, self.acceleration, catalog=self.catalog, collector=self._collector,
            )
        timeline = count(100.0)
        with (
            patch.object(proposal_score_paper, "_label_firewall"),
            patch.object(proposal_score_paper, "monotonic", side_effect=lambda: float(next(timeline))),
            patch.object(proposal_score_paper, "_collect_task4_task5", return_value=(self.sources, self.routes)),
            patch.object(proposal_score_paper, "score_low_frequency", return_value=paper_low),
            patch.object(proposal_score_paper, "classify_stability", return_value=paper_stability),
            patch.object(proposal_score_paper, "build_hard_evidence_surface", return_value=object()),
            patch.object(proposal_score_paper, "build_policies", return_value=paper_policies),
        ):
            existing = proposal_score_paper.compose_proposal_score(self.raw, self.acceleration)

        self._assert_proposal_score_semantic_parity(exp449, existing)
        mutated_sources = dict(existing.sources)
        mutated_sources["Alpha"] = HardSourceBundle(("mutated",), {})
        mutated = proposal_score_paper.ProposalScore(
            existing.raw,
            mutated_sources,
            existing.routes,
            existing.low_frequency,
            existing.stability,
            existing.policies,
            existing.backend,
            existing.stage_seconds,
        )
        with self.assertRaises(AssertionError):
            self._assert_proposal_score_semantic_parity(exp449, mutated)


@dataclass(frozen=True)
class _AttemptFixture:
    dataset: str
    wave: int
    attempt: int
    exception_type: str = "RuntimeError"
    exception_message: str = "fixture failure"


def _recovery_context(root: Path) -> Exp449Context:
    """Build a typed temporary namespace only after the module root guard is patched."""
    context = object.__new__(Exp449Context)
    object.__setattr__(context, "experiment_id", "Exp449")
    object.__setattr__(context, "mode", "smoke")
    object.__setattr__(context, "run_root", root)
    object.__setattr__(context, "database_sha256", "a" * 64)
    object.__setattr__(context, "catalog_sha256", "b" * 64)
    object.__setattr__(context, "code_closure", {"fixture_antigravity.py": "c" * 64})
    object.__setattr__(context, "lineage_sha256", "d" * 64)
    object.__setattr__(context, "codemap_sha256", "e" * 64)
    object.__setattr__(context, "interpreter", "fixture-python")
    object.__setattr__(context, "environment", {"fixture": "deterministic"})
    object.__setattr__(context, "backend_record", {"selected_backend": "cpu"})
    object.__setattr__(context, "seeds_and_constants", {"seed": 7})
    object.__setattr__(context, "historical_parity_status", "not_used")
    return context


class _RecoveryFixture:
    """Real recovery state with only the paper scoring boundary replaced."""

    def __init__(self, directory: Path, *, fail_once: bool = False, fail_twice: bool = False) -> None:
        self.scientific_root = directory / "scientific_fixture_antigravity"
        self.operations_root = directory / "operations_fixture_antigravity"
        self.names = FROZEN_EXP449_CONFIG.smoke_datasets
        self.catalog = {"catalog_sha256": "b" * 64, "datasets": [{"dataset": name} for name in self.names]}
        self.raw = {name: object() for name in self.names}
        self.context = _recovery_context(self.scientific_root)
        self.config = operations.Exp449WaveConfig(
            context=self.context,
            catalog=self.catalog,
            raw_datasets=self.raw,
            acceleration_manifest={"fixture": "acceleration"},
        )
        self.successes: set[str] = set()
        self.attempts: list[_AttemptFixture] = []
        self.revalidated_successes: list[tuple[str, ...]] = []
        self.revalidated_attempts = 0
        self.composer_calls = 0
        self.global_stage_calls = 0
        self.scheduled_identities: list[tuple[str, ...]] = []
        self.fail_once = fail_once
        self.fail_twice = fail_twice

    @property
    def failed_dataset(self) -> str:
        return self.names[-1]

    def verify_success_namespace(self, *_args: object) -> tuple[str, ...]:
        verified = tuple(sorted(self.successes))
        self.revalidated_successes.append(verified)
        return verified

    def verify_attempt_namespace(self, *_args: object) -> tuple[_AttemptFixture, ...]:
        self.revalidated_attempts += 1
        return tuple(self.attempts)

    def compose(self, *_args: object, **_kwargs: object) -> object:
        self.composer_calls += 1
        failure = (self.fail_once and self.composer_calls == 1) or (self.fail_twice and self.composer_calls <= 2)
        scheduled = tuple(name for name in self.names if name not in self.successes)
        self.scheduled_identities.append(scheduled)
        progress = _kwargs.get("_progress_callback")
        if callable(progress):
            for completed, dataset in enumerate(scheduled, 1):
                progress({
                    "stage": "task4_task5", "phase": "tick", "completed": completed,
                    "total": len(scheduled), "dataset": dataset, "elapsed_s": float(completed),
                })
        if failure:
            self.successes.update(name for name in self.names if name != self.failed_dataset)
            self.attempts.append(_AttemptFixture(self.failed_dataset, self.composer_calls, len(self.attempts) + 1))
            raise proposal_score_paper.ProposalScoreIncomplete(
                completed=len(self.successes), failed=1, pending=(self.failed_dataset,), attempts_path=None,
            )
        self.successes.update(self.names)
        self.global_stage_calls += 1
        self.scientific_root.mkdir(parents=True, exist_ok=True)
        (self.scientific_root / "fixture_proposal_payload_antigravity.json").write_bytes(
            json.dumps({"datasets": list(self.names), "scientific": "fixed"}, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        return object()


class Exp449TwoWaveRecoveryTests(unittest.TestCase):
    """The operations boundary owns retries, never success authority or Task6/7 work."""

    def _patches(self, fixture: _RecoveryFixture) -> ExitStack:
        stack = ExitStack()
        stack.enter_context(patch.object(operations, "_operations_root_for_context_antigravity", return_value=fixture.operations_root))
        stack.enter_context(patch.object(operations.Task4Task5SuccessAuthority, "from_verified_context", return_value=object()))
        stack.enter_context(patch.object(operations, "verify_task4_task5_success_namespace", side_effect=fixture.verify_success_namespace))
        stack.enter_context(patch.object(operations, "verify_task4_task5_attempt_namespace", side_effect=fixture.verify_attempt_namespace))
        stack.enter_context(patch.object(
            operations,
            "_read_verified_attempt_payloads_antigravity",
            side_effect=lambda _config, attempts: tuple({
                "dataset": dataset,
                "wave": wave,
                "attempt": attempt,
                "exception_type": exception_type,
                "exception_message": exception_message,
                "raw_identity": {"dataset": dataset},
                "record_sha256": f"{attempt:064x}",
            } for dataset, wave, attempt, exception_type, exception_message in attempts),
        ))
        stack.enter_context(patch.object(operations, "compose_exp449_proposal_score_antigravity", side_effect=fixture.compose))
        return stack

    def test_fail_once_revalidates_success_and_retries_only_unverified_dataset(self) -> None:
        """Recomputing a published success after interruption would change recovery work."""
        with TemporaryDirectory(dir=Path.cwd()) as temporary:
            fixture = _RecoveryFixture(Path(temporary), fail_once=True)
            with self._patches(fixture):
                first = operations.run_exp449_wave_antigravity(fixture.config)
                self.assertEqual((first.status, first.completed_dataset_count, first.failed_dataset_count, first.pending_dataset_count, first.wave_count), ("proposal_score_incomplete", 22, 1, 0, 1))
                self.assertFalse((fixture.operations_root / "proposal_score_incomplete_antigravity.json").exists())
                second = operations.run_exp449_wave_antigravity(fixture.config)
            self.assertEqual((second.status, second.completed_dataset_count, second.wave_count), ("dataset_local_complete", 23, 2))
            self.assertGreaterEqual(len(fixture.revalidated_successes), 4)
            self.assertEqual(fixture.revalidated_successes[1], tuple(name for name in fixture.names if name != fixture.failed_dataset))
            self.assertEqual((fixture.composer_calls, fixture.global_stage_calls), (2, 1))
            self.assertEqual(fixture.scheduled_identities, [fixture.names, (fixture.failed_dataset,)])
            events = [json.loads(line) for line in (fixture.operations_root / "operations_progress_antigravity.jsonl").read_text().splitlines()]
            per_dataset = [event for event in events if "dataset" in event]
            self.assertEqual(tuple(event["dataset"] for event in per_dataset), fixture.names)
            self.assertGreater(per_dataset[1]["eta_seconds"], 0.0)
            first_wave_last = per_dataset[len(fixture.names) - 2]
            self.assertEqual((first_wave_last["completed"], first_wave_last["failed"], first_wave_last["pending"]), (22, 1, 1))

    def test_second_failure_publishes_one_terminal_record_and_third_call_refuses_scoring(self) -> None:
        """Scheduling a third attempt after the frozen retry cap would weaken terminal semantics."""
        with TemporaryDirectory() as temporary:
            fixture = _RecoveryFixture(Path(temporary), fail_twice=True)
            with self._patches(fixture):
                operations.run_exp449_wave_antigravity(fixture.config)
                second = operations.run_exp449_wave_antigravity(fixture.config)
                third = operations.run_exp449_wave_antigravity(fixture.config)
            self.assertEqual((second.status, second.completed_dataset_count, second.failed_dataset_count, second.pending_dataset_count, second.wave_count), ("proposal_score_incomplete", 22, 1, 0, 2))
            self.assertEqual(third, second)
            self.assertEqual((fixture.composer_calls, fixture.global_stage_calls), (2, 0))
            terminal = json.loads((fixture.operations_root / "proposal_score_incomplete_antigravity.json").read_text(encoding="utf-8"))
            self.assertEqual(terminal["verified_successes"], list(fixture.names[:-1]))
            self.assertEqual(terminal["failed"], [fixture.failed_dataset])
            self.assertEqual(terminal["never_attempted"], [])
            self.assertEqual(terminal["context_sha256"], operations.task45_context_payload_antigravity(fixture.context)["context_sha256"])
            self.assertEqual(terminal["catalog_sha256"], "b" * 64)

    def test_terminal_authority_rejects_later_success_or_ledger_mutation_before_scheduling(self) -> None:
        """A terminal record that permits resumed scoring after mutation is not an authority."""
        with TemporaryDirectory() as temporary:
            fixture = _RecoveryFixture(Path(temporary), fail_twice=True)
            with self._patches(fixture):
                operations.run_exp449_wave_antigravity(fixture.config)
                operations.run_exp449_wave_antigravity(fixture.config)
                fixture.successes.add(fixture.failed_dataset)
                with self.assertRaisesRegex(ValueError, "terminal"):
                    operations.run_exp449_wave_antigravity(fixture.config)
            self.assertEqual(fixture.composer_calls, 2)

    def test_attempt_reconciliation_rejects_duplicate_noncontiguous_and_out_of_range_identities(self) -> None:
        """Clamping or coalescing malformed ledger identities would create an unauthorized retry."""
        dataset = FROZEN_EXP449_CONFIG.smoke_datasets[0]
        cases = (
            (_AttemptFixture(dataset, 1, 2),),
            (_AttemptFixture(dataset, 1, 1), _AttemptFixture(dataset, 1, 2)),
            (_AttemptFixture(dataset, 3, 1),),
            (_AttemptFixture(dataset, 2, 1),),
            (_AttemptFixture(dataset, 2, 1), _AttemptFixture(dataset, 1, 2)),
        )
        for attempts in cases:
            with self.subTest(attempts=attempts), TemporaryDirectory() as temporary:
                fixture = _RecoveryFixture(Path(temporary))
                fixture.attempts.extend(attempts)
                with self._patches(fixture):
                    with self.assertRaisesRegex(ValueError, "attempt"):
                        operations.status_exp449_antigravity(fixture.config)

    def test_attempt_records_never_count_as_success_or_unlock_global_composition(self) -> None:
        """Treating an attempt ledger entry as a success would open Task6/7 with a partial catalog."""
        with TemporaryDirectory() as temporary:
            fixture = _RecoveryFixture(Path(temporary))
            fixture.attempts.append(_AttemptFixture(fixture.failed_dataset, 1, 1))
            with self._patches(fixture):
                status = operations.status_exp449_antigravity(fixture.config)
            self.assertEqual((status.status, status.completed_dataset_count, status.failed_dataset_count, status.pending_dataset_count), ("proposal_score_incomplete", 0, 1, 22))
            self.assertEqual(fixture.global_stage_calls, 0)

    def test_progress_log_is_exactly_operational_and_rejects_scientific_fields(self) -> None:
        """A label or score in the operations log would breach the result-neutral boundary."""
        with TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary) / "operations_fixture_antigravity"
            event = {
                "timestamp": "2026-08-14T00:00:00+00:00", "experiment": "Exp449", "mode": "smoke", "wave": 1,
                "completed": 1, "failed": 0, "pending": 22, "elapsed_seconds": 1.0, "eta_seconds": 22.0,
                "dataset": "Alpha",
            }
            operations._write_exp449_progress_event_antigravity(root, event)
            logged = json.loads((root / "operations_progress_antigravity.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(logged, event)
            unsafe = {**event, "scores": [0.1]}
            with self.assertRaisesRegex(ValueError, "forbidden"):
                operations._write_exp449_progress_event_antigravity(root, unsafe)

    def test_interrupted_resume_keeps_fixture_scientific_payload_byte_identical(self) -> None:
        """Retry ordering must not alter the final typed scientific bytes."""
        with TemporaryDirectory() as uninterrupted_dir, TemporaryDirectory() as resumed_dir:
            uninterrupted = _RecoveryFixture(Path(uninterrupted_dir))
            resumed = _RecoveryFixture(Path(resumed_dir), fail_once=True)
            with self._patches(uninterrupted):
                operations.run_exp449_wave_antigravity(uninterrupted.config)
            with self._patches(resumed):
                operations.run_exp449_wave_antigravity(resumed.config)
                operations.run_exp449_wave_antigravity(resumed.config)
            self.assertEqual(
                (uninterrupted.scientific_root / "fixture_proposal_payload_antigravity.json").read_bytes(),
                (resumed.scientific_root / "fixture_proposal_payload_antigravity.json").read_bytes(),
            )

    def test_real_paper_collector_reuses_verified_success_after_one_failure(self) -> None:
        """The actual checkpoint collector, not a composition fake, must avoid rescoring success."""
        from paper.equipment_anomaly_detection import proposal_score_paper as paper_score
        from paper.equipment_anomaly_detection.tests_paper.test_artifacts_runner_paper import Task4Task5FailureBatchingTests

        fixture = Task4Task5FailureBatchingTests("run")
        fixture.setUp()
        try:
            manifest, context, backend = fixture._backend()
            calls: list[str] = []
            failed = [True]

            def score(raw):
                calls.append(raw.dataset)
                if raw.dataset == "alpha" and failed[0]:
                    failed[0] = False
                    raise RuntimeError("fail alpha once")
                return fixture.values[raw.dataset]

            with patch.object(paper_score, "verify_pycatch22_environment_paper", return_value={"pycatch22": "fixture"}), \
                 patch.object(paper_score, "_score_dataset_task4_task5", side_effect=score):
                with self.assertRaises(paper_score.ProposalScoreIncomplete):
                    paper_score._collect_task4_task5(
                        fixture.raw, backend, root=fixture.fixture.root, context=context,
                        catalog=fixture.fixture.catalog, acceleration_manifest=manifest,
                    )
                sources, routes = paper_score._collect_task4_task5(
                    fixture.raw, backend, root=fixture.fixture.root, context=context,
                    catalog=fixture.fixture.catalog, acceleration_manifest=manifest,
                )
            self.assertEqual(calls, ["alpha", "beta", "alpha"])
            self.assertEqual((tuple(sources), tuple(routes)), (("alpha", "beta"), ("alpha", "beta")))
        finally:
            fixture.tearDown()


if __name__ == "__main__":
    unittest.main()
