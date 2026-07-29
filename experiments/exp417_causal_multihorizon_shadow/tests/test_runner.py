"""Synthetic boundary tests for Exp417's label-free population runner."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch
import unittest

import numpy as np

from experiments.exp417_causal_multihorizon_shadow.checkpoints import (
    Exp417CheckpointProvenance,
    Exp417CheckpointStore,
)
from experiments.exp417_causal_multihorizon_shadow.config import FROZEN_CONFIG
from experiments.exp417_causal_multihorizon_shadow.contracts import (
    BackendAudit,
    DatasetDecisionEvidence,
    DatasetResult,
    DatasetTerminalEvidence,
    FrozenLaneSnapshot,
    FrozenRunContext,
    FrozenTrainContract,
    GroupQualification,
    HorizonScore,
    RawTrainDataset,
    SeedDecision,
    TestInstanceDecision,
)
from experiments.exp417_causal_multihorizon_shadow.model import Fold, OOFModelBundle
from experiments.exp417_causal_multihorizon_shadow.tests.test_model import (
    synthetic_folds,
    synthetic_group,
    synthetic_trained_model,
)


def _lanes() -> FrozenLaneSnapshot:
    return FrozenLaneSnapshot(
        hard={"dataset": (1,) + tuple(range(10_000, 12_217))},
        standard={"dataset": (2,) + tuple(range(20_000, 20_805))},
        priority={"dataset": (3,) + tuple(range(30_000, 30_030))},
        automatic_normal={"dataset": (4,) + tuple(range(40_000, 40_034))},
    )


def _context(*, test_size: int = 2) -> FrozenRunContext:
    datasets = ("dataset",) + tuple(f"synthetic-{index:04d}" for index in range(1116))
    return FrozenRunContext(
        database_sha256=FROZEN_CONFIG.database_sha256,
        datasets=datasets,
        lanes=_lanes(),
        test_identity_manifest={
            dataset: tuple((index, 100) for index in range(100 if dataset == "dataset" else test_size))
            for dataset in datasets
        },
        code_commit="synthetic",
        runtime_versions={"python": "synthetic"},
    )


def _group() -> GroupQualification:
    return GroupQualification(
        length=100,
        series_count=100,
        status="qualified",
        oof_candidate_rate=0.01,
        wilson_upper=0.05,
        pooled_synthetic_sensitivity=0.70,
        per_type_synthetic_sensitivity={
            "level_shift": 0.50,
            "gradual_drift": 0.50,
            "amplitude_scaling": 0.50,
            "period_change": 0.50,
        },
        seed_agreement=True,
    )


def _contract(dataset: str = "dataset") -> FrozenTrainContract:
    return FrozenTrainContract(
        dataset=dataset,
        groups=(_group(),),
        backend=BackendAudit("cpu", True, 0.0, 1.0, 1.0, False),
        train_identity_sha256="a" * 64,
    )


def _decisions(
    dataset: str = "dataset", *, candidate: bool = True, count: int = 100
) -> tuple[TestInstanceDecision, ...]:
    decisions = []
    for index in range(count):
        is_candidate = candidate and index == 0
        scores = {
            1: HorizonScore(1, 1.0, 0.01 if is_candidate else 0.5, 0.05 if is_candidate else 0.0),
            2: HorizonScore(2, 1.0, 0.01 if is_candidate else 0.5, 0.05 if is_candidate else 0.0),
            10: HorizonScore(10, 1.0, 0.5, 0.0),
        }
        decisions.append(TestInstanceDecision(
            dataset=dataset, instance_index=index, length=100,
            seed_decisions=tuple(SeedDecision(seed, scores, is_candidate) for seed in FROZEN_CONFIG.model_seeds),
        ))
    return tuple(decisions)


def _result(context: FrozenRunContext, *, candidate: bool = False) -> DatasetResult:
    contract = _contract()
    evidence = DatasetDecisionEvidence(
        dataset="dataset",
        context=context,
        frozen_train_contract=contract,
        database_sha256=context.database_sha256,
        train_identity_sha256=contract.train_identity_sha256,
        decisions=_decisions(candidate=candidate, count=context.test_sizes["dataset"]),
    )
    candidates = evidence.derived_candidates
    return DatasetResult(
        "dataset", context.test_sizes["dataset"], 1, contract.groups, candidates,
        evidence, context.lanes, context, "qualified"
    )


def _checkpoint_material(result: DatasetResult):
    from experiments.exp417_causal_multihorizon_shadow import runner

    contract = result.decision_evidence.frozen_train_contract
    def digest(kind: str) -> str:
        return sha256(f"{result.dataset}:{kind}".encode("utf-8")).hexdigest()
    return runner.CheckpointEvidence(
        raw_train_identity_sha256=digest("raw"),
        repaired_train_identity_sha256=digest("repaired"),
        length_group_identity_sha256={100: digest("group")},
        canonical_model_hashes={100: {seed: digest(f"model:{seed}") for seed in FROZEN_CONFIG.model_seeds}},
        score_calibration_hashes={100: digest("calibration")}, elapsed_seconds=0.0,
        dataset=result.dataset, database_sha256=result.context.database_sha256,
        train_contract_identity_sha256=contract.train_identity_sha256,
        result_evidence_sha256=runner._result_evidence_digest(result),
    )


def _provenance() -> Exp417CheckpointProvenance:
    return Exp417CheckpointProvenance(
        git_commit="c" * 40,
        database_sha256=FROZEN_CONFIG.database_sha256,
        exp291_prediction_sha256=FROZEN_CONFIG.exp291_prediction_sha256,
        exp291_manifest_sha256=FROZEN_CONFIG.exp291_manifest_sha256,
        exp292_prediction_sha256=FROZEN_CONFIG.exp292_prediction_sha256,
        exp292_manifest_sha256=FROZEN_CONFIG.exp292_manifest_sha256,
        frozen_config_sha256="5" * 64,
        preprocessing_contract_sha256="6" * 64,
        fold_contract_sha256="7" * 64,
        model_contract_sha256="8" * 64,
        model_seeds=FROZEN_CONFIG.model_seeds,
        perturbation_seeds=FROZEN_CONFIG.perturbation_seeds,
        runtime_versions={"numpy": "synthetic", "python": "synthetic", "torch": "synthetic"},
        backend_contract_sha256="9" * 64,
    )


def _typed_stage_evidence(contract: FrozenTrainContract):
    from experiments.exp417_causal_multihorizon_shadow import runner

    processed = synthetic_group(count=100, length=100)
    train = RawTrainDataset("dataset", processed.group.series)
    universe = tuple(item.instance_index for item in processed.group.series)
    folds = synthetic_folds(universe)
    horizons = (1, 2, 10)
    models = tuple(
        synthetic_trained_model(seed, fold, horizons)
        for seed in FROZEN_CONFIG.model_seeds
        for fold in folds
    )
    residuals = {
        seed: {
            index: {horizon: np.zeros(100 - horizon, dtype=np.float64) for horizon in horizons}
            for index in universe
        }
        for seed in FROZEN_CONFIG.model_seeds
    }
    bundle = OOFModelBundle(100, universe, horizons, folds, models, residuals)
    final_fold = Fold(universe, (), universe[5:], universe[:5])
    final = tuple(synthetic_trained_model(seed, final_fold, horizons) for seed in FROZEN_CONFIG.model_seeds)
    model_evidence = runner.TrainModelEvidence(
        (processed,), {100: bundle}, {100: final}, contract.backend
    )
    hashes = {100: {model.seed: model.model_sha256 for model in final}}
    scoring = runner.TestScoringEvidence(_decisions(candidate=False), hashes)
    return train, (processed,), model_evidence, scoring


class RunnerOrderTests(unittest.TestCase):
    def test_production_runtime_requires_passing_task7_backend_audit(self) -> None:
        """Production cannot invent a perfect CPU audit or accept a fallback."""
        from experiments.exp417_causal_multihorizon_shadow import runner

        resources = runner.RunnerResources(
            Path("database"), Path("291-prediction"), Path("291-manifest"),
            Path("292-prediction"), Path("292-manifest"),
        )
        with self.assertRaisesRegex(ValueError, "Task 7 backend audit"):
            runner.RunnerRuntime.production(resources)
        fallback = replace(_contract().backend, fallback=True, fallback_reason="parity")
        with self.assertRaisesRegex(ValueError, "Task 7 backend audit"):
            runner.RunnerRuntime.production(replace(resources, backend_audit=fallback))
        runtime = runner.RunnerRuntime.production(replace(resources, backend_audit=_contract().backend))
        self.assertIsInstance(runtime, runner.RunnerRuntime)

    def test_normal_oof_decisions_use_each_seed_query_residual(self) -> None:
        """Reusing seed 41701's query residual would mask seed disagreement."""
        from experiments.exp417_causal_multihorizon_shadow import runner

        horizons = (1, 2, 10)
        bundle = SimpleNamespace(
            universe_indices=(0, 1), horizons=horizons,
            oof_residuals={
                seed: {index: {horizon: np.arange(20, dtype=np.float64) + seed + index + horizon
                               for horizon in horizons}
                       for index in (0, 1)}
                for seed in FROZEN_CONFIG.model_seeds
            },
        )
        query = {
            seed: {horizon: np.full(20, 1.0 if seed == FROZEN_CONFIG.model_seeds[0] else 100.0,
                                    dtype=np.float64) for horizon in horizons}
            for seed in FROZEN_CONFIG.model_seeds
        }

        decisions = runner._residual_decisions(bundle, query)

        self.assertNotEqual(decisions[0].scores[1], decisions[1].scores[1])
    def test_test_sql_static_guard_uses_only_identity_and_values(self) -> None:
        """The runner's TEST boundary cannot depend on a label-bearing query."""
        from experiments.exp417_causal_multihorizon_shadow.inputs import (
            _TEST_VALUES_SQL, assert_label_free_sql_contract,
        )

        assert_label_free_sql_contract()
        self.assertNotIn("label", _TEST_VALUES_SQL.lower())
        self.assertIn("instance_index", _TEST_VALUES_SQL)
        self.assertIn("values_blob", _TEST_VALUES_SQL)

    def test_full_dataset_path_freezes_train_before_loading_values(self) -> None:
        """Moving TEST loading before the frozen-contract stage is a safety breach."""
        from experiments.exp417_causal_multihorizon_shadow import runner

        context = _context()
        contract = _contract()
        events: list[str] = []
        result = _result(context)
        qualification = runner.TrainQualification(contract.groups, contract.backend)
        train, processed, models, scoring = _typed_stage_evidence(contract)
        runtime = runner.RunnerRuntime(
            verify=lambda _: events.append("verify_inputs"),
            load_train=lambda *_: events.append("load_train_values") or train,
            preprocess=lambda _: events.append("preprocess") or processed,
            train_models=lambda _: events.append("train_oof") or models,
            qualify=lambda *_: events.append("qualify_train") or qualification,
            freeze=lambda *_: events.append("freeze_contract") or contract,
            load_test=lambda *_: events.append("load_test_values") or object(),
            score=lambda *_: events.append("score_test") or scoring,
            exclude=lambda evidence: events.append("exclude_lanes") or evidence.derived_candidates,
            workload=lambda _: events.append("dataset_workload") or result,
        )
        with patch.object(runner, "checkpoint", side_effect=lambda *_args, **_kwargs: events.append("checkpoint")):
            self.assertEqual(runner.run_dataset("dataset", context, runtime=runtime), result)

        self.assertEqual(events, [
            "verify_inputs", "load_train_values", "preprocess", "train_oof",
            "qualify_train", "freeze_contract", "load_test_values", "score_test",
            "exclude_lanes", "dataset_workload", "checkpoint",
        ])
        self.assertTrue(all("label" not in event for event in events))

    def test_train_failure_cannot_load_test_values(self) -> None:
        """A pre-contract prediction failure must stop before the TEST boundary."""
        from experiments.exp417_causal_multihorizon_shadow import runner

        context = _context()
        test_loader = Mock(side_effect=AssertionError("TEST values were loaded"))
        runtime = runner.RunnerRuntime(
            verify=lambda _: None, load_train=lambda *_: RawTrainDataset("dataset", _typed_stage_evidence(_contract())[0].series),
            preprocess=lambda train: (_typed_stage_evidence(_contract())[1]),
            train_models=lambda _: (_ for _ in ()).throw(ValueError("synthetic OOF failure")),
            qualify=lambda *_: (_ for _ in ()).throw(AssertionError("qualification was reached")),
            freeze=lambda *_: (_ for _ in ()).throw(AssertionError("freeze was reached")),
            load_test=test_loader, score=lambda *_: (_ for _ in ()).throw(AssertionError("score was reached")),
        )
        result = runner.run_dataset("dataset", context, runtime=runtime)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.decision_evidence.stage, "oof")
        test_loader.assert_not_called()

    def test_diagnostic_preprocessing_is_terminal_before_oof_or_test(self) -> None:
        """A 50--99 TRAIN group cannot be promoted into model or TEST work."""
        from experiments.exp417_causal_multihorizon_shadow import runner

        context = _context()
        train, processed, _models, _scoring = _typed_stage_evidence(_contract())
        diagnostic = replace(processed[0], status="diagnostic_only")
        test_loader = Mock(side_effect=AssertionError("TEST values were loaded"))
        runtime = runner.RunnerRuntime(
            verify=lambda _: None, load_train=lambda *_: train,
            preprocess=lambda _: (diagnostic,),
            train_models=lambda _: (_ for _ in ()).throw(AssertionError("OOF was reached")),
            qualify=lambda *_: (_ for _ in ()).throw(AssertionError("qualification was reached")),
            freeze=lambda *_: (_ for _ in ()).throw(AssertionError("freeze was reached")),
            load_test=test_loader,
            score=lambda *_: (_ for _ in ()).throw(AssertionError("score was reached")),
        )

        result = runner.run_dataset("dataset", context, runtime=runtime)

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.decision_evidence.stage, "preprocessing")
        test_loader.assert_not_called()

    def test_every_noneligible_preprocessing_status_stops_before_test(self) -> None:
        """<50, 50--99, and all-missing groups share the same closed barrier."""
        from experiments.exp417_causal_multihorizon_shadow import runner

        context = _context()
        train, processed, _models, _scoring = _typed_stage_evidence(_contract())
        for status in ("insufficient_train_abstain", "diagnostic_only", "nonfinite_abstain"):
            with self.subTest(status=status):
                channels = processed[0].channels if status == "diagnostic_only" else None
                blocked = replace(processed[0], status=status, channels=channels)
                test_loader = Mock(side_effect=AssertionError("TEST values were loaded"))
                runtime = runner.RunnerRuntime(
                    verify=lambda _: None, load_train=lambda *_: train, preprocess=lambda _: (blocked,),
                    train_models=lambda _: (_ for _ in ()).throw(AssertionError("OOF was reached")),
                    qualify=lambda *_: (_ for _ in ()).throw(AssertionError("qualification was reached")),
                    freeze=lambda *_: (_ for _ in ()).throw(AssertionError("freeze was reached")),
                    load_test=test_loader, score=lambda *_: (_ for _ in ()).throw(AssertionError("score was reached")),
                )
                result = runner.run_dataset("dataset", context, runtime=runtime)
                self.assertEqual(result.decision_evidence.stage, "preprocessing")
                test_loader.assert_not_called()

    def test_backend_fallback_is_terminal_before_qualification_or_test(self) -> None:
        """A failed Task 7 audit cannot reach synthetic qualification or TEST."""
        from experiments.exp417_causal_multihorizon_shadow import runner

        context = _context()
        train, processed, models, _scoring = _typed_stage_evidence(_contract())
        fallback = BackendAudit("cpu", True, 0.0, 1.0, 1.0, True, "parity")
        models = replace(models, backend=fallback)
        test_loader = Mock(side_effect=AssertionError("TEST values were loaded"))
        runtime = runner.RunnerRuntime(
            verify=lambda _: None, load_train=lambda *_: train, preprocess=lambda _: processed,
            train_models=lambda _: models,
            qualify=lambda *_: (_ for _ in ()).throw(AssertionError("qualification was reached")),
            freeze=lambda *_: (_ for _ in ()).throw(AssertionError("freeze was reached")),
            load_test=test_loader, score=lambda *_: (_ for _ in ()).throw(AssertionError("score was reached")),
        )
        result = runner.run_dataset("dataset", context, runtime=runtime)
        self.assertEqual(result.decision_evidence.stage, "backend")
        test_loader.assert_not_called()


class RunnerPopulationTests(unittest.TestCase):
    def test_checkpoint_rejects_cross_dataset_material_reuse(self) -> None:
        """A valid-looking record from one dataset cannot attest another result."""
        from experiments.exp417_causal_multihorizon_shadow import runner

        context = _context()
        first = _result_for_dataset(context, "dataset")
        second = _result_for_dataset(context, context.datasets[1])
        store = Mock()

        with self.assertRaisesRegex(ValueError, "exact dataset result evidence"):
            runner.checkpoint(second, store, _checkpoint_material(first))
        store.write_dataset.assert_not_called()

    def test_terminal_failure_is_checkpointed_without_prediction_evidence(self) -> None:
        """A closed pre-contract failure records its reason, never TEST scores."""
        from experiments.exp417_causal_multihorizon_shadow import runner

        context = replace(_context(), code_commit="c" * 40,
                          runtime_versions={"numpy": "synthetic", "python": "synthetic", "torch": "synthetic"})
        runtime = runner.RunnerRuntime(
            verify=lambda _: None,
            load_train=lambda *_: (_ for _ in ()).throw(ValueError("synthetic TRAIN unavailable")),
            preprocess=lambda _: (_ for _ in ()).throw(AssertionError("preprocess was reached")),
            train_models=lambda _: (_ for _ in ()).throw(AssertionError("OOF was reached")),
            qualify=lambda *_: (_ for _ in ()).throw(AssertionError("qualification was reached")),
            freeze=lambda *_: (_ for _ in ()).throw(AssertionError("freeze was reached")),
            load_test=lambda *_: (_ for _ in ()).throw(AssertionError("TEST was loaded")),
            score=lambda *_: (_ for _ in ()).throw(AssertionError("TEST was scored")),
        )
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "checkpoint"
            store = Exp417CheckpointStore(root, _provenance(), context.datasets)
            result = runner.run_dataset("dataset", context, runtime=runtime)
            # Direct execution has no store; use the same terminal path with the
            # real transactional store to prove its metadata is deliberately empty.
            result = runner._run_dataset("dataset", context, store, runtime)
            reopened = Exp417CheckpointStore(root, _provenance(), context.datasets)
            loaded = reopened.load_completed()["dataset"]
        self.assertEqual(result.status, "failed")
        self.assertIsInstance(loaded.decision_evidence, DatasetTerminalEvidence)
        self.assertEqual(loaded.decision_evidence.stage, "insufficient_train")

    def test_resume_runs_only_missing_datasets_sequentially_and_conserves_population(self) -> None:
        """Re-running completed work or changing frozen dataset order corrupts resume accounting."""
        from experiments.exp417_causal_multihorizon_shadow import runner

        context = _context()
        completed = _result(context)
        store = Mock()
        store.expected_datasets = context.datasets
        store.load_completed.return_value = {"dataset": completed}
        calls: list[str] = []

        def run_missing(dataset: str, supplied_context: FrozenRunContext, supplied_store: object, runtime: object) -> DatasetResult:
            self.assertEqual(supplied_context, context)
            self.assertIs(supplied_store, store)
            calls.append(dataset)
            return _result_for_dataset(supplied_context, dataset)

        synthetic_runtime = runner.RunnerRuntime(
            verify=lambda _: None, load_train=lambda *_: (_ for _ in ()).throw(AssertionError("unused")),
            preprocess=lambda _: (), train_models=lambda _: (_ for _ in ()).throw(AssertionError("unused")),
            qualify=lambda *_: (_ for _ in ()).throw(AssertionError("unused")),
            freeze=lambda *_: (_ for _ in ()).throw(AssertionError("unused")),
            load_test=lambda *_: (_ for _ in ()).throw(AssertionError("unused")),
            score=lambda *_: (_ for _ in ()).throw(AssertionError("unused")),
        )
        with patch.object(runner, "_run_dataset", side_effect=run_missing):
            summary = runner.resume_or_run_population(context, store, runtime=synthetic_runtime)

        self.assertEqual(calls, list(context.datasets[1:]))
        self.assertEqual(tuple(result.dataset for result in summary.results), context.datasets)
        self.assertEqual(summary.total_test, sum(context.test_sizes.values()))
        self.assertTrue(summary.publication_allowed)

    def test_global_workload_gate_precedes_a_local_failed_status(self) -> None:
        """Returning a failed summary first can bypass required global payload clearing."""
        from experiments.exp417_causal_multihorizon_shadow import runner

        context = _context()
        completed = {dataset: object() for dataset in context.datasets}
        store = Mock(expected_datasets=context.datasets)
        store.load_completed.return_value = completed
        global_summary = Mock(status="global_workload_gate_failed")
        synthetic_runtime = runner.RunnerRuntime(
            verify=lambda _: None, load_train=lambda *_: (_ for _ in ()).throw(AssertionError("unused")),
            preprocess=lambda _: (), train_models=lambda _: (_ for _ in ()).throw(AssertionError("unused")),
            qualify=lambda *_: (_ for _ in ()).throw(AssertionError("unused")),
            freeze=lambda *_: (_ for _ in ()).throw(AssertionError("unused")),
            load_test=lambda *_: (_ for _ in ()).throw(AssertionError("unused")),
            score=lambda *_: (_ for _ in ()).throw(AssertionError("unused")),
        )

        with (
            patch.object(runner, "_validate_result"),
            patch.object(runner, "RunSummary", return_value=Mock(status="failed")),
            patch.object(runner, "apply_global_workload", return_value=global_summary) as gate,
        ):
            self.assertIs(runner.resume_or_run_population(context, store, runtime=synthetic_runtime), global_summary)
        gate.assert_called_once()

    def test_checkpoint_writes_only_typed_complete_evidence_to_the_transaction_store(self) -> None:
        """Dropping a checkpoint field would make a resumable result unverifiable."""
        from experiments.exp417_causal_multihorizon_shadow import runner

        context = _context()
        result = _result(context)
        store = Mock()
        material = _checkpoint_material(result)

        runner.checkpoint(result, store, material)

        store.write_dataset.assert_called_once()
        arguments, keyword = store.write_dataset.call_args
        self.assertEqual(arguments, (result,))
        self.assertEqual(keyword["backend_audit"], result.decision_evidence.frozen_train_contract.backend)
        self.assertEqual(keyword["elapsed_seconds"], material.elapsed_seconds)

    def test_resume_writes_and_reloads_typed_evidence_through_a_real_checkpoint_store(self) -> None:
        """Replacing typed evidence with placeholders must not produce a resumable record."""
        from experiments.exp417_causal_multihorizon_shadow import runner

        context = replace(
            _context(),
            code_commit="c" * 40,
            runtime_versions={"numpy": "synthetic", "python": "synthetic", "torch": "synthetic"},
        )
        contract = _contract()
        train, processed, model_evidence, scoring = _typed_stage_evidence(contract)
        contract = replace(contract, train_identity_sha256=model_evidence.train_identity_sha256)
        with TemporaryDirectory() as temporary:
            store = Exp417CheckpointStore(Path(temporary) / "checkpoint", _provenance(), context.datasets)
            qualification = runner.TrainQualification(contract.groups, contract.backend)
            for dataset in context.datasets[1:]:
                completed = _result_for_dataset(context, dataset)
                runner.checkpoint(completed, store, _checkpoint_material(completed))
            runtime = runner.RunnerRuntime(
                verify=lambda _: None, load_train=lambda *_: train, preprocess=lambda _: processed,
                train_models=lambda _: model_evidence, qualify=lambda *_: qualification,
                freeze=lambda *_: contract, load_test=lambda *_: object(), score=lambda *_: scoring,
            )
            summary = runner.resume_or_run_population(context, store, runtime=runtime)
            self.assertEqual(summary.status, "complete")
            self.assertEqual(tuple(store.load_completed()), context.datasets)
            self.assertEqual(tuple(result.dataset for result in store.require_complete()), context.datasets)

    def test_checkpoint_evidence_rejects_missing_or_corrupt_model_bindings(self) -> None:
        """A missing seed or an altered model identity cannot be checkpointed."""
        from experiments.exp417_causal_multihorizon_shadow import runner

        contract = _contract()
        _train, _processed, models, scoring = _typed_stage_evidence(contract)
        contract = replace(contract, train_identity_sha256=models.train_identity_sha256)
        context = _context()
        evidence = DatasetDecisionEvidence(
            "dataset", context, contract, context.database_sha256,
            contract.train_identity_sha256, scoring.decisions,
        )
        bound_result = DatasetResult("dataset", context.test_sizes["dataset"], 1, contract.groups,
                                     evidence.derived_candidates, evidence, context.lanes, context, "qualified")
        with self.assertRaisesRegex(ValueError, "model hashes must cover frozen seeds"):
            runner.TestScoringEvidence(
                scoring.decisions,
                {100: {FROZEN_CONFIG.model_seeds[0]: "d" * 64}},
            )
        with self.assertRaisesRegex(ValueError, "exact final model evidence"):
            runner.build_checkpoint_evidence(
                _train,
                _processed,
                models,
                contract,
                runner.TestScoringEvidence(
                    scoring.decisions,
                    {100: {seed: "d" * 64 for seed in FROZEN_CONFIG.model_seeds}},
                ),
                bound_result,
                0.125,
            )

    def test_prediction_rows_are_label_and_metric_free_with_false_governance_flags(self) -> None:
        """Adding evaluation data or reintroducing any frozen lane must fail publication."""
        from experiments.exp417_causal_multihorizon_shadow import runner

        context = _context()
        results = tuple(_result_for_dataset(context, dataset, candidate=(dataset == "dataset")) for dataset in context.datasets)
        summary = runner.apply_global_workload(results, sum(context.test_sizes.values()), context=context)
        rows = runner.build_prediction_rows(summary)

        self.assertEqual(rows, ({
            "dataset": "dataset", "instance_index": 0,
            "is_hard": False, "is_standard": False,
            "is_priority": False, "is_automatic_normal": False,
        },))
        for row in rows:
            self.assertFalse(set(row) & {"label", "labels", "metric", "precision", "recall", "f1", "tp", "fp"})


def _result_for_dataset(context: FrozenRunContext, dataset: str, *, candidate: bool = False) -> DatasetResult:
    contract = _contract(dataset)
    decisions = _decisions(dataset, candidate=candidate, count=context.test_sizes[dataset])
    evidence = DatasetDecisionEvidence(
        dataset, context, contract, context.database_sha256, contract.train_identity_sha256, decisions
    )
    return DatasetResult(
        dataset, context.test_sizes[dataset], 1, contract.groups, evidence.derived_candidates,
        evidence, context.lanes, context, "qualified"
    )


if __name__ == "__main__":
    unittest.main()
