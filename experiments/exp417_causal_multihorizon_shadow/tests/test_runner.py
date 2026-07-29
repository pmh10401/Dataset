"""Synthetic boundary tests for Exp417's label-free population runner."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
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
    FrozenLaneSnapshot,
    FrozenRunContext,
    FrozenTrainContract,
    GroupQualification,
    HorizonScore,
    RawTrainDataset,
    SeedDecision,
    TestInstanceDecision,
)
from experiments.exp417_causal_multihorizon_shadow.model import OOFModelBundle
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
    final = tuple(
        synthetic_trained_model(seed, folds[0], horizons)
        for seed in FROZEN_CONFIG.model_seeds
    )
    model_evidence = runner.TrainModelEvidence(
        {100: bundle}, {100: final}, contract.backend, contract.train_identity_sha256
    )
    hashes = {100: {model.seed: model.model_sha256 for model in final}}
    scoring = runner.TestScoringEvidence(_decisions(candidate=False), hashes)
    return train, (processed,), model_evidence, scoring


class RunnerOrderTests(unittest.TestCase):
    def test_full_dataset_path_freezes_train_before_loading_values(self) -> None:
        """Moving TEST loading before the frozen-contract stage is a safety breach."""
        from experiments.exp417_causal_multihorizon_shadow import runner

        context = _context()
        contract = _contract()
        events: list[str] = []
        result = _result(context)
        qualification = runner.TrainQualification(contract.groups, contract.backend)
        patches = {
            "verify_inputs": lambda _: events.append("verify_inputs"),
            "load_train_values": lambda *_: events.append("load_train_values") or object(),
            "preprocess": lambda _: events.append("preprocess") or (object(),),
            "train_oof": lambda _: events.append("train_oof") or object(),
            "qualify_train": lambda *_: events.append("qualify_train") or qualification,
            "freeze_contract": lambda *_: events.append("freeze_contract") or contract,
            "load_test_values": lambda *_: events.append("load_test_values") or object(),
            "score_test": lambda *_: events.append("score_test") or runner.TestScoringEvidence(
                _decisions(candidate=False),
                {100: {seed: "d" * 64 for seed in FROZEN_CONFIG.model_seeds}},
            ),
            "exclude_lanes": lambda evidence: events.append("exclude_lanes") or evidence.derived_candidates,
            "dataset_workload": lambda _: events.append("dataset_workload") or result,
            "checkpoint": lambda *_: events.append("checkpoint"),
        }
        with patch.multiple(runner, **patches):
            self.assertEqual(runner.run_dataset("dataset", context), result)

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
        with (
            patch.object(runner, "verify_inputs"),
            patch.object(runner, "load_train_values", return_value=object()),
            patch.object(runner, "preprocess", return_value=(object(),)),
            patch.object(runner, "train_oof", side_effect=ValueError("synthetic OOF failure")),
            patch.object(runner, "load_test_values", test_loader),
        ):
            with self.assertRaisesRegex(ValueError, "synthetic OOF failure"):
                runner.run_dataset("dataset", context)
        test_loader.assert_not_called()


class RunnerPopulationTests(unittest.TestCase):
    def test_resume_runs_only_missing_datasets_sequentially_and_conserves_population(self) -> None:
        """Re-running completed work or changing frozen dataset order corrupts resume accounting."""
        from experiments.exp417_causal_multihorizon_shadow import runner

        context = _context()
        completed = _result(context)
        store = Mock()
        store.expected_datasets = context.datasets
        store.load_completed.return_value = {"dataset": completed}
        calls: list[str] = []

        def run_missing(dataset: str, supplied_context: FrozenRunContext, supplied_store: object) -> DatasetResult:
            self.assertEqual(supplied_context, context)
            self.assertIs(supplied_store, store)
            calls.append(dataset)
            return _result_for_dataset(supplied_context, dataset)

        with patch.object(runner, "_run_dataset", side_effect=run_missing):
            summary = runner.resume_or_run_population(context, store)

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

        with (
            patch.object(runner, "_validate_result"),
            patch.object(runner, "RunSummary", return_value=Mock(status="failed")),
            patch.object(runner, "apply_global_workload", return_value=global_summary) as gate,
        ):
            self.assertIs(runner.resume_or_run_population(context, store), global_summary)
        gate.assert_called_once()

    def test_checkpoint_writes_only_typed_complete_evidence_to_the_transaction_store(self) -> None:
        """Dropping a checkpoint field would make a resumable result unverifiable."""
        from experiments.exp417_causal_multihorizon_shadow import runner

        context = _context()
        result = _result(context)
        store = Mock()
        material = runner.CheckpointEvidence(
            raw_train_identity_sha256="a" * 64,
            repaired_train_identity_sha256="b" * 64,
            length_group_identity_sha256={100: "c" * 64},
            canonical_model_hashes={100: {seed: "d" * 64 for seed in FROZEN_CONFIG.model_seeds}},
            score_calibration_hashes={100: "e" * 64},
            elapsed_seconds=0.125,
        )

        runner.checkpoint(result, store, material)

        store.write_dataset.assert_called_once_with(
            result,
            raw_train_identity_sha256="a" * 64,
            repaired_train_identity_sha256="b" * 64,
            length_group_identity_sha256={100: "c" * 64},
            canonical_model_hashes={100: {seed: "d" * 64 for seed in FROZEN_CONFIG.model_seeds}},
            score_calibration_hashes={100: "e" * 64},
            backend_audit=result.decision_evidence.frozen_train_contract.backend,
            elapsed_seconds=0.125,
        )

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
        target = _result(context)
        completed = {
            dataset: _result_for_dataset(context, dataset)
            for dataset in context.datasets[1:]
        }
        with TemporaryDirectory() as temporary:
            store = Exp417CheckpointStore(Path(temporary) / "checkpoint", _provenance(), context.datasets)
            qualification = runner.TrainQualification(contract.groups, contract.backend)
            with (
                patch.object(store, "load_completed", return_value=completed),
                patch.object(runner, "verify_inputs"),
                patch.object(runner, "load_train_values", return_value=train),
                patch.object(runner, "preprocess", return_value=processed),
                patch.object(runner, "train_oof", return_value=model_evidence),
                patch.object(runner, "qualify_train", return_value=qualification),
                patch.object(runner, "freeze_contract", return_value=contract),
                patch.object(runner, "load_test_values", return_value=object()),
                patch.object(runner, "score_test", return_value=scoring),
                patch.object(runner, "dataset_workload", return_value=target),
            ):
                summary = runner.resume_or_run_population(context, store)
            self.assertEqual(summary.status, "complete")
            self.assertEqual(store.load_completed(), {"dataset": target})

    def test_checkpoint_evidence_rejects_missing_or_corrupt_model_bindings(self) -> None:
        """A missing seed or an altered model identity cannot be checkpointed."""
        from experiments.exp417_causal_multihorizon_shadow import runner

        contract = _contract()
        _train, _processed, models, scoring = _typed_stage_evidence(contract)
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
