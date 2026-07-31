"""CPU-only evidence-parity tests for Exp419 seed concurrency."""

from __future__ import annotations

from dataclasses import replace
import unittest
from unittest.mock import patch

import torch

from experiments.exp417_causal_multihorizon_shadow.contracts import BackendAudit
from experiments.exp417_causal_multihorizon_shadow.runner import (
    RunnerRuntime,
    _production_train_models,
)
from experiments.exp417_causal_multihorizon_shadow.tests.test_model import (
    synthetic_group,
)
from experiments.exp419_deterministic_cpu_parallel.seed_parallel import (
    train_models_seed_parallel,
)


CPU_AUDIT = BackendAudit("cpu", True, 0.0, 1.0, 1.0, False)


class SeedParallelEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # The official worker applies this same contract before it invokes the
        # injected training callable.  The synthetic CPU parity test must not
        # introduce a separate intra-op threading regime.
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

    def test_three_seed_tasks_reassemble_exact_serial_evidence(self) -> None:
        group = synthetic_group()
        sequential = _production_train_models((group,), CPU_AUDIT)
        parallel = train_models_seed_parallel((group,), CPU_AUDIT, 3)

        self.assertEqual(
            parallel.train_identity_sha256, sequential.train_identity_sha256
        )
        self.assertEqual(
            parallel.processed_identity_sha256,
            sequential.processed_identity_sha256,
        )
        length = group.group.length
        self.assertEqual(
            tuple(model.model_sha256 for model in parallel.oof_bundles[length].models),
            tuple(model.model_sha256 for model in sequential.oof_bundles[length].models),
        )
        self.assertEqual(
            tuple(model.stop_epoch for model in parallel.oof_bundles[length].models),
            tuple(model.stop_epoch for model in sequential.oof_bundles[length].models),
        )
        self.assertEqual(
            tuple(model.training_losses for model in parallel.oof_bundles[length].models),
            tuple(model.training_losses for model in sequential.oof_bundles[length].models),
        )
        self.assertEqual(
            tuple(model.validation_losses for model in parallel.oof_bundles[length].models),
            tuple(model.validation_losses for model in sequential.oof_bundles[length].models),
        )
        self.assertEqual(
            tuple(model.model_sha256 for model in parallel.final_models[length]),
            tuple(model.model_sha256 for model in sequential.final_models[length]),
        )
        for seed, residuals_by_index in parallel.oof_bundles[length].oof_residuals.items():
            for index, residuals_by_horizon in residuals_by_index.items():
                for horizon, residual in residuals_by_horizon.items():
                    self.assertEqual(
                        residual.tobytes(),
                        sequential.oof_bundles[length].oof_residuals[seed][index][horizon].tobytes(),
                    )

    def test_invalid_seed_worker_count_fails_before_training(self) -> None:
        with self.assertRaisesRegex(ValueError, "seed_workers"):
            train_models_seed_parallel((synthetic_group(),), CPU_AUDIT, 0)


class RunnerRuntimeInjectionTests(unittest.TestCase):
    def test_production_factory_changes_only_train_model_callable(self) -> None:
        base = RunnerRuntime(
            verify=lambda _context: None,
            load_train=lambda _dataset, _context: None,
            preprocess=lambda _train: (),
            train_models=lambda _processed: None,
            qualify=lambda _dataset, _processed, _models: None,
            freeze=lambda _dataset, _qualification, _models: None,
            load_test=lambda _dataset, _contract, _context: None,
            score=lambda _test, _contract, _models: None,
        )
        replacement = lambda _processed: None
        with patch.object(RunnerRuntime, "production", return_value=base):
            actual = RunnerRuntime.production_with_train_models(object(), replacement)

        self.assertIs(actual.train_models, replacement)
        self.assertEqual(
            replace(actual, train_models=base.train_models), base
        )

