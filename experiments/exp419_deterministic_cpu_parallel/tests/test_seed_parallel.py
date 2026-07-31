"""CPU-only evidence-parity tests for Exp419 seed concurrency."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
import numpy as np

from experiments.exp417_causal_multihorizon_shadow.contracts import (
    BackendAudit,
    RawSeries,
    RawTrainDataset,
)
from experiments.exp417_causal_multihorizon_shadow.inputs import (
    load_raw_train_values_from_connection,
)
from experiments.exp417_causal_multihorizon_shadow.preprocessing import (
    preprocess_length_groups,
)
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


class SessionParityTests(unittest.TestCase):
    """The session loader is plumbing only; TRAIN model evidence must match."""

    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    def test_session_backed_and_reference_train_paths_keep_exact_seed_outputs(self) -> None:
        original = synthetic_group().group.series
        reference = RawTrainDataset(
            "synthetic",
            tuple(
                RawSeries(
                    item.instance_index,
                    tuple(
                        float(value)
                        for value in np.asarray(item.values, dtype="<f4")
                    ),
                )
                for item in original
            ),
        )
        with TemporaryDirectory() as temporary:
            database = Path(temporary) / "session.sqlite"
            connection = sqlite3.connect(database)
            try:
                connection.executescript(
                    """
                    CREATE TABLE datasets (id INTEGER PRIMARY KEY, name TEXT);
                    CREATE TABLE instances (
                        dataset_id INTEGER,
                        split TEXT,
                        label TEXT,
                        instance_index INTEGER,
                        values_blob BLOB
                    );
                    INSERT INTO datasets (id, name) VALUES (1, 'synthetic');
                    """
                )
                connection.executemany(
                    """
                    INSERT INTO instances
                    (dataset_id, split, label, instance_index, values_blob)
                    VALUES (1, 'TRAIN', '0', ?, ?)
                    """,
                    [
                        (
                            item.instance_index,
                            np.asarray(item.values, dtype="<f4").tobytes(),
                        )
                        for item in reference.series
                    ],
                )
                connection.commit()
            finally:
                connection.close()

            live = sqlite3.connect(database)
            try:
                session = load_raw_train_values_from_connection(
                    live,
                    "synthetic",
                    database_sha256="a" * 64,
                    datasets=("synthetic",),
                )
            finally:
                live.close()

        self.assertEqual(session, reference)
        reference_models = _production_train_models(
            preprocess_length_groups(reference), CPU_AUDIT
        )
        session_models = _production_train_models(
            preprocess_length_groups(session), CPU_AUDIT
        )
        self.assertEqual(
            session_models.train_identity_sha256,
            reference_models.train_identity_sha256,
        )
        self.assertEqual(
            session_models.processed_identity_sha256,
            reference_models.processed_identity_sha256,
        )
        length = next(iter(session_models.oof_bundles))
        self.assertEqual(
            tuple(
                model.model_sha256
                for model in session_models.oof_bundles[length].models
            ),
            tuple(
                model.model_sha256
                for model in reference_models.oof_bundles[length].models
            ),
        )
        for seed, by_index in session_models.oof_bundles[length].oof_residuals.items():
            for index, by_horizon in by_index.items():
                for horizon, residual in by_horizon.items():
                    self.assertEqual(
                        residual.tobytes(),
                        reference_models.oof_bundles[length]
                        .oof_residuals[seed][index][horizon]
                        .tobytes(),
                    )
