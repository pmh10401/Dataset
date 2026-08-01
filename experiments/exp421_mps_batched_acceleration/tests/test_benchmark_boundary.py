from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
from types import MappingProxyType, SimpleNamespace
import os

import numpy as np

from experiments.exp417_causal_multihorizon_shadow.contracts import RawSeries, RawTrainDataset
from experiments.exp417_causal_multihorizon_shadow.preprocessing import build_horizons, preprocess_length_groups
from experiments.exp417_causal_multihorizon_shadow.scoring import (
    consensus_candidate,
    score_horizon,
    seed_decision,
)
from experiments.exp421_mps_batched_acceleration import benchmark
from experiments.exp421_mps_batched_acceleration.config import FROZEN_CONFIG
from experiments.exp421_mps_batched_acceleration.contracts import BackendMeasurement, BatchKey, MpsAssessment
from experiments.exp421_mps_batched_acceleration.policy import MpsCapability


class BenchmarkBoundaryTests(unittest.TestCase):
    """Boundary tests never open the private database or start a benchmark."""

    def _fixture(self) -> tuple[sqlite3.Connection, RawTrainDataset]:
        connection = sqlite3.connect(":memory:")
        connection.execute("CREATE TABLE datasets (name TEXT NOT NULL)")
        connection.execute("INSERT INTO datasets(name) VALUES ('A_train'), ('B_train')")
        connection.commit()
        rows = []
        positions = np.arange(64, dtype=np.float64)
        for index in range(100):
            values = np.sin(positions / (2.0 + index % 7)) + index * 0.01
            rows.append(RawSeries(index, tuple(float(value) for value in values)))
        return connection, RawTrainDataset("A_train", tuple(rows))

    @contextmanager
    def _connection_context(self, connection: sqlite3.Connection):
        yield connection

    def _fallback_assessments(self):
        rows = {}
        for size in FROZEN_CONFIG.candidate_batch_sizes:
            measurement = BackendMeasurement(
                batch_size=size,
                backend="cpu",
                transfer_seconds=0.01,
                compute_seconds=0.01,
                synchronize_seconds=0.0,
                total_seconds=0.02,
                finite=True,
                candidate_coordinates_sha256="a" * 64,
            )
            key = BatchKey(size, FROZEN_CONFIG.frozen_model_seeds[0], FROZEN_CONFIG.frozen_perturbation_seeds[0])
            rows[size] = MpsAssessment(
                batch_key=key,
                cpu_measurement=measurement,
                mps_measurement=None,
                parity_report=None,
                speedup=None,
                uses_mps=False,
                fallback_reason="mps_unavailable",
                mps_available=False,
                device="Linux arm64 CPU",
            )
        return rows

    def test_train_loader_is_the_only_values_boundary_and_sample_is_bounded(self):
        connection, raw = self._fixture()
        calls = []

        def loader(connection_arg, dataset, **kwargs):
            calls.append((dataset, kwargs))
            return raw

        with patch.object(benchmark, "load_raw_train_values_from_connection", side_effect=loader), patch.object(
            benchmark, "preprocess_length_groups", return_value=preprocess_length_groups(raw)
        ):
            sample = benchmark._select_train_sample(
                connection,
                database_sha256=FROZEN_CONFIG.exp417_config.database_sha256,
                dataset_names=None,
                max_series=benchmark.MAX_BENCHMARK_SERIES,
                max_batch_bytes=benchmark.MAX_BATCH_BYTES,
            )
        self.assertIsNotNone(sample)
        self.assertLessEqual(len(sample.indices), benchmark.MAX_BENCHMARK_SERIES)
        self.assertEqual(sample.channels.shape[1:], (2, 64))
        self.assertTrue(np.isfinite(sample.channels).all())
        self.assertEqual([item[0] for item in calls], ["A_train"])
        self.assertFalse(hasattr(benchmark, "load_raw_test_values"))
        self.assertNotIn("TEST", benchmark.__dict__)
        connection.close()

    def test_explicit_names_are_sorted_and_exact_lengths_are_not_padded(self):
        connection, raw = self._fixture()
        calls = []

        def loader(connection_arg, dataset, **kwargs):
            calls.append(dataset)
            return raw

        with patch.object(benchmark, "load_raw_train_values_from_connection", side_effect=loader), patch.object(
            benchmark, "preprocess_length_groups", return_value=preprocess_length_groups(raw)
        ):
            sample = benchmark._select_train_sample(
                connection,
                database_sha256=FROZEN_CONFIG.exp417_config.database_sha256,
                dataset_names=["A_train"],
                max_series=10,
                max_batch_bytes=benchmark.MAX_BATCH_BYTES,
            )
        self.assertEqual(calls, ["A_train"])
        self.assertEqual(len(sample.indices), 10)
        self.assertEqual(sample.channels.shape, (10, 2, 64))
        connection.close()

    def test_measurement_rejects_undersized_sample_before_any_fit(self):
        sample = self._fake_measurement_sample()
        undersized = replace(
            sample,
            indices=sample.indices[:16],
            channels=sample.channels[:16],
        )
        with self.assertRaises(benchmark.BenchmarkBoundaryError):
            benchmark._measurements_for_sample(
                undersized,
                capability=MpsCapability(
                    torch_version="unavailable",
                    mps_built=False,
                    mps_available=False,
                    device_name="Linux x86_64 CPU",
                    mlx_version="unavailable",
                    mlx_metal_version="unavailable",
                    mlx_availability="unavailable",
                    mlx_eligibility_reason="package_absent",
                ),
                config=FROZEN_CONFIG,
                max_batch_bytes=benchmark.MAX_BATCH_BYTES,
            )

    def test_coordinate_digest_binds_exact_candidate_coordinates(self):
        first = benchmark._coordinate_digest([1, 4], dataset="A_train", length=64)
        second = benchmark._coordinate_digest([1, 5], dataset="A_train", length=64)
        self.assertNotEqual(first, second)

    def test_public_runner_uses_fresh_root_and_unavailable_mps_falls_back(self):
        connection, raw = self._fixture()
        groups = preprocess_length_groups(raw)
        capability = MpsCapability(
            torch_version="unavailable",
            mps_built=False,
            mps_available=False,
            device_name="Linux x86_64 CPU",
            mlx_version="unavailable",
            mlx_metal_version="unavailable",
            mlx_availability="unavailable",
            mlx_eligibility_reason="package_absent",
        )
        with tempfile.TemporaryDirectory() as temp:
            output_root = Path(temp) / "attempt-001"
            manifest_sentinel = object()

            def fake_source_metadata():
                return "a" * 40, {"experiments/exp421_mps_batched_acceleration/benchmark.py": "b" * 64}, "c" * 64

            evidence = {
                "fit_population_count": 100,
                "fit_population_sha256": "f" * 64,
                "model_state_sha256": "1" * 64,
                "model_stop_sha256": "2" * 64,
                "loss_history_sha256": "3" * 64,
                "oof_sha256": "4" * 64,
                "calibration_sha256": "5" * 64,
                "perturbation_sha256": "6" * 64,
                "qualification_sha256": "7" * 64,
                "synthetic_qualification_sha256": "8" * 64,
                "synthetic_qualification_status": "mps_unavailable",
            }
            with patch.object(benchmark, "verified_read_only_connection", return_value=self._connection_context(connection)), patch.object(
                benchmark, "load_raw_train_values_from_connection", return_value=raw
            ), patch.object(benchmark, "preprocess_length_groups", return_value=groups), patch.object(
                benchmark, "assess_mps_capability", return_value=capability
            ), patch.object(benchmark, "_perturbation_provenance", return_value="d" * 64), patch.object(
                benchmark, "_measurements_for_sample", return_value=(
                    self._fallback_assessments(), np.array([], dtype=np.int64), evidence
                )
            ), patch.object(benchmark, "_source_manifest_metadata", side_effect=fake_source_metadata), patch.object(
                benchmark, "_git_commit", return_value="e" * 40
            ), patch.object(benchmark, "build_manifest", return_value=manifest_sentinel) as build, patch.object(
                benchmark, "write_manifest", return_value=manifest_sentinel
            ) as write:
                result = benchmark.run_train_only_benchmark(
                    Path(temp) / "fixture.db", output_root, config=FROZEN_CONFIG, dataset_names=["A_train"]
                )
            self.assertIs(result, manifest_sentinel)
            self.assertEqual(build.call_args.kwargs["attempt_root"], output_root.resolve())
            self.assertEqual(write.call_args.args[0], output_root.resolve() / "manifest.json")
            provenance = build.call_args.kwargs["provenance"]
            self.assertEqual(provenance["fit_population_count"], 100)
            self.assertEqual(provenance["fit_population_sha256"], "f" * 64)
            self.assertEqual(provenance["process_id"], os.getpid())
            self.assertIn("benchmark_started_at", provenance)
            self.assertIn("benchmark_finished_at", provenance)
            self.assertTrue(provenance["benchmark_finished_at"] >= provenance["benchmark_started_at"])
            connection.close()

    def test_candidate_indices_replay_full_oof_seed_consensus(self):
        """The candidate helper must use the real OOF universe and all seeds."""
        horizons = build_horizons(64)
        indices = tuple(range(200))
        oof_by_seed = {}
        residuals_by_seed = []
        for seed in FROZEN_CONFIG.frozen_model_seeds:
            by_index = {}
            query_by_horizon = {horizon: [] for horizon in horizons}
            for index in indices:
                by_index[index] = {}
                for horizon in horizons:
                    length = 64 - horizon
                    values = (
                        np.full(length, 1000.0, dtype=np.float64)
                        if index == indices[-1]
                        else np.linspace(0.1, 2.0, length, dtype=np.float64) + index * 0.001
                    )
                    by_index[index][horizon] = values
                    query_by_horizon[horizon].append(values)
            oof_by_seed[seed] = by_index
            residuals_by_seed.append(
                {horizon: np.stack(query_by_horizon[horizon]) for horizon in horizons}
            )

        selected, evidence, _ = benchmark._candidate_indices(
            residuals_by_seed,
            indices,
            FROZEN_CONFIG,
            oof_by_seed=oof_by_seed,
        )
        self.assertEqual(selected.tolist(), [indices[-1]])
        self.assertTrue(all(len(value) == 64 and set(value) != {"0"} for value in evidence.values()))

        decisions = []
        for seed in FROZEN_CONFIG.frozen_model_seeds:
            scores = {
                horizon: score_horizon(
                    tuple(oof_by_seed[seed][index][horizon] for index in indices),
                    residuals_by_seed[FROZEN_CONFIG.frozen_model_seeds.index(seed)][horizon][-1],
                    horizon=horizon,
                )
                for horizon in horizons
            }
            decisions.append(seed_decision(scores, seed=seed, config=FROZEN_CONFIG.exp417_config))
        self.assertTrue(all(decision.candidate for decision in decisions))
        self.assertTrue(consensus_candidate(decisions))

    def test_tie_aware_spearman_matches_average_rank_semantics(self):
        left = np.array([1.0, 1.0, 2.0, 3.0], dtype=np.float64)
        right = np.array([2.0, 1.0, 1.0, 3.0], dtype=np.float64)
        expected = float(np.corrcoef(np.array([1.5, 1.5, 3.0, 4.0]), np.array([3.0, 1.5, 1.5, 4.0]))[0, 1])
        self.assertAlmostEqual(benchmark._spearman(left, right), expected)
        self.assertEqual(benchmark._spearman(np.ones(4), right), 0.0)

    def _fake_measurement_sample(self):
        connection, raw = self._fixture()
        with patch.object(benchmark, "load_raw_train_values_from_connection", return_value=raw), patch.object(
            benchmark, "preprocess_length_groups", return_value=preprocess_length_groups(raw)
        ):
            sample = benchmark._select_train_sample(
                connection,
                database_sha256=FROZEN_CONFIG.exp417_config.database_sha256,
                dataset_names=["A_train"],
                max_series=benchmark.MAX_BENCHMARK_SERIES,
                max_batch_bytes=benchmark.MAX_BATCH_BYTES,
            )
        connection.close()
        self.assertIsNotNone(sample)
        return sample

    def _run_fake_mps_measurement(self, *, materialization_error=False, nonfinite=False):
        sample = self._fake_measurement_sample()
        capability = MpsCapability(
            torch_version="2.0.0",
            mps_built=True,
            mps_available=True,
            device_name="Darwin arm64 MPS",
            mlx_version="unavailable",
            mlx_metal_version="unavailable",
            mlx_availability="unavailable",
            mlx_eligibility_reason="package_absent",
        )
        models = tuple(
            SimpleNamespace(
                seed=seed,
                model_sha256=f"{seed:064x}"[-64:],
                stop_epoch=2,
                training_losses=(1.0, 0.5),
                validation_losses=(1.1, 0.6),
            )
            for seed in FROZEN_CONFIG.frozen_model_seeds
        )
        horizons = build_horizons(sample.group.group.length)
        oof = {
            seed: {
                item.instance_index: {
                    horizon: np.ones(sample.group.group.length - horizon, dtype=np.float64)
                    * (item.instance_index + 1)
                    for horizon in horizons
                }
                for item in sample.group.group.series
            }
            for seed in FROZEN_CONFIG.frozen_model_seeds
        }
        class FakeAdapter:
            def __init__(self, seed, device):
                self.seed = seed
                self.device = device

            @classmethod
            def from_trained_seed(cls, model, *, device=None):
                if materialization_error and device.type == "mps":
                    raise RuntimeError("MPS materialization failed")
                return cls(model.seed, device)

            def predict_batch(self, channels):
                return {}

        def fake_timed_predict(adapter, channels):
            if nonfinite and adapter.device.type == "mps":
                raise ValueError("model outputs must be finite")
            return (
                {horizon: np.zeros((channels.shape[0], channels.shape[2] - horizon), dtype=np.float64)
                 for horizon in horizons},
                (0.01, 0.01, 0.0, 0.02),
            )

        def fake_candidates(outputs, indices, config, *, oof_by_seed=None):
            empty = np.asarray([], dtype=np.int64)
            empty.setflags(write=False)
            return empty, MappingProxyType({
                "oof_sha256": "1" * 64,
                "calibration_sha256": "2" * 64,
                "qualification_sha256": "3" * 64,
            }), "4" * 64

        with patch.object(benchmark, "fit_final_seed_models", return_value=models), patch.object(
            benchmark, "train_oof_models", return_value=SimpleNamespace(oof_residuals=oof)
        ), patch.object(benchmark, "FrozenModelAdapter", FakeAdapter), patch.object(
            benchmark, "_timed_predict", side_effect=fake_timed_predict
        ), patch.object(benchmark, "_candidate_indices", side_effect=fake_candidates), patch.object(
            benchmark, "_perturbation_provenance", return_value="a" * 64
        ):
            return benchmark._measurements_for_sample(
                sample,
                capability=capability,
                config=FROZEN_CONFIG,
                max_batch_bytes=benchmark.MAX_BATCH_BYTES,
            )

    def test_mps_materialization_failure_is_explicit_cpu_fallback(self):
        assessments, _, evidence = self._run_fake_mps_measurement(materialization_error=True)
        self.assertTrue(all(not assessment.mps_available for assessment in assessments.values()))
        self.assertTrue(all(assessment.fallback_reason == "mps_unavailable" for assessment in assessments.values()))
        self.assertEqual(evidence["fit_population_count"], 100)
        self.assertNotEqual(evidence["fit_population_sha256"], "0" * 64)

    def test_mps_nonfinite_output_is_available_but_not_selected(self):
        assessments, _, _ = self._run_fake_mps_measurement(nonfinite=True)
        first = assessments[1]
        self.assertTrue(first.mps_available)
        self.assertFalse(first.mps_measurement.finite)
        self.assertEqual(first.fallback_reason, "nonfinite_output")
        self.assertTrue(all(not assessments[size].uses_mps for size in assessments))

    def test_synthetic_qualification_runs_cpu_mps_decisions_and_matches(self):
        sample = self._fake_measurement_sample()
        horizons = build_horizons(sample.group.group.length)
        oof = {
            seed: {
                item.instance_index: {
                    horizon: np.linspace(0.1, 2.0, sample.group.group.length - horizon, dtype=np.float64)
                    + item.instance_index * 0.001
                    for horizon in horizons
                }
                for item in sample.group.group.series
            }
            for seed in FROZEN_CONFIG.frozen_model_seeds
        }

        class PredictAdapter:
            def predict_batch(self, channels):
                return {
                    horizon: np.linspace(
                        0.1, 2.0, channels.shape[2] - horizon, dtype=np.float64
                    )[None, :]
                    for horizon in horizons
                }

        adapters = {seed: PredictAdapter() for seed in FROZEN_CONFIG.frozen_model_seeds}
        digest, matched, status = benchmark._synthetic_qualification_parity(
            sample,
            cpu_adapters=adapters,
            mps_adapters=adapters,
            oof_by_seed=oof,
            config=FROZEN_CONFIG,
        )
        self.assertEqual(status, "compared_pass")
        self.assertTrue(matched)
        self.assertNotEqual(digest, "0" * 64)

    def test_official_exp419_root_is_rejected_before_opening_connection(self):
        opened = []
        with patch.object(benchmark, "verified_read_only_connection", side_effect=lambda *args, **kwargs: opened.append(True)):
            with self.assertRaises(ValueError):
                benchmark.run_train_only_benchmark(
                    Path("/tmp/private.db"),
                    Path("/tmp/exp419_deterministic_cpu_parallel"),
                    config=FROZEN_CONFIG,
                )
        self.assertEqual(opened, [])


if __name__ == "__main__":
    unittest.main()
