from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.exp417_causal_multihorizon_shadow.contracts import RawSeries, RawTrainDataset
from experiments.exp417_causal_multihorizon_shadow.preprocessing import preprocess_length_groups
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

            with patch.object(benchmark, "verified_read_only_connection", return_value=self._connection_context(connection)), patch.object(
                benchmark, "load_raw_train_values_from_connection", return_value=raw
            ), patch.object(benchmark, "preprocess_length_groups", return_value=groups), patch.object(
                benchmark, "assess_mps_capability", return_value=capability
            ), patch.object(benchmark, "_perturbation_provenance", return_value="d" * 64), patch.object(
                benchmark, "_measurements_for_sample", return_value=(self._fallback_assessments(), np.array([], dtype=np.int64))
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
            connection.close()

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
