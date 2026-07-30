"""Synthetic-only parity and adoption-gate tests for Exp417 backends."""

from __future__ import annotations

import json
import inspect
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from experiments.exp417_causal_multihorizon_shadow.backend import (
    BenchmarkTrainGroup,
    TorchBenchmarkMeasurement,
    benchmark_scoring_backend,
    benchmark_torch_devices,
    canonical_mps_fallback_reason,
    load_frozen_backend_audit,
    persistence_numba,
    persistence_numpy,
    probe_mps_host,
    select_benchmark_strata,
    select_canonical_backend,
)
from experiments.exp417_causal_multihorizon_shadow.model import CausalMultiHorizonCNN
from experiments.exp417_causal_multihorizon_shadow.preprocessing import build_horizons
from experiments.exp417_causal_multihorizon_shadow.contracts import (
    BackendAudit,
    LengthGroup,
    ProcessedGroup,
    RawSeries,
    RawTestDataset,
    RawTrainDataset,
)


def train_source(dataset: str, count: int, length: int) -> BenchmarkTrainGroup:
    positions = np.arange(length, dtype=np.float32)
    series = tuple(
        RawSeries(index, tuple((50.0 if index == 0 else index * 0.01) + np.sin(positions / (2.0 + (index % 3)))))
        for index in range(count)
    )
    train = RawTrainDataset(dataset=dataset, series=series)
    channels = {
        item.instance_index: np.stack((np.asarray(item.values, dtype=np.float32), positions / length))
        for item in series
    }
    processed = ProcessedGroup(
        LengthGroup(length=length, series=series), 0.0, 1.0, 0.0, 1.0, "diagnostic_only", channels
    )
    return BenchmarkTrainGroup(train, processed)


def scoring_fixture() -> tuple[dict[int, tuple[np.ndarray, ...]], dict[int, dict[int, np.ndarray]]]:
    oof: dict[int, tuple[np.ndarray, ...]] = {}
    queries: dict[int, dict[int, np.ndarray]] = {0: {}, 1: {}}
    for horizon in (1, 2, 10):
        base = np.arange(10, dtype=np.float64)
        oof[horizon] = tuple(base + (index / 100.0) for index in range(100))
        queries[0][horizon] = np.array([0.0, 100.0, 100.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        queries[1][horizon] = np.arange(10, dtype=np.float64)
    return oof, queries


class PersistenceParityTests(unittest.TestCase):
    def test_numpy_and_numba_keep_exact_persistence_and_candidate_identity(self) -> None:
        for mask in (
            np.array([], dtype=bool), np.array([False, False, False], dtype=bool),
            np.array([True, True, True], dtype=bool), np.array([True, False, True, False]),
            np.random.default_rng(417).random(127) > 0.73,
            np.array([False, True, True, False, True, False, False, True], dtype=bool)[::2],
        ):
            with self.subTest(shape=mask.shape, contiguous=mask.flags.c_contiguous):
                self.assertEqual(persistence_numpy(mask), persistence_numba(mask))
        import experiments.exp417_causal_multihorizon_shadow.backend as backend
        oof, queries = scoring_fixture()
        coordinates, _ = backend._numpy_candidate_coordinates(oof, queries)
        self.assertEqual(coordinates, (0,))
        self.assertTrue(coordinates)
        audit = benchmark_scoring_backend(oof, queries, repeats=1)
        self.assertTrue(audit.candidate_identity)
        self.assertEqual(audit.max_normalized_error, 0.0)
        self.assertEqual(audit.spearman, 1.0)

    def test_divergent_numba_candidate_path_fails_exact_identity(self) -> None:
        import experiments.exp417_causal_multihorizon_shadow.backend as backend

        oof, queries = scoring_fixture()
        original = backend._numba_candidate_coordinates
        try:
            backend._numba_candidate_coordinates = lambda *_args: ((), np.array([0.0]))
            audit = benchmark_scoring_backend(oof, queries, repeats=1)
        finally:
            backend._numba_candidate_coordinates = original
        self.assertFalse(audit.candidate_identity)


class BenchmarkStrataTests(unittest.TestCase):
    def test_train_only_strata_are_deterministic_ordered_and_capped_at_nine(self) -> None:
        sources = tuple(
            train_source(f"dataset-{11 - rank:02d}", 50 + rank, 64 + (rank % 4) * 8)
            for rank in range(12)
        )
        first = select_benchmark_strata(sources)
        self.assertLessEqual(len(first), 9)
        self.assertEqual(first, select_benchmark_strata(tuple(reversed(sources))))
        self.assertEqual(tuple((item.dataset, item.length) for item in first), tuple(sorted(
            (item.dataset, item.length) for item in first
        )))

    def test_strata_reject_test_metadata_and_unbound_processed_groups(self) -> None:
        source = train_source("synthetic", 50, 64)
        with self.assertRaisesRegex(ValueError, "RawTrainDataset"):
            BenchmarkTrainGroup(RawTestDataset("test", source.train.series), source.processed)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "bind"):
            BenchmarkTrainGroup(source.train, train_source("other", 51, 64).processed)


class MPSGateTests(unittest.TestCase):
    @staticmethod
    def measurement(**changes: object) -> TorchBenchmarkMeasurement:
        values: dict[str, object] = dict(candidate_identity=True, max_normalized_error=1e-4,
            spearman=.999, speedup=1.1, finite_outputs=True, fallback=False, environment="host")
        values.update(changes)
        return TorchBenchmarkMeasurement(**values)  # type: ignore[arg-type]

    def test_each_failed_gate_is_cpu_with_its_exact_reason(self) -> None:
        for field, value, reason in (
            ("candidate_identity", False, "candidate_identity"), ("max_normalized_error", .000100001, "max_normalized_error"),
            ("spearman", .998999, "spearman"), ("speedup", 1.09999, "speedup"),
            ("finite_outputs", False, "nonfinite_outputs"), ("fallback", True, "device_fallback"),
        ):
            with self.subTest(field=field):
                audit = benchmark_torch_devices(measurement=self.measurement(**{field: value}))
                self.assertEqual((audit.backend, audit.fallback_reason), ("cpu", reason))

    def test_isolated_evidence_cannot_positively_adopt_mps(self) -> None:
        isolated = benchmark_torch_devices(measurement=self.measurement(environment="isolated"))
        unavailable = benchmark_torch_devices(measurement=self.measurement(environment="isolated", available=False))
        self.assertEqual((isolated.backend, isolated.fallback_reason), ("cpu", "isolated_environment"))
        self.assertEqual(unavailable.fallback_reason, "isolated_mps_unavailable")
        self.assertEqual(select_canonical_backend((isolated,)), "cpu")


class CanonicalSelectionTests(unittest.TestCase):
    def test_selection_requires_every_host_mps_audit_and_is_order_independent(self) -> None:
        numba = BackendAudit("numba", True, 0.0, 1.0, 10.0, False)
        passed = BackendAudit("mps", True, 0.0, 1.0, 1.1, False)
        failed = BackendAudit("cpu", True, 0.0, 1.0, 1.0, True, "speedup")
        self.assertEqual(select_canonical_backend((numba, passed)), "mps")
        self.assertEqual(select_canonical_backend((passed, failed, numba)), "cpu")
        self.assertEqual(select_canonical_backend((numba, failed, passed)), "cpu")
        self.assertEqual(canonical_mps_fallback_reason((passed, failed)), "speedup")

    def test_canonical_failure_reason_has_fixed_precedence(self) -> None:
        unavailable = BackendAudit("cpu", True, 0.0, 1.0, 1.0, True, "mps_unavailable")
        slow = BackendAudit("cpu", True, 0.0, 1.0, 1.0, True, "speedup")
        self.assertEqual(canonical_mps_fallback_reason((slow, unavailable)), "mps_unavailable")
        with self.assertRaisesRegex(ValueError, "MPS"):
            select_canonical_backend(())


class FrozenTask7ArtifactTests(unittest.TestCase):
    _FROZEN_PAYLOAD = {
        "config_identity_sha256": "bbf06cdca6336bc62e642c445264d949e6be34d14c4b7abb9f45134479ccd498",
        "execution_audit": {
            "backend": "cpu",
            "candidate_identity": True,
            "fallback": False,
            "fallback_reason": None,
            "max_normalized_error": 4.0948700075205124e-07,
            "spearman": 0.9999999999845686,
            "speedup": 0.6709759390840965,
        },
        "host": {"architecture": "arm64", "device": "Apple M2 Max"},
        "mps_adoption_audit": {
            "backend": "cpu",
            "candidate_identity": True,
            "fallback": True,
            "fallback_reason": "speedup",
            "max_normalized_error": 4.0948700075205124e-07,
            "spearman": 0.9999999999845686,
            "speedup": 0.6709759390840965,
        },
        "numba": {"candidate_identity": True, "speedup": 32.49217872476399},
        "runtime": {"torch": "2.13.0"},
        "schema": "exp417.task7-backend-audit.v1",
        "selected_backend": "cpu",
        "task7_commit": "81c6ff36da7b8cdf09fc625adb2b87ad7202e05f",
    }

    @classmethod
    def _bytes(cls, payload: dict[str, object] | None = None) -> bytes:
        return (json.dumps(payload or cls._FROZEN_PAYLOAD, sort_keys=True, separators=(",", ":")) + "\n").encode()

    def test_only_independently_pinned_task7_artifact_is_accepted(self) -> None:
        """Recomputing a mutable file-local digest must not authorize forged evidence."""
        with TemporaryDirectory() as directory:
            path = Path(directory) / "task7_backend_audit.json"
            path.write_bytes(self._bytes())
            audit = load_frozen_backend_audit(path)
            self.assertEqual(audit.backend, "cpu")
            self.assertFalse(audit.fallback)
            self.assertEqual(audit.speedup, 0.6709759390840965)
            from experiments.exp417_causal_multihorizon_shadow.runner import (
                RunnerResources,
                RunnerRuntime,
            )
            resources = RunnerResources(
                Path("database"), Path("291-prediction"), Path("291-manifest"),
                Path("292-prediction"), Path("292-manifest"), path,
            )
            self.assertIsInstance(RunnerRuntime.production(resources), RunnerRuntime)

            forged = json.loads(self._bytes())
            forged["execution_audit"]["speedup"] = 99.0
            forged["mps_adoption_audit"]["speedup"] = 99.0
            forged["payload_sha256"] = "self-declared-and-recomputed"
            path.write_bytes(self._bytes(forged))
            with self.assertRaisesRegex(ValueError, "trusted digest"):
                load_frozen_backend_audit(path)

    def test_wrong_frozen_config_identity_is_rejected_even_with_valid_json(self) -> None:
        """An artifact from another threshold/config contract cannot authorize execution."""
        with TemporaryDirectory() as directory:
            path = Path(directory) / "task7_backend_audit.json"
            forged = dict(self._FROZEN_PAYLOAD)
            forged["config_identity_sha256"] = "0" * 64
            path.write_bytes(self._bytes(forged))
            with self.assertRaisesRegex(ValueError, "trusted digest|config identity"):
                load_frozen_backend_audit(path)


class HostMPSProbeTests(unittest.TestCase):
    def test_host_probe_reports_only_its_own_environment(self) -> None:
        probe = probe_mps_host()
        self.assertEqual(probe.environment, "host")
        if probe.available:
            self.assertTrue(probe.forward_backward_synchronized)
        else:
            self.assertEqual(probe.reason, "mps_unavailable")

    def test_typed_synthetic_train_strata_benchmark_is_finite_or_fails_closed(self) -> None:
        torch.manual_seed(41701)
        audit = benchmark_torch_devices(select_benchmark_strata((train_source("synthetic", 102, 64),)), repeats=1)
        self.assertEqual(select_canonical_backend((audit,)), audit.backend)
        self.assertTrue(np.isfinite((audit.max_normalized_error, audit.spearman, audit.speedup)).all())

    def test_cpu_candidate_uses_unmodified_model_residuals_without_calibration_branch(self) -> None:
        import experiments.exp417_causal_multihorizon_shadow.backend as backend

        self.assertNotIn("calibrate_candidate", inspect.signature(backend._is_candidate).parameters)
        source = train_source("synthetic", 102, 64)
        indices = tuple(item.instance_index for item in source.processed.group.series)
        inputs = torch.from_numpy(np.stack([source.processed.channels[index] for index in indices]).copy())
        torch.manual_seed(41701)
        _scores, coordinates, _elapsed, _fallback = backend._run_torch_stage(
            CausalMultiHorizonCNN(build_horizons(64)), inputs, torch.device("cpu"), "synthetic", indices, 1
        )
        self.assertIn(("synthetic", 0), coordinates)
        self.assertLess(len(coordinates), len(indices))


if __name__ == "__main__":
    unittest.main()
