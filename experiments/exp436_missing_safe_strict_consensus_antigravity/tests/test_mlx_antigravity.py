"""TDD coverage for Exp436's optional scalar MLX readiness gate."""

from __future__ import annotations

import unittest
from unittest.mock import patch
from contextlib import contextmanager
import sys
import types

import numpy as np

from experiments.exp436_missing_safe_strict_consensus_antigravity import mlx_antigravity as mlx_module
from experiments.exp436_missing_safe_strict_consensus_antigravity.mlx_antigravity import (
    MlxEngine,
    MlxFixture,
    benchmark_mlx_adoption,
)


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        self.value += 1.0
        return self.value


class _Engine:
    backend = "mlx"
    mlx_version = "fake-mlx"
    mlx_metal_version = "fake-metal"
    device = "fake-gpu-device"
    device_info = "fake-info"
    metal_available = True

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, ...], tuple[int, ...], int, int]] = []

    def score(self, query, references, k, *, batch_size=16):
        query_array = np.asarray(query, dtype=np.float32)
        reference_array = np.asarray(references, dtype=np.float32)
        self.calls.append((query_array.shape, reference_array.shape, int(k), int(batch_size)))
        distances = np.sqrt(np.mean((reference_array - query_array) ** 2, axis=1, dtype=np.float32), dtype=np.float32)
        return float(np.median(np.partition(distances, int(k) - 1)[:int(k)]))


class _Timer:
    def __init__(self, device_seconds: dict[int, float] | None = None) -> None:
        self.calls: list[tuple[str, int]] = []
        self.device_seconds = device_seconds or {}

    def __call__(self, kind, batch_size, call, clock):
        del clock
        call()
        self.calls.append((kind, batch_size))
        return 2.0 if kind == "cpu" else self.device_seconds.get(batch_size, 1.0)


class MlxReadinessTests(unittest.TestCase):
    def test_fixture_and_engine_reject_every_nonfrozen_k(self) -> None:
        query = np.asarray([0.0, 0.0], dtype=np.float32)
        references = np.asarray([[0., 0.], [1., 1.], [2., 2.], [3., 3.], [4., 4.]], dtype=np.float32)
        for k in (1, 2, 4):
            with self.subTest(boundary="fixture", k=k):
                with self.assertRaisesRegex(ValueError, "frozen"):
                    MlxFixture(query, references, 0.5, k=k)

        @contextmanager
        def stream(device):
            del device
            yield

        engine = object.__new__(MlxEngine)
        engine.gpu = "gpu:0"
        engine.mx = types.SimpleNamespace(
            stream=stream, float32=np.float32, array=lambda value, dtype=None: np.asarray(value, dtype=dtype),
            square=np.square, mean=lambda value, axis=None: np.mean(value, axis=axis, dtype=np.float32),
            sqrt=lambda value: np.sqrt(value, dtype=np.float32), sort=np.sort, concatenate=np.concatenate,
            eval=lambda value: None, synchronize=lambda device: None,
        )
        for k in (1, 2, 4):
            with self.subTest(boundary="engine", k=k):
                with self.assertRaisesRegex(ValueError, "frozen"):
                    engine.score(query, references, k)

    def test_scalar_protocol_retains_five_real_pairs_and_two_warmups(self) -> None:
        engine = _Engine()
        fixture = MlxFixture(
            np.asarray([0.1, 0.1], dtype=np.float32),
            np.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float32),
            threshold=0.5,
        )
        evidence = benchmark_mlx_adoption((fixture,), engine=engine, clock=_Clock())
        self.assertTrue(evidence.available)
        self.assertEqual(tuple(item.batch_size for item in evidence.batches), (16, 32, 64, 128, 256))
        self.assertTrue(all(item.warmup_pairs_completed == 2 for item in evidence.batches))
        self.assertTrue(all(len(item.timing_pairs) == 5 for item in evidence.batches))
        self.assertTrue(all(shape == (2,) for shape, _, _, _ in engine.calls))

    def test_all_gate_boundaries_and_partial_runtime_failure_are_truthful(self) -> None:
        fixture = MlxFixture(np.asarray([0.1, 0.1], dtype=np.float32), np.asarray([[0., 0.], [1., 1.], [2., 2.]], dtype=np.float32), 1.0)

        class Mixed(_Engine):
            def score(self, query, references, k, *, batch_size=16):
                if batch_size == 16:
                    raise RuntimeError("lost")
                value = super().score(query, references, k, batch_size=batch_size)
                return value + (10.0 if batch_size == 32 else 0.0)

        timer = _Timer({64: 1.0, 128: 2.0, 256: 2.0, 32: 0.1})
        evidence = benchmark_mlx_adoption((fixture,), engine=Mixed(), timer=timer, clock=_Clock())
        failed = evidence.batches[0]
        self.assertEqual(failed.state, "runtime_failed")
        self.assertEqual(failed.timing_pairs, ())
        self.assertIsNone(failed.absolute_error_max)
        mismatch = evidence.batches[1]
        self.assertEqual(mismatch.reason, "candidate_identity_mismatch")
        self.assertFalse(mismatch.passed)
        self.assertEqual(evidence.selected_batch_size, 64)
        self.assertEqual(len(timer.calls), 40)

    def test_speed_gate_requires_one_point_one_and_fixture_hashes_bind_thresholds(self) -> None:
        fixture = MlxFixture(np.asarray([0.1, 0.1], dtype=np.float32), np.asarray([[0., 0.], [1., 1.], [2., 2.]], dtype=np.float32), 0.5)
        slow = benchmark_mlx_adoption((fixture,), engine=_Engine(), timer=_Timer({size: 2.0 / 1.09 for size in (16, 32, 64, 128, 256)}), clock=_Clock())
        self.assertFalse(slow.passed)
        self.assertTrue(all(item.reason == "speedup_below_threshold" for item in slow.batches))
        changed = MlxFixture(fixture.query, fixture.references, 0.6)
        fast = benchmark_mlx_adoption((fixture,), engine=_Engine(), timer=_Timer(), clock=_Clock())
        changed_evidence = benchmark_mlx_adoption((changed,), engine=_Engine(), timer=_Timer(), clock=_Clock())
        self.assertNotEqual(fast.batches[0].fixture_input_sha256, changed_evidence.batches[0].fixture_input_sha256)

    def test_fixture_errors_propagate_and_cpu_provenance_is_not_an_accelerator(self) -> None:
        with self.assertRaisesRegex(ValueError, "fixtures"):
            benchmark_mlx_adoption((), engine=_Engine(), clock=_Clock())
        cpu_like = _Engine()
        cpu_like.device = "cpu:0"
        evidence = benchmark_mlx_adoption((MlxFixture(np.zeros(2, np.float32), np.zeros((3, 2), np.float32), 0.0),), engine=cpu_like, clock=_Clock())
        self.assertFalse(evidence.available)

    def test_first_warmup_mismatch_and_timed_json_work_cannot_be_erased(self) -> None:
        fixture = MlxFixture(np.asarray([0.1, 0.1], dtype=np.float32), np.asarray([[0., 0.], [1., 1.], [2., 2.]], dtype=np.float32), 1.0)

        class FirstMismatch(_Engine):
            def __init__(self) -> None:
                super().__init__()
                self.by_batch: dict[int, int] = {}

            def score(self, query, references, k, *, batch_size=16):
                count = self.by_batch.get(batch_size, 0)
                self.by_batch[batch_size] = count + 1
                value = super().score(query, references, k, batch_size=batch_size)
                return value + (10.0 if count == 0 else 0.0)

        in_timed_call = False
        original_dumps = mlx_module.json.dumps

        def guarded_dumps(*args, **kwargs):
            if in_timed_call:
                raise AssertionError("candidate JSON ran inside timed callable")
            return original_dumps(*args, **kwargs)

        def timer(kind, batch_size, call, clock):
            nonlocal in_timed_call
            del batch_size, clock
            in_timed_call = True
            try:
                call()
            finally:
                in_timed_call = False
            return 2.0 if kind == "cpu" else 1.0

        with patch.object(mlx_module.json, "dumps", side_effect=guarded_dumps):
            evidence = benchmark_mlx_adoption((fixture,), engine=FirstMismatch(), timer=timer, clock=_Clock())
        self.assertTrue(all(item.state == "completed" for item in evidence.batches))
        self.assertTrue(all(item.reason == "candidate_identity_mismatch" for item in evidence.batches))

    def test_each_invalid_protocol_position_preserves_only_prior_pairs_and_threshold_is_strict(self) -> None:
        fixture = MlxFixture(np.asarray([0.1, 0.1], dtype=np.float32), np.asarray([[0., 0.], [1., 1.], [2., 2.]], dtype=np.float32), 1.0)
        fixtures = (fixture,)
        cpu_scores = mlx_module._cpu_score_suite(fixtures)
        coordinates = mlx_module._candidate_coordinates(fixtures, cpu_scores)
        input_sha, output_sha = mlx_module._digest_fixtures(fixtures), mlx_module._fixture_output_sha(cpu_scores, coordinates)
        for position in range(7):
            class Sequenced(_Engine):
                def __init__(self) -> None:
                    super().__init__()
                    self.protocol_calls = 0

                def score(self, query, references, k, *, batch_size=16):
                    value = super().score(query, references, k, batch_size=batch_size)
                    result = float("nan") if self.protocol_calls == position else value
                    self.protocol_calls += 1
                    return result

            batch = mlx_module._runtime_batch("mlx", 16, fixtures, input_sha, output_sha, cpu_scores, coordinates, Sequenced(), _Clock(), _Timer())
            self.assertEqual(batch.state, "runtime_failed")
            self.assertEqual(batch.warmup_pairs_completed, min(position, 2))
            self.assertEqual(len(batch.timing_pairs), max(0, position - 2))
            self.assertIsNone(batch.normalized_error_max)
        equal = MlxFixture(fixture.query, fixture.references, cpu_scores[0])
        evidence = benchmark_mlx_adoption((equal,), engine=_Engine(), timer=_Timer(), clock=_Clock())
        self.assertTrue(evidence.passed)

    def test_mlx_engine_uses_the_explicit_gpu_stream_sync_and_mlx_metal_distribution(self) -> None:
        streams: list[object] = []
        synchronized: list[object] = []

        class Metal:
            @staticmethod
            def is_available() -> bool:
                return True

        core = types.ModuleType("mlx.core")
        core.__version__ = "mlx-module-version"
        core.float32 = np.float32
        core.gpu = "gpu:0"
        core.metal = Metal()
        core.device_info = lambda: {"name": "mock-gpu"}

        @contextmanager
        def stream(device):
            streams.append(device)
            yield

        core.stream = stream
        core.array = lambda value, dtype=None: np.asarray(value, dtype=dtype)
        core.square = np.square
        core.mean = lambda value, axis=None: np.mean(value, axis=axis, dtype=np.float32)
        core.sqrt = lambda value: np.sqrt(value, dtype=np.float32)
        core.sort = np.sort
        core.concatenate = np.concatenate
        core.eval = lambda value: None
        core.synchronize = lambda device: synchronized.append(device)
        package = types.ModuleType("mlx")
        package.core = core
        previous = {name: sys.modules.get(name) for name in ("mlx", "mlx.core")}
        sys.modules["mlx"] = package
        sys.modules["mlx.core"] = core
        try:
            with patch.object(mlx_module.metadata, "version", side_effect=lambda name: {"mlx": "1.2", "mlx-metal": "3.4"}[name]):
                engine = MlxEngine()
                score = engine.score(np.asarray([0., 0.], np.float32), np.asarray([[0., 0.], [1., 1.], [2., 2.], [3., 3.]], np.float32), 3, batch_size=2)
            self.assertEqual(engine.device, "gpu:0")
            self.assertEqual(engine.mlx_version, "1.2")
            self.assertEqual(engine.mlx_metal_version, "3.4")
            self.assertEqual(score, 1.0)
            self.assertEqual(streams, ["gpu:0"])
            self.assertEqual(synchronized, ["gpu:0"])
        finally:
            for name, value in previous.items():
                if value is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = value


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
