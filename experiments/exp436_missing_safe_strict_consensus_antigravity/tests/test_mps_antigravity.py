"""TDD coverage for Exp436's optional scalar PyTorch/MPS readiness gate."""

from __future__ import annotations

import unittest
from unittest.mock import patch
from contextlib import contextmanager
import sys
import types

import numpy as np

from experiments.exp436_missing_safe_strict_consensus_antigravity import mps_antigravity as mps_module
from experiments.exp436_missing_safe_strict_consensus_antigravity.mps_antigravity import (
    MpsEngine,
    MpsFixture,
    benchmark_mps_adoption,
)


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        self.value += 1.0
        return self.value


class _Engine:
    backend = "mps"
    pytorch_version = "fake-torch"
    device = "mps"
    device_info = "fake-info"
    metal_available = True

    def score(self, query, references, k, *, batch_size=16):
        distances = np.sqrt(np.mean((np.asarray(references, dtype=np.float32) - np.asarray(query, dtype=np.float32)) ** 2, axis=1, dtype=np.float32), dtype=np.float32)
        return float(np.median(np.partition(distances, int(k) - 1)[:int(k)]))


class _Timer:
    def __call__(self, kind, batch_size, call, clock):
        del batch_size, clock
        call()
        return 2.0 if kind == "cpu" else 1.0


class MpsReadinessTests(unittest.TestCase):
    def test_fixture_and_engine_reject_every_nonfrozen_k(self) -> None:
        query = np.asarray([0.0, 0.0], dtype=np.float32)
        references = np.asarray([[0., 0.], [1., 1.], [2., 2.], [3., 3.], [4., 4.]], dtype=np.float32)
        for k in (1, 2, 4):
            with self.subTest(boundary="fixture", k=k):
                with self.assertRaisesRegex(ValueError, "frozen"):
                    MpsFixture(query, references, 0.5, k=k)

        class Tensor(np.ndarray):
            def __new__(cls, value): return np.asarray(value, dtype=np.float32).view(cls)
            def detach(self): return self
            def __getitem__(self, key): return Tensor(super().__getitem__(key))
            def to(self, device):
                self.to_device = device
                return self
            def numpy(self): return np.asarray(self)

        @contextmanager
        def no_grad():
            yield

        engine = object.__new__(MpsEngine)
        engine.torch = types.SimpleNamespace(
            float32=np.float32, no_grad=no_grad,
            as_tensor=lambda value, dtype=None, device=None: Tensor(np.asarray(value, dtype=dtype)),
            sqrt=lambda value: Tensor(np.sqrt(value, dtype=np.float32)),
            mean=lambda value, dim=None: Tensor(np.mean(value, axis=dim, dtype=np.float32)),
            sort=lambda value: types.SimpleNamespace(values=Tensor(np.sort(value))),
            cat=lambda values: Tensor(np.concatenate(values)),
            mps=types.SimpleNamespace(synchronize=lambda: None),
        )
        for k in (1, 2, 4):
            with self.subTest(boundary="engine", k=k):
                with self.assertRaisesRegex(ValueError, "frozen"):
                    engine.score(query, references, k)

    def test_scalar_mps_fixture_requires_threshold_and_returns_backend_evidence(self) -> None:
        fixture = MpsFixture(
            np.asarray([0.1, 0.1], dtype=np.float32),
            np.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float32),
            threshold=0.5,
        )
        evidence = benchmark_mps_adoption((fixture,), engine=_Engine(), clock=_Clock())
        self.assertEqual(evidence.backend, "mps")
        self.assertTrue(evidence.available)
        self.assertEqual(len(evidence.batches), 5)

    def test_mps_runtime_failure_keeps_sibling_batches_and_lazy_module_has_no_torch_import(self) -> None:
        fixture = MpsFixture(np.asarray([0.1, 0.1], dtype=np.float32), np.asarray([[0., 0.], [1., 1.], [2., 2.]], dtype=np.float32), 0.5)

        class FailsOne(_Engine):
            def score(self, query, references, k, *, batch_size=16):
                if batch_size == 16:
                    raise RuntimeError("device lost")
                return super().score(query, references, k, batch_size=batch_size)

        evidence = benchmark_mps_adoption((fixture,), engine=FailsOne(), timer=_Timer(), clock=_Clock())
        self.assertEqual(evidence.batches[0].state, "runtime_failed")
        self.assertTrue(any(item.passed for item in evidence.batches[1:]))
        self.assertNotIn("torch", mps_module.__dict__)

    def test_every_protocol_output_is_retained_and_timed_cpu_does_no_json_work(self) -> None:
        fixture = MpsFixture(np.asarray([0.1, 0.1], dtype=np.float32), np.asarray([[0., 0.], [1., 1.], [2., 2.]], dtype=np.float32), 1.0)
        fixtures = (fixture,)
        cpu_scores = mps_module._cpu_score_suite(fixtures)
        coordinates = mps_module._candidate_coordinates(fixtures, cpu_scores)
        input_sha, output_sha = mps_module._digest_fixtures(fixtures), mps_module._fixture_output_sha(cpu_scores, coordinates)
        for position in range(7):
            class Sequenced(_Engine):
                def __init__(self) -> None:
                    self.calls = 0

                def score(self, query, references, k, *, batch_size=16):
                    value = super().score(query, references, k, batch_size=batch_size)
                    result = value + (10.0 if self.calls == position else 0.0)
                    self.calls += 1
                    return result

            batch = mps_module._batch(16, fixtures, input_sha, output_sha, cpu_scores, coordinates, Sequenced(), _Clock(), _Timer())
            self.assertEqual(batch.state, "completed")
            self.assertEqual(batch.reason, "candidate_identity_mismatch")
            self.assertFalse(batch.passed)
        in_timed_call = False
        original_dumps = mps_module.json.dumps

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

        with patch.object(mps_module.json, "dumps", side_effect=guarded_dumps):
            evidence = benchmark_mps_adoption(fixtures, engine=_Engine(), timer=timer, clock=_Clock())
        self.assertTrue(evidence.passed)

    def test_invalid_protocol_position_preserves_only_prior_pairs_and_threshold_is_strict(self) -> None:
        fixture = MpsFixture(np.asarray([0.1, 0.1], dtype=np.float32), np.asarray([[0., 0.], [1., 1.], [2., 2.]], dtype=np.float32), 1.0)
        fixtures = (fixture,)
        cpu_scores = mps_module._cpu_score_suite(fixtures)
        coordinates = mps_module._candidate_coordinates(fixtures, cpu_scores)
        input_sha, output_sha = mps_module._digest_fixtures(fixtures), mps_module._fixture_output_sha(cpu_scores, coordinates)
        for position in range(7):
            class Sequenced(_Engine):
                def __init__(self) -> None:
                    self.calls = 0

                def score(self, query, references, k, *, batch_size=16):
                    value = super().score(query, references, k, batch_size=batch_size)
                    result = float("nan") if self.calls == position else value
                    self.calls += 1
                    return result

            batch = mps_module._batch(16, fixtures, input_sha, output_sha, cpu_scores, coordinates, Sequenced(), _Clock(), _Timer())
            self.assertEqual(batch.state, "runtime_failed")
            self.assertEqual(batch.warmup_pairs_completed, min(position, 2))
            self.assertEqual(len(batch.timing_pairs), max(0, position - 2))
            self.assertIsNone(batch.absolute_error_max)
        score = cpu_scores[0]
        equal = MpsFixture(fixture.query, fixture.references, score)
        evidence = benchmark_mps_adoption((equal,), engine=_Engine(), timer=_Timer(), clock=_Clock())
        self.assertTrue(evidence.passed)

    def test_mps_fixture_errors_propagate(self) -> None:
        with self.assertRaisesRegex(ValueError, "fixtures"):
            benchmark_mps_adoption((), engine=_Engine(), clock=_Clock())

    def test_cpu_like_injected_mps_device_cannot_be_available_or_selected(self) -> None:
        fixture = MpsFixture(np.asarray([0.1, 0.1], dtype=np.float32), np.asarray([[0., 0.], [1., 1.], [2., 2.]], dtype=np.float32), 0.5)
        engine = _Engine()
        engine.device = "cpu"
        evidence = benchmark_mps_adoption((fixture,), engine=engine, timer=_Timer(), clock=_Clock())
        self.assertFalse(evidence.available)
        self.assertEqual(evidence.batches, ())

    def test_mps_engine_keeps_global_nearest_three_across_reference_chunks(self) -> None:
        synchronized: list[bool] = []

        class Tensor(np.ndarray):
            def __new__(cls, value):
                return np.asarray(value, dtype=np.float32).view(cls)

            def detach(self):
                return self

            def __getitem__(self, key):
                return Tensor(super().__getitem__(key))

            def to(self, device):
                self.to_device = device
                return self

            def numpy(self):
                return np.asarray(self)

        class MpsBackend:
            @staticmethod
            def is_built(): return True

            @staticmethod
            def is_available(): return True

        torch = types.ModuleType("torch")
        torch.__version__ = "fake-torch"
        torch.float32 = np.float32
        torch.backends = types.SimpleNamespace(mps=MpsBackend())
        torch.mps = types.SimpleNamespace(synchronize=lambda: synchronized.append(True))
        torch.as_tensor = lambda value, dtype=None, device=None: Tensor(np.asarray(value, dtype=dtype))
        torch.mean = lambda value, dim=None: Tensor(np.mean(value, axis=dim, dtype=np.float32))
        torch.sqrt = lambda value: Tensor(np.sqrt(value, dtype=np.float32))
        torch.sort = lambda value: types.SimpleNamespace(values=Tensor(np.sort(value)))
        torch.cat = lambda values: Tensor(np.concatenate(values))
        @contextmanager
        def no_grad():
            yield
        torch.no_grad = no_grad
        previous = sys.modules.get("torch")
        sys.modules["torch"] = torch
        try:
            engine = MpsEngine()
            score = engine.score(np.asarray([0., 0.], np.float32), np.asarray([[0., 0.], [4., 4.], [1., 1.], [2., 2.], [3., 3.]], np.float32), 3, batch_size=2)
            self.assertEqual(score, 1.0)
            self.assertEqual(synchronized, [True])
        finally:
            if previous is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = previous


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
