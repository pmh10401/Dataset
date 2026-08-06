"""TDD coverage for Exp435's optional MLX parity and adoption gate."""

from __future__ import annotations

from dataclasses import dataclass
import builtins
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from experiments.exp435_train_only_consensus_recovery_antigravity.contracts_antigravity import (
    BackendDecision,
)
from experiments.exp435_train_only_consensus_recovery_antigravity.mlx_antigravity import (
    MlxEngine,
    MlxUnavailable,
    benchmark_mlx_adoption,
    score_channel_mlx,
    score_with_backend,
)
from experiments.exp435_train_only_consensus_recovery_antigravity import cli_antigravity as cli_module
from experiments.exp435_train_only_consensus_recovery_antigravity import mlx_antigravity as mlx_module
from experiments.exp435_train_only_consensus_recovery_antigravity.features_antigravity import (
    score_channel_cpu,
)


class _StepClock:
    """Deterministic clock used to avoid wall-time assertions in gate tests."""

    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        self.value += 1.0
        return self.value


class FakeMlxEngine:
    """Small deterministic engine double with the production engine protocol."""

    def __init__(
        self,
        *,
        error: float = 0.0,
        candidate_flip: bool = False,
        speedup: float = 1.2,
        nonfinite: bool = False,
    ) -> None:
        self.error = float(error)
        self.candidate_flip = bool(candidate_flip)
        self.speedup = float(speedup)
        self.nonfinite = bool(nonfinite)
        self.mlx_version = "fake-mlx"
        self.mlx_metal_version = "fake-metal"
        self.device = "fake-metal-device"
        self.device_info = "fake-device-info"
        self.metal_available = True

    def score(self, query: object, references: object, k: int, *, batch_size: int = 16) -> np.ndarray:
        del batch_size
        query_array = np.asarray(query, dtype=np.float32)
        if query_array.ndim == 1:
            query_array = query_array[None, :]
        references_array = np.asarray(references, dtype=np.float32)
        output = []
        for row in query_array:
            delta = np.asarray(references_array - row[None, :], dtype=np.float32)
            distances = np.sqrt(np.mean(delta * delta, axis=1, dtype=np.float32), dtype=np.float32)
            nearest = np.partition(distances, int(k) - 1)[: int(k)]
            output.append(np.float32(np.median(nearest)))
        result = np.asarray(output, dtype=np.float32)
        if self.candidate_flip:
            result = result + np.float32(10.0)
        else:
            result = result + np.float32(self.error)
        if self.nonfinite:
            result[0] = np.float32(np.nan)
        return result

    def timed_call(self, kind: str, batch_size: int, call, clock) -> float:
        """Run ``call`` and expose reproducible CPU/MLX timings to the gate."""

        del batch_size
        start = clock()
        call()
        end = clock()
        measured = max(end - start, 1e-9)
        return measured if kind == "cpu" else measured / self.speedup


def _fixtures() -> tuple[dict[str, object], ...]:
    queries = np.asarray(
        [[0.1, 0.1], [1.1, 1.1], [2.1, 2.1], [0.5, 0.5]], dtype=np.float32
    )
    references = np.asarray(
        [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32
    )
    return ({"queries": queries, "references": references, "thresholds": np.asarray([0.95] * 4, dtype=np.float32)},)


class MlxAdoptionTests(unittest.TestCase):
    @property
    def fixtures(self) -> tuple[dict[str, object], ...]:
        return _fixtures()

    def test_adopts_only_exact_candidate_identity_and_speed(self) -> None:
        decision = benchmark_mlx_adoption(
            fixtures=self.fixtures,
            engine=FakeMlxEngine(error=1e-6, candidate_flip=False, speedup=1.2),
            clock=_StepClock(),
        )
        self.assertTrue(decision.adopted)
        self.assertEqual(decision.backend, "mlx")
        self.assertEqual(decision.smoke_seed, 43501)
        self.assertEqual(decision.selected_batch_size, 16)

    def test_readiness_fixture_uses_one_query_row_per_production_score_call(self) -> None:
        with patch.object(
            mlx_module,
            "benchmark_mlx_adoption",
            return_value=BackendDecision.cpu_fallback("fixture"),
        ) as benchmark:
            cli_module._synthetic_mlx_decision()
        fixture = benchmark.call_args.args[0][0]
        self.assertEqual(fixture.queries.shape[0], 1)

    def test_candidate_flip_forces_cpu_even_when_fast(self) -> None:
        decision = benchmark_mlx_adoption(
            fixtures=self.fixtures,
            engine=FakeMlxEngine(error=0.0, candidate_flip=True, speedup=10.0),
            clock=_StepClock(),
        )
        self.assertFalse(decision.adopted)
        self.assertEqual(decision.reason, "candidate_identity_mismatch")

    def test_injected_unavailable_engine_returns_cpu(self) -> None:
        with patch(
            "experiments.exp435_train_only_consensus_recovery_antigravity.mlx_antigravity.MlxEngine",
            side_effect=MlxUnavailable("mlx_unavailable"),
        ):
            decision = benchmark_mlx_adoption(fixtures=self.fixtures, engine=None)
        self.assertFalse(decision.adopted)
        self.assertEqual(decision.backend, "cpu")
        self.assertEqual(decision.reason, "mlx_unavailable")

    def test_nonfinite_output_fails_closed(self) -> None:
        decision = benchmark_mlx_adoption(
            fixtures=self.fixtures,
            engine=FakeMlxEngine(nonfinite=True, speedup=10.0),
            clock=_StepClock(),
        )
        self.assertFalse(decision.adopted)
        self.assertEqual(decision.reason, "nonfinite_output")
        self.assertFalse(decision.finite_outputs)

    def test_absolute_error_gate_is_strict(self) -> None:
        decision = benchmark_mlx_adoption(
            fixtures=self.fixtures,
            engine=FakeMlxEngine(error=1.0001e-5, speedup=10.0),
            clock=_StepClock(),
        )
        self.assertFalse(decision.adopted)
        self.assertEqual(decision.reason, "absolute_error_exceeded")

    def test_speed_gate_is_strict(self) -> None:
        decision = benchmark_mlx_adoption(
            fixtures=self.fixtures,
            engine=FakeMlxEngine(speedup=1.099),
            clock=_StepClock(),
        )
        self.assertFalse(decision.adopted)
        self.assertEqual(decision.reason, "speedup_below_threshold")

    def test_one_failed_batch_does_not_poison_a_passing_batch(self) -> None:
        class FailsOnlyFirstBatch(FakeMlxEngine):
            def score(self, query, references, k, *, batch_size=16):
                if batch_size == 16:
                    raise RuntimeError("forced kernel failure")
                return super().score(query, references, k, batch_size=batch_size)

        decision = benchmark_mlx_adoption(
            fixtures=self.fixtures,
            engine=FailsOnlyFirstBatch(speedup=1.2),
            clock=_StepClock(),
        )
        self.assertTrue(decision.adopted)
        self.assertNotEqual(decision.selected_batch_size, 16)
        first = next(item for item in decision.batch_evidence if item.batch_size == 16)
        self.assertFalse(first.parity_passed)
        self.assertIn("mlx_runtime_error", first.reason)

    def test_one_candidate_mismatch_batch_does_not_poison_a_passing_batch(self) -> None:
        class FlipsOnlyFirstBatch(FakeMlxEngine):
            def score(self, query, references, k, *, batch_size=16):
                output = super().score(query, references, k, batch_size=batch_size)
                if batch_size == 16:
                    output = output + np.float32(10.0)
                return output

        decision = benchmark_mlx_adoption(
            fixtures=self.fixtures,
            engine=FlipsOnlyFirstBatch(speedup=1.2),
            clock=_StepClock(),
        )
        self.assertTrue(decision.adopted)
        self.assertNotEqual(decision.selected_batch_size, 16)
        first = next(item for item in decision.batch_evidence if item.batch_size == 16)
        self.assertEqual(first.reason, "candidate_identity_mismatch")

    def test_normalized_error_uses_the_one_e_minus_twelve_floor(self) -> None:
        queries = np.zeros((2, 2), dtype=np.float32)
        references = np.zeros((3, 2), dtype=np.float32)
        decision = benchmark_mlx_adoption(
            fixtures=(
                {
                    "queries": queries,
                    "references": references,
                    "thresholds": np.asarray([1.0, 1.0], dtype=np.float32),
                },
            ),
            engine=FakeMlxEngine(error=1e-6, speedup=10.0),
            clock=_StepClock(),
        )
        self.assertFalse(decision.adopted)
        self.assertEqual(decision.reason, "normalized_error_exceeded")
        self.assertGreater(decision.normalized_error_max or 0.0, 1e-5)

    def test_normalized_error_boundary_above_one_e_minus_five_fails(self) -> None:
        class RelativeError(FakeMlxEngine):
            def score(self, query, references, k, *, batch_size=16):
                output = super().score(query, references, k, batch_size=batch_size)
                return np.asarray(output * np.float32(1.0 + 1.0001e-5), dtype=np.float32)

        decision = benchmark_mlx_adoption(
            fixtures=self.fixtures,
            engine=RelativeError(speedup=10.0),
            clock=_StepClock(),
        )
        self.assertFalse(decision.adopted)
        self.assertEqual(decision.reason, "normalized_error_exceeded")
        self.assertGreater(decision.normalized_error_max or 0.0, 1e-5)

    def test_all_batches_failing_never_adopts(self) -> None:
        decision = benchmark_mlx_adoption(
            fixtures=self.fixtures,
            engine=FakeMlxEngine(error=1.0, speedup=100.0),
            clock=_StepClock(),
        )
        self.assertFalse(decision.adopted)
        self.assertEqual(len(decision.timings), 5)
        self.assertIn(decision.reason, {"absolute_error_exceeded", "candidate_identity_mismatch"})

    def test_shape_only_failure_has_precise_top_level_reason(self) -> None:
        class WrongShape(FakeMlxEngine):
            def score(self, query, references, k, *, batch_size=16):
                return np.asarray([1.0], dtype=np.float32)

        decision = benchmark_mlx_adoption(
            fixtures=self.fixtures,
            engine=WrongShape(speedup=10.0),
            clock=_StepClock(),
        )
        self.assertFalse(decision.adopted)
        self.assertEqual(decision.reason, "score_shape_mismatch")

    def test_timing_tie_chooses_smaller_batch(self) -> None:
        class Tied(FakeMlxEngine):
            def timed_call(self, kind: str, batch_size: int, call, clock) -> float:
                del batch_size
                call()
                return 1.0 if kind == "cpu" else 0.5

        decision = benchmark_mlx_adoption(
            fixtures=self.fixtures,
            engine=Tied(speedup=2.0),
            clock=_StepClock(),
        )
        self.assertTrue(decision.adopted)
        self.assertEqual(decision.selected_batch_size, 16)

    def test_backend_decision_serializes_gate_evidence(self) -> None:
        decision = benchmark_mlx_adoption(
            fixtures=self.fixtures,
            engine=FakeMlxEngine(error=1e-6, speedup=1.2),
            clock=_StepClock(),
        )
        payload = decision.to_dict()
        for key in (
            "smoke_seed",
            "mlx_version",
            "mlx_metal_version",
            "device",
            "timings",
            "parity_passed",
            "reason",
            "normalized_error_max",
            "batch_evidence",
            "metal_available",
            "device_info",
        ):
            self.assertIn(key, payload)

    def test_score_with_cpu_decision_never_calls_mlx(self) -> None:
        decision = BackendDecision.cpu_fallback("mlx_unavailable")
        score = score_with_backend(
            decision,
            np.asarray([0.0, 0.0], dtype=np.float32),
            np.asarray([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
        )
        self.assertEqual(score, 0.0)

    def test_mlx_batch_score_matches_the_float32_cpu_reference(self) -> None:
        fixture = _fixtures()[0]
        output = score_channel_mlx(
            fixture["queries"],
            fixture["references"],
            3,
            batch_size=2,
            engine=FakeMlxEngine(),
        )
        expected = np.asarray(
            [score_channel_cpu(row, fixture["references"], 3) for row in fixture["queries"]],
            dtype=np.float32,
        )
        self.assertEqual(output.dtype, np.dtype(np.float32))
        np.testing.assert_array_equal(output, expected)

    def test_adopted_decision_uses_mlx_and_runtime_failure_fails_closed(self) -> None:
        decision = benchmark_mlx_adoption(
            fixtures=self.fixtures,
            engine=FakeMlxEngine(speedup=1.2),
            clock=_StepClock(),
        )
        self.assertTrue(decision.adopted)
        query = np.asarray([0.1, 0.1], dtype=np.float32)
        references = np.asarray(
            [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32
        )
        expected = score_channel_cpu(query, references, 3)
        self.assertEqual(
            score_with_backend(decision, query, references, 3, engine=FakeMlxEngine()),
            expected,
        )

        class Broken(FakeMlxEngine):
            def score(self, query, references, k, *, batch_size=16):
                raise RuntimeError("device lost")

        with self.assertRaisesRegex(MlxUnavailable, "mlx_runtime_error:RuntimeError:device lost"):
            score_with_backend(decision, query, references, 3, engine=Broken())

        class OtherDevice(FakeMlxEngine):
            def __init__(self):
                super().__init__()
                self.device = "other-metal-device"

        class OtherHardwareInfo(FakeMlxEngine):
            def __init__(self):
                super().__init__()
                self.device_info = "other-device-info"

        class NoMetal(FakeMlxEngine):
            def __init__(self):
                super().__init__()
                self.metal_available = False

        for incompatible in (OtherDevice(), OtherHardwareInfo(), NoMetal()):
            with self.subTest(incompatible=type(incompatible).__name__):
                with self.assertRaises(MlxUnavailable):
                    score_with_backend(decision, query, references, 3, engine=incompatible)

    def test_injected_engine_failure_is_explicit(self) -> None:
        class UnavailableEngine:
            def score(self, query, references, k, *, batch_size=16):
                raise MlxUnavailable("mlx_unavailable")

        with self.assertRaises(MlxUnavailable):
            score_channel_mlx(
                np.asarray([[0.0, 0.0]], dtype=np.float32),
                np.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float32),
                3,
                engine=UnavailableEngine(),
            )

    def test_mlx_reference_batches_are_exact_in_readiness_and_production(self) -> None:
        """One selected batch size bounds references, not production queries."""

        class Metal:
            @staticmethod
            def is_available() -> bool:
                return True

            @staticmethod
            def device_info() -> dict[str, object]:
                return {"architecture": "reference-batching-stub"}

        class Core(types.ModuleType):
            __version__ = "0.30.0"
            float32 = np.float32
            metal = Metal()

            def __init__(self, name: str) -> None:
                super().__init__(name)
                self.transfers: list[np.ndarray] = []

            @staticmethod
            def default_device() -> object:
                return "Device(gpu,0)"

            def array(self, value, dtype=None):
                array = np.asarray(value, dtype=dtype)
                self.transfers.append(array.copy())
                return array

            @staticmethod
            def square(value):
                return np.square(value)

            @staticmethod
            def mean(value, axis=None):
                return np.mean(value, axis=axis, dtype=np.float32)

            @staticmethod
            def sqrt(value):
                return np.sqrt(value, dtype=np.float32)

            @staticmethod
            def sort(value, axis=1):
                return np.sort(value, axis=axis)

            @staticmethod
            def concatenate(values, axis=0):
                return np.concatenate(values, axis=axis)

            @staticmethod
            def eval(value):
                return None

            @staticmethod
            def synchronize():
                return None

        core = Core("mlx.core")
        package = types.ModuleType("mlx")
        package.core = core
        previous = {name: sys.modules.get(name) for name in ("mlx", "mlx.core")}
        sys.modules["mlx"] = package
        sys.modules["mlx.core"] = core
        try:
            references = np.stack(
                (np.arange(513, dtype=np.float32), np.arange(513, dtype=np.float32)), axis=1
            )
            queries = np.asarray([[-1.0, -1.0]], dtype=np.float32)

            class RecordingMlxEngine(MlxEngine):
                def __init__(self) -> None:
                    super().__init__()
                    self.reference_batches: list[tuple[int, tuple[int, ...]]] = []

                def score(self, query, refs, k, *, batch_size=16):
                    before = len(core.transfers)
                    output = super().score(query, refs, k, batch_size=batch_size)
                    transfers = core.transfers[before:]
                    chunk_rows = tuple(
                        transfer.shape[0]
                        for transfer in transfers
                        if transfer.ndim == 2 and transfer.shape[0] > 0 and transfer[0, 0] >= 0
                    )
                    self.reference_batches.append((int(batch_size), chunk_rows))
                    return output

                def timed_call(self, kind, batch_size, call, clock):
                    del batch_size, clock
                    call()
                    return 1.0 if kind == "cpu" else 0.5

            with patch(
                "experiments.exp435_train_only_consensus_recovery_antigravity.mlx_antigravity.metadata.version",
                return_value="0.30.0",
            ):
                engine = RecordingMlxEngine()

            output = score_channel_mlx(queries, references, 3, batch_size=128, engine=engine)
            expected = np.asarray([score_channel_cpu(queries[0], references, 3)], dtype=np.float32)
            np.testing.assert_array_equal(output, expected)
            self.assertEqual(engine.reference_batches, [(128, (128, 128, 128, 128, 1))])

            engine.reference_batches.clear()
            decision = benchmark_mlx_adoption(
                fixtures=({"queries": queries, "references": references, "thresholds": 2.0},),
                engine=engine,
                clock=_StepClock(),
            )
            self.assertTrue(decision.adopted)
            self.assertTrue(all(item.candidate_identity_match for item in decision.batch_evidence))
            self.assertTrue(all(item.parity_passed for item in decision.batch_evidence))
            for batch_size in (16, 32, 64, 128, 256):
                expected_chunks = (batch_size,) * (513 // batch_size)
                if 513 % batch_size:
                    expected_chunks += (513 % batch_size,)
                matching = [chunks for size, chunks in engine.reference_batches if size == batch_size]
                self.assertEqual(matching, [expected_chunks] * 7)
        finally:
            for name, value in previous.items():
                if value is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = value

    def test_selected_batch_size_reaches_production_engine(self) -> None:
        class SelectedBatchEngine(FakeMlxEngine):
            def __init__(self) -> None:
                super().__init__(speedup=2.0)
                self.batch_sizes: list[int] = []

            def score(self, query, references, k, *, batch_size=16):
                self.batch_sizes.append(int(batch_size))
                return super().score(query, references, k, batch_size=batch_size)

            def timed_call(self, kind, batch_size, call, clock):
                del clock
                call()
                return 1.0 if kind == "cpu" else (0.25 if batch_size == 64 else 0.5)

        engine = SelectedBatchEngine()
        decision = benchmark_mlx_adoption(fixtures=self.fixtures, engine=engine, clock=_StepClock())
        self.assertTrue(decision.adopted)
        self.assertEqual(decision.selected_batch_size, 64)
        engine.batch_sizes.clear()
        score = score_with_backend(
            decision,
            np.asarray([0.1, 0.1], dtype=np.float32),
            np.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32),
            3,
            engine=engine,
        )
        self.assertEqual(score, score_channel_cpu(np.asarray([0.1, 0.1], dtype=np.float32), np.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32), 3))
        self.assertEqual(engine.batch_sizes, [64])

    def test_mlx_engine_constructs_from_standard_bundled_metal_stub(self) -> None:
        class Metal:
            @staticmethod
            def is_available() -> bool:
                return True

            @staticmethod
            def device_info() -> dict[str, object]:
                return {"architecture": "stub-gpu", "memory_size": 123}

        class Core(types.ModuleType):
            __version__ = "0.30.0"
            float32 = np.float32
            metal = Metal()

            @staticmethod
            def default_device() -> object:
                return "Device(gpu,0)"

            @staticmethod
            def array(value, dtype=None):
                return np.asarray(value, dtype=dtype)

            @staticmethod
            def square(value):
                return np.square(value)

            @staticmethod
            def mean(value, axis=None):
                return np.mean(value, axis=axis, dtype=np.float32)

            @staticmethod
            def sqrt(value):
                return np.sqrt(value, dtype=np.float32)

            @staticmethod
            def sort(value, axis=1):
                return np.sort(value, axis=axis)

            @staticmethod
            def eval(value):
                return None

            @staticmethod
            def synchronize():
                return None

        package = types.ModuleType("mlx")
        core = Core("mlx.core")
        package.core = core
        previous = {name: sys.modules.get(name) for name in ("mlx", "mlx.core")}
        sys.modules["mlx"] = package
        sys.modules["mlx.core"] = core
        try:
            with patch("experiments.exp435_train_only_consensus_recovery_antigravity.mlx_antigravity.metadata.version", return_value="0.30.0"):
                engine = MlxEngine()
            self.assertTrue(engine.metal_available)
            self.assertEqual(engine.mlx_version, "0.30.0")
            self.assertEqual(engine.mlx_metal_version, "bundled-with-mlx-0.30.0")
            self.assertEqual(engine.device, "Device(gpu,0)")
            self.assertIn("stub-gpu", engine.device_info or "")
            result = engine.score(
                np.asarray([[0.1, 0.1]], dtype=np.float32),
                np.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float32),
                3,
            )
            self.assertEqual(result.dtype, np.dtype(np.float32))
            self.assertEqual(result.shape, (1,))
        finally:
            for name, value in previous.items():
                if value is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = value

    def test_mlx_engine_prefers_core_device_info_over_legacy_metal_probe(self) -> None:
        calls: list[str] = []

        class Metal:
            @staticmethod
            def is_available() -> bool:
                return True

            @staticmethod
            def device_info() -> dict[str, object]:
                calls.append("legacy")
                raise AssertionError("legacy device_info must not be called")

        core = types.ModuleType("mlx.core")
        core.__version__ = "0.32.0"
        core.metal = Metal()
        core.default_device = lambda: "Device(gpu,0)"

        def device_info() -> dict[str, object]:
            calls.append("modern")
            return {"memory_size": 456, "architecture": "modern-stub-gpu"}

        core.device_info = device_info
        package = types.ModuleType("mlx")
        package.core = core
        previous = {name: sys.modules.get(name) for name in ("mlx", "mlx.core")}
        sys.modules["mlx"] = package
        sys.modules["mlx.core"] = core
        try:
            with patch(
                "experiments.exp435_train_only_consensus_recovery_antigravity.mlx_antigravity.metadata.version",
                return_value="0.32.0",
            ):
                engine = MlxEngine()
            self.assertEqual(calls, ["modern"])
            self.assertEqual(
                engine.device_info,
                '{"architecture":"modern-stub-gpu","memory_size":456}',
            )
        finally:
            for name, value in previous.items():
                if value is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = value

    def test_mlx_engine_uses_legacy_device_info_when_modern_api_is_absent(self) -> None:
        calls: list[str] = []

        class Metal:
            @staticmethod
            def is_available() -> bool:
                return True

            @staticmethod
            def device_info() -> dict[str, object]:
                calls.append("legacy")
                return {"architecture": "legacy-stub-gpu"}

        core = types.ModuleType("mlx.core")
        core.__version__ = "0.30.0"
        core.metal = Metal()
        core.default_device = lambda: "Device(gpu,0)"
        package = types.ModuleType("mlx")
        package.core = core
        previous = {name: sys.modules.get(name) for name in ("mlx", "mlx.core")}
        sys.modules["mlx"] = package
        sys.modules["mlx.core"] = core
        try:
            with patch(
                "experiments.exp435_train_only_consensus_recovery_antigravity.mlx_antigravity.metadata.version",
                return_value="0.30.0",
            ):
                engine = MlxEngine()
            self.assertEqual(calls, ["legacy"])
            self.assertEqual(engine.device_info, '{"architecture":"legacy-stub-gpu"}')
        finally:
            for name, value in previous.items():
                if value is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = value

    def test_modern_device_info_probe_failure_preserves_cpu_fallback_provenance(self) -> None:
        calls: list[str] = []

        class Metal:
            @staticmethod
            def is_available() -> bool:
                return True

            @staticmethod
            def device_info() -> dict[str, object]:
                calls.append("legacy")
                return {"architecture": "must-not-be-used"}

        core = types.ModuleType("mlx.core")
        core.__version__ = "0.32.0"
        core.metal = Metal()
        core.default_device = lambda: "Device(gpu,0)"

        def device_info() -> dict[str, object]:
            calls.append("modern")
            raise RuntimeError("modern probe failed")

        core.device_info = device_info
        package = types.ModuleType("mlx")
        package.core = core
        previous = {name: sys.modules.get(name) for name in ("mlx", "mlx.core")}
        sys.modules["mlx"] = package
        sys.modules["mlx.core"] = core
        try:
            with patch(
                "experiments.exp435_train_only_consensus_recovery_antigravity.mlx_antigravity.metadata.version",
                return_value="0.32.0",
            ):
                decision = benchmark_mlx_adoption(self.fixtures)
            self.assertEqual(calls, ["modern"])
            self.assertFalse(decision.adopted)
            self.assertEqual(decision.reason, "mlx_device_info_failed:RuntimeError")
            self.assertEqual(decision.mlx_version, "0.32.0")
            self.assertEqual(decision.device, "Device(gpu,0)")
            self.assertTrue(decision.metal_available)
            self.assertIsNone(decision.device_info)
        finally:
            for name, value in previous.items():
                if value is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = value

    def test_direct_absence_test_injects_import_failure_independent_of_host(self) -> None:
        original_import = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name == "mlx.core" or name.startswith("mlx."):
                raise ModuleNotFoundError("forced absence")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=blocked):
            with self.assertRaises(MlxUnavailable) as context:
                MlxEngine()
        self.assertEqual(context.exception.reason, "mlx_unavailable")

    def test_partial_discovery_is_preserved_in_cpu_fallback_evidence(self) -> None:
        class Metal:
            @staticmethod
            def is_available() -> bool:
                return False

            @staticmethod
            def device_info() -> dict[str, object]:
                return {"architecture": "stub-cpu"}

        core = types.ModuleType("mlx.core")
        core.__version__ = "0.30.1"
        core.metal = Metal()
        core.default_device = lambda: "Device(cpu,0)"
        package = types.ModuleType("mlx")
        package.core = core
        previous = {name: sys.modules.get(name) for name in ("mlx", "mlx.core")}
        sys.modules["mlx"] = package
        sys.modules["mlx.core"] = core
        try:
            with patch("experiments.exp435_train_only_consensus_recovery_antigravity.mlx_antigravity.metadata.version", return_value="0.30.1"):
                decision = benchmark_mlx_adoption(self.fixtures)
            self.assertFalse(decision.adopted)
            self.assertEqual(decision.reason, "mlx_metal_unavailable")
            self.assertEqual(decision.mlx_version, "0.30.1")
            self.assertEqual(decision.mlx_metal_version, "bundled-with-mlx-0.30.1")
            self.assertEqual(decision.device, "Device(cpu,0)")
            self.assertFalse(decision.metal_available)
            self.assertIn("stub-cpu", decision.device_info or "")
        finally:
            for name, value in previous.items():
                if value is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = value


if __name__ == "__main__":
    unittest.main()
