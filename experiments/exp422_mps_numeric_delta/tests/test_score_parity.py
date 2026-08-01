"""Synthetic score-path parity tests for Exp422's bounded TRAIN stage."""

from __future__ import annotations

from hashlib import sha256
from types import MappingProxyType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.exp417_causal_multihorizon_shadow.contracts import RawSeries, RawTrainDataset
from experiments.exp417_causal_multihorizon_shadow.preprocessing import preprocess_length_groups
from experiments.exp421_mps_batched_acceleration.config import FROZEN_CONFIG as EXP421_CONFIG
from experiments.exp422_mps_numeric_delta.capability import DeviceCapability
from experiments.exp422_mps_numeric_delta.config import FROZEN_CONFIG
from experiments.exp422_mps_numeric_delta.contracts import TimingEvidence
import experiments.exp422_mps_numeric_delta.score_parity as score_parity
from experiments.exp422_mps_numeric_delta.score_parity import (
    ScoreStageResult,
    TrainSample,
    _ScoreBackendError,
    _qualification_evidence,
    run_score_parity,
)
from experiments.exp422_mps_numeric_delta.contracts import StageResult


def _sample() -> TrainSample:
    positions = np.arange(64, dtype=np.float64)
    raw = RawTrainDataset(
        "Adiac",
        tuple(
            RawSeries(
                index,
                tuple(float(value) for value in np.sin(positions / (3.0 + index % 5)) + index * 0.02),
            )
            for index in range(100)
        ),
    )
    group = preprocess_length_groups(raw)[0]
    indices = tuple(range(64))
    raw_by_index = {index: np.asarray(raw.series[index].values, dtype=np.float64) for index in indices}
    digest = sha256(b"synthetic-score-fixture").hexdigest()
    return TrainSample(
        dataset="Adiac",
        length=64,
        train_count=100,
        raw_waveform_sha256=digest,
        bounded_series_count=64,
        group=group,
        indices=indices,
        channels=np.stack([group.channels[index] for index in indices]),
        raw_by_index=raw_by_index,
        max_series=64,
        max_batch_bytes=256 * 1024 * 1024,
    )


def _capability(available: bool) -> DeviceCapability:
    return DeviceCapability(
        torch_version="2.13.0",
        mps_built=available,
        mps_available=available,
        device_name="Darwin arm64 MPS" if available else "Darwin arm64 CPU",
        mlx_version="unavailable",
        mlx_metal_version="unavailable",
        mlx_availability="unavailable",
        fallback_reason=None if available else "mps_unavailable",
    )


class FakeAdapter:
    def __init__(self, model, device: torch.device):
        self.model = model
        self.device = device

    @classmethod
    def from_trained_seed(cls, model, *, device):
        return cls(model, device)


class ScoreParityTests(unittest.TestCase):
    def test_unavailable_mps_skips_before_cpu_fit(self) -> None:
        """A missing capability must not start expensive TRAIN fitting."""
        with patch.object(score_parity, "fit_final_seed_models") as fit:
            result = run_score_parity(_sample(), _capability(False), FROZEN_CONFIG)
        fit.assert_not_called()
        self.assertEqual(result.stage.state, "skipped")
        self.assertEqual(result.stage.reason, "mps_unavailable")
        self.assertEqual(result.reports, ())

    def test_score_parity_fits_once_and_reports_every_exp422_batch_policy(self) -> None:
        """A refit per backend/batch or inherited Exp421 policy changes the experiment."""
        sample = _sample()
        horizons = (1, 2, 6)
        models = tuple(
            SimpleNamespace(
                seed=seed,
                horizons=horizons,
                model_sha256=sha256(str(seed).encode()).hexdigest(),
                stop_epoch=3,
                training_losses=(3.0, 2.0, 1.0),
                validation_losses=(3.0, 2.0, 1.0),
            )
            for seed in FROZEN_CONFIG.model_seeds
        )
        oof = {
            seed: {
                index: {
                    horizon: np.linspace(
                        0.01 + index * 1e-4,
                        1.0 + seed * 1e-8,
                        sample.length - horizon,
                        dtype=np.float64,
                    )
                    for horizon in horizons
                }
                for index in range(sample.train_count)
            }
            for seed in FROZEN_CONFIG.model_seeds
        }
        oof_bundle = SimpleNamespace(oof_residuals=MappingProxyType(oof))

        qualification_sizes: list[int] = []

        def predict(adapters, channels, *, batch_size, max_batch_bytes):
            backend = next(iter(adapters.values())).device.type
            rows = channels.shape[0]
            raw_outputs = {
                seed: {
                    horizon: np.tile(
                        np.linspace(0.05, 0.95, sample.length - horizon, dtype=np.float64),
                        (rows, 1),
                    )
                    for horizon in horizons
                }
                for seed in FROZEN_CONFIG.model_seeds
            }
            residual_outputs = {
                seed: {
                    horizon: values.copy()
                    for horizon, values in by_horizon.items()
                }
                for seed, by_horizon in raw_outputs.items()
            }
            timing = (
                TimingEvidence("cpu", 0.002, 0.018, 0.0, 0.02)
                if backend == "cpu"
                else TimingEvidence("mps", 0.002, 0.007, 0.001, 0.01)
            )
            chunks = tuple(
                min(batch_size, rows - start) for start in range(0, rows, batch_size)
            )
            return MappingProxyType(raw_outputs), MappingProxyType(residual_outputs), timing, chunks

        def qualification(sample_arg, cpu_arg, mps_arg, caches_arg, config_arg, *, batch_size):
            qualification_sizes.append(batch_size)
            return "p" * 64, "c" * 64, "c" * 64, True

        exp421_batches_before = EXP421_CONFIG.candidate_batch_sizes
        with patch.object(score_parity, "fit_final_seed_models", return_value=models) as fit, patch.object(
            score_parity, "train_oof_models", return_value=oof_bundle
        ) as train_oof, patch.object(score_parity, "FrozenModelAdapter", FakeAdapter), patch.object(
            score_parity, "_predict_for_policy", side_effect=predict
        ), patch.object(score_parity, "_qualification_evidence", side_effect=qualification):
            result = run_score_parity(sample, _capability(True), FROZEN_CONFIG)

        fit.assert_called_once_with(sample.group, device=torch.device("cpu"), config=FROZEN_CONFIG.exp417_config)
        train_oof.assert_called_once_with(sample.group, device=torch.device("cpu"), config=FROZEN_CONFIG.exp417_config)
        self.assertEqual(EXP421_CONFIG.candidate_batch_sizes, exp421_batches_before)
        self.assertEqual(tuple(report.batch_size for report in result.reports), (16, 32, 64, 128, 256))
        self.assertEqual(tuple(report.chunk_sizes for report in result.reports), ((16,), (32,), (64,), (), ()))
        self.assertTrue(all(report.candidate_indices_match for report in result.reports[:3]))
        self.assertTrue(all(report.qualification_match for report in result.reports[:3]))
        self.assertTrue(all(report.spearman == 1.0 for report in result.reports[:3]))
        self.assertTrue(all(report.speedup == 2.0 for report in result.reports[:3]))
        self.assertTrue(all(report.cpu_forward_sha256 == report.mps_forward_sha256 for report in result.reports[:3]))
        self.assertTrue(all(report.cpu_residual_sha256 == report.mps_residual_sha256 for report in result.reports[:3]))
        self.assertTrue(all(report.cpu_forward_sha256 != report.cpu_residual_sha256 for report in result.reports[:3]))
        self.assertTrue(all(report.passes for report in result.reports[:3]))
        self.assertEqual(tuple(report.status for report in result.reports[3:]), ("failed", "failed"))
        self.assertEqual(result.stage.state, "failed")
        self.assertEqual(qualification_sizes, [16, 32, 64])
        self.assertEqual(len(result.oof_residual_sha256), 64)
        self.assertEqual(len(result.calibration_sha256), 64)
        self.assertEqual(len(result.perturbation_sha256), 64)

    def test_policies_larger_than_bounded_sample_fail_closed_without_downshift(self) -> None:
        """A 64-row execution must never be reported under the 128/256 policy labels."""
        sample = _sample()
        horizons = (1, 2, 6)
        models = tuple(
            SimpleNamespace(
                seed=seed,
                horizons=horizons,
                model_sha256=sha256(str(seed).encode()).hexdigest(),
                stop_epoch=3,
                training_losses=(3.0, 2.0, 1.0),
                validation_losses=(3.0, 2.0, 1.0),
            )
            for seed in FROZEN_CONFIG.model_seeds
        )
        oof = {
            seed: {
                index: {
                    horizon: np.linspace(0.01, 1.0, sample.length - horizon, dtype=np.float64)
                    for horizon in horizons
                }
                for index in range(sample.train_count)
            }
            for seed in FROZEN_CONFIG.model_seeds
        }
        def predict(adapters, channels, *, batch_size, max_batch_bytes):
            rows = len(channels)
            raw_outputs = {
                seed: {
                    horizon: np.tile(
                        np.linspace(0.05, 0.95, sample.length - horizon, dtype=np.float64),
                        (rows, 1),
                    )
                    for horizon in horizons
                }
                for seed in FROZEN_CONFIG.model_seeds
            }
            residual_outputs = {
                seed: {horizon: values.copy() for horizon, values in by_horizon.items()}
                for seed, by_horizon in raw_outputs.items()
            }
            backend = "cpu" if next(iter(adapters.values())).device.type == "cpu" else "mps"
            return MappingProxyType(raw_outputs), MappingProxyType(residual_outputs), TimingEvidence(backend, 0.0, 0.01, 0.0, 0.01), (batch_size,)

        with patch.object(score_parity, "fit_final_seed_models", return_value=models), patch.object(
            score_parity,
            "train_oof_models",
            return_value=SimpleNamespace(oof_residuals=MappingProxyType(oof)),
        ), patch.object(score_parity, "FrozenModelAdapter", FakeAdapter), patch.object(
            score_parity, "_predict_for_policy", side_effect=predict
        ), patch.object(score_parity, "_qualification_evidence", return_value=("p" * 64, "c" * 64, "m" * 64, True)):
            result = run_score_parity(sample, _capability(True), FROZEN_CONFIG)
        self.assertEqual(result.stage.state, "failed")
        self.assertEqual(result.reports[3].status, "failed")
        self.assertEqual(result.reports[3].fallback_reason, "fixed_batch_sample_too_small")
        self.assertEqual(result.reports[4].status, "failed")
        self.assertEqual(result.reports[4].fallback_reason, "fixed_batch_sample_too_small")
        self.assertEqual(result.reports[3].chunk_sizes, ())
        self.assertEqual(result.reports[4].chunk_sizes, ())

    def test_score_stage_requires_exact_five_batch_reports(self) -> None:
        """A completed score result cannot silently omit or duplicate a policy row."""
        with self.assertRaisesRegex(ValueError, "exact.*batch"):
            ScoreStageResult(
                _sample(),
                (),
                StageResult.failed("score_parity_gate_failed"),
                *(["a" * 64] * 8),
            )

    def test_qualification_uses_repaired_processed_series(self) -> None:
        """Qualification must start from Exp417's repaired TRAIN waveform, not raw bytes."""
        base = _sample()
        raw_by_index = dict(base.raw_by_index)
        raw_by_index[0] = raw_by_index[0] + 123.0
        sample = TrainSample(
            dataset=base.dataset,
            length=base.length,
            train_count=base.train_count,
            raw_waveform_sha256=base.raw_waveform_sha256,
            bounded_series_count=base.bounded_series_count,
            group=base.group,
            indices=base.indices,
            channels=base.channels,
            raw_by_index=raw_by_index,
            max_series=base.max_series,
            max_batch_bytes=base.max_batch_bytes,
        )
        seen: list[np.ndarray] = []

        class FakeCache(dict):
            digest = "d" * 64

        def fake_cached(_dataset, _index, series, _scale, _seed):
            seen.append(np.asarray(series, dtype=np.float64).copy())
            return FakeCache(
                {
                    name: np.asarray(series, dtype=np.float64).copy()
                    for name in (
                        "level_shift",
                        "gradual_drift",
                        "amplitude_scaling",
                        "period_change",
                        "secondary_spike",
                    )
                }
            )

        def fake_predict(adapters, channels, *, batch_size, max_batch_bytes):
            rows = len(channels)
            outputs = {
                seed: {
                    horizon: np.zeros((rows, sample.length - horizon), dtype=np.float64)
                    for horizon in (1, 2, 6)
                }
                for seed in FROZEN_CONFIG.model_seeds
            }
            backend = "cpu" if next(iter(adapters.values())).device.type == "cpu" else "mps"
            return MappingProxyType(outputs), MappingProxyType(outputs), TimingEvidence(backend, 0.0, 0.01, 0.0, 0.01), (batch_size,)

        with patch.object(score_parity, "build_cached_perturbations", side_effect=fake_cached), patch.object(
            score_parity, "_predict_for_policy", side_effect=fake_predict
        ), patch.object(
            score_parity, "_score_outputs", return_value=(np.zeros(10), (), "a" * 64)
        ):
            _qualification_evidence(
                sample,
                {seed: SimpleNamespace(device=torch.device("cpu")) for seed in FROZEN_CONFIG.model_seeds},
                {seed: SimpleNamespace(device=torch.device("mps")) for seed in FROZEN_CONFIG.model_seeds},
                {},
                FROZEN_CONFIG,
                batch_size=16,
            )
        repaired = np.asarray(base.group.group.series[0].values, dtype=np.float64)
        self.assertTrue(seen)
        np.testing.assert_array_equal(seen[0], repaired)
        self.assertFalse(np.array_equal(seen[0], raw_by_index[0]))

    def test_mps_policy_failure_is_an_explicit_failed_row(self) -> None:
        """MPS inference errors must become a per-policy fallback row, not escape."""
        sample = _sample()
        horizons = (1, 2, 6)
        models = tuple(
            SimpleNamespace(
                seed=seed,
                horizons=horizons,
                model_sha256=sha256(str(seed).encode()).hexdigest(),
                stop_epoch=3,
                training_losses=(3.0, 2.0, 1.0),
                validation_losses=(3.0, 2.0, 1.0),
            )
            for seed in FROZEN_CONFIG.model_seeds
        )
        oof = {
            seed: {
                index: {
                    horizon: np.linspace(0.01, 1.0, sample.length - horizon, dtype=np.float64)
                    for horizon in horizons
                }
                for index in range(sample.train_count)
            }
            for seed in FROZEN_CONFIG.model_seeds
        }

        def predict(adapters, channels, *, batch_size, max_batch_bytes):
            if next(iter(adapters.values())).device.type == "mps":
                raise RuntimeError("synthetic MPS allocation failure")
            rows = len(channels)
            outputs = {
                seed: {
                    horizon: np.ones((rows, sample.length - horizon), dtype=np.float64)
                    for horizon in horizons
                }
                for seed in FROZEN_CONFIG.model_seeds
            }
            return MappingProxyType(outputs), MappingProxyType(outputs), TimingEvidence("cpu", 0.0, 0.01, 0.0, 0.01), (batch_size,)

        with patch.object(score_parity, "fit_final_seed_models", return_value=models), patch.object(
            score_parity, "train_oof_models", return_value=SimpleNamespace(oof_residuals=MappingProxyType(oof))
        ), patch.object(score_parity, "FrozenModelAdapter", FakeAdapter), patch.object(
            score_parity, "_predict_for_policy", side_effect=predict
        ), patch.object(score_parity, "_qualification_evidence", return_value=("p" * 64, "c" * 64, "m" * 64, True)):
            result = run_score_parity(sample, _capability(True), FROZEN_CONFIG)
        self.assertEqual(result.stage.state, "failed")
        self.assertEqual(result.reports[0].status, "failed")
        self.assertEqual(result.reports[0].fallback_reason, "mps_runtime_error")

    def test_mps_qualification_failure_is_explicit(self) -> None:
        """Qualification-time MPS errors carry a backend-specific fallback reason."""
        class Cached(dict):
            digest = "d" * 64

        cached = Cached(
            {
                name: np.zeros(64, dtype=np.float64)
                for name in (
                    "level_shift",
                    "gradual_drift",
                    "amplitude_scaling",
                    "period_change",
                    "secondary_spike",
                )
            }
        )
        outputs = {
            seed: {
                horizon: np.zeros((16, 64 - horizon), dtype=np.float64)
                for horizon in (1, 2, 6)
            }
            for seed in FROZEN_CONFIG.model_seeds
        }
        cpu_return = (
            MappingProxyType(outputs),
            MappingProxyType(outputs),
            TimingEvidence("cpu", 0.0, 0.01, 0.0, 0.01),
            (16,),
        )
        with patch.object(score_parity, "build_cached_perturbations", return_value=cached), patch.object(
            score_parity, "_predict_for_policy", side_effect=[cpu_return, RuntimeError("mps failure")]
        ):
            with self.assertRaisesRegex(_ScoreBackendError, "mps_qualification_runtime_error"):
                _qualification_evidence(
                    _sample(),
                    {seed: SimpleNamespace(device=torch.device("cpu")) for seed in FROZEN_CONFIG.model_seeds},
                    {seed: SimpleNamespace(device=torch.device("mps")) for seed in FROZEN_CONFIG.model_seeds},
                    {},
                    FROZEN_CONFIG,
                    batch_size=16,
                )


if __name__ == "__main__":
    unittest.main()
