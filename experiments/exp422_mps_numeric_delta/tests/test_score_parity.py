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
from experiments.exp422_mps_numeric_delta.score_parity import TrainSample, run_score_parity


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

        def predict(adapters, channels, *, batch_size, max_batch_bytes):
            backend = next(iter(adapters.values())).device.type
            rows = channels.shape[0]
            outputs = {
                seed: {
                    horizon: np.tile(
                        np.linspace(0.05, 0.95, sample.length - horizon, dtype=np.float64),
                        (rows, 1),
                    )
                    for horizon in horizons
                }
                for seed in FROZEN_CONFIG.model_seeds
            }
            timing = (
                TimingEvidence("cpu", 0.002, 0.018, 0.0, 0.02)
                if backend == "cpu"
                else TimingEvidence("mps", 0.002, 0.007, 0.001, 0.01)
            )
            chunks = tuple(
                min(batch_size, rows - start) for start in range(0, rows, batch_size)
            )
            return MappingProxyType(outputs), timing, chunks

        exp421_batches_before = EXP421_CONFIG.candidate_batch_sizes
        with patch.object(score_parity, "fit_final_seed_models", return_value=models) as fit, patch.object(
            score_parity, "train_oof_models", return_value=oof_bundle
        ) as train_oof, patch.object(score_parity, "FrozenModelAdapter", FakeAdapter), patch.object(
            score_parity, "_predict_for_policy", side_effect=predict
        ):
            result = run_score_parity(sample, _capability(True), FROZEN_CONFIG)

        fit.assert_called_once_with(sample.group, device=torch.device("cpu"), config=FROZEN_CONFIG.exp417_config)
        train_oof.assert_called_once_with(sample.group, device=torch.device("cpu"), config=FROZEN_CONFIG.exp417_config)
        self.assertEqual(EXP421_CONFIG.candidate_batch_sizes, exp421_batches_before)
        self.assertEqual(tuple(report.batch_size for report in result.reports), (16, 32, 64, 128, 256))
        self.assertEqual(tuple(report.chunk_sizes for report in result.reports), ((16, 16, 16, 16), (32, 32), (64,), (64,), (64,)))
        self.assertTrue(all(report.candidate_indices_match for report in result.reports))
        self.assertTrue(all(report.qualification_match for report in result.reports))
        self.assertTrue(all(report.spearman == 1.0 for report in result.reports))
        self.assertTrue(all(report.speedup == 2.0 for report in result.reports))
        self.assertTrue(all(report.passes for report in result.reports))
        self.assertEqual(result.stage.state, "passed")
        self.assertEqual(len(result.oof_residual_sha256), 64)
        self.assertEqual(len(result.calibration_sha256), 64)
        self.assertEqual(len(result.perturbation_sha256), 64)


if __name__ == "__main__":
    unittest.main()
