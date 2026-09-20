"""TRAIN-only boundary tests for the bounded Exp422 score stage."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.exp417_causal_multihorizon_shadow.contracts import RawSeries, RawTrainDataset
from experiments.exp422_mps_numeric_delta.config import FROZEN_CONFIG
from experiments.exp422_mps_numeric_delta.contracts import StageResult
import experiments.exp422_mps_numeric_delta.score_parity as score_parity
from experiments.exp422_mps_numeric_delta.score_parity import (
    decide_stage_two,
    select_train_sample,
    validate_dataset_name,
)


def _raw_dataset(name: str = "Adiac", *, count: int = 100, length: int = 64) -> RawTrainDataset:
    positions = np.arange(length, dtype=np.float64)
    series = tuple(
        RawSeries(
            index,
            tuple(
                float(value)
                for value in np.sin(positions / (2.0 + index % 7)) + index * 0.01
            ),
        )
        for index in range(count)
    )
    return RawTrainDataset(name, series)


class ScoreBoundaryTests(unittest.TestCase):
    def test_score_selector_rejects_test_and_evaluation_tokens(self) -> None:
        """Removing token validation would permit an evaluation namespace."""
        for name in ("TEST_Adiac", "Adiac_evaluation", "Adiac-label"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_dataset_name(name)
        self.assertEqual(validate_dataset_name("Adiac"), "Adiac")

    def test_exp422_does_not_import_exp419_checkpoint_paths(self) -> None:
        """Adding the official Exp419 checkpoint module must cross this boundary."""
        source = Path(score_parity.__file__).read_text(encoding="utf-8")
        self.assertNotIn("exp419_deterministic_cpu_parallel.checkpoint", source)
        self.assertNotIn("load_raw_test_values", source)

    def test_score_stage_is_skipped_when_core_failed(self) -> None:
        """A failed or skipped core gate must prevent all stage-two work."""
        self.assertEqual(decide_stage_two(StageResult.failed("mps_unavailable")), "skipped")
        self.assertEqual(decide_stage_two(StageResult.skipped("core_failed")), "skipped")
        self.assertEqual(decide_stage_two(StageResult.passed()), "run")

    def test_selector_uses_one_verified_connection_and_caps_exact_length_sample(self) -> None:
        """A widened selector or second connection could escape the preregistered bound."""
        connection = sqlite3.connect(":memory:")
        connection.execute("CREATE TABLE datasets (name TEXT NOT NULL)")
        connection.execute("INSERT INTO datasets(name) VALUES ('Adiac')")
        connection.commit()
        raw = _raw_dataset()
        trace: list[str] = []
        connection.set_trace_callback(trace.append)
        opened: list[tuple[Path, str]] = []
        loaded: list[tuple[str, tuple[str, ...]]] = []

        @contextmanager
        def verified(database: Path, digest: str):
            opened.append((database, digest))
            yield connection

        def load_train(connection_arg, dataset, *, database_sha256, datasets):
            self.assertIs(connection_arg, connection)
            loaded.append((dataset, datasets))
            self.assertEqual(database_sha256, FROZEN_CONFIG.exp417_config.database_sha256)
            return raw

        with tempfile.TemporaryDirectory() as temp, patch.object(
            score_parity, "verified_read_only_connection", verified
        ), patch.object(
            score_parity, "load_raw_train_values_from_connection", load_train
        ):
            database = Path(temp) / "fixture.sqlite"
            sample = select_train_sample(database, FROZEN_CONFIG)

        self.assertEqual(opened, [(database, FROZEN_CONFIG.exp417_config.database_sha256)])
        self.assertEqual(loaded, [("Adiac", ("Adiac",))])
        self.assertEqual(sample.dataset, "Adiac")
        self.assertEqual(sample.length, 64)
        self.assertEqual(sample.train_count, 100)
        self.assertEqual(sample.bounded_series_count, 64)
        self.assertEqual(sample.indices, tuple(range(64)))
        self.assertEqual(sample.channels.shape, (64, 2, 64))
        self.assertEqual(len(sample.raw_waveform_sha256), 64)
        self.assertFalse(sample.channels.flags.writeable)
        normalized_trace = tuple(" ".join(item.lower().split()) for item in trace)
        self.assertIn("pragma quick_check", normalized_trace)
        self.assertFalse(any(" split = 'test'" in item for item in normalized_trace))
        connection.close()

    def test_selector_rejects_nonfinite_series_instead_of_repairing_it(self) -> None:
        """The diagnostic may not hide nonfinite input behind TRAIN repair."""
        connection = sqlite3.connect(":memory:")
        connection.execute("CREATE TABLE datasets (name TEXT NOT NULL)")
        connection.execute("INSERT INTO datasets(name) VALUES ('Adiac')")
        connection.commit()
        raw = _raw_dataset()
        rows = list(raw.series)
        damaged = list(rows[3].values)
        damaged[4] = float("nan")
        rows[3] = RawSeries(rows[3].instance_index, tuple(damaged))
        nonfinite = RawTrainDataset(raw.dataset, tuple(rows))

        @contextmanager
        def verified(_database: Path, _digest: str):
            yield connection

        with patch.object(score_parity, "verified_read_only_connection", verified), patch.object(
            score_parity, "load_raw_train_values_from_connection", return_value=nonfinite
        ):
            with self.assertRaisesRegex(ValueError, "finite"):
                select_train_sample(Path("fixture.sqlite"), FROZEN_CONFIG)
        connection.close()


if __name__ == "__main__":
    unittest.main()
