"""Task 6 frozen pipeline tests using synthetic TRAIN-only inputs."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
import unittest
from unittest.mock import patch

import numpy as np

from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity import (
    cli_antigravity as cli,
)
from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.qualification_antigravity import (
    blocked_fold_indices_antigravity,
)


@lru_cache(maxsize=1)
def synthetic_train96_antigravity() -> np.ndarray:
    x = np.arange(96, dtype=np.float64)
    base = 0.01 * x + np.sin(2.0 * np.pi * x / 21.0) + 0.3 * np.cos(2.0 * np.pi * x / 9.0)
    rows = base[None, :] + np.random.default_rng(46861).normal(
        scale=0.01,
        size=(69, 96),
    )
    result = np.asarray(rows, dtype=np.float64, order="C")
    result.setflags(write=False)
    return result


def synthetic_train140_antigravity() -> np.ndarray:
    x = np.linspace(0.0, 95.0, 140, dtype=np.float64)
    base = 0.01 * x + np.sin(2.0 * np.pi * x / 21.0) + 0.3 * np.cos(2.0 * np.pi * x / 9.0)
    rows = base[None, :] + np.random.default_rng(46862).normal(
        scale=0.01,
        size=(69, 140),
    )
    result = np.asarray(rows, dtype=np.float64, order="C")
    result.setflags(write=False)
    return result


class PipelineAntigravityTest(unittest.TestCase):
    def test_public_blocked_fold_shapes_and_two_seed_seed_major_identity(self) -> None:
        folds = blocked_fold_indices_antigravity()
        self.assertEqual(
            tuple(len(query) for _fit, query in folds),
            (14, 14, 14, 14, 13),
        )
        self.assertEqual(
            tuple(len(fit) for fit, _query in folds),
            (55, 55, 55, 55, 56),
        )
        first_queries = tuple(query.tobytes(order="C") for _fit, query in folds)
        self.assertEqual(
            first_queries,
            tuple(
                values.tobytes(order="C")
                for values in np.array_split(np.arange(69, dtype=np.int64), 5)
            ),
        )
        for fit, query in folds:
            self.assertEqual(
                tuple(int(value) for value in fit),
                tuple(value for value in range(69) if value not in set(query.tolist())),
            )

    def test_synthetic_pipeline_calls_direct_stages_once_and_publishes_then_inspects(
        self,
    ) -> None:
        train96 = synthetic_train96_antigravity()
        train140 = synthetic_train140_antigravity()
        calls = {
            "load": 0,
            "interpolate": 0,
            "authority": 0,
            "score": [],
            "qualify": 0,
            "evaluate": 0,
            "publish": 0,
            "inspect": 0,
        }

        def load(_path: Path):
            calls["load"] += 1
            return train96

        def interpolate(_values: np.ndarray):
            calls["interpolate"] += 1
            self.assertIs(_values, train96)
            return train140

        real_capture = cli._load_pipeline_dependencies_antigravity()[2]

        def capture():
            calls["authority"] += 1
            return real_capture()

        real_score = cli._load_pipeline_dependencies_antigravity()[5]
        real_qualify = cli._load_pipeline_dependencies_antigravity()[6]
        real_evaluate = cli._load_pipeline_dependencies_antigravity()[7]
        real_publish = cli._load_pipeline_dependencies_antigravity()[8]
        real_inspect = cli._load_pipeline_dependencies_antigravity()[9]

        def score(*args, **kwargs):
            calls["score"].append(
                (
                    kwargs["seed"],
                    kwargs["fold_index"],
                    args[1].tobytes(order="C"),
                    args[2].tobytes(order="C"),
                ),
            )
            return real_score(*args, **kwargs)

        def qualify(records):
            calls["qualify"] += 1
            return real_qualify(records)

        def evaluate(qualification):
            calls["evaluate"] += 1
            return real_evaluate(qualification)

        def publish(root, records, qualification, decision):
            calls["publish"] += 1
            return real_publish(root, records, qualification, decision)

        def inspect(root):
            calls["inspect"] += 1
            return real_inspect(root)

        dependencies = (
            load,
            interpolate,
            capture,
            blocked_fold_indices_antigravity,
            np,
            score,
            qualify,
            evaluate,
            publish,
            inspect,
        )
        with patch.object(cli, "_load_pipeline_dependencies_antigravity", return_value=dependencies):
            with self.assertRaisesRegex(ValueError, "fixed Task 6 private root"):
                cli._evaluate_at_root_antigravity(Path("/tmp/synthetic-exp468-task6"))

        # The actual pipeline remains root-agnostic behind this test seam; use
        # the same temporary path by replacing the fixed-root validator only.
        with tempfile_directory_antigravity() as root:
            with (
                patch.object(cli, "_load_pipeline_dependencies_antigravity", return_value=dependencies),
                patch.object(cli, "_validate_output_root_antigravity"),
            ):
                payload = cli._evaluate_at_root_antigravity(root)
            self.assertEqual(payload["experiment"], "Exp468")
            self.assertEqual(payload["dashboard_snapshot"]["private_reproducibility"]["comparison_status"], "not_yet_compared")
            for leaf in root.iterdir():
                raw = leaf.read_bytes()
                self.assertNotIn(str(root).encode(), raw)
                self.assertNotIn(b"run_index", raw)
                self.assertNotIn(b"timestamp", raw)
                self.assertNotIn(b"created_at", raw)

        self.assertEqual(calls["load"], 1)
        self.assertEqual(calls["interpolate"], 1)
        self.assertEqual(calls["authority"], 1)
        self.assertEqual(
            [(seed, fold_index) for seed, fold_index, _fit, _query in calls["score"]],
            [
                (seed, fold_index)
                for seed in (20260721, 20261721)
                for fold_index in range(5)
            ],
        )
        for fold_index, (fit_indices, query_indices) in enumerate(folds := blocked_fold_indices_antigravity()):
            first = calls["score"][fold_index]
            second = calls["score"][5 + fold_index]
            self.assertEqual(first[2:], second[2:])
            self.assertEqual(first[2], fit_indices.tobytes(order="C"))
            self.assertEqual(first[3], query_indices.tobytes(order="C"))
        # Publication and retained-input inspection have their own replay
        # contracts.  This test only counts direct pipeline stages.
        self.assertEqual(calls["qualify"], 1)
        self.assertEqual(calls["evaluate"], 1)
        self.assertEqual(calls["publish"], 1)
        self.assertEqual(calls["inspect"], 1)


class tempfile_directory_antigravity:
    """Small context manager that returns a Path without exposing real roots."""

    def __enter__(self) -> Path:
        import tempfile

        self._context = tempfile.TemporaryDirectory(prefix="exp468-task6-pipeline-")
        return Path(self._context.name) / "result"

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._context.cleanup()


if __name__ == "__main__":
    unittest.main()
