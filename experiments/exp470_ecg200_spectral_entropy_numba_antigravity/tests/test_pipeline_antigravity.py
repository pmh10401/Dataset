from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np


@dataclass(frozen=True)
class _DecisionAntigravity:
    passed: bool = True
    eligible_for_private_reproducibility: bool = True
    failure_reasons: tuple[str, ...] = ()


class Exp470PipelineTestAntigravity(unittest.TestCase):
    def test_real_private_constants_stay_absent_before_and_after_no_run_logic_antigravity(self) -> None:
        """Break caught: a Task 6 test creates either real private root even without evaluation permission."""
        from scratch import run_exp470_ecg200_spectral_entropy_numba_antigravity as launcher

        before = tuple(root.exists() for root in (launcher.PRIVATE_RUN1_ANTIGRAVITY, launcher.PRIVATE_RUN2_ANTIGRAVITY))
        self.assertEqual(before, (False, False))
        with self.assertRaisesRegex(PermissionError, "private checkpoint approval is required"):
            launcher.run_private_sequence_antigravity("a" * 40, "not-approved")
        after = tuple(root.exists() for root in (launcher.PRIVATE_RUN1_ANTIGRAVITY, launcher.PRIVATE_RUN2_ANTIGRAVITY))
        self.assertEqual(after, (False, False))

    def test_evaluate_pipeline_rejects_existing_unrelated_root_before_loader_antigravity(self) -> None:
        """Break caught: a later evaluation can overwrite or consume an unrelated destination."""
        from experiments.exp470_ecg200_spectral_entropy_numba_antigravity import cli_antigravity

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "occupied_antigravity"
            root.mkdir()
            (root / "keep.txt").write_text("preserve", encoding="utf-8")
            loader = mock.Mock()
            with self.assertRaisesRegex(ValueError, "empty"):
                cli_antigravity.evaluate_pipeline_antigravity(root, "a" * 40, loader=loader)
            loader.assert_not_called()
            self.assertEqual((root / "keep.txt").read_text(encoding="utf-8"), "preserve")

    def test_evaluate_pipeline_calls_exact_ten_seed_fold_records_then_publishes_seven_leaves_antigravity(self) -> None:
        """Break caught: evaluation skips/reorders a frozen fold, bypasses qualification, or does not publish exactly seven leaves."""
        from experiments.exp470_ecg200_spectral_entropy_numba_antigravity import cli_antigravity
        from experiments.exp470_ecg200_spectral_entropy_numba_antigravity.artifacts_antigravity import LEAF_NAMES_ANTIGRAVITY

        calls: list[tuple[int, int]] = []
        records: dict[tuple[int, int], object] = {}
        qualification = object()
        decision = _DecisionAntigravity()

        def score(train, fit, query, *, seed, fold_index, authority):
            self.assertEqual(train.shape, (69, 140))
            self.assertEqual(fit.dtype, np.dtype(np.int64))
            self.assertEqual(query.dtype, np.dtype(np.int64))
            calls.append((seed, fold_index))
            return (seed, fold_index)

        def publish(root, actual_records, actual_qualification, actual_decision):
            self.assertEqual(tuple(actual_records), tuple((seed, fold) for seed in (20260721, 20261721) for fold in range(5)))
            self.assertIs(actual_qualification, qualification)
            self.assertIs(actual_decision, decision)
            root.mkdir()
            for name in LEAF_NAMES_ANTIGRAVITY:
                (root / name).write_bytes(name.encode("utf-8"))
            return mock.Mock(hashes={name: "a" * 64 for name in LEAF_NAMES_ANTIGRAVITY[1:]})

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "publication_antigravity"
            payload = cli_antigravity.evaluate_pipeline_antigravity(
                root,
                "a" * 40,
                loader=lambda: np.zeros((69, 96), dtype=np.float64),
                interpolate=lambda values: np.zeros((69, 140), dtype=np.float64),
                authority_factory=lambda: object(),
                scorer=score,
                qualifier=lambda actual: qualification,
                decider=lambda actual: decision,
                publisher=publish,
            )
            self.assertEqual(tuple(calls), tuple((seed, fold) for seed in (20260721, 20261721) for fold in range(5)))
            self.assertEqual(tuple(sorted(path.name for path in root.iterdir())), tuple(sorted(LEAF_NAMES_ANTIGRAVITY)))
            self.assertEqual(payload["root"], str(root))
            self.assertEqual(payload["decision"]["passed"], True)

    def test_private_token_rejects_before_any_git_subprocess_or_root_check_antigravity(self) -> None:
        """Break caught: an unapproved sequence can observe or change private-run state."""
        from scratch import run_exp470_ecg200_spectral_entropy_numba_antigravity as launcher

        with mock.patch.object(launcher, "_require_clean_reviewed_commit_antigravity") as clean, \
             mock.patch.object(launcher, "_evaluate_then_inspect_antigravity") as evaluate, \
             mock.patch.object(launcher, "_require_private_roots_absent_antigravity") as roots:
            with self.assertRaisesRegex(PermissionError, "private checkpoint approval is required"):
                launcher.run_private_sequence_antigravity("a" * 40, "wrong")
        clean.assert_not_called()
        evaluate.assert_not_called()
        roots.assert_not_called()

    def test_private_sequence_orders_clean_run1_run2_then_compares_raw_seven_leaf_bytes_antigravity(self) -> None:
        """Break caught: commit checks or byte equality occur in the wrong order."""
        from scratch import run_exp470_ecg200_spectral_entropy_numba_antigravity as launcher

        events: list[str] = []
        first = Path("first_antigravity")
        second = Path("second_antigravity")
        with tempfile.TemporaryDirectory() as directory:
            run1 = Path(directory) / "run1_antigravity"
            run2 = Path(directory) / "run2_antigravity"
            with mock.patch.object(launcher, "PRIVATE_RUN1_ANTIGRAVITY", run1), \
                 mock.patch.object(launcher, "PRIVATE_RUN2_ANTIGRAVITY", run2), \
                 mock.patch.object(launcher, "_require_private_roots_absent_antigravity", side_effect=lambda: events.append("roots")), \
                 mock.patch.object(launcher, "_require_clean_reviewed_commit_antigravity", side_effect=lambda commit: events.append("clean")), \
                 mock.patch.object(launcher, "_evaluate_then_inspect_antigravity", side_effect=lambda root, commit: (events.append("run1" if root == run1 else "run2") or (first if root == run1 else second))), \
                 mock.patch.object(launcher, "_require_byte_identical_antigravity", side_effect=lambda a, b: events.append("bytes")):
                actual = launcher.run_private_sequence_antigravity("a" * 40, "TASK_6_PRIVATE_RUN_1_2_APPROVED")
        self.assertEqual(actual, (first, second))
        self.assertEqual(events, ["clean", "roots", "run1", "clean", "run2", "bytes"])

    def test_private_sequence_stops_before_second_run_on_first_failure_or_dirty_commit_antigravity(self) -> None:
        """Break caught: a failed first run or dirty reviewed commit still permits run two."""
        from scratch import run_exp470_ecg200_spectral_entropy_numba_antigravity as launcher

        with tempfile.TemporaryDirectory() as directory:
            run1 = Path(directory) / "run1_antigravity"
            run2 = Path(directory) / "run2_antigravity"
            with mock.patch.object(launcher, "PRIVATE_RUN1_ANTIGRAVITY", run1), \
                 mock.patch.object(launcher, "PRIVATE_RUN2_ANTIGRAVITY", run2), \
                 mock.patch.object(launcher, "_require_private_roots_absent_antigravity"), \
                 mock.patch.object(launcher, "_require_clean_reviewed_commit_antigravity", side_effect=ValueError("dirty")), \
                 mock.patch.object(launcher, "_evaluate_then_inspect_antigravity") as evaluate:
                with self.assertRaisesRegex(ValueError, "dirty"):
                    launcher.run_private_sequence_antigravity("a" * 40, "TASK_6_PRIVATE_RUN_1_2_APPROVED")
            evaluate.assert_not_called()
            with mock.patch.object(launcher, "PRIVATE_RUN1_ANTIGRAVITY", run1), \
                 mock.patch.object(launcher, "PRIVATE_RUN2_ANTIGRAVITY", run2), \
                 mock.patch.object(launcher, "_require_private_roots_absent_antigravity"), \
                 mock.patch.object(launcher, "_require_clean_reviewed_commit_antigravity"), \
                 mock.patch.object(launcher, "_evaluate_then_inspect_antigravity", side_effect=RuntimeError("run1")) as evaluate:
                with self.assertRaisesRegex(RuntimeError, "run1"):
                    launcher.run_private_sequence_antigravity("a" * 40, "TASK_6_PRIVATE_RUN_1_2_APPROVED")
            self.assertEqual(evaluate.call_count, 1)

    def test_byte_comparison_requires_each_exact_leaf_and_raw_byte_equality_antigravity(self) -> None:
        """Break caught: helper trusts hashes/decision or omits one of the seven publication leaves."""
        from scratch import run_exp470_ecg200_spectral_entropy_numba_antigravity as launcher
        from experiments.exp470_ecg200_spectral_entropy_numba_antigravity.artifacts_antigravity import LEAF_NAMES_ANTIGRAVITY

        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first_antigravity"
            second = Path(directory) / "second_antigravity"
            first.mkdir()
            second.mkdir()
            for name in LEAF_NAMES_ANTIGRAVITY:
                (first / name).write_bytes(name.encode("utf-8"))
                (second / name).write_bytes(name.encode("utf-8"))
            launcher._require_byte_identical_antigravity(first, second)
            (second / LEAF_NAMES_ANTIGRAVITY[-1]).write_bytes(b"different")
            with self.assertRaisesRegex(ValueError, "byte"):
                launcher._require_byte_identical_antigravity(first, second)
            (second / LEAF_NAMES_ANTIGRAVITY[-1]).unlink()
            with self.assertRaisesRegex(ValueError, "leaf"):
                launcher._require_byte_identical_antigravity(first, second)

    def test_launcher_git_and_child_cli_commands_use_exact_repository_root_antigravity(self) -> None:
        """Break caught: launcher git or child CLI commands inherit an arbitrary caller working directory."""
        from scratch import run_exp470_ecg200_spectral_entropy_numba_antigravity as launcher

        root = Path(launcher.__file__).resolve().parents[1]
        git_responses = (
            subprocess.CompletedProcess(("git",), 0, stdout="a" * 40 + "\n", stderr=""),
            subprocess.CompletedProcess(("git",), 0, stdout="", stderr=""),
        )
        with mock.patch.object(launcher.subprocess, "run", side_effect=git_responses) as run:
            launcher._require_clean_reviewed_commit_antigravity("a" * 40)
        self.assertTrue(all(call.kwargs["cwd"] == root for call in run.call_args_list))
        payload = b'{"decision":{},"hashes":{},"root":"temporary"}\n'
        completed = subprocess.CompletedProcess(("python",), 0, stdout=payload, stderr=b"")
        with mock.patch.object(launcher.subprocess, "run", return_value=completed) as run:
            launcher._run_cli_antigravity("inspect", Path("temporary_antigravity"))
        self.assertEqual(run.call_args.kwargs["cwd"], root)


if __name__ == "__main__":
    unittest.main()
