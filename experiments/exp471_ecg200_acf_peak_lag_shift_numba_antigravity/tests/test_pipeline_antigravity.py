from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
import unittest
from unittest import mock

import numpy as np


class Exp471PipelineTestAntigravity(unittest.TestCase):
    def test_task6_preflight_does_not_create_private_roots_antigravity(self) -> None:
        """Break caught: implementation or its tests create either real private root."""
        from scratch import run_exp471_ecg200_acf_peak_lag_shift_numba_antigravity as launcher

        before = tuple(root.exists() for root in (launcher.PRIVATE_RUN1_ANTIGRAVITY, launcher.PRIVATE_RUN2_ANTIGRAVITY))
        self.assertEqual(before, (False, False))

    def test_private_helper_requires_checkpoint_antigravity(self) -> None:
        """Break caught: the private sequence runs without the explicit checkpoint token."""
        from scratch import run_exp471_ecg200_acf_peak_lag_shift_numba_antigravity as launcher

        with self.assertRaisesRegex(PermissionError, "checkpoint"):
            launcher.run_private_sequence_antigravity("a" * 40, approval_token=None)

    def test_private_token_rejects_before_any_git_subprocess_or_root_check_antigravity(self) -> None:
        """Break caught: an unapproved sequence can observe or change private-run state."""
        from scratch import run_exp471_ecg200_acf_peak_lag_shift_numba_antigravity as launcher

        with mock.patch.object(launcher, "_require_clean_reviewed_commit_antigravity") as clean, \
             mock.patch.object(launcher, "_evaluate_then_inspect_antigravity") as evaluate, \
             mock.patch.object(launcher, "_require_private_roots_absent_antigravity") as roots:
            with self.assertRaisesRegex(PermissionError, "checkpoint"):
                launcher.run_private_sequence_antigravity("a" * 40, "not-approved")
        clean.assert_not_called()
        evaluate.assert_not_called()
        roots.assert_not_called()

    def test_private_sequence_orders_clean_run1_run2_then_compares_raw_seven_leaf_bytes_antigravity(self) -> None:
        """Break caught: commit checks or byte equality occur in the wrong order."""
        from scratch import run_exp471_ecg200_acf_peak_lag_shift_numba_antigravity as launcher

        events: list[str] = []
        first = "first_antigravity"
        second = "second_antigravity"
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
        from scratch import run_exp471_ecg200_acf_peak_lag_shift_numba_antigravity as launcher

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
        from scratch import run_exp471_ecg200_acf_peak_lag_shift_numba_antigravity as launcher
        from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity.artifacts_antigravity import LEAF_NAMES_ANTIGRAVITY

        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first_antigravity"
            second = Path(directory) / "second_antigravity"
            first.mkdir()
            second.mkdir()
            for name in LEAF_NAMES_ANTIGRAVITY:
                (first / name).write_bytes(name.encode("utf-8"))
                (second / name).write_bytes(name.encode("utf-8"))
            launcher._require_byte_identical_antigravity(first, second)
            (second / "extra_antigravity.json").write_bytes(b"extra")
            with self.assertRaisesRegex(ValueError, "leaf"):
                launcher._require_byte_identical_antigravity(first, second)
            (second / "extra_antigravity.json").unlink()
            (second / LEAF_NAMES_ANTIGRAVITY[-1]).write_bytes(b"different")
            with self.assertRaisesRegex(ValueError, "byte"):
                launcher._require_byte_identical_antigravity(first, second)
            (second / LEAF_NAMES_ANTIGRAVITY[-1]).unlink()
            with self.assertRaisesRegex(ValueError, "leaf"):
                launcher._require_byte_identical_antigravity(first, second)

    def test_launcher_git_and_child_cli_commands_use_exact_repository_root_antigravity(self) -> None:
        """Break caught: launcher git or child CLI commands inherit an arbitrary caller working directory."""
        from scratch import run_exp471_ecg200_acf_peak_lag_shift_numba_antigravity as launcher

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

    def test_evaluate_pipeline_rejects_nonfixed_root_before_loader_antigravity(self) -> None:
        """Break caught: direct evaluation can publish outside the two reviewed Task 6 roots."""
        from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity import cli_antigravity

        loader = mock.Mock()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "fixed Task 6"):
                cli_antigravity.evaluate_pipeline_antigravity(
                    Path(directory) / "arbitrary_antigravity",
                    "a" * 40,
                    loader=loader,
                )
        loader.assert_not_called()

    def test_evaluate_pipeline_builds_ten_records_then_publishes_and_semantically_inspects_antigravity(self) -> None:
        """Break caught: evaluate skips a fold or returns before exact-leaf semantic replay."""
        from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity import cli_antigravity

        calls: list[tuple[int, int]] = []
        events: list[str] = []
        qualification = object()
        decision = mock.Mock(
            passed=True,
            eligible_for_private_reproducibility=True,
            failure_reasons=(),
        )
        inspected = mock.Mock(
            decision=decision,
            hashes={"context_antigravity.json": "a" * 64},
        )

        def score(train, fit, query, *, seed, fold_index, authority):
            self.assertEqual(train.shape, (69, 140))
            self.assertEqual(fit.dtype, np.dtype(np.int64))
            self.assertEqual(query.dtype, np.dtype(np.int64))
            calls.append((seed, fold_index))
            return (seed, fold_index)

        hashes = {"context_antigravity.json": "a" * 64}
        def publish(root, records, actual_qualification, actual_decision):
            events.append("publish")
            self.assertEqual(
                tuple(records),
                tuple((seed, fold) for seed in (20260721, 20261721) for fold in range(5)),
            )
            self.assertIs(actual_qualification, qualification)
            self.assertIs(actual_decision, decision)
            root.mkdir()
            for name in (
                "context_antigravity.json", "dashboard_snapshot_antigravity.json", "decision_antigravity.json",
                "failed_folds_antigravity.jsonl", "method_fold_metrics_antigravity.jsonl", "method_summary_antigravity.jsonl",
                "manifest_antigravity.json",
            ):
                (root / name).write_bytes(name.encode("utf-8"))
            return mock.Mock(hashes=hashes)

        def inspect(root):
            events.append("inspect")
            self.assertTrue(root.is_dir())
            return inspected

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run1_antigravity"
            with mock.patch.object(cli_antigravity, "_PRIVATE_ROOTS_ANTIGRAVITY", (root,)):
                payload = cli_antigravity.evaluate_pipeline_antigravity(
                    root,
                    "a" * 40,
                    loader=lambda: np.zeros((69, 96), dtype=np.float64),
                    interpolate=lambda values: np.zeros((69, 140), dtype=np.float64),
                    authority_factory=lambda: object(),
                    scorer=score,
                    qualifier=lambda records: qualification,
                    decider=lambda actual: decision,
                    publisher=publish,
                    inspector=inspect,
                )
        self.assertEqual(
            tuple(calls),
            tuple((seed, fold) for seed in (20260721, 20261721) for fold in range(5)),
        )
        self.assertEqual(events, ["publish", "inspect"])
        self.assertEqual(payload["decision"]["passed"], True)
        self.assertEqual(payload["hashes"], {"context_antigravity.json": "a" * 64})


if __name__ == "__main__":
    unittest.main()
