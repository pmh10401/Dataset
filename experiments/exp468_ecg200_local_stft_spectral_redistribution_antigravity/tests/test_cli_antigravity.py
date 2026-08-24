"""Task 6 CLI and launcher contract tests.

These tests are deliberately written before the Task 6 implementation.  They
cover the command surface and the boundaries that can be checked without
touching the real ECG200 private roots.
"""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


_REPOSITORY_ANTIGRAVITY = Path(__file__).resolve().parents[3]
_LAUNCHER_ANTIGRAVITY = _REPOSITORY_ANTIGRAVITY / (
    "scratch/run_exp468_ecg200_local_stft_spectral_redistribution_antigravity.py"
)
_COMMIT_ANTIGRAVITY = "a" * 40
_RUN1_ROOT_ANTIGRAVITY = Path(
    "/private/tmp/exp468_task6_private_run1_antigravity",
)
_RUN2_ROOT_ANTIGRAVITY = Path(
    "/private/tmp/exp468_task6_private_run2_antigravity",
)


class CliAntigravityTest(unittest.TestCase):
    def test_exact_cli_surface_accepts_only_fixed_evaluate_and_inspect(self) -> None:
        from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.cli_antigravity import (
            parser_antigravity,
        )

        parser = parser_antigravity()
        evaluate = parser.parse_args(
            [
                "evaluate",
                "--output",
                str(_RUN1_ROOT_ANTIGRAVITY),
                "--reviewed-commit",
                _COMMIT_ANTIGRAVITY,
            ],
        )
        inspect = parser.parse_args(
            ["inspect", "--root", str(_RUN1_ROOT_ANTIGRAVITY)],
        )
        self.assertEqual(
            (evaluate.command, evaluate.output, evaluate.reviewed_commit),
            ("evaluate", str(_RUN1_ROOT_ANTIGRAVITY), _COMMIT_ANTIGRAVITY),
        )
        self.assertEqual(
            (inspect.command, inspect.root),
            ("inspect", str(_RUN1_ROOT_ANTIGRAVITY)),
        )
        rejected = (
            (),
            ("evaluate",),
            ("evaluate", "--output", str(_RUN1_ROOT_ANTIGRAVITY)),
            (
                "evaluate",
                "--output",
                str(_RUN1_ROOT_ANTIGRAVITY),
                "--reviewed-commit",
                "A" * 40,
            ),
            (
                "evaluate",
                "--output",
                str(_RUN1_ROOT_ANTIGRAVITY),
                "--reviewed-commit",
                "a" * 39,
            ),
            (
                "evaluate",
                "--output",
                str(_RUN1_ROOT_ANTIGRAVITY),
                "--reviewed-commit",
                _COMMIT_ANTIGRAVITY,
                "--seed",
                "1",
            ),
            ("inspect",),
            ("inspect", "--root", str(_RUN1_ROOT_ANTIGRAVITY), "--retry"),
            ("inspect", "--output", str(_RUN1_ROOT_ANTIGRAVITY)),
        )
        for argv in rejected:
            with (
                self.subTest(argv=argv),
                redirect_stderr(io.StringIO()),
                redirect_stdout(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                parser.parse_args(argv)

    def test_fixed_roots_are_create_only_and_symlinks_are_rejected(self) -> None:
        from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity import (
            cli_antigravity as cli,
        )

        for root in (_RUN1_ROOT_ANTIGRAVITY, _RUN2_ROOT_ANTIGRAVITY):
            with self.subTest(root=root):
                with patch.object(cli, "_PRIVATE_ROOTS_ANTIGRAVITY", (root,)):
                    cli._validate_output_root_antigravity(root)
                    with tempfile.TemporaryDirectory(prefix="exp468-task6-cli-") as directory:
                        existing = Path(directory) / "existing"
                        existing.mkdir()
                        with self.assertRaisesRegex(ValueError, "fixed Task 6 private root"):
                            cli._validate_output_root_antigravity(existing)
                        link = Path(directory) / "link"
                        link.symlink_to(existing, target_is_directory=True)
                        with self.assertRaisesRegex(ValueError, "symlink"):
                            cli._validate_output_root_antigravity(link)

        with self.assertRaisesRegex(ValueError, "fixed Task 6 private root"):
            cli._validate_output_root_antigravity(Path("/private/tmp/other"))

    def test_evaluate_and_inspect_emit_same_canonical_payload_without_path_or_run_metadata(
        self,
    ) -> None:
        from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity import (
            cli_antigravity as cli,
        )

        payload = {
            "experiment": "Exp468",
            "kind": "ecg200_local_stft_train_qualification",
            "context": {"dataset": "ECG200", "data_split": "TRAIN_only"},
            "qualification": {"label_query_count": 0},
            "decision": {"status": "qualification_failed_blocked_before_private_run"},
            "dashboard_snapshot": {
                "private_reproducibility": {
                    "comparison_status": "not_yet_compared",
                    "run_1": "not_run",
                    "run_2": "not_run",
                },
            },
            "artifact_sha256": {"context_antigravity.json": "a" * 64},
        }
        with (
            patch.object(cli, "_validate_output_root_antigravity"),
            patch.object(cli, "_validate_inspect_root_antigravity"),
            patch.object(cli, "_validate_clean_current_commit_antigravity"),
            patch.object(cli, "_evaluate_at_root_antigravity", return_value=payload),
            patch.object(cli, "_inspect_at_root_antigravity", return_value=payload),
        ):
            outputs: list[str] = []
            for argv in (
                [
                    "evaluate",
                    "--output",
                    str(_RUN1_ROOT_ANTIGRAVITY),
                    "--reviewed-commit",
                    _COMMIT_ANTIGRAVITY,
                ],
                ["inspect", "--root", str(_RUN1_ROOT_ANTIGRAVITY)],
            ):
                output = io.StringIO()
                with redirect_stdout(output):
                    self.assertEqual(cli.main_antigravity(argv), 0)
                outputs.append(output.getvalue())
        expected = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ) + "\n"
        self.assertEqual(outputs, [expected, expected])
        self.assertNotIn(str(_RUN1_ROOT_ANTIGRAVITY), outputs[0])

    def test_inspect_path_does_not_import_or_call_train_pipeline(self) -> None:
        script = "\n".join(
            (
                "import sys",
                "from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity import cli_antigravity as cli",
                "mods = (",
                " 'experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.sources_antigravity',",
                " 'experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.interpolation_antigravity',",
                " 'experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.scoring_antigravity',",
                " 'experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.qualification_antigravity',",
                " 'experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.artifacts_antigravity',",
                ")",
                "cli.parser_antigravity().parse_args(['inspect', '--root', '/private/tmp/exp468_task6_private_run1_antigravity'])",
                "assert not any(name in sys.modules for name in mods)",
            ),
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=_REPOSITORY_ANTIGRAVITY,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_clean_current_commit_rejects_wrong_head_and_dirty_porcelain(self) -> None:
        from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity import (
            cli_antigravity as cli,
        )

        with (
            patch.object(cli.subprocess, "check_output", side_effect=("b" * 40, "")),
            self.assertRaisesRegex(ValueError, "exact clean current commit"),
        ):
            cli._validate_clean_current_commit_antigravity(_COMMIT_ANTIGRAVITY)
        with (
            patch.object(cli.subprocess, "check_output", side_effect=("a" * 40, " M file\n")),
            self.assertRaisesRegex(ValueError, "exact clean current commit"),
        ):
            cli._validate_clean_current_commit_antigravity(_COMMIT_ANTIGRAVITY)

    def test_inspect_at_root_and_main_use_delayed_replay_only(self) -> None:
        from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity import (
            cli_antigravity as cli,
        )

        payload = {
            "experiment": "Exp468",
            "kind": "ecg200_local_stft_train_qualification",
            "context": {"dataset": "ECG200", "data_split": "TRAIN_only"},
        }
        root = Path("/private/tmp/exp468_task6_private_run1_antigravity")
        with (
            patch.object(cli, "_validate_inspect_root_antigravity"),
            patch.object(cli, "_load_pipeline_dependencies_antigravity", side_effect=AssertionError("pipeline import")),
            patch.object(cli, "_load_replay_inspector_antigravity", return_value=lambda _root: object()),
            patch.object(cli, "_payload_from_inspected_antigravity", return_value=payload),
        ):
            self.assertEqual(cli._inspect_at_root_antigravity(root), payload)

        output = io.StringIO()
        with (
            redirect_stdout(output),
            patch.object(cli, "_validate_official_root_absent_antigravity"),
            patch.object(cli, "_validate_inspect_root_antigravity"),
            patch.object(cli, "_load_pipeline_dependencies_antigravity", side_effect=AssertionError("pipeline import")),
            patch.object(cli, "_load_replay_inspector_antigravity", return_value=lambda _root: object()),
            patch.object(cli, "_payload_from_inspected_antigravity", return_value=payload),
        ):
            self.assertEqual(cli.main_antigravity(["inspect", "--root", str(root)]), 0)
        self.assertEqual(json.loads(output.getvalue()), payload)

    def test_launcher_sequence_uses_fixed_interpreter_cwd_independent_paths_and_stops_on_failure(
        self,
    ) -> None:
        from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity import (
            cli_antigravity as cli,
        )
        # The launcher owns this helper; the test patches process execution so
        # no private root or real evaluation is started.
        launcher = __import__(
            "scratch.run_exp468_ecg200_local_stft_spectral_redistribution_antigravity",
            fromlist=["run_private_sequence_antigravity"],
        )
        calls: list[tuple[list[str], str | None]] = []

        class Completed:
            returncode = 0

        def make_run(run1: Path, run2: Path):
            def run(command, *, cwd, check):
                calls.append((list(command), str(cwd)))
                if command[2] == "evaluate":
                    root = Path(command[4])
                    root.mkdir(parents=True, exist_ok=False)
                    for leaf in launcher._LEAF_NAMES_ANTIGRAVITY:
                        (root / leaf).write_text("{}\n", encoding="utf-8")
                return Completed()

            return run

        with tempfile.TemporaryDirectory(prefix="exp468-task6-launcher-") as directory:
            base = Path(directory)
            run1 = base / "run1"
            run2 = base / "run2"
            state = base / "sequence-state.json"
            next_state = base / "sequence-next.json"
            clean_calls: list[str] = []

            def clean(reviewed_commit: str) -> None:
                clean_calls.append(reviewed_commit)

            with (
                patch.object(launcher, "_RUN1_ROOT_ANTIGRAVITY", run1),
                patch.object(launcher, "_RUN2_ROOT_ANTIGRAVITY", run2),
                patch.object(launcher, "_SEQUENCE_STATE_PATH_ANTIGRAVITY", state),
                patch.object(launcher, "_SEQUENCE_NEXT_PATH_ANTIGRAVITY", next_state),
                patch.object(cli, "_validate_clean_current_commit_antigravity", side_effect=clean),
                patch.object(launcher.subprocess, "run", side_effect=make_run(run1, run2)),
            ):
                launcher.run_private_sequence_antigravity(_COMMIT_ANTIGRAVITY)

            self.assertEqual(len(calls), 4)
            self.assertEqual(len(clean_calls), 4)
            self.assertEqual(clean_calls, [_COMMIT_ANTIGRAVITY] * 4)
            self.assertEqual(
                [call[0][2:] for call in calls],
                [
                    [
                        "evaluate",
                        "--output",
                        str(run1),
                        "--reviewed-commit",
                        _COMMIT_ANTIGRAVITY,
                    ],
                    ["inspect", "--root", str(run1)],
                    [
                        "evaluate",
                        "--output",
                        str(run2),
                        "--reviewed-commit",
                        _COMMIT_ANTIGRAVITY,
                    ],
                    ["inspect", "--root", str(run2)],
                ],
            )
            self.assertTrue(all(call[0][0] == launcher._FIXED_EXECUTABLE_ANTIGRAVITY for call in calls))
            self.assertTrue(all(call[1] == str(launcher._REPOSITORY_ANTIGRAVITY) for call in calls))
            self.assertEqual(json.loads(state.read_text(encoding="utf-8"))["stage"], "run2_inspected")
            self.assertEqual(json.loads(state.read_text(encoding="utf-8"))["reviewed_commit"], _COMMIT_ANTIGRAVITY)
            self.assertFalse(next_state.exists())

        calls.clear()
        with tempfile.TemporaryDirectory(prefix="exp468-task6-launcher-fail-") as directory:
            base = Path(directory)
            run1 = base / "run1"
            run2 = base / "run2"
            state = base / "sequence-state.json"
            next_state = base / "sequence-next.json"
            def fail_on_second(command, *, cwd, check):
                calls.append((list(command), str(cwd)))
                if len(calls) == 2:
                    return type("Failed", (), {"returncode": 1})()
                if command[2] == "evaluate":
                    root = Path(command[4])
                    root.mkdir(parents=True, exist_ok=False)
                    for leaf in launcher._LEAF_NAMES_ANTIGRAVITY:
                        (root / leaf).write_text("{}\n", encoding="utf-8")
                return Completed()

            with (
                patch.object(launcher, "_RUN1_ROOT_ANTIGRAVITY", run1),
                patch.object(launcher, "_RUN2_ROOT_ANTIGRAVITY", run2),
                patch.object(launcher, "_SEQUENCE_STATE_PATH_ANTIGRAVITY", state),
                patch.object(launcher, "_SEQUENCE_NEXT_PATH_ANTIGRAVITY", next_state),
                patch.object(cli, "_validate_clean_current_commit_antigravity"),
                patch.object(launcher.subprocess, "run", side_effect=fail_on_second),
                self.assertRaises(subprocess.CalledProcessError),
            ):
                launcher.run_private_sequence_antigravity(_COMMIT_ANTIGRAVITY)
            self.assertEqual(len(calls), 2)
            self.assertEqual(json.loads(state.read_text(encoding="utf-8"))["stage"], "run1_evaluated")
            self.assertFalse(run2.exists())


if __name__ == "__main__":
    unittest.main()
