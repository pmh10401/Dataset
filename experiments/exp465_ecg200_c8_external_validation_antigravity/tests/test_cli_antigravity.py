from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np


_COMMIT_ANTIGRAVITY = "a" * 40
_FIXED_EXECUTABLE_ANTIGRAVITY = (
    "/Users/mac/Documents/Dataset/venvs/"
    "detection-coverage-antigravity-aeon140-py314/bin/python"
)
_REPOSITORY_ANTIGRAVITY = Path(__file__).resolve().parents[3]
_LAUNCHER_ANTIGRAVITY = (
    _REPOSITORY_ANTIGRAVITY
    / "scratch/run_exp465_ecg200_c8_external_validation_antigravity.py"
)
_OFFICIAL_ROOT_ANTIGRAVITY = Path(
    "/Users/mac/Documents/Dataset/"
    "exp465_ecg200_c8_external_validation_antigravity_20260822"
)


def _train96_antigravity() -> np.ndarray:
    values = np.random.default_rng(46505).normal(size=(69, 96))
    values.setflags(write=False)
    return values


class CliAntigravityTest(unittest.TestCase):
    def _run_payload_antigravity(self, cli: object, argv: list[str]) -> dict[str, object]:
        output = io.StringIO()
        with redirect_stdout(output):
            code = cli.main_antigravity(argv)
        self.assertEqual(code, 0)
        return json.loads(output.getvalue())

    def _private_module_antigravity(self, root: Path) -> types.ModuleType:
        private = types.ModuleType("_exp465_cli_private_test_antigravity")
        private.CAPABILITY_ANTIGRAVITY = "exp465_cli_private_test_antigravity_v1"
        private.root_antigravity = root
        return private

    def test_exact_surface_accepts_only_reviewed_evaluate_or_bare_inspect_antigravity(self) -> None:
        """Break caught: an override or alias reaches the fixed experiment."""
        from experiments.exp465_ecg200_c8_external_validation_antigravity.cli_antigravity import (
            OFFICIAL_ROOT_ANTIGRAVITY,
            parser_antigravity,
        )

        self.assertEqual(OFFICIAL_ROOT_ANTIGRAVITY, _OFFICIAL_ROOT_ANTIGRAVITY)
        parser = parser_antigravity()
        parsed = parser.parse_args(["evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY])
        self.assertEqual((parsed.command, parsed.reviewed_commit), ("evaluate", _COMMIT_ANTIGRAVITY))
        self.assertEqual(parser.parse_args(["inspect"]).command, "inspect")
        rejected = (
            (), ("--help",), ("-h",), ("eval", "--reviewed-commit", _COMMIT_ANTIGRAVITY),
            ("evaluate",), ("evaluate", "--reviewed-commit", "A" * 40),
            ("evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY, "extra"),
            ("evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY, "--root", "/tmp/x"),
            ("evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY, "--path", "/tmp/x"),
            ("evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY, "--seed", "1"),
            ("evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY, "--fold", "0"),
            ("evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY, "--gate", "0.1"),
            ("evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY, "--resume"),
            ("evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY, "--retry"),
            ("inspect", "extra"), ("inspect", "--root", "/tmp/x"), ("readiness",),
        )
        for argv in rejected:
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as exited:
                parser.parse_args(argv)
            self.assertNotEqual(exited.exception.code, 0)

    def test_parser_and_bare_inspect_are_lazy_and_label_free_antigravity(self) -> None:
        """Break caught: parser or root-free inspect imports science or opens data."""
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys\n"
                    "from experiments.exp465_ecg200_c8_external_validation_antigravity import cli_antigravity as cli\n"
                    "mods=(\n"
                    " 'experiments.exp465_ecg200_c8_external_validation_antigravity.sources_antigravity',\n"
                    " 'experiments.exp465_ecg200_c8_external_validation_antigravity.interpolation_antigravity',\n"
                    " 'experiments.exp465_ecg200_c8_external_validation_antigravity.scoring_antigravity',\n"
                    " 'experiments.exp465_ecg200_c8_external_validation_antigravity.qualification_antigravity',\n"
                    " 'experiments.exp465_ecg200_c8_external_validation_antigravity.evaluation_antigravity',\n"
                    " 'experiments.exp465_ecg200_c8_external_validation_antigravity.artifacts_antigravity',\n"
                    " 'experiments.exp465_ecg200_c8_external_validation_antigravity.replay_antigravity',\n"
                    ")\n"
                    "assert not any(m in sys.modules for m in mods)\n"
                    "cli.parser_antigravity().parse_args(['inspect'])\n"
                    "assert not any(m in sys.modules for m in mods)\n"
                    "try:\n cli.main_antigravity(['inspect'])\nexcept ValueError: pass\n"
                    "assert not any(m in sys.modules for m in mods)\n"
                    "assert 'sqlite3' not in sys.modules\n"
                    "assert not any(m == 'aeon' or m.startswith('aeon.') for m in sys.modules)\n"
                ),
            ],
            cwd=_REPOSITORY_ANTIGRAVITY,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_launcher_reexecs_fixed_runtime_from_outside_cwd_antigravity(self) -> None:
        """Break caught: launcher depends on the caller CWD or arbitrary interpreter."""
        with tempfile.TemporaryDirectory(prefix="exp465_launcher_cwd_antigravity_") as temporary:
            completed = subprocess.run(
                [sys.executable, str(_LAUNCHER_ANTIGRAVITY), "inspect"],
                cwd=temporary,
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("private Task 5 root capability is required", completed.stderr)
        self.assertFalse(_OFFICIAL_ROOT_ANTIGRAVITY.exists())

    def test_launcher_accepts_only_preregistered_private_capabilities_antigravity(self) -> None:
        """Break caught: the real launcher cannot enter private qualification or accepts an arbitrary root."""
        with tempfile.TemporaryDirectory(prefix="exp465_launcher_private_capability_antigravity_") as temporary:
            environment = dict(__import__("os").environ)
            environment["EXP465_TASK5_PRIVATE_CAPABILITY_ANTIGRAVITY"] = "run1"
            completed = subprocess.run(
                [sys.executable, str(_LAUNCHER_ANTIGRAVITY), "inspect"],
                cwd=temporary,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            hashlib.sha256(completed.stdout.encode("utf-8")).hexdigest(),
            "0436fbbd727815daf2cecdac353b380d0624b923ff74f43a03197536f894f891",
        )
        self.assertNotIn("root capability is required", completed.stderr)
        environment["EXP465_TASK5_PRIVATE_CAPABILITY_ANTIGRAVITY"] = "/tmp/arbitrary"
        rejected = subprocess.run(
            [sys.executable, str(_LAUNCHER_ANTIGRAVITY), "inspect"],
            cwd=_REPOSITORY_ANTIGRAVITY,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertIn("private capability is not preregistered", rejected.stderr)

    def test_private_seam_injects_only_one_nonofficial_path_antigravity(self) -> None:
        """Break caught: the private seam replaces dependencies or selects official output."""
        from experiments.exp465_ecg200_c8_external_validation_antigravity import cli_antigravity as cli

        with tempfile.TemporaryDirectory(prefix="exp465_cli_seam_antigravity_") as temporary:
            private = self._private_module_antigravity(Path(temporary) / "result_antigravity")
            private.scorer_antigravity = lambda: None
            with patch.dict(sys.modules, {private.__name__: private}), self.assertRaisesRegex(ValueError, "only root_antigravity"):
                cli.main_antigravity(["inspect"])
            private = self._private_module_antigravity(_OFFICIAL_ROOT_ANTIGRAVITY)
            with patch.dict(sys.modules, {private.__name__: private}), self.assertRaisesRegex(ValueError, "non-official Path"):
                cli.main_antigravity(["inspect"])

    def test_create_only_private_root_precedes_validation_and_scoring_antigravity(self) -> None:
        """Break caught: an existing private root reaches clean validation or science."""
        from experiments.exp465_ecg200_c8_external_validation_antigravity import cli_antigravity as cli

        with tempfile.TemporaryDirectory(prefix="exp465_cli_existing_root_antigravity_") as temporary:
            root = Path(temporary) / "result_antigravity"
            root.mkdir()
            private = self._private_module_antigravity(root)
            with patch.dict(sys.modules, {private.__name__: private}), patch.object(cli, "_validate_clean_current_commit_antigravity", side_effect=AssertionError("validation should not run")), self.assertRaisesRegex(ValueError, "create-only"):
                cli.main_antigravity(["evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY])

    def test_clean_commit_and_duplicate_process_guards_antigravity(self) -> None:
        """Break caught: a dirty, wrong, or concurrent evaluation starts scoring."""
        from experiments.exp465_ecg200_c8_external_validation_antigravity import cli_antigravity as cli

        with patch.object(cli.subprocess, "check_output", side_effect=["b" * 40 + "\n", "", " 3 1 scratch/run_exp465_ecg200_c8_external_validation_antigravity.py evaluate\n"]), patch.object(cli.os, "getpid", return_value=99):
            with self.assertRaisesRegex(ValueError, "duplicate Exp465 process"):
                cli._validate_clean_current_commit_antigravity("b" * 40)
        with patch.object(cli.subprocess, "check_output", side_effect=["b" * 40 + "\n", "dirty\n"]):
            with self.assertRaisesRegex(ValueError, "exact clean current commit"):
                cli._validate_clean_current_commit_antigravity("b" * 40)

    def test_private_evaluate_then_inspect_uses_real_lower_layers_and_label_zero_antigravity(self) -> None:
        """Break caught: CLI skips loader/folds/publish/replay or inspect recomputes science."""
        from experiments.exp465_ecg200_c8_external_validation_antigravity import cli_antigravity as cli
        from experiments.exp465_ecg200_c8_external_validation_antigravity.artifacts_antigravity import _LEAVES_ANTIGRAVITY

        with tempfile.TemporaryDirectory(prefix="exp465_cli_private_lower_layers_antigravity_") as temporary:
            root = Path(temporary) / "result_antigravity"
            private = self._private_module_antigravity(root)
            with patch.dict(sys.modules, {private.__name__: private}), patch.object(cli, "_validate_clean_current_commit_antigravity", return_value=None), patch(
                "experiments.exp465_ecg200_c8_external_validation_antigravity.sources_antigravity.load_ecg200_normal_train_antigravity",
                return_value=_train96_antigravity(),
            ) as loader:
                evaluated = self._run_payload_antigravity(cli, ["evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY])
                self.assertEqual(loader.call_count, 1)
                with patch(
                    "experiments.exp465_ecg200_c8_external_validation_antigravity.scoring_antigravity.score_fold_methods_antigravity",
                    side_effect=AssertionError("inspect must not score"),
                ), patch(
                    "experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.scoring_antigravity.score_fold_methods_antigravity",
                    side_effect=AssertionError("inspect must not call Exp464 scorer"),
                ), patch(
                    "experiments.exp465_ecg200_c8_external_validation_antigravity.qualification_antigravity.score_exp464_fold_methods_antigravity",
                    side_effect=AssertionError("inspect must not call cached Exp464 scorer"),
                ):
                    inspected = self._run_payload_antigravity(cli, ["inspect"])
            self.assertEqual(evaluated, inspected)
            self.assertEqual(evaluated["experiment"], "Exp465")
            self.assertEqual(evaluated["decision"]["label_query_count"], 0)
            self.assertEqual(tuple(sorted(item.name for item in root.iterdir())), tuple(sorted(_LEAVES_ANTIGRAVITY)))
            self.assertEqual(tuple(sorted(evaluated["leaf_sha256"])), tuple(sorted(_LEAVES_ANTIGRAVITY)))
            self.assertFalse(_OFFICIAL_ROOT_ANTIGRAVITY.exists())


if __name__ == "__main__":
    unittest.main()
