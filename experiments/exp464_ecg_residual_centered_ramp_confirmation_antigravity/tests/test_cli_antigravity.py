from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.tests.test_artifacts_antigravity import (
    LEAVES_ANTIGRAVITY,
    publication_inputs_antigravity,
)


_COMMIT_ANTIGRAVITY = "a" * 40
_FIXED_EXECUTABLE_ANTIGRAVITY = (
    "/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python"
)
_FIXED_PREFIX_ANTIGRAVITY = (
    "/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314"
)
_REPOSITORY_ANTIGRAVITY = Path(__file__).resolve().parents[3]
_LAUNCHER_ANTIGRAVITY = (
    _REPOSITORY_ANTIGRAVITY
    / "scratch/run_exp464_ecg_residual_centered_ramp_confirmation_antigravity.py"
)
_OFFICIAL_ROOT_ANTIGRAVITY = Path(
    "/Users/mac/Documents/Dataset/exp464_ecg_residual_centered_ramp_confirmation_antigravity_20260821"
)


class CliAntigravityTest(unittest.TestCase):
    def _source_free_lower_layer_inputs_antigravity(self):
        authority, qualification, _decision = publication_inputs_antigravity()
        return authority, qualification

    def _run_main_payload_antigravity(
        self,
        cli,
        argv: list[str],
    ) -> dict[str, object]:
        output = io.StringIO()
        with redirect_stdout(output):
            code = cli.main_antigravity(argv)
        self.assertEqual(code, 0)
        return json.loads(output.getvalue())

    def _run_clean_snapshot_public_dispatch_antigravity(self) -> dict[str, object]:
        package_relative = Path(
            "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity"
        )
        package_source = _REPOSITORY_ANTIGRAVITY / package_relative
        with tempfile.TemporaryDirectory(prefix="exp464_public_dispatch_snapshot_antigravity_") as temporary:
            snapshot_root = Path(temporary) / "repo"
            official_root = Path(temporary) / "official_root_antigravity"
            subprocess.run(
                ["git", "worktree", "add", "--detach", str(snapshot_root), "HEAD"],
                cwd=_REPOSITORY_ANTIGRAVITY,
                text=True,
                capture_output=True,
                check=True,
            )
            try:
                shutil.copytree(
                    package_source,
                    snapshot_root / package_relative,
                    dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                )
                subprocess.run(
                    ["git", "add", str(package_relative)],
                    cwd=snapshot_root,
                    text=True,
                    capture_output=True,
                    check=True,
                )
                dirty = subprocess.check_output(
                    ["git", "status", "--porcelain"],
                    cwd=snapshot_root,
                    text=True,
                ).strip()
                if dirty:
                    subprocess.run(
                        [
                            "git",
                            "-c",
                            "user.name=Codex",
                            "-c",
                            "user.email=codex@example.com",
                            "commit",
                            "-m",
                            "snapshot exp464 task5 public dispatch antigravity",
                        ],
                        cwd=snapshot_root,
                        text=True,
                        capture_output=True,
                        check=True,
                    )
                script = r"""
from __future__ import annotations
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity import artifacts_antigravity as artifacts
from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity import cli_antigravity as cli
from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity import replay_antigravity as replay
from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.tests.test_artifacts_antigravity import publication_inputs_antigravity

authority, qualification, _decision = publication_inputs_antigravity()
official_root = Path(sys.argv[1])
reviewed_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
evaluate_output = io.StringIO()
inspect_output = io.StringIO()
with (
    patch.object(cli, "OFFICIAL_ROOT_ANTIGRAVITY", official_root),
    patch.object(artifacts, "OFFICIAL_ROOT_ANTIGRAVITY", official_root),
    patch.object(replay, "OFFICIAL_ROOT_ANTIGRAVITY", official_root),
    patch(
        "experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.predecessor_antigravity.capture_exp463_authority_antigravity",
        return_value=authority,
    ),
    patch(
        "experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.qualification_antigravity.qualify_methods_antigravity",
        return_value=qualification,
    ),
):
    with contextlib.redirect_stdout(evaluate_output):
        evaluate_code = cli.main_antigravity(["evaluate", "--reviewed-commit", reviewed_commit])
    with contextlib.redirect_stdout(inspect_output):
        inspect_code = cli.main_antigravity(["inspect"])
print(
    json.dumps(
        {
            "evaluate_code": evaluate_code,
            "inspect_code": inspect_code,
            "evaluate_payload": json.loads(evaluate_output.getvalue()),
            "inspect_payload": json.loads(inspect_output.getvalue()),
            "official_root_exists": official_root.is_dir(),
            "official_root_entries": sorted(entry.name for entry in official_root.iterdir()),
        },
        sort_keys=True,
    )
)
"""
                completed = subprocess.run(
                    [sys.executable, "-c", script, str(official_root)],
                    cwd=snapshot_root,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                return json.loads(completed.stdout)
            finally:
                subprocess.run(
                    ["git", "worktree", "remove", "--force", str(snapshot_root)],
                    cwd=_REPOSITORY_ANTIGRAVITY,
                    text=True,
                    capture_output=True,
                    check=False,
                )

    def test_exact_surface_accepts_only_reviewed_evaluate_or_bare_inspect_antigravity(self) -> None:
        """Break caught: aliases, overrides, or extra tokens reach execution."""
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.cli_antigravity import (
            OFFICIAL_ROOT_ANTIGRAVITY,
            parser_antigravity,
        )

        self.assertEqual(OFFICIAL_ROOT_ANTIGRAVITY, _OFFICIAL_ROOT_ANTIGRAVITY)
        parser = parser_antigravity()
        parsed = parser.parse_args(["evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY])
        self.assertEqual(parsed.command, "evaluate")
        self.assertEqual(parsed.reviewed_commit, _COMMIT_ANTIGRAVITY)
        self.assertEqual(parser.parse_args(["inspect"]).command, "inspect")

        rejected = (
            (),
            ("--help",),
            ("-h",),
            ("eval", "--reviewed-commit", _COMMIT_ANTIGRAVITY),
            ("evaluate", "--reviewed-com", _COMMIT_ANTIGRAVITY),
            ("evaluate", f"--reviewed-commit={_COMMIT_ANTIGRAVITY}"),
            ("evaluate", "--reviewed-commit", "A" * 40),
            ("evaluate", "--reviewed-commit", "a" * 39),
            ("evaluate",),
            ("evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY, "extra"),
            ("evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY, "--root", "/tmp/x"),
            ("evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY, "--path", "/tmp/x"),
            ("evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY, "--seed", "1"),
            ("evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY, "--fold", "0"),
            ("evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY, "--gate", "0.1"),
            ("evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY, "--resume"),
            ("evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY, "--retry"),
            ("inspect", "extra"),
            ("inspect", "--reviewed-commit", _COMMIT_ANTIGRAVITY),
            ("inspect", "--root", "/tmp/x"),
            ("readiness",),
            ("resume",),
            ("retry",),
        )
        for argv in rejected:
            with (
                self.subTest(argv=argv),
                redirect_stderr(io.StringIO()),
                redirect_stdout(io.StringIO()),
                self.assertRaises(SystemExit) as exited,
            ):
                parser.parse_args(argv)
            self.assertNotEqual(exited.exception.code, 0)

    def test_parser_and_dispatch_are_lazy_and_label_free_antigravity(self) -> None:
        """Break caught: parse/inspect imports a scientific or label-capable phase."""
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys\n"
                    "from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity import cli_antigravity as cli\n"
                    "mods=(\n"
                    " 'experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.sources_antigravity',\n"
                    " 'experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.scoring_antigravity',\n"
                    " 'experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.qualification_antigravity',\n"
                    " 'experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.evaluation_antigravity',\n"
                    " 'experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.artifacts_antigravity',\n"
                    " 'experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.replay_antigravity',\n"
                    ")\n"
                    "assert not any(m in sys.modules for m in mods)\n"
                    "cli.parser_antigravity().parse_args(['inspect'])\n"
                    "assert not any(m in sys.modules for m in mods)\n"
                    "try:\n"
                    "    cli.parser_antigravity().parse_args(['inspect', 'extra'])\n"
                    "except SystemExit:\n"
                    "    pass\n"
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

        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity import (
            cli_antigravity as cli,
        )

        canonical_payload = {
            "decision": {"label_query_count": 0},
            "experiment": "Exp464",
        }
        with (
            patch.object(cli, "_load_evaluate_antigravity", side_effect=AssertionError("evaluate loaded")),
            patch.object(
                cli,
                "_load_inspect_antigravity",
                return_value=lambda: canonical_payload,
            ) as inspect,
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(cli.main_antigravity(["inspect"]), 0)
            inspect.assert_called_once_with()

        with (
            patch.object(cli, "_validate_clean_current_commit_antigravity", return_value=None) as validate,
            patch.object(
                cli,
                "_load_evaluate_antigravity",
                return_value=lambda commit: {"reviewed_commit": commit},
            ) as evaluate,
            patch.object(cli, "_load_inspect_antigravity", side_effect=AssertionError("inspect loaded")),
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(cli.main_antigravity(["evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY]), 0)
            validate.assert_called_once_with(_COMMIT_ANTIGRAVITY)
            evaluate.assert_called_once_with()

    def test_private_seam_rejects_dependency_and_validation_injection_antigravity(self) -> None:
        """Break caught: private seam replaces frozen CLI dependencies instead of root only."""
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity import cli_antigravity as cli

        with tempfile.TemporaryDirectory(prefix="exp464_cli_forbidden_seam_antigravity_") as temporary:
            private = types.ModuleType("_exp464_cli_private_test_antigravity")
            private.CAPABILITY_ANTIGRAVITY = "exp464_cli_private_test_antigravity_v1"
            private.root_antigravity = Path(temporary) / "result_antigravity"
            private.evaluate_dependencies_antigravity = lambda: None
            private.inspect_antigravity = lambda root: None
            private.validate_clean_current_commit_antigravity = lambda commit: None
            private.validate_no_duplicate_process_antigravity = lambda: None
            with (
                patch.dict(sys.modules, {private.__name__: private}),
                patch.object(cli, "_validate_clean_current_commit_antigravity", return_value=None),
                self.assertRaisesRegex(ValueError, "private test seam may inject only root_antigravity"),
            ):
                cli.main_antigravity(["evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY])
            with (
                patch.dict(sys.modules, {private.__name__: private}),
                self.assertRaisesRegex(ValueError, "private test seam may inject only root_antigravity"),
            ):
                cli.main_antigravity(["inspect"])

    def test_private_root_public_dispatch_evaluate_then_inspect_uses_real_lower_layer_path_antigravity(self) -> None:
        """Break caught: public dispatch skips the frozen lower layers or replay-backed equality."""
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity import cli_antigravity as cli

        authority, qualification = self._source_free_lower_layer_inputs_antigravity()
        with tempfile.TemporaryDirectory(prefix="exp464_cli_private_antigravity_") as temporary:
            root = Path(temporary) / "result_antigravity"
            private = types.ModuleType("_exp464_cli_private_test_antigravity")
            private.CAPABILITY_ANTIGRAVITY = "exp464_cli_private_test_antigravity_v1"
            private.root_antigravity = root
            with (
                patch.dict(sys.modules, {private.__name__: private}),
                patch.object(cli, "_validate_clean_current_commit_antigravity", return_value=None),
                patch(
                    "experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.predecessor_antigravity.capture_exp463_authority_antigravity",
                    return_value=authority,
                ),
                patch(
                    "experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.qualification_antigravity.qualify_methods_antigravity",
                    return_value=qualification,
                ),
            ):
                evaluated = self._run_main_payload_antigravity(
                    cli,
                    ["evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY],
                )
                inspected = self._run_main_payload_antigravity(cli, ["inspect"])
            self.assertEqual(evaluated, inspected)
            self.assertEqual(evaluated["experiment"], "Exp464")
            self.assertEqual(evaluated["decision"]["label_query_count"], 0)
            self.assertEqual(tuple(sorted(entry.name for entry in root.iterdir())), tuple(sorted(LEAVES_ANTIGRAVITY)))
            self.assertEqual(tuple(sorted(evaluated["leaf_sha256"])), tuple(sorted(LEAVES_ANTIGRAVITY)))
            before = {entry.name: entry.read_bytes() for entry in root.iterdir()}
            with (
                patch.dict(sys.modules, {private.__name__: private}),
                patch.object(cli, "_validate_clean_current_commit_antigravity", return_value=None),
                patch(
                    "experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.predecessor_antigravity.capture_exp463_authority_antigravity",
                    return_value=authority,
                ),
                patch(
                    "experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.qualification_antigravity.qualify_methods_antigravity",
                    side_effect=AssertionError("qualification should not rerun"),
                ),
                self.assertRaisesRegex(ValueError, "create-only"),
            ):
                cli.main_antigravity(["evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY])
            self.assertEqual({entry.name: entry.read_bytes() for entry in root.iterdir()}, before)
            self.assertFalse(_OFFICIAL_ROOT_ANTIGRAVITY.exists())

    def test_public_dispatch_official_evaluate_then_inspect_uses_internal_lifecycle_authority_antigravity(self) -> None:
        """Break caught: public dispatch still needs helper-only official lifecycle wiring."""
        summary = self._run_clean_snapshot_public_dispatch_antigravity()
        self.assertEqual(summary["evaluate_code"], 0)
        self.assertEqual(summary["inspect_code"], 0)
        self.assertTrue(summary["official_root_exists"])
        self.assertEqual(
            tuple(summary["official_root_entries"]),
            tuple(sorted(LEAVES_ANTIGRAVITY)),
        )
        self.assertEqual(summary["evaluate_payload"], summary["inspect_payload"])
        self.assertEqual(
            tuple(sorted(summary["evaluate_payload"]["leaf_sha256"])),
            tuple(sorted(LEAVES_ANTIGRAVITY)),
        )
        self.assertEqual(summary["evaluate_payload"]["decision"]["label_query_count"], 0)

    def test_public_interfaces_do_not_expose_official_lifecycle_minting_antigravity(self) -> None:
        """Break caught: Task 5 callers can still mint official authority through public helpers."""
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity import artifacts_antigravity as artifacts
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity import cli_antigravity as cli

        self.assertFalse(hasattr(cli, "_evaluate_official_antigravity"))
        self.assertFalse(hasattr(cli, "_inspect_official_antigravity"))
        self.assertNotIn("_official_lifecycle_authority_antigravity", getattr(artifacts, "__all__", ()))

    def test_private_root_existing_path_fails_before_validation_or_qualification_antigravity(self) -> None:
        """Break caught: existing private root reaches validation or heavy compute."""
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity import cli_antigravity as cli

        with tempfile.TemporaryDirectory(prefix="exp464_cli_existing_root_antigravity_") as temporary:
            root = Path(temporary) / "result_antigravity"
            root.mkdir()
            private = types.ModuleType("_exp464_cli_private_test_antigravity")
            private.CAPABILITY_ANTIGRAVITY = "exp464_cli_private_test_antigravity_v1"
            private.root_antigravity = root
            with (
                patch.dict(sys.modules, {private.__name__: private}),
                patch.object(
                    cli,
                    "_validate_clean_current_commit_antigravity",
                    side_effect=AssertionError("clean validation should not run"),
                ),
                patch(
                    "experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.qualification_antigravity.qualify_methods_antigravity",
                    side_effect=AssertionError("qualification should not run"),
                ),
                self.assertRaisesRegex(ValueError, "create-only"),
            ):
                cli.main_antigravity(["evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY])

    def test_private_root_symlink_fails_before_validation_or_qualification_antigravity(self) -> None:
        """Break caught: symlink private root reaches validation or heavy compute."""
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity import cli_antigravity as cli

        with tempfile.TemporaryDirectory(prefix="exp464_cli_symlink_root_antigravity_") as temporary:
            link_root = Path(temporary) / "result_antigravity"
            target_root = Path(temporary) / "target_antigravity"
            target_root.mkdir()
            link_root.symlink_to(target_root, target_is_directory=True)
            private = types.ModuleType("_exp464_cli_private_test_antigravity")
            private.CAPABILITY_ANTIGRAVITY = "exp464_cli_private_test_antigravity_v1"
            private.root_antigravity = link_root
            with (
                patch.dict(sys.modules, {private.__name__: private}),
                patch.object(
                    cli,
                    "_validate_clean_current_commit_antigravity",
                    side_effect=AssertionError("clean validation should not run"),
                ),
                patch(
                    "experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.qualification_antigravity.qualify_methods_antigravity",
                    side_effect=AssertionError("qualification should not run"),
                ),
                self.assertRaisesRegex(ValueError, "non-symlink"),
            ):
                cli.main_antigravity(["evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY])

    def test_private_root_rebound_fails_before_qualification_antigravity(self) -> None:
        """Break caught: root created after preflight still reaches heavy compute."""
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity import cli_antigravity as cli

        authority, _qualification = self._source_free_lower_layer_inputs_antigravity()
        with tempfile.TemporaryDirectory(prefix="exp464_cli_rebound_root_antigravity_") as temporary:
            root = Path(temporary) / "result_antigravity"
            private = types.ModuleType("_exp464_cli_private_test_antigravity")
            private.CAPABILITY_ANTIGRAVITY = "exp464_cli_private_test_antigravity_v1"
            private.root_antigravity = root

            def rebound_validate_antigravity(_commit: str) -> None:
                root.mkdir()

            with (
                patch.dict(sys.modules, {private.__name__: private}),
                patch.object(
                    cli,
                    "_validate_clean_current_commit_antigravity",
                    side_effect=rebound_validate_antigravity,
                ),
                patch(
                    "experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.predecessor_antigravity.capture_exp463_authority_antigravity",
                    return_value=authority,
                ),
                patch(
                    "experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.qualification_antigravity.qualify_methods_antigravity",
                    side_effect=AssertionError("qualification should not run"),
                ),
                self.assertRaisesRegex(ValueError, "create-only"),
            ):
                cli.main_antigravity(["evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY])

    def test_private_root_guard_rejects_resolved_official_alias_antigravity(self) -> None:
        """Break caught: a lexical alias selects the official root."""
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity import cli_antigravity as cli

        alias = (
            _OFFICIAL_ROOT_ANTIGRAVITY.parent
            / ".."
            / _OFFICIAL_ROOT_ANTIGRAVITY.parent.name
            / _OFFICIAL_ROOT_ANTIGRAVITY.name
        )
        private = types.ModuleType("_exp464_cli_private_test_antigravity")
        private.CAPABILITY_ANTIGRAVITY = "exp464_cli_private_test_antigravity_v1"
        private.root_antigravity = alias
        with (
            patch.dict(sys.modules, {private.__name__: private}),
            patch.object(
                cli,
                "_validate_clean_current_commit_antigravity",
                side_effect=AssertionError("clean validation should not run"),
            ),
            self.assertRaisesRegex(ValueError, "private test root must be a non-official Path"),
        ):
            cli.main_antigravity(["evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY])
        with (
            patch.dict(sys.modules, {private.__name__: private}),
            self.assertRaisesRegex(ValueError, "private test root must be a non-official Path"),
        ):
            cli.main_antigravity(["inspect"])
        self.assertFalse(_OFFICIAL_ROOT_ANTIGRAVITY.exists())

    def test_private_root_guard_rejects_non_path_before_validation_antigravity(self) -> None:
        """Break caught: non-Path private root reaches validation."""
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity import cli_antigravity as cli

        private = types.ModuleType("_exp464_cli_private_test_antigravity")
        private.CAPABILITY_ANTIGRAVITY = "exp464_cli_private_test_antigravity_v1"
        private.root_antigravity = "not-a-path"
        with (
            patch.dict(sys.modules, {private.__name__: private}),
            patch.object(
                cli,
                "_validate_clean_current_commit_antigravity",
                side_effect=AssertionError("clean validation should not run"),
            ),
            self.assertRaisesRegex(ValueError, "private test root must be a non-official Path"),
        ):
            cli.main_antigravity(["evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY])

    def test_duplicate_process_guard_ignores_current_ancestor_wrapper_chain_antigravity(self) -> None:
        """Break caught: the current shell/time wrapper trips the duplicate-process guard before science starts."""
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity import cli_antigravity as cli

        process_output = "\n".join(
            (
                "300 200 /Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python -c helper_antigravity",
                f"200 100 zsh -lc /usr/bin/time -p -o logs/run1_corrected_evaluate_time_antigravity.log /Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python {str(_LAUNCHER_ANTIGRAVITY)!s} evaluate --reviewed-commit { _COMMIT_ANTIGRAVITY }",
                f"100 1 /usr/bin/time -p -o logs/run1_corrected_evaluate_time_antigravity.log /Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python {str(_LAUNCHER_ANTIGRAVITY)!s} evaluate --reviewed-commit { _COMMIT_ANTIGRAVITY }",
            ),
        )
        with (
            patch.object(cli.os, "getpid", return_value=300),
            patch.object(cli.subprocess, "check_output", return_value=process_output),
        ):
            cli._validate_no_duplicate_process_antigravity()

    def test_duplicate_process_guard_still_rejects_independent_launcher_antigravity(self) -> None:
        """Break caught: excluding ancestors also masks a genuinely separate active Exp464 launcher."""
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity import cli_antigravity as cli

        process_output = "\n".join(
            (
                "300 200 /Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python -c helper_antigravity",
                "200 1 zsh -lc helper_antigravity",
                f"999 1 /Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python {str(_LAUNCHER_ANTIGRAVITY)!s} evaluate --reviewed-commit {_COMMIT_ANTIGRAVITY}",
            ),
        )
        with (
            patch.object(cli.os, "getpid", return_value=300),
            patch.object(cli.subprocess, "check_output", return_value=process_output),
            self.assertRaisesRegex(ValueError, "duplicate Exp464 process is already active"),
        ):
            cli._validate_no_duplicate_process_antigravity()

    def test_inspect_dependency_path_avoids_source_sqlite_and_aeon_imports_antigravity(self) -> None:
        """Break caught: inspect imports TRAIN source, SQLite, or aeon."""
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from pathlib import Path\n"
                    "import contextlib, io, json, sys, tempfile, types\n"
                    "from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity import cli_antigravity as cli\n"
                    "from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity import replay_antigravity as replay\n"
                    "private = types.ModuleType('_exp464_cli_private_test_antigravity')\n"
                    "private.CAPABILITY_ANTIGRAVITY = 'exp464_cli_private_test_antigravity_v1'\n"
                    "with tempfile.TemporaryDirectory(prefix='exp464_inspect_stub_antigravity_') as temporary:\n"
                    "    private.root_antigravity = Path(temporary) / 'result_antigravity'\n"
                    "    replay.inspect_replay_at_antigravity = lambda root, official_lifecycle_authority=None: {'experiment': 'Exp464', 'decision': {'label_query_count': 0}, 'root': root.name}\n"
                    "    sys.modules[private.__name__] = private\n"
                    "    output = io.StringIO()\n"
                    "    with contextlib.redirect_stdout(output):\n"
                    "        code = cli.main_antigravity(['inspect'])\n"
                    "    forbidden = (\n"
                    "        'experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.sources_antigravity',\n"
                    "        'sqlite3',\n"
                    "    )\n"
                    "    print(json.dumps({'code': code, 'payload': json.loads(output.getvalue()), 'forbidden_loaded': [m for m in forbidden if m in sys.modules], 'aeon_loaded': any(m == 'aeon' or m.startswith('aeon.') for m in sys.modules)}, sort_keys=True))\n"
                ),
            ],
            cwd=_REPOSITORY_ANTIGRAVITY,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        proof = json.loads(completed.stdout)
        self.assertEqual(proof["code"], 0)
        self.assertEqual(proof["payload"]["decision"]["label_query_count"], 0)
        self.assertEqual(proof["forbidden_loaded"], [])
        self.assertFalse(proof["aeon_loaded"])

    def test_launcher_imports_package_from_outside_repository_cwd_antigravity(self) -> None:
        """Break caught: launcher relies on repository CWD."""
        self.assertTrue(_LAUNCHER_ANTIGRAVITY.is_file())
        with tempfile.TemporaryDirectory(prefix="exp464_launcher_cwd_antigravity_") as temporary:
            completed = subprocess.run(
                [sys.executable, str(_LAUNCHER_ANTIGRAVITY), "inspect", "extra"],
                cwd=temporary,
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("only bare inspect", completed.stderr)
        self.assertNotIn("ModuleNotFoundError", completed.stderr)

    def test_launcher_reexecs_exact_fixed_interpreter_antigravity(self) -> None:
        """Break caught: launcher accepts caller runtime or a different interpreter."""
        script = r'''
from __future__ import annotations
import json
import os
from pathlib import Path
import runpy
import sys

launcher = Path(sys.argv[1])
captured = {}

class CapturedExecAntigravity(Exception):
    pass

def capture_execve_antigravity(executable, argv, env):
    captured.update(executable=executable, argv=argv, marker=env.get("EXP464_FIXED_RUNTIME_REEXEC_ANTIGRAVITY"))
    raise CapturedExecAntigravity

os.execve = capture_execve_antigravity
sys.argv = [str(launcher), "inspect", "extra"]
try:
    runpy.run_path(str(launcher), run_name="__main__")
except CapturedExecAntigravity:
    pass
print(json.dumps(captured, sort_keys=True))
'''
        with tempfile.TemporaryDirectory(prefix="exp464_launcher_reexec_antigravity_") as temporary:
            completed = subprocess.run(
                ["/usr/bin/python3", "-c", script, str(_LAUNCHER_ANTIGRAVITY)],
                cwd=temporary,
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        captured = json.loads(completed.stdout)
        self.assertEqual(captured["executable"], _FIXED_EXECUTABLE_ANTIGRAVITY)
        self.assertEqual(
            captured["argv"],
            [_FIXED_EXECUTABLE_ANTIGRAVITY, str(_LAUNCHER_ANTIGRAVITY), "inspect", "extra"],
        )
        self.assertEqual(captured["marker"], "1")
        self.assertEqual(_FIXED_PREFIX_ANTIGRAVITY, str(Path(_FIXED_EXECUTABLE_ANTIGRAVITY).parent.parent))


if __name__ == "__main__":
    unittest.main()
