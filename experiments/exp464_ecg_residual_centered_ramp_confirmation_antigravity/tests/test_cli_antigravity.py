from __future__ import annotations

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
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
                self.assertRaisesRegex(ValueError, "private test seam may inject only root_antigravity"),
            ):
                cli._evaluate_antigravity(_COMMIT_ANTIGRAVITY)
            with (
                patch.dict(sys.modules, {private.__name__: private}),
                self.assertRaisesRegex(ValueError, "private test seam may inject only root_antigravity"),
            ):
                cli._inspect_antigravity()

    def test_private_root_evaluate_then_inspect_uses_real_lower_layer_path_antigravity(self) -> None:
        """Break caught: CLI uses seam-level dependency replacement instead of frozen lower layers."""
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity import cli_antigravity as cli

        authority, qualification = self._source_free_lower_layer_inputs_antigravity()
        with tempfile.TemporaryDirectory(prefix="exp464_cli_private_antigravity_") as temporary:
            root = Path(temporary) / "result_antigravity"
            private = types.ModuleType("_exp464_cli_private_test_antigravity")
            private.CAPABILITY_ANTIGRAVITY = "exp464_cli_private_test_antigravity_v1"
            private.root_antigravity = root
            with (
                patch.dict(sys.modules, {private.__name__: private}),
                patch(
                    "experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.predecessor_antigravity.capture_exp463_authority_antigravity",
                    return_value=authority,
                ),
                patch(
                    "experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.qualification_antigravity.qualify_methods_antigravity",
                    return_value=qualification,
                ),
            ):
                evaluated = cli._evaluate_antigravity(_COMMIT_ANTIGRAVITY)
                inspected = cli._inspect_antigravity()
            self.assertEqual(evaluated, inspected)
            self.assertEqual(evaluated["experiment"], "Exp464")
            self.assertEqual(evaluated["decision"]["label_query_count"], 0)
            self.assertEqual(tuple(sorted(entry.name for entry in root.iterdir())), tuple(sorted(LEAVES_ANTIGRAVITY)))
            self.assertEqual(tuple(evaluated["leaf_sha256"]), LEAVES_ANTIGRAVITY)
            before = {entry.name: entry.read_bytes() for entry in root.iterdir()}
            with (
                patch.dict(sys.modules, {private.__name__: private}),
                patch(
                    "experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.predecessor_antigravity.capture_exp463_authority_antigravity",
                    return_value=authority,
                ),
                patch(
                    "experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.qualification_antigravity.qualify_methods_antigravity",
                    return_value=qualification,
                ),
                self.assertRaises(ValueError),
            ):
                cli._evaluate_antigravity(_COMMIT_ANTIGRAVITY)
            self.assertEqual({entry.name: entry.read_bytes() for entry in root.iterdir()}, before)
            self.assertFalse(_OFFICIAL_ROOT_ANTIGRAVITY.exists())

    def test_task5_public_path_cannot_create_official_root_without_internal_authority_antigravity(self) -> None:
        """Break caught: Task 5 path can create the official root without Task 6 authority."""
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity import cli_antigravity as cli
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity import artifacts_antigravity as artifacts
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity import replay_antigravity as replay

        authority, qualification = self._source_free_lower_layer_inputs_antigravity()
        with tempfile.TemporaryDirectory(prefix="exp464_cli_official_guard_antigravity_") as temporary:
            official_root = Path(temporary) / "official_root_antigravity"
            with ExitStack() as stack:
                stack.enter_context(patch.object(cli, "OFFICIAL_ROOT_ANTIGRAVITY", official_root))
                stack.enter_context(patch.object(artifacts, "OFFICIAL_ROOT_ANTIGRAVITY", official_root))
                stack.enter_context(patch.object(replay, "OFFICIAL_ROOT_ANTIGRAVITY", official_root))
                stack.enter_context(
                    patch(
                        "experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.predecessor_antigravity.capture_exp463_authority_antigravity",
                        return_value=authority,
                    )
                )
                stack.enter_context(
                    patch(
                        "experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.qualification_antigravity.qualify_methods_antigravity",
                        return_value=qualification,
                    )
                )
                with self.assertRaisesRegex(ValueError, "official result root requires internal official lifecycle authority"):
                    cli._evaluate_antigravity(_COMMIT_ANTIGRAVITY)
                self.assertFalse(official_root.exists())
                with self.assertRaisesRegex(ValueError, "official result root requires internal official lifecycle authority"):
                    cli._inspect_antigravity()
                self.assertFalse(official_root.exists())

    def test_authorized_official_lifecycle_uses_public_official_root_with_internal_authority_antigravity(self) -> None:
        """Break caught: Task 6 official evaluate/inspect still needs later product changes."""
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity import cli_antigravity as cli
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity import artifacts_antigravity as artifacts
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity import replay_antigravity as replay

        authority, qualification = self._source_free_lower_layer_inputs_antigravity()
        with tempfile.TemporaryDirectory(prefix="exp464_cli_official_lifecycle_antigravity_") as temporary:
            official_root = Path(temporary) / "official_root_antigravity"
            with ExitStack() as stack:
                stack.enter_context(patch.object(cli, "OFFICIAL_ROOT_ANTIGRAVITY", official_root))
                stack.enter_context(patch.object(artifacts, "OFFICIAL_ROOT_ANTIGRAVITY", official_root))
                stack.enter_context(patch.object(replay, "OFFICIAL_ROOT_ANTIGRAVITY", official_root))
                stack.enter_context(
                    patch(
                        "experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.predecessor_antigravity.capture_exp463_authority_antigravity",
                        return_value=authority,
                    )
                )
                stack.enter_context(
                    patch(
                        "experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.qualification_antigravity.qualify_methods_antigravity",
                        return_value=qualification,
                    )
                )
                evaluated = cli._evaluate_official_antigravity(_COMMIT_ANTIGRAVITY)
                inspected = cli._inspect_official_antigravity()
                self.assertEqual(official_root, cli.OFFICIAL_ROOT_ANTIGRAVITY)
            self.assertEqual(evaluated, inspected)
            self.assertTrue(official_root.is_dir())
            self.assertEqual(tuple(evaluated["leaf_sha256"]), LEAVES_ANTIGRAVITY)
            self.assertEqual(evaluated["decision"]["label_query_count"], 0)

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
            patch.object(cli, "_validate_clean_current_commit_antigravity", return_value=None),
            self.assertRaisesRegex(ValueError, "private test root must be a non-official Path"),
        ):
            cli.main_antigravity(["evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY])
        with (
            patch.dict(sys.modules, {private.__name__: private}),
            self.assertRaisesRegex(ValueError, "private test root must be a non-official Path"),
        ):
            cli.main_antigravity(["inspect"])
        self.assertFalse(_OFFICIAL_ROOT_ANTIGRAVITY.exists())

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
