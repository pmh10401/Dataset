from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from experiments.exp461_ecg_period_signature_qualification_antigravity.tests.test_artifacts_antigravity import (
    LEAVES_ANTIGRAVITY,
    publication_inputs_antigravity,
)


_COMMIT_ANTIGRAVITY = "a" * 40
_FIXED_EXECUTABLE_ANTIGRAVITY = (
    "/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python"
)
_REPOSITORY_ANTIGRAVITY = Path(__file__).resolve().parents[3]
_LAUNCHER_ANTIGRAVITY = (
    _REPOSITORY_ANTIGRAVITY
    / "scratch/run_exp461_ecg_period_signature_qualification_antigravity.py"
)
_OFFICIAL_ROOT_ANTIGRAVITY = Path(
    "/Users/mac/Documents/Dataset/exp461_ecg_period_signature_qualification_antigravity_20260820"
)


class CliAntigravityTest(unittest.TestCase):
    def test_exact_surface_accepts_only_reviewed_evaluate_or_bare_inspect_antigravity(self) -> None:
        """Break caught: aliases, abbreviations, overrides, retries, or extra tokens reach execution."""
        from experiments.exp461_ecg_period_signature_qualification_antigravity.cli_antigravity import (
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
            ("evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY, "--source", "x"),
            ("evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY, "--seed", "1"),
            ("evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY, "--fold", "0"),
            ("evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY, "--threshold", "0.1"),
            ("evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY, "--resume"),
            ("evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY, "--retry"),
            ("inspect", "extra"),
            ("inspect", "--reviewed-commit", _COMMIT_ANTIGRAVITY),
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

    def test_parser_and_dispatch_are_lazy_antigravity(self) -> None:
        """Break caught: parsing imports scientific phases, or dispatch loads the unselected phase."""
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys\n"
                    "from experiments.exp461_ecg_period_signature_qualification_antigravity import cli_antigravity as cli\n"
                    "mods=(\n"
                    " 'experiments.exp461_ecg_period_signature_qualification_antigravity.sources_antigravity',\n"
                    " 'experiments.exp461_ecg_period_signature_qualification_antigravity.scoring_antigravity',\n"
                    " 'experiments.exp461_ecg_period_signature_qualification_antigravity.qualification_antigravity',\n"
                    " 'experiments.exp461_ecg_period_signature_qualification_antigravity.evaluation_antigravity',\n"
                    " 'experiments.exp461_ecg_period_signature_qualification_antigravity.artifacts_antigravity',\n"
                    " 'experiments.exp461_ecg_period_signature_qualification_antigravity.replay_antigravity',\n"
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

        from experiments.exp461_ecg_period_signature_qualification_antigravity import (
            cli_antigravity as cli,
        )

        with (
            patch.object(
                cli,
                "_load_evaluate_antigravity",
                side_effect=AssertionError("evaluate phase loaded"),
            ),
            patch.object(
                cli,
                "_load_inspect_antigravity",
                return_value=lambda: {"phase": "inspect", "label_query_count": 0},
            ) as inspect,
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(cli.main_antigravity(["inspect"]), 0)
            inspect.assert_called_once_with()

        with (
            patch.object(
                cli,
                "_validate_clean_current_commit_antigravity",
                return_value=None,
            ) as validate,
            patch.object(
                cli,
                "_load_evaluate_antigravity",
                return_value=lambda commit: {"phase": "evaluate", "commit": commit},
            ) as evaluate,
            patch.object(
                cli,
                "_load_inspect_antigravity",
                side_effect=AssertionError("inspect phase loaded"),
            ),
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(
                cli.main_antigravity(["evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY]),
                0,
            )
            validate.assert_called_once_with(_COMMIT_ANTIGRAVITY)
            evaluate.assert_called_once_with()

    def test_private_evaluate_then_inspect_returns_same_payload_and_seven_leaves_antigravity(self) -> None:
        """Break caught: CLI skips qualification/evaluation/publication/replay or overwrites a private root."""
        from experiments.exp461_ecg_period_signature_qualification_antigravity import (
            cli_antigravity as cli,
        )
        from experiments.exp461_ecg_period_signature_qualification_antigravity.artifacts_antigravity import (
            publish_result_antigravity,
        )
        from experiments.exp461_ecg_period_signature_qualification_antigravity.evaluation_antigravity import (
            evaluate_qualification_antigravity,
        )
        from experiments.exp461_ecg_period_signature_qualification_antigravity.replay_antigravity import (
            inspect_replay_at_antigravity,
        )

        authority, qualification, _decision = publication_inputs_antigravity()
        with tempfile.TemporaryDirectory(prefix="exp461_cli_private_antigravity_") as temporary:
            root = Path(temporary) / "result_antigravity"
            self.assertNotEqual(
                root.resolve(strict=False),
                _OFFICIAL_ROOT_ANTIGRAVITY.resolve(strict=False),
            )
            evaluated = cli._evaluate_with_dependencies_antigravity(
                root,
                lambda: authority,
                lambda: qualification,
                evaluate_qualification_antigravity,
                publish_result_antigravity,
            )
            inspected = cli._inspect_with_dependencies_antigravity(
                root,
                inspect_replay_at_antigravity,
            )
            self.assertEqual(evaluated, inspected)
            self.assertEqual(evaluated["experiment"], "Exp461")
            self.assertEqual(evaluated["label_query_count"], 0)
            self.assertEqual(
                tuple(sorted(entry.name for entry in root.iterdir())),
                tuple(sorted(LEAVES_ANTIGRAVITY)),
            )
            before = {entry.name: entry.read_bytes() for entry in root.iterdir()}
            with self.assertRaisesRegex(ValueError, "publication destination must not exist"):
                cli._evaluate_with_dependencies_antigravity(
                    root,
                    lambda: authority,
                    lambda: qualification,
                    evaluate_qualification_antigravity,
                    publish_result_antigravity,
                )
            self.assertEqual({entry.name: entry.read_bytes() for entry in root.iterdir()}, before)

    def test_private_root_guard_rejects_resolved_official_alias_antigravity(self) -> None:
        """Break caught: a lexical path alias lets private execution select the official root."""
        from experiments.exp461_ecg_period_signature_qualification_antigravity import (
            cli_antigravity as cli,
        )

        official_root_existed_before_antigravity = _OFFICIAL_ROOT_ANTIGRAVITY.exists()
        alias = (
            _OFFICIAL_ROOT_ANTIGRAVITY.parent
            / ".."
            / _OFFICIAL_ROOT_ANTIGRAVITY.parent.name
            / _OFFICIAL_ROOT_ANTIGRAVITY.name
        )
        private = types.ModuleType("_exp461_cli_private_test_antigravity")
        private.CAPABILITY_ANTIGRAVITY = "exp461_cli_private_test_antigravity_v1"
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
        self.assertEqual(_OFFICIAL_ROOT_ANTIGRAVITY.exists(), official_root_existed_before_antigravity)

    def test_inspect_dependency_path_avoids_source_sqlite_and_aeon_imports_antigravity(self) -> None:
        """Break caught: CLI inspect itself imports TRAIN source, SQLite, or aeon instead of a replay dependency."""
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from pathlib import Path\n"
                    "import contextlib, io, json, sys, tempfile, types\n"
                    "from experiments.exp461_ecg_period_signature_qualification_antigravity import cli_antigravity as cli\n"
                    "private = types.ModuleType('_exp461_cli_private_test_antigravity')\n"
                    "private.CAPABILITY_ANTIGRAVITY = 'exp461_cli_private_test_antigravity_v1'\n"
                    "with tempfile.TemporaryDirectory(prefix='exp461_inspect_stub_antigravity_') as temporary:\n"
                    "    private.root_antigravity = Path(temporary) / 'result_antigravity'\n"
                    "    private.inspect_antigravity = lambda root: {'root': root.name, 'label_query_count': 0}\n"
                    "    sys.modules[private.__name__] = private\n"
                    "    output = io.StringIO()\n"
                    "    with contextlib.redirect_stdout(output):\n"
                    "        code = cli.main_antigravity(['inspect'])\n"
                    "    forbidden = (\n"
                    "        'experiments.exp461_ecg_period_signature_qualification_antigravity.sources_antigravity',\n"
                    "        'experiments.exp461_ecg_period_signature_qualification_antigravity.scoring_antigravity',\n"
                    "        'experiments.exp461_ecg_period_signature_qualification_antigravity.qualification_antigravity',\n"
                    "        'sqlite3',\n"
                    "    )\n"
                    "    print(json.dumps({\n"
                    "        'code': code,\n"
                    "        'payload': json.loads(output.getvalue()),\n"
                    "        'forbidden_loaded': [m for m in forbidden if m in sys.modules],\n"
                    "        'aeon_loaded': any(m == 'aeon' or m.startswith('aeon.') for m in sys.modules),\n"
                    "    }, sort_keys=True))\n"
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
        self.assertEqual(proof["payload"]["label_query_count"], 0)
        self.assertEqual(proof["forbidden_loaded"], [])
        self.assertFalse(proof["aeon_loaded"])

    def test_launcher_imports_package_from_outside_repository_cwd_antigravity(self) -> None:
        """Break caught: launcher relies on repository CWD before reaching the exact parser."""
        self.assertTrue(_LAUNCHER_ANTIGRAVITY.is_file())
        with tempfile.TemporaryDirectory(prefix="exp461_launcher_cwd_antigravity_") as temporary:
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

    def test_launcher_reexecs_the_exact_fixed_interpreter_antigravity(self) -> None:
        """Break caught: launcher accepts the caller runtime or re-execs a different interpreter."""
        self.assertTrue(_LAUNCHER_ANTIGRAVITY.is_file())
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
    captured.update(executable=executable, argv=argv, marker=env.get("EXP461_FIXED_RUNTIME_REEXEC_ANTIGRAVITY"))
    raise CapturedExecAntigravity

os.execve = capture_execve_antigravity
sys.argv = [str(launcher), "inspect", "extra"]
try:
    runpy.run_path(str(launcher), run_name="__main__")
except CapturedExecAntigravity:
    pass
print(json.dumps(captured, sort_keys=True))
'''
        with tempfile.TemporaryDirectory(prefix="exp461_launcher_reexec_antigravity_") as temporary:
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


if __name__ == "__main__":
    unittest.main()
