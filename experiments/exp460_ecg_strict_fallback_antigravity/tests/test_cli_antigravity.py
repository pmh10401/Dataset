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

from experiments.exp460_ecg_strict_fallback_antigravity.tests.test_artifacts_antigravity import (
    expected_dashboard_antigravity,
)


_COMMIT_ANTIGRAVITY = "a" * 40
_FIXED_EXECUTABLE_ANTIGRAVITY = (
    "/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python"
)
_REPOSITORY_ANTIGRAVITY = Path(__file__).resolve().parents[3]
_LAUNCHER_ANTIGRAVITY = (
    _REPOSITORY_ANTIGRAVITY
    / "scratch/run_exp460_ecg_strict_fallback_antigravity.py"
)
_OFFICIAL_ROOT_ANTIGRAVITY = Path(
    "/Users/mac/Documents/Dataset/exp460_ecg_strict_fallback_antigravity_20260820"
)
_EXPECTED_LEAVES_ANTIGRAVITY = (
    "context_antigravity.json",
    "rule_comparisons_antigravity.jsonl",
    "failed_folds_antigravity.jsonl",
    "decision_antigravity.json",
    "dashboard_snapshot_antigravity.json",
    "artifact_sha256_antigravity.json",
)


class CliAntigravityTest(unittest.TestCase):
    def test_exact_surface_accepts_only_reviewed_evaluate_or_bare_inspect_antigravity(
        self,
    ) -> None:
        """Break caught: aliases, overrides, abbreviations, retries, or extra tokens reach execution."""
        from experiments.exp460_ecg_strict_fallback_antigravity.cli_antigravity import (
            OFFICIAL_ROOT_ANTIGRAVITY,
            parser_antigravity,
        )

        self.assertEqual(OFFICIAL_ROOT_ANTIGRAVITY, _OFFICIAL_ROOT_ANTIGRAVITY)
        parser = parser_antigravity()
        parsed = parser.parse_args(
            ["evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY]
        )
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
            ("evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY, "--resume"),
            ("evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY, "--retry"),
            ("evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY, "--override"),
            ("inspect", "extra"),
            ("inspect", "--reviewed-commit", _COMMIT_ANTIGRAVITY),
            ("new",),
            ("resume",),
            ("retry",),
            ("override",),
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
                    "from experiments.exp460_ecg_strict_fallback_antigravity import cli_antigravity as cli\n"
                    "mods=(\n"
                    " 'experiments.exp460_ecg_strict_fallback_antigravity.predecessor_antigravity',\n"
                    " 'experiments.exp460_ecg_strict_fallback_antigravity.evaluation_antigravity',\n"
                    " 'experiments.exp460_ecg_strict_fallback_antigravity.artifacts_antigravity',\n"
                    " 'experiments.exp460_ecg_strict_fallback_antigravity.replay_antigravity',\n"
                    " 'experiments.exp460_ecg_strict_fallback_antigravity.dashboard_export_antigravity',\n"
                    ")\n"
                    "assert not any(m in sys.modules for m in mods)\n"
                    "cli.parser_antigravity().parse_args(['inspect'])\n"
                    "assert not any(m in sys.modules for m in mods)\n"
                    "try:\n"
                    "    cli.parser_antigravity().parse_args(['inspect', 'extra'])\n"
                    "except SystemExit:\n"
                    "    pass\n"
                    "assert not any(m in sys.modules for m in mods)\n"
                ),
            ],
            cwd=_REPOSITORY_ANTIGRAVITY,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

        from experiments.exp460_ecg_strict_fallback_antigravity import (
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
                return_value=lambda: expected_dashboard_antigravity(),
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
                return_value=lambda commit: expected_dashboard_antigravity(),
            ) as evaluate,
            patch.object(
                cli,
                "_load_inspect_antigravity",
                side_effect=AssertionError("inspect phase loaded"),
            ),
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(
                cli.main_antigravity(
                    ["evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY]
                ),
                0,
            )
            validate.assert_called_once_with(_COMMIT_ANTIGRAVITY)
            evaluate.assert_called_once_with()

    def test_private_evaluate_then_inspect_returns_exact_payload_and_six_leaves_antigravity(
        self,
    ) -> None:
        """Break caught: CLI skips capture/evaluation/publication/replay or overwrites an existing root."""
        from experiments.exp460_ecg_strict_fallback_antigravity.artifacts_antigravity import (
            publish_result_antigravity,
        )
        from experiments.exp460_ecg_strict_fallback_antigravity import (
            cli_antigravity as cli,
        )
        from experiments.exp460_ecg_strict_fallback_antigravity.evaluation_antigravity import (
            compare_fallback_rules_antigravity,
        )
        from experiments.exp460_ecg_strict_fallback_antigravity.predecessor_antigravity import (
            capture_exp459_fallback_authority_antigravity,
        )
        from experiments.exp460_ecg_strict_fallback_antigravity.replay_antigravity import (
            inspect_replay_at_antigravity,
        )

        with tempfile.TemporaryDirectory(
            prefix="exp460_cli_private_antigravity_"
        ) as temporary:
            root = Path(temporary) / "result_antigravity"
            self.assertNotEqual(root, _OFFICIAL_ROOT_ANTIGRAVITY)
            evaluated = cli._evaluate_with_dependencies_antigravity(
                root,
                capture_exp459_fallback_authority_antigravity,
                compare_fallback_rules_antigravity,
                publish_result_antigravity,
            )
            inspected = cli._inspect_with_dependencies_antigravity(
                root,
                capture_exp459_fallback_authority_antigravity,
                inspect_replay_at_antigravity,
            )
            self.assertEqual(evaluated, expected_dashboard_antigravity())
            self.assertEqual(inspected, expected_dashboard_antigravity())
            self.assertEqual(evaluated, inspected)
            self.assertEqual(
                tuple(sorted(entry.name for entry in root.iterdir())),
                tuple(sorted(_EXPECTED_LEAVES_ANTIGRAVITY)),
            )
            before = {
                entry.name: entry.read_bytes() for entry in root.iterdir()
            }
            with self.assertRaisesRegex(
                ValueError, "publication destination must not exist"
            ):
                cli._evaluate_with_dependencies_antigravity(
                    root,
                    capture_exp459_fallback_authority_antigravity,
                    compare_fallback_rules_antigravity,
                    publish_result_antigravity,
                )
            self.assertEqual(
                {entry.name: entry.read_bytes() for entry in root.iterdir()},
                before,
            )

    def test_launcher_imports_package_from_outside_repository_cwd_antigravity(
        self,
    ) -> None:
        """Break caught: launcher relies on repository CWD before reaching the exact parser."""
        self.assertTrue(_LAUNCHER_ANTIGRAVITY.is_file())
        with tempfile.TemporaryDirectory(
            prefix="exp460_launcher_cwd_antigravity_"
        ) as temporary:
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
    captured.update(executable=executable, argv=argv, marker=env.get("EXP460_FIXED_RUNTIME_REEXEC_ANTIGRAVITY"))
    raise CapturedExecAntigravity

os.execve = capture_execve_antigravity
sys.argv = [str(launcher), "inspect", "extra"]
try:
    runpy.run_path(str(launcher), run_name="__main__")
except CapturedExecAntigravity:
    pass
print(json.dumps(captured, sort_keys=True))
'''
        with tempfile.TemporaryDirectory(
            prefix="exp460_launcher_reexec_antigravity_"
        ) as temporary:
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
            [
                _FIXED_EXECUTABLE_ANTIGRAVITY,
                str(_LAUNCHER_ANTIGRAVITY),
                "inspect",
                "extra",
            ],
        )
        self.assertEqual(captured["marker"], "1")

    def test_launcher_private_evaluate_then_inspect_from_outside_cwd_antigravity(
        self,
    ) -> None:
        """Break caught: launcher success requires CWD tricks or a public root override."""
        self.assertTrue(_LAUNCHER_ANTIGRAVITY.is_file())
        script = r'''
from __future__ import annotations
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import runpy
import sys
import types

launcher = Path(sys.argv[1])
root = Path(sys.argv[2])
private = types.ModuleType("_exp460_cli_private_test_antigravity")
private.CAPABILITY_ANTIGRAVITY = "exp460_cli_private_test_antigravity_v1"
private.root_antigravity = root

def validate_clean_current_commit_antigravity(reviewed_commit):
    if reviewed_commit != "b" * 40:
        raise AssertionError("unexpected reviewed commit")

private.validate_clean_current_commit_antigravity = validate_clean_current_commit_antigravity
sys.modules[private.__name__] = private
os.chdir(root.parent)

def invoke_antigravity(argv):
    sys.argv = [str(launcher), *argv]
    output = io.StringIO()
    with redirect_stdout(output):
        try:
            runpy.run_path(str(launcher), run_name="__main__")
        except SystemExit as exited:
            if exited.code != 0:
                raise
    return json.loads(output.getvalue())

evaluated = invoke_antigravity(["evaluate", "--reviewed-commit", "b" * 40])
inspected = invoke_antigravity(["inspect"])
print(json.dumps({
    "evaluated": evaluated,
    "inspected": inspected,
    "leaves": sorted(entry.name for entry in root.iterdir()),
}, sort_keys=True))
'''
        with tempfile.TemporaryDirectory(
            prefix="exp460_launcher_private_antigravity_"
        ) as temporary:
            root = Path(temporary) / "result_antigravity"
            self.assertNotEqual(root, _OFFICIAL_ROOT_ANTIGRAVITY)
            completed = subprocess.run(
                [sys.executable, "-c", script, str(_LAUNCHER_ANTIGRAVITY), str(root)],
                cwd=temporary,
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        proof = json.loads(completed.stdout)
        self.assertEqual(proof["evaluated"], expected_dashboard_antigravity())
        self.assertEqual(proof["inspected"], expected_dashboard_antigravity())
        self.assertEqual(proof["evaluated"], proof["inspected"])
        self.assertEqual(proof["leaves"], sorted(_EXPECTED_LEAVES_ANTIGRAVITY))


if __name__ == "__main__":
    unittest.main()
