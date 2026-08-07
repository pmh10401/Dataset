from pathlib import Path
import subprocess
import sys
import unittest

from experiments.exp438_frequency_complexity_evidence_antigravity.cli_antigravity import (
    CliDependenciesAntigravity,
    build_parser_antigravity,
    main_with_dependencies_antigravity,
)


class Exp438CliAntigravityTests(unittest.TestCase):
    def test_parser_exposes_exactly_five_commands_and_no_overrides(self) -> None:
        parser = build_parser_antigravity()
        for command in ("readiness", "new", "resume", "inspect", "evaluate"):
            self.assertEqual(parser.parse_args([command]).command, command)
        for forbidden in (
            ["new", "--workers", "1"],
            ["new", "--backend", "cpu"],
            ["new", "--threshold", "0.5"],
            ["evaluate", "--alpha", "0.1"],
        ):
            with self.assertRaises(SystemExit):
                parser.parse_args(forbidden)

    def test_dispatch_calls_only_the_selected_dependency(self) -> None:
        called = []
        dependencies = CliDependenciesAntigravity(
            **{
                command: (lambda command=command: called.append(command))
                for command in ("readiness", "new", "resume", "inspect", "evaluate")
            }
        )
        self.assertEqual(main_with_dependencies_antigravity(["inspect"], dependencies), 0)
        self.assertEqual(called, ["inspect"])
        self.assertNotIn(
            "experiments.exp438_frequency_complexity_evidence_antigravity.evaluation_antigravity",
            sys.modules,
        )

    def test_wrapper_help_works_outside_repository_cwd(self) -> None:
        root = Path(__file__).resolve().parents[3]
        wrapper = root / "scratch/run_exp438_frequency_complexity_evidence_antigravity.py"
        completed = subprocess.run(
            [sys.executable, str(wrapper), "--help"],
            cwd="/tmp",
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("readiness", completed.stdout)
        self.assertIn("evaluate", completed.stdout)


if __name__ == "__main__":
    unittest.main()
