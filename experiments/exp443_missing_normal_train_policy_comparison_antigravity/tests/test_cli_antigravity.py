"""Behavior-first tests for the fixed Exp443 command boundary."""
from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from io import StringIO
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


REVIEWED_COMMIT_ANTIGRAVITY = "a" * 40


@dataclass(frozen=True)
class _FixtureResultAntigravity:
    state: str
    values: tuple[int, ...] = (1, 2)


class CliAntigravityTests(unittest.TestCase):
    def _module_antigravity(self):
        from experiments.exp443_missing_normal_train_policy_comparison_antigravity import cli_antigravity

        return cli_antigravity

    def _dependencies_antigravity(self, *, calls: list[object] | None = None, failure: Exception | None = None):
        module = self._module_antigravity()
        recorded = [] if calls is None else calls

        def action(name: str, result: object):
            def invoke(*arguments: object):
                recorded.append((name, *arguments))
                if failure is not None and name == "new":
                    raise failure
                return result
            return invoke

        def import_module(name: str):
            recorded.append(("import", name))

            class EvaluatorAntigravity:
                @staticmethod
                def evaluate_publication_antigravity(attempt: Path):
                    recorded.append(("evaluate", str(attempt)))
                    return _FixtureResultAntigravity("evaluated")

            return EvaluatorAntigravity

        return module.CliDependenciesAntigravity(
            readiness=action("readiness", {"state": "ready", "nested": (1, 2)}),
            new=action("new", _FixtureResultAntigravity("complete")),
            resume=action("resume", {"state": "complete"}),
            inspect=action("inspect", _FixtureResultAntigravity("published")),
            import_module=import_module,
        )

    def _main_antigravity(self, arguments: list[str], dependencies):
        module = self._module_antigravity()
        stdout = StringIO()
        stderr = StringIO()
        code = module.main_antigravity(arguments, dependencies=dependencies, stdout=stdout, stderr=stderr)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_parser_has_only_five_commands_and_no_scientific_or_resource_override(self) -> None:
        module = self._module_antigravity()
        parser = module.build_parser_antigravity()
        choices = next(action.choices for action in parser._actions if getattr(action, "choices", None))
        self.assertEqual(tuple(choices), ("readiness", "new", "resume", "inspect", "evaluate"))
        help_text = parser.format_help() + "\n".join(choice.format_help() for choice in choices.values())
        for forbidden in (
            "--workers", "--policy", "--alpha", "--kernel", "--seed",
            "--run-root", "--output", "--database", "--backend",
        ):
            self.assertNotIn(forbidden, help_text)

    def test_required_reviewed_commit_and_resume_attempt_are_fail_closed(self) -> None:
        dependencies = self._dependencies_antigravity()
        for arguments in (["readiness"], ["new"], ["resume", "--reviewed-commit", REVIEWED_COMMIT_ANTIGRAVITY]):
            code, _stdout, stderr = self._main_antigravity(list(arguments), dependencies)
            self.assertNotEqual(code, 0)
            self.assertTrue(stderr)

    def test_success_output_is_one_json_safe_status_object(self) -> None:
        dependencies = self._dependencies_antigravity()
        code, stdout, stderr = self._main_antigravity(
            ["new", "--reviewed-commit", REVIEWED_COMMIT_ANTIGRAVITY], dependencies,
        )
        self.assertEqual(code, 0)
        self.assertEqual(stderr, "")
        payload = json.loads(stdout)
        self.assertEqual(payload["schema"], "exp443.cli-status.v1")
        self.assertEqual(payload["command"], "new")
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["result"], {"state": "complete", "values": [1, 2]})
        self.assertEqual(stdout.count("\n"), 1)

    def test_action_failure_is_json_safe_and_returns_nonzero(self) -> None:
        dependencies = self._dependencies_antigravity(failure=ValueError("fixture failure"))
        code, stdout, stderr = self._main_antigravity(
            ["new", "--reviewed-commit", REVIEWED_COMMIT_ANTIGRAVITY], dependencies,
        )
        self.assertEqual(code, 1)
        self.assertEqual(stdout, "")
        payload = json.loads(stderr)
        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["error_type"], "ValueError")
        self.assertEqual(payload["error"], "fixture failure")

    def test_non_evaluate_commands_never_import_the_evaluator(self) -> None:
        commands = (
            ["readiness", "--reviewed-commit", REVIEWED_COMMIT_ANTIGRAVITY],
            ["new", "--reviewed-commit", REVIEWED_COMMIT_ANTIGRAVITY],
            ["resume", "--reviewed-commit", REVIEWED_COMMIT_ANTIGRAVITY, "--attempt", "attempt_0000_antigravity"],
            ["inspect"],
        )
        for arguments in commands:
            calls: list[object] = []
            code, _stdout, _stderr = self._main_antigravity(list(arguments), self._dependencies_antigravity(calls=calls))
            self.assertEqual(code, 0)
            self.assertFalse(any(call[0] == "import" for call in calls), calls)

    def test_evaluate_inspects_publication_before_lazy_import_and_evaluation(self) -> None:
        calls: list[object] = []
        code, stdout, stderr = self._main_antigravity(["evaluate"], self._dependencies_antigravity(calls=calls))
        self.assertEqual(code, 0)
        self.assertEqual(stderr, "")
        self.assertEqual([call[0] for call in calls], ["inspect", "import", "evaluate"])
        self.assertEqual(calls[1][1], "experiments.exp443_missing_normal_train_policy_comparison_antigravity.evaluation_antigravity")
        self.assertEqual(json.loads(stdout)["result"]["state"], "evaluated")

    def test_evaluate_does_not_import_when_publication_inspection_fails(self) -> None:
        module = self._module_antigravity()
        calls: list[object] = []

        def inspect(_attempt: Path):
            calls.append(("inspect",))
            raise ValueError("publication absent")

        dependencies = module.CliDependenciesAntigravity(
            readiness=lambda _commit: {}, new=lambda _commit: {}, resume=lambda _commit, _attempt: {},
            inspect=inspect,
            import_module=lambda name: calls.append(("import", name)),
        )
        code, _stdout, stderr = self._main_antigravity(["evaluate"], dependencies)
        self.assertEqual(code, 1)
        self.assertEqual([call[0] for call in calls], ["inspect"])
        self.assertIn("publication absent", stderr)

    def test_resume_passes_only_an_exact_safe_attempt_basename(self) -> None:
        valid_calls: list[object] = []
        valid = self._dependencies_antigravity(calls=valid_calls)
        code, _stdout, _stderr = self._main_antigravity(
            ["resume", "--reviewed-commit", REVIEWED_COMMIT_ANTIGRAVITY, "--attempt", "attempt_0042_antigravity"], valid,
        )
        self.assertEqual(code, 0)
        self.assertIn(("resume", REVIEWED_COMMIT_ANTIGRAVITY, "attempt_0042_antigravity"), valid_calls)
        for invalid in (
            "/tmp/attempt_0000_antigravity", "../attempt_0000_antigravity",
            "attempt_0_antigravity", "attempt_0000", "attempt_0000_antigravity/..",
        ):
            calls: list[object] = []
            code, _stdout, _stderr = self._main_antigravity(
                ["resume", "--reviewed-commit", REVIEWED_COMMIT_ANTIGRAVITY, "--attempt", invalid],
                self._dependencies_antigravity(calls=calls),
            )
            self.assertNotEqual(code, 0, invalid)
            self.assertFalse(any(call[0] == "resume" for call in calls), invalid)

    def test_cli_source_has_no_evaluator_import_statement(self) -> None:
        module = self._module_antigravity()
        source = Path(module.__file__).read_text(encoding="utf-8")
        self.assertNotIn("from .evaluation_antigravity", source)
        self.assertNotIn("import evaluation_antigravity", source)

    def test_inspect_calls_only_the_read_only_inspection_dependency(self) -> None:
        calls: list[object] = []
        code, _stdout, _stderr = self._main_antigravity(
            ["inspect"], self._dependencies_antigravity(calls=calls),
        )
        self.assertEqual(code, 0)
        self.assertEqual([call[0] for call in calls], ["inspect"])

    def test_wrapper_is_fixed_to_official_interpreter_and_works_outside_repo_cwd(self) -> None:
        repository = Path(__file__).resolve().parents[3]
        wrapper = repository / "scratch/run_exp443_missing_normal_train_policy_comparison_antigravity.py"
        self.assertTrue(wrapper.is_file())
        source = wrapper.read_text(encoding="utf-8")
        self.assertNotIn(".venv", source)
        specification = importlib.util.spec_from_file_location("exp443_wrapper_antigravity", wrapper)
        self.assertIsNotNone(specification)
        self.assertIsNotNone(specification.loader)
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        with self.assertRaisesRegex(RuntimeError, "official interpreter"):
            module._validate_interpreter_antigravity(executable="/tmp/not-the-official-python-antigravity")
        expected = Path(module.OFFICIAL_PYTHON_ANTIGRAVITY)
        wrong = Path(__import__("sys")._base_executable)
        self.assertNotEqual(wrong.absolute(), expected.absolute())
        failed = subprocess.run(
            [str(wrong), str(wrapper), "inspect"], text=True,
            capture_output=True, check=False,
        )
        self.assertNotEqual(failed.returncode, 0)
        self.assertNotIn("Traceback", failed.stderr)
        failure_payload = json.loads(failed.stderr)
        self.assertEqual(failure_payload["schema"], "exp443.wrapper-status.v1")
        self.assertEqual(failure_payload["command"], "inspect")
        self.assertEqual(failure_payload["status"], "error")
        with tempfile.TemporaryDirectory(prefix="exp443_cli_cwd_antigravity_") as temporary:
            completed = subprocess.run(
                [str(expected), str(wrapper), "--help"], cwd=temporary,
                text=True, capture_output=True, check=False,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("readiness", completed.stdout)
        self.assertIn("evaluate", completed.stdout)


if __name__ == "__main__":
    unittest.main()
