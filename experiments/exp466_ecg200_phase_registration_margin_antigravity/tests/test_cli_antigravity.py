from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


_COMMIT_ANTIGRAVITY = "a" * 40
_REPOSITORY_ANTIGRAVITY = Path(__file__).resolve().parents[3]
_LAUNCHER_ANTIGRAVITY = _REPOSITORY_ANTIGRAVITY / (
    "scratch/run_exp466_ecg200_phase_registration_margin_antigravity.py"
)
_RUN1_ROOT_ANTIGRAVITY = Path(
    "/private/tmp/exp466_task6_private_run1_antigravity",
)
_RUN2_ROOT_ANTIGRAVITY = Path(
    "/private/tmp/exp466_task6_private_run2_antigravity",
)
_LEAVES_ANTIGRAVITY = (
    "artifact_sha256_antigravity.json",
    "context_antigravity.json",
    "dashboard_snapshot_antigravity.json",
    "decision_antigravity.json",
    "failed_folds_antigravity.jsonl",
    "method_fold_metrics_antigravity.jsonl",
    "method_summary_antigravity.jsonl",
)


@dataclass(frozen=True, slots=True)
class _DecisionAntigravity:
    status: str = "qualification_failed_blocked_before_fn_validation"
    primary_candidate: str = "C9"
    eligible_for_exp467_design: bool = False
    primary_passed_count: int = 9
    primary_required_count: int = 10
    failure_reasons: tuple[str, ...] = ("c9_not_10_of_10",)
    routing_applied: bool = False
    champion_modified: bool = False
    pending_modified: bool = False
    promotion_applied: bool = False
    test_labels_opened: bool = False
    label_query_count: int = 0


def _private_module_antigravity(root: object) -> types.ModuleType:
    private = types.ModuleType("_exp466_cli_private_test_antigravity")
    private.CAPABILITY_ANTIGRAVITY = "exp466_task6_private_capability_v1"
    private.root_antigravity = root
    return private


def inspected_antigravity(*, label_query_count: int = 0) -> types.SimpleNamespace:
    qualification = types.SimpleNamespace(
        normal_trial_counts={20260721: 69, 20261721: 69},
        normal_confirmations={20260721: 0, 20261721: 0},
        label_query_count=label_query_count,
        official_evaluate_count=0,
        official_inspect_count=0,
    )
    return types.SimpleNamespace(
        qualification=qualification,
        decision=_DecisionAntigravity(label_query_count=label_query_count),
        dashboard={
            "status": "qualification_failed_blocked_before_fn_validation",
            "html_antigravity": "<section>검증 결과</section>",
        },
        leaf_sha256=tuple(
            (name, f"{index + 1:064x}")
            for index, name in enumerate(_LEAVES_ANTIGRAVITY)
        ),
        artifact_map_sha256="f" * 64,
    )


class CliAntigravityTest(unittest.TestCase):
    def test_exact_surface_accepts_only_reviewed_commands_antigravity(self) -> None:
        """Break caught: an override, alias, or tuning control reaches the CLI."""
        from experiments.exp466_ecg200_phase_registration_margin_antigravity.cli_antigravity import (
            parser_antigravity,
        )

        parser = parser_antigravity()
        parsed = parser.parse_args(
            ["evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY],
        )
        self.assertEqual(
            (parsed.command, parsed.reviewed_commit),
            ("evaluate", _COMMIT_ANTIGRAVITY),
        )
        self.assertEqual(parser.parse_args(["inspect"]).command, "inspect")
        rejected = (
            (),
            ("--help",),
            ("eval", "--reviewed-commit", _COMMIT_ANTIGRAVITY),
            ("evaluate",),
            ("evaluate", "--root", "/tmp/x"),
            ("evaluate", "--reviewed-commit", "A" * 40),
            ("evaluate", "--reviewed-commit", "a" * 39),
            ("evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY, "extra"),
            ("evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY, "--seed", "1"),
            ("evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY, "--warp", "1.1"),
            ("evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY, "--retry"),
            ("inspect", "--root", "/tmp/x"),
            ("inspect", "--retry"),
        )
        for argv in rejected:
            with (
                self.subTest(argv=argv),
                redirect_stderr(io.StringIO()),
                redirect_stdout(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                parser.parse_args(argv)

    def test_root_free_parser_and_inspect_failure_are_lazy_antigravity(self) -> None:
        """Break caught: parsing or missing capability imports data/scoring code."""
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "\n".join(
                    (
                        "import sys",
                        "package='experiments.exp466_ecg200_phase_registration_margin_antigravity'",
                        "mods=tuple(f'{package}.{name}' for name in (",
                        " 'sources_antigravity','interpolation_antigravity',",
                        " 'scoring_antigravity','qualification_antigravity',",
                        " 'evaluation_antigravity','artifacts_antigravity','replay_antigravity'))",
                        "from experiments.exp466_ecg200_phase_registration_margin_antigravity import cli_antigravity as cli",
                        "assert not any(item in sys.modules for item in mods)",
                        "cli.parser_antigravity().parse_args(['inspect'])",
                        "assert not any(item in sys.modules for item in mods)",
                        "try: cli.main_antigravity(['inspect'])",
                        "except ValueError as error: assert 'capability' in str(error)",
                        "else: raise AssertionError('missing capability was accepted')",
                        "assert not any(item in sys.modules for item in mods)",
                        "assert not any(item == 'aeon' or item.startswith('aeon.') for item in sys.modules)",
                    ),
                ),
            ],
            cwd=_REPOSITORY_ANTIGRAVITY,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_private_capability_accepts_only_two_fixed_roots_antigravity(self) -> None:
        """Break caught: production CLI accepts an arbitrary result root."""
        from experiments.exp466_ecg200_phase_registration_margin_antigravity import (
            cli_antigravity as cli,
        )

        with self.assertRaisesRegex(ValueError, "capability is required"):
            cli._private_test_root_antigravity()
        for forbidden in (None, "/private/tmp/x", Path("/private/tmp/x")):
            private = _private_module_antigravity(forbidden)
            with (
                self.subTest(forbidden=repr(forbidden)),
                patch.dict(sys.modules, {private.__name__: private}),
                self.assertRaisesRegex(ValueError, "must select a fixed Task 6 private root"),
            ):
                cli._private_test_root_antigravity()
        extra = _private_module_antigravity(_RUN1_ROOT_ANTIGRAVITY)
        extra.tuning_antigravity = 1
        with (
            patch.dict(sys.modules, {extra.__name__: extra}),
            self.assertRaisesRegex(ValueError, "may inject only root_antigravity"),
        ):
            cli._private_test_root_antigravity()
        for allowed in (_RUN1_ROOT_ANTIGRAVITY, _RUN2_ROOT_ANTIGRAVITY):
            private = _private_module_antigravity(allowed)
            with patch.dict(sys.modules, {private.__name__: private}):
                self.assertIs(cli._private_test_root_antigravity(), allowed)

    def test_launcher_is_cwd_independent_and_rejects_arbitrary_capability_antigravity(self) -> None:
        """Break caught: launcher depends on CWD or accepts an arbitrary root."""
        with tempfile.TemporaryDirectory(prefix="exp466_launcher_antigravity_") as temporary:
            missing = subprocess.run(
                [sys.executable, str(_LAUNCHER_ANTIGRAVITY), "inspect"],
                cwd=temporary,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(missing.returncode, 0)
            self.assertIn("fixed Task 6 private root capability is required", missing.stderr)

            environment = dict(os.environ)
            environment["EXP466_TASK6_PRIVATE_CAPABILITY_ANTIGRAVITY"] = "/tmp/x"
            arbitrary = subprocess.run(
                [sys.executable, str(_LAUNCHER_ANTIGRAVITY), "inspect"],
                cwd=temporary,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertIn("private capability is not preregistered", arbitrary.stderr)

    def test_slots_decision_projects_to_canonical_label_free_payload_antigravity(self) -> None:
        """Break caught: slots decision cannot be serialized after publication."""
        from experiments.exp466_ecg200_phase_registration_margin_antigravity import (
            cli_antigravity as cli,
        )

        payload = cli._result_payload_from_inspected_antigravity(
            inspected_antigravity(),
        )
        self.assertEqual(payload["experiment"], "Exp466")
        self.assertEqual(payload["decision"]["label_query_count"], 0)
        self.assertIs(payload["decision"]["test_labels_opened"], False)
        self.assertEqual(len(payload["leaf_sha256"]), 7)
        with self.assertRaisesRegex(ValueError, "authority boundary|opened labels"):
            cli._result_payload_from_inspected_antigravity(
                inspected_antigravity(label_query_count=1),
            )

    def test_evaluate_and_inspect_emit_identical_canonical_stdout_antigravity(self) -> None:
        """Break caught: command branches emit different result projections."""
        from experiments.exp466_ecg200_phase_registration_margin_antigravity import (
            cli_antigravity as cli,
        )

        payload = cli._result_payload_from_inspected_antigravity(inspected_antigravity())
        with tempfile.TemporaryDirectory(prefix="exp466_cli_stdout_antigravity_") as temporary:
            root = Path(temporary) / "unused_result"
            outputs: list[str] = []
            for command in (
                ["evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY],
                ["inspect"],
            ):
                output = io.StringIO()
                with (
                    patch.object(cli, "_private_test_root_antigravity", return_value=root),
                    patch.object(cli, "_validate_clean_current_commit_antigravity"),
                    patch.object(cli, "_load_evaluate_antigravity", return_value=lambda _root: payload),
                    patch.object(cli, "_load_inspect_antigravity", return_value=lambda _root: payload),
                    redirect_stdout(output),
                ):
                    self.assertEqual(cli.main_antigravity(command), 0)
                outputs.append(output.getvalue())
        expected = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ) + "\n"
        self.assertEqual(outputs, [expected, expected])


if __name__ == "__main__":
    unittest.main()
