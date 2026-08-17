"""Closed CLI and pre-official candidate-flow tests for Exp457 Task 5."""

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

from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.tests.test_evaluation_antigravity import (
    decisions_antigravity,
    exp456_authority_antigravity,
)
from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.tests.test_router_antigravity import (
    qualification_antigravity,
)


_COMMIT_ANTIGRAVITY = "a" * 40
_REPOSITORY_ANTIGRAVITY = Path(__file__).resolve().parents[3]
_LAUNCHER_ANTIGRAVITY = (
    _REPOSITORY_ANTIGRAVITY
    / "scratch/run_exp457_ecg_train_synthetic_confidence_routing_antigravity.py"
)
_OFFICIAL_ROOT_ANTIGRAVITY = Path(
    "/Users/mac/Documents/Dataset/exp457_ecg_train_synthetic_confidence_routing_antigravity_20260818"
)


def _qualifications_antigravity():
    datasets = tuple(f"ECG5000_normal_{index}" for index in range(1, 6))
    return tuple(
        qualification_antigravity(dataset, dataset == "ECG5000_normal_2")
        for dataset in datasets
    )


class CliAntigravityTest(unittest.TestCase):
    def test_exact_surface_accepts_only_reviewed_evaluate_or_bare_inspect_antigravity(self) -> None:
        """Break caught: an alias, override, or abbreviated option reaches execution."""
        from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.cli_antigravity import (
            parser_antigravity,
        )

        parser = parser_antigravity()
        self.assertEqual(
            parser.parse_args(
                ["evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY]
            ).command,
            "evaluate",
        )
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
            (
                "evaluate",
                "--reviewed-commit",
                _COMMIT_ANTIGRAVITY,
                "--reviewed-commit",
                "b" * 40,
            ),
            ("evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY, "--root", "/tmp/x"),
            ("evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY, "--override"),
            ("inspect", "extra"),
            ("inspect", "--reviewed-commit", _COMMIT_ANTIGRAVITY),
            ("readiness",),
            ("new",),
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

    def test_dispatch_loads_only_the_selected_phase_antigravity(self) -> None:
        """Break caught: inspect imports qualification or evaluate imports replay."""
        from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity import (
            cli_antigravity as cli,
        )

        with (
            patch.object(
                cli,
                "_load_evaluate_antigravity",
                side_effect=AssertionError("evaluate phase loaded"),
            ),
            patch.object(cli, "_load_inspect_antigravity", return_value=lambda: 0) as inspect,
        ):
            self.assertEqual(cli.main_antigravity(["inspect"]), 0)
            inspect.assert_called_once_with()
        with (
            patch.object(
                cli,
                "_load_evaluate_antigravity",
                return_value=lambda commit: 0,
            ) as evaluate,
            patch.object(
                cli,
                "_load_inspect_antigravity",
                side_effect=AssertionError("inspect phase loaded"),
            ),
        ):
            self.assertEqual(
                cli.main_antigravity(
                    ["evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY]
                ),
                0,
            )
            evaluate.assert_called_once_with()

    def test_candidate_flow_qualifies_routes_evaluates_and_publishes_once_antigravity(self) -> None:
        """Break caught: candidate flow skips Gate 1, changes stage order, or writes the official root."""
        from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity import (
            cli_antigravity as cli,
        )
        from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.artifacts_antigravity import (
            publish_result_antigravity,
        )
        from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.dashboard_export_antigravity import (
            export_dashboard_antigravity,
        )
        from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.evaluation_antigravity import (
            evaluate_routes_antigravity,
        )
        from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.router_antigravity import (
            route_ecg_antigravity,
        )

        authority = exp456_authority_antigravity()
        qualifications = _qualifications_antigravity()
        calls: list[str] = []
        with tempfile.TemporaryDirectory(prefix="exp457_cli_candidate_antigravity_") as temporary:
            root = Path(temporary) / "result"
            self.assertNotEqual(root, _OFFICIAL_ROOT_ANTIGRAVITY)
            result, published = cli._evaluate_with_dependencies_antigravity(
                _COMMIT_ANTIGRAVITY,
                lambda commit: (
                    calls.append("validate"),
                    self.assertEqual(commit, _COMMIT_ANTIGRAVITY),
                ),
                lambda: (calls.append("qualify"), qualifications)[1],
                lambda: (calls.append("authority"), authority)[1],
                lambda feature, qualification: (
                    calls.append(f"route:{qualification.dataset}"),
                    route_ecg_antigravity(feature, qualification),
                )[1],
                lambda captured, routes: (
                    calls.append("evaluate"),
                    self.assertIs(captured, authority),
                    evaluate_routes_antigravity(captured, routes),
                )[2],
                lambda destination, evidence: (
                    calls.append("publish"),
                    publish_result_antigravity(destination, evidence),
                )[1],
                root,
            )
            self.assertEqual(
                calls,
                [
                    "validate",
                    "qualify",
                    "authority",
                    *(f"route:ECG5000_normal_{index}" for index in range(1, 6)),
                    "evaluate",
                    "publish",
                ],
            )
            self.assertEqual(len(published.leaf_sha256), 9)
            self.assertEqual(authority.label_query_count, 0)
            self.assertEqual(
                tuple((row.dataset, row.route) for row in result.decisions),
                (
                    ("ECG5000_normal_1", "baseline"),
                    ("ECG5000_normal_2", "candidate"),
                    ("ECG5000_normal_3", "baseline"),
                    ("ECG5000_normal_4", "baseline"),
                    ("ECG5000_normal_5", "baseline"),
                ),
            )
            output = io.StringIO()
            with redirect_stdout(output):
                cli.emit_dashboard_json_antigravity(result, export_dashboard_antigravity)
            self.assertEqual(
                output.getvalue(),
                json.dumps(
                    export_dashboard_antigravity(result),
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n",
            )

    def test_candidate_flow_rejects_wrong_qualification_order_or_label_query_antigravity(self) -> None:
        """Break caught: partial Gate 1 evidence or TEST-label authority controls a result."""
        from copy import copy

        from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity import (
            cli_antigravity as cli,
        )

        authority = exp456_authority_antigravity()
        label_mutant = copy(authority)
        object.__setattr__(label_mutant, "label_query_count", 1)
        qualifications = _qualifications_antigravity()
        cases = (
            (qualifications[:-1], authority),
            (qualifications[::-1], authority),
            (qualifications, label_mutant),
        )
        for candidate_qualifications, candidate_authority in cases:
            with self.subTest(
                qualification_count=len(candidate_qualifications),
                labels=candidate_authority.label_query_count,
            ), tempfile.TemporaryDirectory(prefix="exp457_cli_reject_antigravity_") as temporary:
                with self.assertRaises(ValueError):
                    cli._evaluate_with_dependencies_antigravity(
                        _COMMIT_ANTIGRAVITY,
                        lambda _commit: None,
                        lambda: candidate_qualifications,
                        lambda: candidate_authority,
                        lambda _feature, _qualification: self.fail("route reached"),
                        lambda _authority, _routes: self.fail("evaluate reached"),
                        lambda _root, _evidence: self.fail("publish reached"),
                        Path(temporary) / "result",
                    )

    def test_inspect_uses_public_replay_and_emits_only_dashboard_payload_antigravity(self) -> None:
        """Break caught: inspect bypasses replay or emits raw TRAIN/TEST evidence."""
        from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity import (
            cli_antigravity as cli,
        )
        from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.dashboard_export_antigravity import (
            export_dashboard_antigravity,
        )
        from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.evaluation_antigravity import (
            evaluate_routes_antigravity,
        )

        authority = exp456_authority_antigravity()
        result = evaluate_routes_antigravity(authority, decisions_antigravity(authority))
        with tempfile.TemporaryDirectory(prefix="exp457_cli_inspect_antigravity_") as temporary:
            root = Path(temporary) / "result"
            called: list[Path] = []
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(
                    cli._inspect_with_dependencies_antigravity(
                        root,
                        lambda destination: (called.append(destination), result)[1],
                        export_dashboard_antigravity,
                    ),
                    0,
                )
            self.assertEqual(called, [root])
            payload = json.loads(output.getvalue())
            self.assertEqual(payload["experiment_id"], "Exp457")
            self.assertNotIn("values", output.getvalue())
            self.assertNotIn("labels", output.getvalue())
            self.assertNotIn("pvalues", output.getvalue())

    def test_launcher_imports_package_from_outside_repository_cwd_antigravity(self) -> None:
        """Break caught: launcher relies on the caller's working directory."""
        self.assertTrue(_LAUNCHER_ANTIGRAVITY.is_file())
        with tempfile.TemporaryDirectory(prefix="exp457_launcher_cwd_antigravity_") as temporary:
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


if __name__ == "__main__":
    unittest.main()
