"""Closed CLI and private candidate-flow tests for Exp458 Task 5."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.tests.test_artifacts_antigravity import (
    publication_evidence_antigravity,
)
from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.tests.test_evaluation_antigravity import (
    DATASET_ANTIGRAVITY,
    exp456_authority_antigravity,
)
from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.tests.test_router_antigravity import (
    qualification_antigravity,
)


_COMMIT_ANTIGRAVITY = "a" * 40
_REPOSITORY_ANTIGRAVITY = Path(__file__).resolve().parents[3]
_LAUNCHER_ANTIGRAVITY = (
    _REPOSITORY_ANTIGRAVITY
    / "scratch/run_exp458_ecg_normal2_rank_confidence_routing_antigravity.py"
)
_OFFICIAL_ROOT_ANTIGRAVITY = Path(
    "/Users/mac/Documents/Dataset/exp458_ecg_normal2_rank_confidence_routing_antigravity_20260818"
)


class CliAntigravityTest(unittest.TestCase):
    def test_exact_surface_accepts_only_reviewed_evaluate_or_bare_inspect_antigravity(self) -> None:
        """Break caught: an alias, override, retry, abbreviation, or extra token reaches execution."""
        from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.cli_antigravity import (
            parser_antigravity,
        )

        parser = parser_antigravity()
        self.assertEqual(
            parser.parse_args(
                ["evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY]
            ).command,
            "evaluate",
        )
        self.assertEqual(
            parser.parse_args(
                ["evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY]
            ).reviewed_commit,
            _COMMIT_ANTIGRAVITY,
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
        """Break caught: inspect imports qualification/evaluation or evaluate imports replay."""
        from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity import (
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
                "_validate_clean_current_commit_antigravity",
                return_value=None,
            ) as validate,
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
            validate.assert_called_once_with(_COMMIT_ANTIGRAVITY)
            evaluate.assert_called_once_with()

    def test_parser_and_help_paths_do_not_load_candidate_dependencies_antigravity(self) -> None:
        """Break caught: parsing or rejected help imports TRAIN qualification or Exp456 evaluation."""
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys\n"
                    "from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity import cli_antigravity as cli\n"
                    "mods=('experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.qualification_antigravity',"
                    "'experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.evaluation_antigravity',"
                    "'experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.replay_antigravity')\n"
                    "assert not any(m in sys.modules for m in mods)\n"
                    "cli.parser_antigravity().parse_args(['inspect'])\n"
                    "assert not any(m in sys.modules for m in mods)\n"
                    "try:\n    cli.parser_antigravity().parse_args(['--help'])\nexcept SystemExit:\n    pass\n"
                    "assert not any(m in sys.modules for m in mods)"
                ),
            ],
            cwd=_REPOSITORY_ANTIGRAVITY,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_candidate_flow_qualifies_routes_evaluates_publishes_and_replays_once_antigravity(self) -> None:
        """Break caught: candidate flow skips a required stage, skips replay, or writes the official root."""
        from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity import (
            cli_antigravity as cli,
        )
        from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.artifacts_antigravity import (
            publish_result_antigravity,
        )
        from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.evaluation_antigravity import (
            evaluate_routes_antigravity,
        )
        from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.replay_antigravity import (
            inspect_result_at_antigravity,
        )
        from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.router_antigravity import (
            route_normal2_antigravity,
        )

        authority = exp456_authority_antigravity()
        feature = {item.dataset: item for item in authority.features}[DATASET_ANTIGRAVITY]
        qualification = qualification_antigravity(True)
        calls: list[str] = []
        with tempfile.TemporaryDirectory(prefix="exp458_cli_candidate_antigravity_") as temporary:
            root = Path(temporary) / "result"
            self.assertNotEqual(root, _OFFICIAL_ROOT_ANTIGRAVITY)
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(
                    cli._evaluate_with_dependencies_antigravity(
                        _COMMIT_ANTIGRAVITY,
                        root,
                        lambda: (calls.append("qualify"), qualification)[1],
                        lambda: (calls.append("authority"), authority)[1],
                        lambda received_feature, received_qualification: (
                            calls.append("route"),
                            self.assertEqual(received_feature, feature),
                            route_normal2_antigravity(
                                received_feature, received_qualification
                            ),
                        )[2],
                        lambda captured, decision: (
                            calls.append("evaluate"),
                            self.assertIs(captured, authority),
                            evaluate_routes_antigravity(captured, decision),
                        )[2],
                        lambda destination, evidence: (
                            calls.append("publish"),
                            publish_result_antigravity(destination, evidence),
                        )[1],
                        lambda destination: (
                            calls.append("inspect"),
                            inspect_result_at_antigravity(destination),
                        )[1],
                    ),
                    0,
                )
            self.assertEqual(
                calls,
                ["qualify", "authority", "route", "evaluate", "publish", "inspect"],
            )
            payload = json.loads(output.getvalue())
            self.assertEqual(payload["experiment_id"], "Exp458")
            self.assertEqual(payload["normal2_route"]["route"], "candidate")
            final = payload["methods"][-1]
            self.assertEqual(
                (final["tp"], final["fp"], final["fn"], final["tn"], final["f1"]),
                (617, 66, 829, 70788, 0.5796148426491311),
            )
            self.assertTrue(root.is_dir())
            self.assertEqual(len({entry.name for entry in root.iterdir()}), 9)
            self.assertNotEqual(root.resolve(), _OFFICIAL_ROOT_ANTIGRAVITY.resolve())

    def test_evaluate_fails_closed_on_runtime_mismatch_before_qualifier_or_root_antigravity(self) -> None:
        """Break caught: public evaluate can create a root or qualify under the wrong runtime."""
        from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity import (
            cli_antigravity as cli,
        )

        runtime_fields = ("executable", "prefix", "python_version", "numpy_version", "aeon_version")
        for field in runtime_fields:
            with self.subTest(field=field), tempfile.TemporaryDirectory(prefix="exp458_runtime_reject_antigravity_") as temporary:
                root = Path(temporary) / "result"
                with (
                    patch.object(
                        cli,
                        "observe_runtime_provenance_antigravity",
                        side_effect=ValueError(f"{field} differs from frozen runtime"),
                        create=True,
                    ) as runtime,
                    self.assertRaisesRegex(ValueError, field),
                ):
                    cli._evaluate_with_dependencies_antigravity(
                        _COMMIT_ANTIGRAVITY,
                        root,
                        lambda: self.fail("qualifier reached after runtime mismatch"),
                        lambda: self.fail("authority reached after runtime mismatch"),
                        lambda _feature, _qualification: self.fail("router reached after runtime mismatch"),
                        lambda _authority, _decision: self.fail("evaluator reached after runtime mismatch"),
                        lambda _root, _evidence: self.fail("publisher reached after runtime mismatch"),
                        lambda _root: self.fail("inspector reached after runtime mismatch"),
                    )
                runtime.assert_called_once_with()
                self.assertFalse(root.exists())

    def test_candidate_flow_routes_baseline_when_qualification_fails_antigravity(self) -> None:
        """Break caught: failed TRAIN qualification is published as a passing candidate."""
        from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity import (
            cli_antigravity as cli,
        )
        from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.artifacts_antigravity import (
            publish_result_antigravity,
        )
        from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.evaluation_antigravity import (
            evaluate_routes_antigravity,
        )
        from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.router_antigravity import (
            route_normal2_antigravity,
        )

        authority = exp456_authority_antigravity()
        failing = qualification_antigravity(False)
        with tempfile.TemporaryDirectory(prefix="exp458_cli_baseline_antigravity_") as temporary:
            root = Path(temporary) / "result"
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(
                    cli._evaluate_with_dependencies_antigravity(
                        _COMMIT_ANTIGRAVITY,
                        root,
                        lambda: failing,
                        lambda: authority,
                        route_normal2_antigravity,
                        evaluate_routes_antigravity,
                        publish_result_antigravity,
                        lambda destination: evaluate_routes_antigravity(
                            authority,
                            route_normal2_antigravity(
                                {item.dataset: item for item in authority.features}[
                                    DATASET_ANTIGRAVITY
                                ],
                                failing,
                            ),
                        ),
                    ),
                    0,
                )
            payload = json.loads(output.getvalue())
            self.assertEqual(payload["normal2_route"]["route"], "baseline")
            self.assertEqual(payload["methods"][-1]["fn"], 836)
            self.assertFalse(payload["passed"])

    def test_evaluate_rejects_wrong_authority_or_replay_mismatch_antigravity(self) -> None:
        """Break caught: label-query authority or an unreplayable publication is accepted."""
        from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity import (
            cli_antigravity as cli,
        )
        from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.evaluation_antigravity import (
            evaluate_routes_antigravity,
        )
        from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.router_antigravity import (
            route_normal2_antigravity,
        )

        authority = exp456_authority_antigravity()
        label_mutant = replace(authority)
        object.__setattr__(label_mutant, "label_query_count", 1)
        with tempfile.TemporaryDirectory(prefix="exp458_cli_reject_antigravity_") as temporary:
            with self.assertRaises(ValueError):
                cli._evaluate_with_dependencies_antigravity(
                    _COMMIT_ANTIGRAVITY,
                    Path(temporary) / "result",
                    lambda: qualification_antigravity(True),
                    lambda: label_mutant,
                    route_normal2_antigravity,
                    evaluate_routes_antigravity,
                    lambda _root, _evidence: self.fail("publish reached"),
                    lambda _root: self.fail("inspect reached"),
                )
        with tempfile.TemporaryDirectory(prefix="exp458_cli_replay_reject_antigravity_") as temporary:
            with self.assertRaises(ValueError):
                cli._evaluate_with_dependencies_antigravity(
                    _COMMIT_ANTIGRAVITY,
                    Path(temporary) / "result",
                    lambda: qualification_antigravity(True),
                    lambda: authority,
                    route_normal2_antigravity,
                    evaluate_routes_antigravity,
                    lambda _root, _evidence: publication_evidence_antigravity(),
                    lambda _root: object(),
                )

    def test_inspect_uses_public_replay_and_emits_only_dashboard_payload_antigravity(self) -> None:
        """Break caught: inspect bypasses replay or emits raw TRAIN/TEST evidence."""
        from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity import (
            cli_antigravity as cli,
        )
        from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.dashboard_export_antigravity import (
            export_dashboard_antigravity,
        )
        from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.evaluation_antigravity import (
            evaluate_routes_antigravity,
        )
        from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.router_antigravity import (
            route_normal2_antigravity,
        )

        authority = exp456_authority_antigravity()
        feature = {item.dataset: item for item in authority.features}[DATASET_ANTIGRAVITY]
        result = evaluate_routes_antigravity(
            authority,
            route_normal2_antigravity(feature, qualification_antigravity(True)),
        )
        with tempfile.TemporaryDirectory(prefix="exp458_cli_inspect_antigravity_") as temporary:
            root = Path(temporary) / "result"
            called: list[Path] = []
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(
                    cli._inspect_with_dependencies_antigravity(
                        root,
                        lambda destination: (called.append(destination), result)[1],
                    ),
                    0,
                )
            self.assertEqual(called, [root])
            payload = json.loads(output.getvalue())
            self.assertEqual(payload["experiment_id"], "Exp458")
            self.assertNotIn("values", output.getvalue())
            self.assertNotIn("labels", output.getvalue())
            self.assertNotIn("pvalues", output.getvalue())
            self.assertEqual(payload, export_dashboard_antigravity(result))

    def test_launcher_imports_package_from_outside_repository_cwd_antigravity(self) -> None:
        """Break caught: launcher relies on the caller's working directory."""
        self.assertTrue(_LAUNCHER_ANTIGRAVITY.is_file())
        with tempfile.TemporaryDirectory(prefix="exp458_launcher_cwd_antigravity_") as temporary:
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

    def test_launcher_private_synthetic_evaluate_then_inspect_from_outside_cwd_antigravity(self) -> None:
        """Break caught: launcher success path is untested outside CWD or requires a public override."""
        self.assertTrue(_LAUNCHER_ANTIGRAVITY.is_file())
        with tempfile.TemporaryDirectory(prefix="exp458_launcher_success_antigravity_") as temporary:
            root = Path(temporary) / "synthetic_result_antigravity"
            script = r'''
from __future__ import annotations

from contextlib import redirect_stdout
from pathlib import Path
import io
import json
import runpy
import sys
import types

launcher = Path(sys.argv[1])
root = Path(sys.argv[2])
private = types.ModuleType("_exp458_cli_private_test_antigravity")
private.CAPABILITY_ANTIGRAVITY = "exp458_cli_private_test_antigravity_v1"
private.root_antigravity = root

def validate_clean_current_commit_antigravity(reviewed_commit):
    if reviewed_commit != "b" * 40:
        raise AssertionError("unexpected reviewed commit")

def evaluate_dependencies_antigravity():
    from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.artifacts_antigravity import publish_result_antigravity
    from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.evaluation_antigravity import evaluate_routes_antigravity
    from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.replay_antigravity import inspect_result_at_antigravity
    from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.router_antigravity import route_normal2_antigravity
    from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.tests.test_evaluation_antigravity import exp456_authority_antigravity
    from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.tests.test_router_antigravity import qualification_antigravity

    return (
        lambda: qualification_antigravity(True),
        exp456_authority_antigravity,
        route_normal2_antigravity,
        evaluate_routes_antigravity,
        publish_result_antigravity,
        inspect_result_at_antigravity,
    )

private.validate_clean_current_commit_antigravity = validate_clean_current_commit_antigravity
private.evaluate_dependencies_antigravity = evaluate_dependencies_antigravity
sys.modules[private.__name__] = private

def invoke(argv):
    output = io.StringIO()
    old_argv = sys.argv
    try:
        sys.argv = [str(launcher), *argv]
        with redirect_stdout(output):
            try:
                runpy.run_path(str(launcher), run_name="__main__")
            except SystemExit as exited:
                code = 0 if exited.code is None else exited.code
            else:
                code = 0
    finally:
        sys.argv = old_argv
    return code, json.loads(output.getvalue())

evaluate_code, evaluate_payload = invoke(["evaluate", "--reviewed-commit", "b" * 40])
inspect_code, inspect_payload = invoke(["inspect"])
print(json.dumps({
    "evaluate_code": evaluate_code,
    "inspect_code": inspect_code,
    "payloads_equal": evaluate_payload == inspect_payload,
    "root_exists": root.is_dir(),
    "leaf_count": len(tuple(root.iterdir())),
    "route": evaluate_payload["normal2_route"]["route"],
}, sort_keys=True))
'''
            completed = subprocess.run(
                [sys.executable, "-c", script, str(_LAUNCHER_ANTIGRAVITY), str(root)],
                cwd=temporary,
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        proof = json.loads(completed.stdout)
        self.assertEqual(proof["evaluate_code"], 0)
        self.assertEqual(proof["inspect_code"], 0)
        self.assertTrue(proof["payloads_equal"])
        self.assertTrue(proof["root_exists"])
        self.assertEqual(proof["leaf_count"], 9)
        self.assertEqual(proof["route"], "candidate")


if __name__ == "__main__":
    unittest.main()
