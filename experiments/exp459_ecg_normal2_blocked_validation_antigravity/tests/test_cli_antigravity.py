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

from experiments.exp459_ecg_normal2_blocked_validation_antigravity.tests.test_artifacts_antigravity import (
    authority_antigravity,
    decision_antigravity,
)
from experiments.exp459_ecg_normal2_blocked_validation_antigravity.tests.test_evaluation_antigravity import (
    qualification_antigravity,
)


_COMMIT_ANTIGRAVITY = "a" * 40
_REPOSITORY_ANTIGRAVITY = Path(__file__).resolve().parents[3]
_LAUNCHER_ANTIGRAVITY = (
    _REPOSITORY_ANTIGRAVITY
    / "scratch/run_exp459_ecg_normal2_blocked_validation_antigravity.py"
)
_OFFICIAL_ROOT_ANTIGRAVITY = Path(
    "/Users/mac/Documents/Dataset/exp459_ecg_normal2_blocked_validation_antigravity_20260818"
)


class CliAntigravityTest(unittest.TestCase):
    def test_exact_surface_accepts_only_reviewed_evaluate_or_bare_inspect_antigravity(self) -> None:
        """Break caught: aliases, overrides, abbreviations, retries, or extra tokens reach execution."""
        from experiments.exp459_ecg_normal2_blocked_validation_antigravity.cli_antigravity import (
            parser_antigravity,
        )

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
            ("evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY, "--root", "/tmp/x"),
            ("evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY, "--threshold", "0"),
            ("evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY, "--seed", "1"),
            ("evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY, "--resume"),
            ("inspect", "extra"),
            ("inspect", "--reviewed-commit", _COMMIT_ANTIGRAVITY),
            ("readiness",),
            ("new",),
            ("retry",),
            ("resume",),
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
        """Break caught: inspect loads evaluate code, or evaluate loads inspect code before dispatch."""
        from experiments.exp459_ecg_normal2_blocked_validation_antigravity import (
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
            patch.object(cli, "_load_evaluate_antigravity", return_value=lambda commit: 0) as evaluate,
            patch.object(
                cli,
                "_load_inspect_antigravity",
                side_effect=AssertionError("inspect phase loaded"),
            ),
        ):
            self.assertEqual(
                cli.main_antigravity(["evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY]),
                0,
            )
            validate.assert_called_once_with(_COMMIT_ANTIGRAVITY)
            evaluate.assert_called_once_with()

    def test_parser_and_rejected_paths_do_not_load_hydra_source_or_replay_antigravity(self) -> None:
        """Break caught: parsing alone imports TRAIN source, Hydra scoring, evaluation, or replay."""
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys\n"
                    "from experiments.exp459_ecg_normal2_blocked_validation_antigravity import cli_antigravity as cli\n"
                    "mods=(\n"
                    " 'experiments.exp459_ecg_normal2_blocked_validation_antigravity.sources_antigravity',\n"
                    " 'experiments.exp459_ecg_normal2_blocked_validation_antigravity.qualification_antigravity',\n"
                    " 'experiments.exp459_ecg_normal2_blocked_validation_antigravity.evaluation_antigravity',\n"
                    " 'experiments.exp459_ecg_normal2_blocked_validation_antigravity.replay_antigravity',\n"
                    ")\n"
                    "assert not any(m in sys.modules for m in mods)\n"
                    "cli.parser_antigravity().parse_args(['inspect'])\n"
                    "assert not any(m in sys.modules for m in mods)\n"
                    "try:\n"
                    "    cli.parser_antigravity().parse_args(['inspect', 'extra'])\n"
                    "except SystemExit:\n"
                    "    pass\n"
                    "assert not any(m in sys.modules for m in mods)\n"
                    "assert not any(m == 'aeon' or m.startswith('aeon.') for m in sys.modules)\n"
                ),
            ],
            cwd=_REPOSITORY_ANTIGRAVITY,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_synthetic_evaluate_qualifies_evaluates_publishes_replays_with_authority_once_antigravity(
        self,
    ) -> None:
        """Break caught: evaluate skips authority/replay, trusts qualification drift, or writes official root."""
        from experiments.exp459_ecg_normal2_blocked_validation_antigravity import (
            cli_antigravity as cli,
        )
        from experiments.exp459_ecg_normal2_blocked_validation_antigravity.artifacts_antigravity import (
            publish_result_antigravity,
        )
        from experiments.exp459_ecg_normal2_blocked_validation_antigravity.evaluation_antigravity import (
            evaluate_blocked_qualification_antigravity,
        )
        from experiments.exp459_ecg_normal2_blocked_validation_antigravity.replay_antigravity import (
            inspect_replay_at_antigravity,
        )

        calls: list[str] = []
        authority = authority_antigravity()
        with tempfile.TemporaryDirectory(prefix="exp459_cli_synthetic_antigravity_") as temporary:
            root = Path(temporary) / "result_antigravity"
            self.assertNotEqual(root, _OFFICIAL_ROOT_ANTIGRAVITY)
            output = io.StringIO()
            with (
                patch.object(cli, "observe_runtime_provenance_antigravity", return_value=object()),
                redirect_stdout(output),
            ):
                self.assertEqual(
                    cli._evaluate_with_dependencies_antigravity(
                        _COMMIT_ANTIGRAVITY,
                        root,
                        lambda: (calls.append("authority"), authority)[1],
                        lambda: (calls.append("qualify"), qualification_antigravity())[1],
                        lambda qualification: (
                            calls.append("evaluate"),
                            evaluate_blocked_qualification_antigravity(qualification),
                        )[1],
                        lambda destination, decision, trusted: (
                            calls.append("publish"),
                            publish_result_antigravity(destination, decision, trusted),
                        )[1],
                        lambda destination, trusted: (
                            calls.append("inspect"),
                            inspect_replay_at_antigravity(destination, trusted),
                        )[1],
                    ),
                    0,
                )

            self.assertEqual(calls, ["authority", "qualify", "evaluate", "publish", "inspect"])
            self.assertTrue(root.is_dir())
            self.assertEqual(len(tuple(root.iterdir())), 7)
            payload = json.loads(output.getvalue())
            self.assertEqual(payload["experiment_id"], "Exp459")
            self.assertEqual(payload["status"], "stable")
            self.assertTrue(payload["passed"])
            self.assertEqual(payload["label_query_count"], 0)
            self.assertEqual(len(payload["fold_records"]), 10)
            self.assertEqual(len(payload["seed_summaries"]), 2)
            self.assertNotEqual(root.resolve(), _OFFICIAL_ROOT_ANTIGRAVITY.resolve())

    def test_evaluate_fails_closed_on_runtime_or_authority_before_qualification_or_root_antigravity(
        self,
    ) -> None:
        """Break caught: wrong runtime or missing authority can qualify TRAIN or create a root."""
        from experiments.exp459_ecg_normal2_blocked_validation_antigravity import (
            cli_antigravity as cli,
        )

        with tempfile.TemporaryDirectory(prefix="exp459_runtime_reject_antigravity_") as temporary:
            root = Path(temporary) / "result_antigravity"
            with (
                patch.object(
                    cli,
                    "observe_runtime_provenance_antigravity",
                    side_effect=ValueError("python_version differs from frozen runtime"),
                    create=True,
                ),
                self.assertRaisesRegex(ValueError, "python_version"),
            ):
                cli._evaluate_with_dependencies_antigravity(
                    _COMMIT_ANTIGRAVITY,
                    root,
                    lambda: self.fail("authority loaded after runtime mismatch"),
                    lambda: self.fail("qualifier reached after runtime mismatch"),
                    lambda _qualification: self.fail("evaluator reached after runtime mismatch"),
                    lambda _root, _decision, _authority: self.fail("publisher reached after runtime mismatch"),
                    lambda _root, _authority: self.fail("inspector reached after runtime mismatch"),
                )
            self.assertFalse(root.exists())

        with tempfile.TemporaryDirectory(prefix="exp459_authority_reject_antigravity_") as temporary:
            root = Path(temporary) / "result_antigravity"
            with (
                patch.object(cli, "observe_runtime_provenance_antigravity", return_value=object()),
                self.assertRaisesRegex(ValueError, "trusted frozen validation authority"),
            ):
                cli._evaluate_with_dependencies_antigravity(
                    _COMMIT_ANTIGRAVITY,
                    root,
                    lambda: object(),
                    lambda: self.fail("qualifier reached after invalid authority"),
                    lambda _qualification: self.fail("evaluator reached after invalid authority"),
                    lambda _root, _decision, _authority: self.fail("publisher reached after invalid authority"),
                    lambda _root, _authority: self.fail("inspector reached after invalid authority"),
                )
            self.assertFalse(root.exists())

    def test_public_evaluate_invalid_authority_does_not_import_qualification_or_source_antigravity(
        self,
    ) -> None:
        """Break caught: public evaluate imports TRAIN qualification/source before authority validation."""
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from pathlib import Path\n"
                    "import json, sys, tempfile, types\n"
                    "from experiments.exp459_ecg_normal2_blocked_validation_antigravity import cli_antigravity as cli\n"
                    "private = types.ModuleType('_exp459_cli_private_test_antigravity')\n"
                    "private.CAPABILITY_ANTIGRAVITY = 'exp459_cli_private_test_antigravity_v1'\n"
                    "private.root_antigravity = Path(tempfile.mkdtemp(prefix='exp459_invalid_authority_antigravity_')) / 'result_antigravity'\n"
                    "private.validate_clean_current_commit_antigravity = lambda reviewed_commit: None\n"
                    "private.authority_antigravity = lambda: object()\n"
                    "sys.modules[private.__name__] = private\n"
                    "cli.observe_runtime_provenance_antigravity = lambda: {}\n"
                    "try:\n"
                    "    cli.main_antigravity(['evaluate', '--reviewed-commit', 'c' * 40])\n"
                    "except ValueError as error:\n"
                    "    error_text = str(error)\n"
                    "else:\n"
                    "    error_text = 'no error'\n"
                    "forbidden = (\n"
                    "    'experiments.exp459_ecg_normal2_blocked_validation_antigravity.sources_antigravity',\n"
                    "    'experiments.exp459_ecg_normal2_blocked_validation_antigravity.qualification_antigravity',\n"
                    ")\n"
                    "print(json.dumps({'error': error_text, 'forbidden_loaded': [m for m in forbidden if m in sys.modules]}, sort_keys=True))\n"
                ),
            ],
            cwd=_REPOSITORY_ANTIGRAVITY,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        proof = json.loads(completed.stdout)
        self.assertIn("trusted frozen validation authority", proof["error"])
        self.assertEqual(proof["forbidden_loaded"], [])

    def test_evaluate_rejects_qualification_that_differs_from_frozen_authority_antigravity(self) -> None:
        """Break caught: actual qualification differs from the authority but is still published."""
        from experiments.exp459_ecg_normal2_blocked_validation_antigravity import (
            cli_antigravity as cli,
        )
        from experiments.exp459_ecg_normal2_blocked_validation_antigravity.evaluation_antigravity import (
            evaluate_blocked_qualification_antigravity,
        )

        with tempfile.TemporaryDirectory(prefix="exp459_authority_mismatch_antigravity_") as temporary:
            root = Path(temporary) / "result_antigravity"
            with (
                patch.object(cli, "observe_runtime_provenance_antigravity", return_value=object()),
                self.assertRaisesRegex(ValueError, "differs from trusted authority"),
            ):
                cli._evaluate_with_dependencies_antigravity(
                    _COMMIT_ANTIGRAVITY,
                    root,
                    authority_antigravity,
                    lambda: qualification_antigravity(failing_seed=20261721),
                    evaluate_blocked_qualification_antigravity,
                    lambda _root, _decision, _authority: self.fail("publish reached after authority mismatch"),
                    lambda _root, _authority: self.fail("inspect reached after authority mismatch"),
                )
            self.assertFalse(root.exists())

    def test_inspect_uses_authority_replay_and_avoids_source_sqlite_hydra_imports_antigravity(self) -> None:
        """Break caught: inspect bypasses replay or imports TRAIN source/SQLite/Hydra."""
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from pathlib import Path\n"
                    "import contextlib, io, json, sys, tempfile, types\n"
                    "from experiments.exp459_ecg_normal2_blocked_validation_antigravity.artifacts_antigravity import publish_result_antigravity\n"
                    "from experiments.exp459_ecg_normal2_blocked_validation_antigravity.tests.test_artifacts_antigravity import authority_antigravity, decision_antigravity\n"
                    "from experiments.exp459_ecg_normal2_blocked_validation_antigravity import cli_antigravity as cli\n"
                    "authority = authority_antigravity()\n"
                    "with tempfile.TemporaryDirectory(prefix='exp459_inspect_boundary_antigravity_') as temporary:\n"
                    "    root = Path(temporary) / 'result_antigravity'\n"
                    "    publish_result_antigravity(root, decision_antigravity(), authority)\n"
                    "    private = types.ModuleType('_exp459_cli_private_test_antigravity')\n"
                    "    private.CAPABILITY_ANTIGRAVITY = 'exp459_cli_private_test_antigravity_v1'\n"
                    "    private.root_antigravity = root\n"
                    "    private.authority_antigravity = lambda: authority\n"
                    "    sys.modules[private.__name__] = private\n"
                    "    output = io.StringIO()\n"
                    "    with contextlib.redirect_stdout(output):\n"
                    "        code = cli.main_antigravity(['inspect'])\n"
                    "    forbidden = (\n"
                    "        'experiments.exp459_ecg_normal2_blocked_validation_antigravity.sources_antigravity',\n"
                    "        'experiments.exp459_ecg_normal2_blocked_validation_antigravity.qualification_antigravity',\n"
                    "        'sqlite3',\n"
                    "    )\n"
                    "    proof = {\n"
                    "        'code': code,\n"
                    "        'payload': json.loads(output.getvalue()),\n"
                    "        'forbidden_loaded': [m for m in forbidden if m in sys.modules],\n"
                    "        'aeon_loaded': any(m == 'aeon' or m.startswith('aeon.') for m in sys.modules),\n"
                    "    }\n"
                    "    print(json.dumps(proof, sort_keys=True))\n"
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
        self.assertEqual(proof["payload"]["experiment_id"], "Exp459")
        self.assertEqual(proof["payload"]["status"], "stable")
        self.assertEqual(proof["forbidden_loaded"], [])
        self.assertFalse(proof["aeon_loaded"])

    def test_launcher_imports_package_from_outside_repository_cwd_antigravity(self) -> None:
        """Break caught: launcher relies on the caller's working directory."""
        self.assertTrue(_LAUNCHER_ANTIGRAVITY.is_file())
        with tempfile.TemporaryDirectory(prefix="exp459_launcher_cwd_antigravity_") as temporary:
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
        """Break caught: launcher success path requires CWD tricks or public root overrides."""
        self.assertTrue(_LAUNCHER_ANTIGRAVITY.is_file())
        with tempfile.TemporaryDirectory(prefix="exp459_launcher_success_antigravity_") as temporary:
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
private = types.ModuleType("_exp459_cli_private_test_antigravity")
private.CAPABILITY_ANTIGRAVITY = "exp459_cli_private_test_antigravity_v1"
private.root_antigravity = root

def validate_clean_current_commit_antigravity(reviewed_commit):
    if reviewed_commit != "b" * 40:
        raise AssertionError("unexpected reviewed commit")

def authority_antigravity():
    from experiments.exp459_ecg_normal2_blocked_validation_antigravity.tests.test_artifacts_antigravity import authority_antigravity as authority
    return authority()

def evaluate_dependencies_antigravity():
    from experiments.exp459_ecg_normal2_blocked_validation_antigravity.artifacts_antigravity import publish_result_antigravity
    from experiments.exp459_ecg_normal2_blocked_validation_antigravity.evaluation_antigravity import evaluate_blocked_qualification_antigravity
    from experiments.exp459_ecg_normal2_blocked_validation_antigravity.replay_antigravity import inspect_replay_at_antigravity
    from experiments.exp459_ecg_normal2_blocked_validation_antigravity.tests.test_evaluation_antigravity import qualification_antigravity

    return (
        qualification_antigravity,
        evaluate_blocked_qualification_antigravity,
        publish_result_antigravity,
        inspect_replay_at_antigravity,
    )

private.validate_clean_current_commit_antigravity = validate_clean_current_commit_antigravity
private.authority_antigravity = authority_antigravity
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
    "status": evaluate_payload["status"],
    "label_query_count": evaluate_payload["label_query_count"],
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
        self.assertEqual(proof["leaf_count"], 7)
        self.assertEqual(proof["status"], "stable")
        self.assertEqual(proof["label_query_count"], 0)

    def test_version_controlled_frozen_authority_matches_exact_private_runs_antigravity(self) -> None:
        """Break caught: final CLI authority drifts from the exact two-run private equality evidence."""
        import inspect

        from experiments.exp459_ecg_normal2_blocked_validation_antigravity import (
            frozen_validation_authority_antigravity as authority_module,
        )
        from experiments.exp459_ecg_normal2_blocked_validation_antigravity.evaluation_antigravity import (
            FrozenValidationAuthorityAntigravity,
            validate_frozen_validation_authority_antigravity,
        )

        authority = validate_frozen_validation_authority_antigravity(
            authority_module.FROZEN_VALIDATION_AUTHORITY_ANTIGRAVITY
        )
        self.assertIsInstance(authority, FrozenValidationAuthorityAntigravity)
        self.assertEqual(
            authority.canonical_digest,
            "f661c4ce6c7fd6625ac5cf767db688ce7ede7a0d0557721c9e9fa7e7ea6508ee",
        )
        self.assertEqual(len(authority.fold_plan), 5)
        self.assertEqual(len(authority.fold_evidence), 10)
        self.assertEqual(len(authority.seed_summaries), 2)
        self.assertEqual(authority.decision.status, "unstable")
        self.assertFalse(authority.decision.passed)
        self.assertEqual(
            [(record.seed, record.fold_index) for record in authority.fold_evidence if not record.passed],
            [(20260721, 3), (20261721, 4)],
        )
        self.assertEqual(
            authority.fold_evidence[3].per_type_auc["period_change"],
            0.8208339620743423,
        )
        self.assertEqual(
            authority.fold_evidence[9].per_type_auc["period_change"],
            0.8004124541019063,
        )
        source = inspect.getsource(authority_module)
        for forbidden in (
            "qualify_blocked_normal2_antigravity",
            "load_normal2_train_antigravity",
            "_score_fold_antigravity",
            "sqlite3",
            "Hydra",
            "aeon",
            "univariate_ts.db",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
