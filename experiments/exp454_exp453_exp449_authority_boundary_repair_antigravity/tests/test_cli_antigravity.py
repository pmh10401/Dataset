"""Closed command and production-path CLI tests for Exp454 Task 5."""

from __future__ import annotations

from contextlib import redirect_stderr
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import textwrap
import unittest
from unittest.mock import patch


_COMMIT_ANTIGRAVITY = "a" * 40
_FIXED_PYTHON_ANTIGRAVITY = (
    "/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python"
)
_FIXED_PREFIX_ANTIGRAVITY = (
    "/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314"
)
_THREADS_ANTIGRAVITY = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "AEON_NUM_THREADS",
    "MLX_NUM_THREADS",
)
_ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[3]
_LAUNCHER_ANTIGRAVITY = (
    _ROOT_ANTIGRAVITY / "scratch/run_exp454_exp453_exp449_authority_boundary_repair_antigravity.py"
)
_PACKAGE_ANTIGRAVITY = "experiments.exp454_exp453_exp449_authority_boundary_repair_antigravity"
_OFFICIAL_ROOT_ANTIGRAVITY = (
    "/Users/mac/Documents/Dataset/exp454_exp453_exp449_authority_boundary_repair_evaluation_antigravity_20260814"
)
_OFFICIAL_RECEIPTS_ANTIGRAVITY = (
    "/Users/mac/Documents/Dataset/exp454_authority_boundary_repair_lifecycle_receipts_antigravity_20260814"
)
_OFFICIAL_AUDIT_ANTIGRAVITY = (
    "/Users/mac/Documents/Dataset/exp454_authority_boundary_repair_failure_audit_antigravity.json"
)


def _stub_dispatch_script_antigravity(destination: Path) -> Path:
    destination.write_text(
        textwrap.dedent(
            f"""\
            from __future__ import annotations

            import json
            import os
            from pathlib import Path
            import sys

            _scratch = Path({str(_ROOT_ANTIGRAVITY / "scratch")!r})
            if str(_scratch) not in sys.path:
                sys.path.insert(0, str(_scratch))
            from exp454_runtime_binding_antigravity import ensure_fixed_runtime_antigravity

            ensure_fixed_runtime_antigravity(Path(__file__), tuple(sys.argv[1:]))

            for _name in {list(_THREADS_ANTIGRAVITY)!r}:
                os.environ[_name] = "1"

            _repository = Path({str(_ROOT_ANTIGRAVITY)!r})
            if str(_repository) not in sys.path:
                sys.path.insert(0, str(_repository))

            from {_PACKAGE_ANTIGRAVITY} import cli_antigravity as cli

            def _result(command, commit=None):
                return {{
                    "command": command,
                    "commit": commit,
                    "executable": sys.executable,
                    "prefix": sys.prefix,
                    "threads": {{name: os.environ[name] for name in {list(_THREADS_ANTIGRAVITY)!r}}},
                }}

            cli._load_readiness_antigravity = lambda: lambda commit: _result("readiness", commit)
            cli._validate_reviewed_map_antigravity = lambda commit: None
            cli._load_evaluate_antigravity = lambda: lambda commit: _result("evaluate", commit)
            cli._load_inspect_antigravity = lambda: lambda: _result("inspect")
            raise SystemExit(cli.main_antigravity())
            """
        ),
        encoding="utf-8",
    )
    return destination


class CliAntigravityTests(unittest.TestCase):
    def test_parser_exposes_only_three_exact_forms_antigravity(self) -> None:
        """Break caught: an abbreviation, override, extra command, or inspect argument reaches a phase."""
        from experiments.exp454_exp453_exp449_authority_boundary_repair_antigravity.cli_antigravity import (
            build_parser_antigravity,
        )

        parser = build_parser_antigravity()
        self.assertEqual(
            parser.parse_args(["readiness", "--reviewed-commit", _COMMIT_ANTIGRAVITY]).command,
            "readiness",
        )
        self.assertEqual(
            parser.parse_args(["evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY]).command,
            "evaluate",
        )
        self.assertEqual(parser.parse_args(["inspect"]).command, "inspect")
        rejected = (
            ("read", "--reviewed-commit", _COMMIT_ANTIGRAVITY),
            ("readiness", "--reviewed-com", _COMMIT_ANTIGRAVITY),
            ("readiness", "-r", _COMMIT_ANTIGRAVITY),
            ("readiness", "--commit", _COMMIT_ANTIGRAVITY),
            ("readiness", "--reviewed", _COMMIT_ANTIGRAVITY),
            ("predict", "--reviewed-commit", _COMMIT_ANTIGRAVITY),
            ("new",),
            ("resume",),
            ("retry",),
            ("override",),
            ("evaluate", "--reviewed-commit", "A" * 40),
            ("evaluate", "--reviewed-commit", "a" * 39),
            ("evaluate", "--reviewed-commit", "a" * 41),
            ("evaluate",),
            (
                "readiness",
                "--reviewed-commit",
                "a" * 40,
                "--reviewed-commit",
                "b" * 40,
            ),
            (
                "evaluate",
                "--reviewed-commit",
                "a" * 40,
                "--reviewed-commit",
                "b" * 40,
            ),
            ("evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY, "--output", "/tmp"),
            ("evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY, "--override"),
            ("inspect", "--reviewed-commit", _COMMIT_ANTIGRAVITY),
            ("inspect", "extra"),
        )
        for arguments in rejected:
            with self.subTest(arguments=arguments), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser.parse_args(arguments)

    def test_safe_commands_leave_evaluator_and_label_modules_unloaded_antigravity(self) -> None:
        """Break caught: help, readiness, or inspect eagerly imports evaluator or TEST-label code."""
        probe = (
            "import json,sys;sys.path.insert(0,%r);" % str(_ROOT_ANTIGRAVITY)
            + "from experiments.exp454_exp453_exp449_authority_boundary_repair_antigravity import cli_antigravity as c;"
            "c.build_parser_antigravity().parse_args(['--help'])"
        )
        result = subprocess.run(
            (_FIXED_PYTHON_ANTIGRAVITY, "-c", probe),
            cwd="/tmp",
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        loaded_probe = (
            "import json,sys;sys.path.insert(0,%r);" % str(_ROOT_ANTIGRAVITY)
            + "from experiments.exp454_exp453_exp449_authority_boundary_repair_antigravity import cli_antigravity as c;"
            "print(json.dumps(sorted(name for name in sys.modules if name.endswith("
            "('_antigravity.evaluation_antigravity','_antigravity.sources_antigravity',"
            "'_antigravity.dashboard_export_antigravity','_antigravity.replay_antigravity')))))"
        )
        loaded = subprocess.run(
            (_FIXED_PYTHON_ANTIGRAVITY, "-c", loaded_probe),
            cwd="/tmp",
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(loaded.returncode, 0, loaded.stderr)
        self.assertEqual(json.loads(loaded.stdout), [])
        from experiments.exp454_exp453_exp449_authority_boundary_repair_antigravity import (
            cli_antigravity as cli,
        )

        with (
            patch.object(cli, "_load_readiness_antigravity", return_value=lambda value: {"commit": value}),
            patch.object(cli, "_load_inspect_antigravity", return_value=lambda: {"replay": "safe"}),
            patch.object(cli, "_load_evaluate_antigravity", side_effect=AssertionError("evaluator imported")),
            patch.object(cli, "_print_result_antigravity"),
        ):
            self.assertEqual(cli.main_antigravity(["readiness", "--reviewed-commit", _COMMIT_ANTIGRAVITY]), 0)
            self.assertEqual(cli.main_antigravity(["inspect"]), 0)

    def test_evaluate_runs_clean_reviewed_map_gate_before_evaluator_import_antigravity(self) -> None:
        """Break caught: evaluator import happens before the clean reviewed-map gate."""
        from experiments.exp454_exp453_exp449_authority_boundary_repair_antigravity import (
            cli_antigravity as cli,
        )

        events: list[tuple[str, str]] = []

        def gate(commit: str) -> None:
            events.append(("gate", commit))

        def evaluator():
            events.append(("evaluator_import", ""))
            return lambda commit: {"commit": commit}

        with (
            patch.object(cli, "_validate_reviewed_map_antigravity", side_effect=gate),
            patch.object(cli, "_load_evaluate_antigravity", side_effect=evaluator),
            patch.object(cli, "_print_result_antigravity"),
        ):
            self.assertEqual(cli.main_antigravity(["evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY]), 0)
        self.assertEqual(events, [("gate", _COMMIT_ANTIGRAVITY), ("evaluator_import", "")])

    def test_launcher_binds_threads_then_cli_and_runs_outside_repository_antigravity(self) -> None:
        """Break caught: launcher depends on cwd, imports CLI before binding, or skips thread pins."""
        source = _LAUNCHER_ANTIGRAVITY.read_text(encoding="utf-8")
        bind_at = source.index("ensure_fixed_runtime_antigravity")
        thread_at = source.index("OMP_NUM_THREADS")
        cli_at = source.index("cli_antigravity")
        package_at = source.index(_PACKAGE_ANTIGRAVITY)
        self.assertLess(bind_at, thread_at)
        self.assertLess(thread_at, cli_at)
        self.assertLess(thread_at, package_at)
        self.assertNotIn("evaluation_antigravity", source)
        self.assertNotIn("replay_antigravity", source)
        self.assertNotIn("readiness_antigravity", source.split("ensure_fixed_runtime_antigravity", 1)[1])

        environment = {**os.environ, **{name: "999" for name in _THREADS_ANTIGRAVITY}}
        help_result = subprocess.run(
            (_FIXED_PYTHON_ANTIGRAVITY, str(_LAUNCHER_ANTIGRAVITY), "--help"),
            cwd="/tmp",
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("readiness", help_result.stdout)
        self.assertIn("evaluate", help_result.stdout)
        self.assertIn("inspect", help_result.stdout)

        probe = (
            "import json,os,runpy;runpy.run_path(%r);"
            "print(json.dumps({name:os.environ[name] for name in %r},sort_keys=True))"
            % (str(_LAUNCHER_ANTIGRAVITY), _THREADS_ANTIGRAVITY)
        )
        result = subprocess.run(
            (_FIXED_PYTHON_ANTIGRAVITY, "-c", probe),
            cwd="/tmp",
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {name: "1" for name in _THREADS_ANTIGRAVITY})

    def test_subprocess_dispatch_paths_use_synthetic_seams_antigravity(self) -> None:
        """Break caught: a closed command reaches real labels or official evaluation paths."""
        with tempfile.TemporaryDirectory(prefix="exp454_cli_dispatch_antigravity_") as temporary:
            script = _stub_dispatch_script_antigravity(Path(temporary) / "dispatch_cli_antigravity.py")
            commands = (
                ("readiness", ["readiness", "--reviewed-commit", _COMMIT_ANTIGRAVITY], _COMMIT_ANTIGRAVITY),
                ("evaluate", ["evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY], _COMMIT_ANTIGRAVITY),
                ("inspect", ["inspect"], None),
            )
            for command, arguments, commit in commands:
                with self.subTest(command=command):
                    result = subprocess.run(
                        (_FIXED_PYTHON_ANTIGRAVITY, str(script), *arguments),
                        cwd="/tmp",
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    payload = json.loads(result.stdout)
                    self.assertEqual(payload["command"], command)
                    self.assertEqual(payload["commit"], commit)
                    self.assertEqual(payload["executable"], _FIXED_PYTHON_ANTIGRAVITY)
                    self.assertEqual(payload["prefix"], _FIXED_PREFIX_ANTIGRAVITY)
                    self.assertEqual(payload["threads"], {name: "1" for name in _THREADS_ANTIGRAVITY})
                    self.assertNotIn("exp454_exp453_exp449_authority_boundary_repair_evaluation", result.stdout)
                    self.assertNotIn("exp454_authority_boundary_repair_lifecycle_receipts", result.stdout)
                    self.assertNotIn("exp454_authority_boundary_repair_failure_audit", result.stdout)

    def test_cli_evaluate_subprocess_loads_real_normalized_capture_antigravity(self) -> None:
        """Break caught: production CLI evaluate uses a fixture converter or skips Task 2 capture."""
        official_paths = tuple(
            Path(value)
            for value in (
                _OFFICIAL_ROOT_ANTIGRAVITY,
                _OFFICIAL_RECEIPTS_ANTIGRAVITY,
                _OFFICIAL_AUDIT_ANTIGRAVITY,
            )
        )

        def snapshot(path: Path) -> object:
            if not path.exists():
                return None
            stat = path.lstat()
            contents = (
                tuple(sorted((leaf.name, leaf.read_bytes()) for leaf in path.iterdir()))
                if path.is_dir()
                else path.read_bytes()
            )
            return (stat.st_dev, stat.st_ino, stat.st_mode, stat.st_nlink, contents)

        before = tuple(snapshot(path) for path in official_paths)
        expected_exists = tuple(path.exists() for path in official_paths)
        with tempfile.TemporaryDirectory(prefix="exp454_cli_production_antigravity_") as temporary:
            site = Path(temporary)
            (site / "sitecustomize.py").write_text(
                textwrap.dedent(
                    f"""\
                    from __future__ import annotations

                    import builtins
                    import sqlite3

                    _real_import = builtins.__import__
                    _official_root = {_OFFICIAL_ROOT_ANTIGRAVITY!r}
                    _official_receipts = {_OFFICIAL_RECEIPTS_ANTIGRAVITY!r}
                    _official_audit = {_OFFICIAL_AUDIT_ANTIGRAVITY!r}

                    def _block_test_labels(*_args, **_kwargs):
                        raise AssertionError("TEST labels were queried")

                    sqlite3.connect = _block_test_labels

                    def _hook(name, globals=None, locals=None, fromlist=(), level=0):
                        module = _real_import(name, globals, locals, fromlist, level)
                        if (
                            name.endswith("cli_antigravity")
                            and "exp454_exp453_exp449_authority_boundary_repair_antigravity" in name
                        ):
                            module._validate_reviewed_map_antigravity = lambda commit: None
                            real_load = module._load_evaluate_antigravity

                            def _load():
                                evaluate = real_load()

                                def _wrapped(commit):
                                    import inspect
                                    import json
                                    from pathlib import Path
                                    import sys

                                    eval_name = (
                                        "experiments.exp454_exp453_exp449_authority_boundary_repair_antigravity"
                                        ".evaluation_antigravity"
                                    )
                                    eval_mod = sys.modules[eval_name]
                                    source = inspect.getsource(eval_mod.evaluate_antigravity)
                                    from experiments.exp454_exp453_exp449_authority_boundary_repair_antigravity.predecessor_antigravity import (
                                        capture_normalized_target_authority_antigravity,
                                    )

                                    authority = capture_normalized_target_authority_antigravity()
                                    fixture = [
                                        loaded
                                        for loaded in sys.modules
                                        if "test_fixture" in loaded
                                        or "fixture_converter" in loaded
                                        or (
                                            ".tests." in loaded
                                            and "converter" in loaded
                                        )
                                    ]
                                    print(
                                        json.dumps(
                                            {{
                                                "production_evaluate_loaded": Path(eval_mod.__file__).name
                                                == "evaluation_antigravity.py",
                                                "evaluate_is_production": evaluate is eval_mod.evaluate_antigravity,
                                                "evaluate_wires_normalized_capture": (
                                                    "capture_normalized_target_authority_antigravity" in source
                                                ),
                                                "capture_type": type(authority).__name__,
                                                "dataset_count": len(authority.target_dataset_names),
                                                "row_count": len(authority.target_baseline_rows),
                                                "row_type": type(authority.target_baseline_rows[0]).__name__,
                                                "fixture_converter_modules": fixture,
                                                "official_root_exists": Path(_official_root).exists(),
                                                "official_receipts_exist": Path(_official_receipts).exists(),
                                                "official_audit_exists": Path(_official_audit).exists(),
                                            }},
                                            sort_keys=True,
                                            separators=(",", ":"),
                                        ),
                                        flush=True,
                                    )
                                    raise SystemExit(0)

                                return _wrapped

                            module._load_evaluate_antigravity = _load
                        return module

                    builtins.__import__ = _hook
                    """
                ),
                encoding="utf-8",
            )
            environment = {
                **os.environ,
                "PYTHONPATH": str(site),
                **{name: "1" for name in _THREADS_ANTIGRAVITY},
            }
            result = subprocess.run(
                (
                    _FIXED_PYTHON_ANTIGRAVITY,
                    str(_LAUNCHER_ANTIGRAVITY),
                    "evaluate",
                    "--reviewed-commit",
                    _COMMIT_ANTIGRAVITY,
                ),
                cwd="/tmp",
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            payload = json.loads(result.stdout)
            self.assertTrue(payload["production_evaluate_loaded"])
            self.assertTrue(payload["evaluate_is_production"])
            self.assertTrue(payload["evaluate_wires_normalized_capture"])
            self.assertEqual(payload["capture_type"], "NormalizedTargetAuthorityAntigravity")
            self.assertEqual(payload["dataset_count"], 48)
            self.assertEqual(payload["row_count"], 48)
            self.assertEqual(payload["row_type"], "DatasetBaselineAntigravity")
            self.assertEqual(payload["fixture_converter_modules"], [])
            self.assertEqual(
                (
                    payload["official_root_exists"],
                    payload["official_receipts_exist"],
                    payload["official_audit_exists"],
                ),
                expected_exists,
            )
            self.assertEqual(tuple(snapshot(path) for path in official_paths), before)


if __name__ == "__main__":
    unittest.main()
