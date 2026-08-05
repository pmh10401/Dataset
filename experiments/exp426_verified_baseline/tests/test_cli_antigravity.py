"""Frozen command boundary and readiness side-effect tests for Exp426."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import ast
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from experiments.exp426_verified_baseline.cli_antigravity import (
    _CliDependencies,
    _revalidate_readiness_for_execution,
    build_parser,
    main,
)
from experiments.exp426_verified_baseline.tests import (
    test_reporting_antigravity as reporting_fixtures,
    test_runtime_antigravity as runtime_fixtures,
)
from experiments.exp426_verified_baseline.runtime_antigravity import run_exp426_new


class Exp426CliTests(unittest.TestCase):
    HEAD = "1" * 40

    @classmethod
    def _readiness_payload(cls, *, git_commit: str | None = None) -> dict[str, object]:
        return {
            "schema": "exp426.readiness.v1",
            "state": "ready",
            "git_commit": cls.HEAD if git_commit is None else git_commit,
            "workers": 12,
            "prefetch": 24,
            "inner_threads": 1,
            "backend": "cpu",
            "mlx_adopted": False,
            "mps_adopted": False,
            "output_root_absent": True,
            "checkpoint_root_absent": True,
            "database": {
                "sha256": "8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c",
                "full_hash_count": 1,
                "cleanup_status": "removed",
                "elapsed_seconds": 1.0,
            },
            "lanes": {
                "hard": "73dff56e26429158bd2672a898d87266aa38e73f0a50d15d4f7c7082e3a01848",
                "standard": "f02259ae77fdcbccb8922fb9f90d5918401020d5d7d902212c2e424b878d0d3b",
                "priority": "f02259ae77fdcbccb8922fb9f90d5918401020d5d7d902212c2e424b878d0d3b",
                "automatic_normal": "73dff56e26429158bd2672a898d87266aa38e73f0a50d15d4f7c7082e3a01848",
                "elapsed_seconds": 1.0,
            },
            "source": {
                "records": 1117,
                "importable": 983,
                "terminal": 134,
                "root_sha256": "f7526bf638f4c1051a5970271bdbbccd753986f7d067ec1260817da884c1b757",
                "progress_sha256": "5e9aa4a4b7240ac89c078bdc1dce4b41bda6b6dbc505dd9ad49782556da7bd84",
                "record_set_sha256": "d74a0def7788db89a1d3e32f46662abf07c44c7cbbce5e02612b510e2fdf4258",
                "elapsed_seconds": 1.0,
            },
            "scientific_parity": True,
            "one_worker_seconds": 2.0,
            "twelve_worker_seconds": 1.0,
            "observed_worker_processes": 12,
            "one_worker_processes": 1,
            "test_values_opened": 0,
            "test_labels_opened": 0,
            "focused_tests": {"returncode": 0},
            "artifact_round_trip": True,
        }

    @classmethod
    def _write_readiness(cls, path: Path, *, git_commit: str | None = None) -> None:
        path.write_text(json.dumps(
            cls._readiness_payload(git_commit=git_commit),
            sort_keys=True, separators=(",", ":"),
        ))

    @staticmethod
    def _dependencies(parent: Path, calls: list[str]) -> _CliDependencies:
        execution_commits: list[str | None] = []

        def execute(
            command: str, _output: Path, _checkpoint: Path,
            readiness_commit: str | None,
        ):
            calls.append(command)
            execution_commits.append(readiness_commit)
            return {
                "state": "complete_review_only",
                "completed_datasets": 3,
                "workers": 12,
            }
        execute.execution_commits = execution_commits  # type: ignore[attr-defined]

        def readiness():
            calls.append("readiness")
            return Exp426CliTests._readiness_payload()

        return _CliDependencies(
            frozen_output=parent / "output_antigravity",
            frozen_checkpoint=parent / "checkpoint_antigravity",
            frozen_readiness=parent / "readiness_antigravity.json",
            execute=execute,
            readiness=readiness,
            current_head=lambda: Exp426CliTests.HEAD,
        )

    @staticmethod
    def _main(argv: list[str], dependencies: _CliDependencies):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(argv, _dependencies=dependencies)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_parser_has_four_commands_and_no_worker_option(self) -> None:
        parser = build_parser()
        help_text = parser.format_help()
        self.assertIn("new", help_text)
        self.assertIn("resume", help_text)
        self.assertIn("inspect", help_text)
        self.assertIn("readiness", help_text)
        self.assertNotIn("--workers", help_text)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            parser.parse_args(["new", "--workers", "1"])
        self.assertEqual(raised.exception.code, 2)

    def test_new_rejects_arbitrary_or_existing_roots_before_execution(self) -> None:
        with TemporaryDirectory() as directory:
            parent = Path(directory)
            calls: list[str] = []
            dependencies = self._dependencies(parent, calls)
            code, _stdout, stderr = self._main([
                "new", "--output-root", str(parent / "wrong"),
                "--checkpoint-root", str(dependencies.frozen_checkpoint),
            ], dependencies)
            self.assertEqual(code, 2)
            self.assertIn("frozen", stderr)
            self.assertEqual(calls, [])
            self.assertFalse((parent / "wrong").exists())

            dependencies.frozen_output.mkdir()
            code, _stdout, stderr = self._main([
                "new", "--output-root", str(dependencies.frozen_output),
                "--checkpoint-root", str(dependencies.frozen_checkpoint),
            ], dependencies)
            self.assertEqual(code, 2)
            self.assertIn("absent", stderr)
            self.assertEqual(calls, [])

    def test_resume_requires_both_existing_roots_before_execution(self) -> None:
        with TemporaryDirectory() as directory:
            parent = Path(directory)
            calls: list[str] = []
            dependencies = self._dependencies(parent, calls)
            code, _stdout, stderr = self._main([
                "resume", "--output-root", str(dependencies.frozen_output),
                "--checkpoint-root", str(dependencies.frozen_checkpoint),
            ], dependencies)
            self.assertEqual(code, 2)
            self.assertIn("absent", stderr)
            self.assertEqual(calls, [])

    def test_new_executes_only_for_exact_absent_frozen_roots(self) -> None:
        with TemporaryDirectory() as directory:
            parent = Path(directory)
            calls: list[str] = []
            dependencies = self._dependencies(parent, calls)
            self._write_readiness(dependencies.frozen_readiness)
            code, stdout, stderr = self._main([
                "new", "--output-root", str(dependencies.frozen_output),
                "--checkpoint-root", str(dependencies.frozen_checkpoint),
            ], dependencies)
            self.assertEqual((code, stderr), (0, ""))
            self.assertEqual(calls, ["new"])
            self.assertEqual(
                dependencies.execute.execution_commits,  # type: ignore[attr-defined]
                [self.HEAD],
            )
            self.assertEqual(json.loads(stdout)["workers"], 12)

    def test_new_rejects_missing_readiness_before_execution(self) -> None:
        with TemporaryDirectory() as directory:
            parent = Path(directory)
            calls: list[str] = []
            dependencies = self._dependencies(parent, calls)
            code, _stdout, stderr = self._main([
                "new", "--output-root", str(dependencies.frozen_output),
                "--checkpoint-root", str(dependencies.frozen_checkpoint),
            ], dependencies)
            self.assertEqual(code, 2)
            self.assertIn("readiness", stderr)
            self.assertEqual(calls, [])

    def test_new_rejects_readiness_from_a_different_commit(self) -> None:
        with TemporaryDirectory() as directory:
            parent = Path(directory)
            calls: list[str] = []
            dependencies = self._dependencies(parent, calls)
            self._write_readiness(dependencies.frozen_readiness, git_commit="2" * 40)
            code, _stdout, stderr = self._main([
                "new", "--output-root", str(dependencies.frozen_output),
                "--checkpoint-root", str(dependencies.frozen_checkpoint),
            ], dependencies)
            self.assertEqual(code, 2)
            self.assertIn("commit", stderr)
            self.assertEqual(calls, [])

    def test_execution_revalidates_readiness_after_head_handoff(self) -> None:
        with TemporaryDirectory() as directory:
            readiness = Path(directory) / "readiness_antigravity.json"
            self._write_readiness(readiness)
            verified = _revalidate_readiness_for_execution(
                readiness,
                validated_commit=self.HEAD,
                current_head=self.HEAD,
            )
            self.assertEqual(verified["git_commit"], self.HEAD)
            with self.assertRaisesRegex(ValueError, "changed before execution"):
                _revalidate_readiness_for_execution(
                    readiness,
                    validated_commit=self.HEAD,
                    current_head="2" * 40,
                )

    def test_matching_incomplete_resume_executes_and_inspect_is_read_only(self) -> None:
        with TemporaryDirectory() as directory:
            parent = Path(directory)
            calls: list[str] = []
            datasets = ("dataset-00",)
            fake = runtime_fixtures._FakeRuntime(
                runtime_fixtures.Exp426RuntimeTests.classifications(datasets, imported=1)
            )
            output, checkpoint = parent / "output_antigravity", parent / "checkpoint_antigravity"

            def crash(name: str, _dataset: str | None) -> None:
                if name == "after_worker_result":
                    raise RuntimeError("leave incomplete")

            with self.assertRaisesRegex(RuntimeError, "leave incomplete"):
                run_exp426_new(
                    output, checkpoint,
                    runtime_fixtures.Exp426RuntimeTests.provenance(datasets),
                    runtime_fixtures.Exp426RuntimeTests.checkpoint_context(datasets),
                    object(), runtime_fixtures._FakeDatabase({}), object(),
                    _dependencies=fake.dependencies(), _hook=crash,
                )
            base = self._dependencies(parent, calls)
            dependencies = _CliDependencies(
                output, checkpoint, base.frozen_readiness,
                base.execute, base.readiness,
                reporting_fixtures._PublicationExpectations(1, 1, 0),
            )
            before = reporting_fixtures.Exp426ReportingTests._tree_state(parent)
            inspect_code, inspect_stdout, inspect_stderr = self._main([
                "inspect", "--output-root", str(output),
                "--checkpoint-root", str(checkpoint),
            ], dependencies)
            after_inspect = reporting_fixtures.Exp426ReportingTests._tree_state(parent)
            self.assertEqual((inspect_code, inspect_stderr), (1, ""))
            self.assertNotEqual(json.loads(inspect_stdout)["state"], "failed")
            self.assertEqual(before, after_inspect)
            resume_code, _resume_stdout, resume_stderr = self._main([
                "resume", "--output-root", str(output),
                "--checkpoint-root", str(checkpoint),
            ], dependencies)
            self.assertEqual((resume_code, resume_stderr), (0, ""))
            self.assertEqual(calls, ["resume"])

    def test_resume_rejects_a_failed_root_before_execution(self) -> None:
        with TemporaryDirectory() as directory:
            parent = Path(directory)
            calls: list[str] = []
            _datasets, _provenance, _context, output, checkpoint = (
                reporting_fixtures.Exp426ReportingTests._fixture(parent)
            )
            progress_path = output / "progress.json"
            progress = json.loads(progress_path.read_text())
            progress["state"] = "failed"
            progress["counts"]["error"] = 1
            progress_path.write_text(json.dumps(
                progress, sort_keys=True, separators=(",", ":"),
            ))
            base = self._dependencies(parent, calls)
            dependencies = _CliDependencies(
                output, checkpoint, base.frozen_readiness,
                base.execute, base.readiness,
                reporting_fixtures.Exp426ReportingTests.expectations,
            )
            code, _stdout, stderr = self._main([
                "resume", "--output-root", str(output),
                "--checkpoint-root", str(checkpoint),
            ], dependencies)
            self.assertEqual(code, 2)
            self.assertIn("failed runtime root", stderr)
            self.assertEqual(calls, [])

    def test_resume_allows_checkpoint_one_record_ahead_of_runtime_progress(self) -> None:
        with TemporaryDirectory() as directory:
            parent = Path(directory)
            calls: list[str] = []
            datasets = ("dataset-00",)
            fake = runtime_fixtures._FakeRuntime(
                runtime_fixtures.Exp426RuntimeTests.classifications(datasets, imported=1)
            )
            output, checkpoint = parent / "output_antigravity", parent / "checkpoint_antigravity"
            run_exp426_new(
                output, checkpoint,
                runtime_fixtures.Exp426RuntimeTests.provenance(datasets),
                runtime_fixtures.Exp426RuntimeTests.checkpoint_context(datasets),
                object(), runtime_fixtures._FakeDatabase({}), object(),
                _dependencies=fake.dependencies(),
            )
            progress_path = output / "progress.json"
            progress = json.loads(progress_path.read_text())
            progress.update({
                "state": "importing",
                "current_dataset": datasets[0],
                "counts": {
                    "total": 1, "completed": 0, "remaining": 1,
                    "imported": 0, "verified_train_abstentions": 0,
                    "abstained": 0, "failed": 0, "success": 0, "error": 0,
                },
            })
            progress_path.write_text(
                json.dumps(progress, sort_keys=True, separators=(",", ":"))
            )
            base = self._dependencies(parent, calls)
            dependencies = _CliDependencies(
                output, checkpoint, base.frozen_readiness,
                base.execute, base.readiness,
                reporting_fixtures._PublicationExpectations(1, 1, 0),
            )
            code, _stdout, stderr = self._main([
                "resume", "--output-root", str(output),
                "--checkpoint-root", str(checkpoint),
            ], dependencies)
            self.assertEqual((code, stderr), (0, ""))
            self.assertEqual(calls, ["resume"])

    def test_readiness_writes_only_explicit_file_and_never_overwrites(self) -> None:
        with TemporaryDirectory() as directory:
            parent = Path(directory)
            calls: list[str] = []
            dependencies = self._dependencies(parent, calls)
            code, stdout, stderr = self._main([
                "readiness", "--readiness-output", str(dependencies.frozen_readiness),
            ], dependencies)
            self.assertEqual((code, stderr), (0, ""))
            self.assertEqual(calls, ["readiness"])
            self.assertEqual(json.loads(stdout)["state"], "ready")
            files = [path for path in parent.rglob("*") if path.is_file()]
            self.assertEqual(files, [dependencies.frozen_readiness])
            before = dependencies.frozen_readiness.read_bytes()

            code, _stdout, stderr = self._main([
                "readiness", "--readiness-output", str(dependencies.frozen_readiness),
            ], dependencies)
            self.assertEqual(code, 2)
            self.assertIn("absent", stderr)
            self.assertEqual(dependencies.frozen_readiness.read_bytes(), before)
            self.assertEqual(calls, ["readiness"])

    def test_complete_inspect_and_resume_are_read_only_and_skip_execution(self) -> None:
        with TemporaryDirectory() as directory:
            parent = Path(directory)
            calls: list[str] = []
            dependencies = self._dependencies(parent, calls)
            _datasets, provenance, context, output, checkpoint = (
                reporting_fixtures.Exp426ReportingTests._fixture(parent)
            )
            reporting_fixtures.publish_exp426(
                output, checkpoint, provenance, context,
                _expectations=reporting_fixtures.Exp426ReportingTests.expectations,
            )
            dependencies = _CliDependencies(
                output, checkpoint, dependencies.frozen_readiness,
                dependencies.execute, dependencies.readiness,
                reporting_fixtures.Exp426ReportingTests.expectations,
            )
            before = reporting_fixtures.Exp426ReportingTests._tree_state(parent)
            inspect_code, inspect_stdout, inspect_stderr = self._main([
                "inspect", "--output-root", str(output),
                "--checkpoint-root", str(checkpoint),
            ], dependencies)
            resume_code, resume_stdout, resume_stderr = self._main([
                "resume", "--output-root", str(output),
                "--checkpoint-root", str(checkpoint),
            ], dependencies)
            after = reporting_fixtures.Exp426ReportingTests._tree_state(parent)
            self.assertEqual((inspect_code, inspect_stderr), (0, ""))
            self.assertEqual((resume_code, resume_stderr), (0, ""))
            self.assertEqual(json.loads(inspect_stdout)["state"], "complete_review_only")
            self.assertEqual(json.loads(resume_stdout)["state"], "complete_review_only")
            self.assertEqual(calls, [])
            self.assertEqual(before, after)

    def test_complete_resume_can_finish_publication_without_execution(self) -> None:
        with TemporaryDirectory() as directory:
            parent = Path(directory)
            calls: list[str] = []
            _datasets, _provenance, _context, output, checkpoint = (
                reporting_fixtures.Exp426ReportingTests._fixture(parent)
            )
            base = self._dependencies(parent, calls)
            dependencies = _CliDependencies(
                output, checkpoint, base.frozen_readiness,
                base.execute, base.readiness,
                reporting_fixtures.Exp426ReportingTests.expectations,
            )
            self.assertFalse((output / "publication_antigravity").exists())
            code, stdout, stderr = self._main([
                "resume", "--output-root", str(output),
                "--checkpoint-root", str(checkpoint),
            ], dependencies)
            self.assertEqual((code, stderr), (0, ""))
            self.assertEqual(json.loads(stdout)["state"], "complete_review_only")
            self.assertTrue((output / "publication_antigravity").is_dir())
            self.assertEqual(calls, [])

    def test_cli_modules_do_not_import_label_or_evaluation_code(self) -> None:
        root = Path(__file__).resolve().parents[3]
        paths = (
            root / "experiments/exp426_verified_baseline/cli_antigravity.py",
            root / "scratch/run_exp426_verified_baseline_antigravity.py",
        )
        for path in paths:
            source = path.read_text()
            tree = ast.parse(source)
            imported: list[str] = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imported.append(node.module or "")
                    imported.extend(alias.name for alias in node.names)
            normalized = " ".join(imported).lower()
            self.assertNotIn("load_test", normalized)
            self.assertNotIn("label", normalized)
            self.assertNotIn("evaluation", normalized)
            self.assertNotIn("--workers", source)


if __name__ == "__main__":
    unittest.main()
