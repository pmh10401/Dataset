"""Public-literal lifecycle tests for Exp471 reporting and its closed CLI."""

from __future__ import annotations

from contextlib import ExitStack
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from experiments.exp471_exp467_evaluation_recovery_antigravity import (
    evaluation_antigravity as evaluation,
)
from experiments.exp471_exp467_evaluation_recovery_antigravity import (
    reporting_antigravity as reporting,
)
from experiments.exp471_exp467_evaluation_recovery_antigravity.cli_antigravity import (
    main_antigravity,
)
from experiments.exp471_exp467_evaluation_recovery_antigravity.config_antigravity import (
    METHOD_ORDER_ANTIGRAVITY,
)
from experiments.exp471_exp467_evaluation_recovery_antigravity.contracts_antigravity import (
    canonical_json_bytes_exp471_antigravity,
)
from experiments.exp471_exp467_evaluation_recovery_antigravity.tests.test_evaluation_antigravity import (
    _LABELS_ANTIGRAVITY,
    _authority_antigravity,
    _evaluation_patches_antigravity,
    _publish_cache_antigravity,
    _session_antigravity,
)


def _worker_provenance_antigravity() -> dict[str, object]:
    return {
        "schema": "exp471-public-worker-provenance-antigravity.v1",
        "detected_logical_cpus": 4,
        "default_cpu_workers": 3,
        "benchmark_decision": {
            "single_worker_seconds": 2.0,
            "parallel_seconds": 1.0,
            "byte_parity": True,
            "selected_workers": 3,
        },
        "execution_decision": {
            "selected_workers": 3,
            "fallback_reason": "",
        },
    }


def _artifact_antigravity(base: Path):
    cache_root = base / "score_cache_antigravity"
    evaluation_root = base / "evaluation_antigravity"
    authority = _authority_antigravity()
    session = _session_antigravity(authority)
    cache = _publish_cache_antigravity(cache_root, authority)
    with ExitStack() as stack:
        for patcher in _evaluation_patches_antigravity(
            cache_root, evaluation_root
        ):
            stack.enter_context(patcher)
        artifact = evaluation.evaluate_exp471_antigravity(
            session,
            cache,
            label_loader=lambda _datasets: _LABELS_ANTIGRAVITY,
        )
    return artifact


class ReportingTestsAntigravity(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(
            prefix="exp471_reporting_cli_antigravity_"
        )
        self.base = Path(self.temporary.name).resolve()
        self.artifact = _artifact_antigravity(self.base / "artifact_antigravity")
        self.report_root = self.base / "report_antigravity"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_two_argument_public_interface_reads_durable_worker_provenance(self) -> None:
        """The specified two-argument API must bind the reopened worker decision."""

        from experiments.exp471_exp467_evaluation_recovery_antigravity import (
            score_cache_antigravity,
        )

        status = {
            **_worker_provenance_antigravity(),
            "inner_numerical_threads": 1,
            "acceleration": {"selected_backend": "numba_cpu"},
        }
        with mock.patch.object(
            score_cache_antigravity,
            "status_score_cache_antigravity",
            return_value=status,
        ) as worker_status:
            authority = reporting.write_exp471_report_antigravity(
                self.artifact, self.report_root
            )
        self.assertRegex(authority.report_manifest_sha256, r"^[0-9a-f]{64}$")
        self.assertEqual(worker_status.call_count, 1)

    def test_report_closes_all_ten_methods_axes_tables_and_artifact_map(self) -> None:
        """Missing methods, swapped curve axes, or an unbound leaf must fail."""

        authority = reporting.write_exp471_report_antigravity(
            self.artifact,
            self.report_root,
            worker_provenance=_worker_provenance_antigravity(),
        )
        reopened = reporting.read_exp471_report_antigravity(
            self.artifact, self.report_root
        )
        self.assertEqual(reopened, authority)
        self.assertEqual(
            set(authority.report_file_sha256),
            {
                path.relative_to(self.report_root).as_posix()
                for path in self.report_root.rglob("*")
                if path.is_file()
                and path.name
                not in {
                    reporting._COMPLETE_MARKER_NAME_ANTIGRAVITY,
                    reporting._ARTIFACT_MAP_NAME_ANTIGRAVITY,
                }
            },
        )
        metrics = json.loads(
            (
                self.report_root
                / "exp450_report_artifacts_antigravity"
                / "exp450_metric_tables_antigravity.json"
            ).read_text(encoding="utf-8")
        )
        curves = json.loads(
            (
                self.report_root
                / "exp450_report_artifacts_antigravity"
                / "exp450_curve_tables_antigravity.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(tuple(metrics["method_order"]), METHOD_ORDER_ANTIGRAVITY)
        self.assertEqual(set(metrics["macro"]), set(METHOD_ORDER_ANTIGRAVITY))
        self.assertEqual(set(metrics["micro"]), set(METHOD_ORDER_ANTIGRAVITY))
        self.assertEqual(curves["roc"]["x_name"], "false_positive_rate")
        self.assertEqual(curves["roc"]["y_name"], "true_positive_rate")
        self.assertEqual(curves["precision_recall"]["x_name"], "recall")
        self.assertEqual(curves["precision_recall"]["y_name"], "precision")
        self.assertEqual(
            {row["method"] for row in curves["roc"]["rows"]},
            set(METHOD_ORDER_ANTIGRAVITY),
        )

    def test_changed_leaf_and_artifact_map_digest_are_rejected(self) -> None:
        """Changed report bytes or a rehashed foreign map cannot retain authority."""

        reporting.write_exp471_report_antigravity(
            self.artifact,
            self.report_root,
            worker_provenance=_worker_provenance_antigravity(),
        )
        leaf = (
            self.report_root
            / "exp450_report_artifacts_antigravity"
            / "exp450_metric_tables_antigravity.csv"
        )
        original_leaf = leaf.read_bytes()
        leaf.write_bytes(original_leaf + b"changed-public-literal\n")
        with self.assertRaisesRegex(ValueError, "leaf digest"):
            reporting.read_exp471_report_antigravity(
                self.artifact, self.report_root
            )
        leaf.write_bytes(original_leaf)

        artifact_map = self.report_root / reporting._ARTIFACT_MAP_NAME_ANTIGRAVITY
        value = json.loads(artifact_map.read_text(encoding="utf-8"))
        value["files"][next(iter(value["files"]))] = "f" * 64
        artifact_map.write_bytes(canonical_json_bytes_exp471_antigravity(value))
        with self.assertRaisesRegex(ValueError, "artifact map"):
            reporting.read_exp471_report_antigravity(
                self.artifact, self.report_root
            )

    def test_exact_reuse_preserves_every_inode_and_skips_renderer(self) -> None:
        """A second exact report call must not write or render any report leaf."""

        render_calls: list[str] = []

        def render(frozen, root):
            render_calls.append("render")
            return reporting.write_exp450_report_antigravity(frozen, root)

        first = reporting.write_exp471_report_antigravity(
            self.artifact,
            self.report_root,
            worker_provenance=_worker_provenance_antigravity(),
            renderer=render,
        )
        before = {
            path.relative_to(self.report_root).as_posix(): (
                path.stat().st_ino,
                path.read_bytes(),
            )
            for path in self.report_root.rglob("*")
            if path.is_file()
        }
        second = reporting.write_exp471_report_antigravity(
            self.artifact,
            self.report_root,
            worker_provenance=_worker_provenance_antigravity(),
            renderer=render,
        )
        after = {
            path.relative_to(self.report_root).as_posix(): (
                path.stat().st_ino,
                path.read_bytes(),
            )
            for path in self.report_root.rglob("*")
            if path.is_file()
        }
        self.assertEqual(first, second)
        self.assertEqual(before, after)
        self.assertEqual(render_calls, ["render"])

    def test_markerless_foreign_destination_and_symlinks_fail_closed(self) -> None:
        """A foreign reservation or linked destination must never be reclaimed."""

        self.report_root.mkdir()
        foreign = self.report_root / "foreign_antigravity.txt"
        foreign.write_bytes(b"foreign-public-literal\n")
        before = foreign.read_bytes()
        with self.assertRaisesRegex(ValueError, "markerless|foreign"):
            reporting.write_exp471_report_antigravity(
                self.artifact,
                self.report_root,
                worker_provenance=_worker_provenance_antigravity(),
            )
        self.assertEqual(foreign.read_bytes(), before)

        linked = self.base / "linked_report_antigravity"
        linked.symlink_to(self.report_root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            reporting.write_exp471_report_antigravity(
                self.artifact,
                linked,
                worker_provenance=_worker_provenance_antigravity(),
            )

    def test_fault_before_rename_is_invisible_and_after_rename_reopens(self) -> None:
        """Only a fully marker-bearing rename may become externally visible."""

        def before(stage: str) -> None:
            if stage == "before_rename":
                raise RuntimeError("public fault before rename")

        with self.assertRaisesRegex(RuntimeError, "before rename"):
            reporting.write_exp471_report_antigravity(
                self.artifact,
                self.report_root,
                worker_provenance=_worker_provenance_antigravity(),
                fault=before,
            )
        self.assertFalse(os.path.lexists(self.report_root))

        after_calls: list[str] = []

        def after(stage: str) -> None:
            after_calls.append(stage)
            if stage == "after_rename":
                raise RuntimeError("public fault after rename")

        authority = reporting.write_exp471_report_antigravity(
            self.artifact,
            self.report_root,
            worker_provenance=_worker_provenance_antigravity(),
            fault=after,
        )
        self.assertIn("after_rename", after_calls)
        self.assertEqual(
            authority,
            reporting.read_exp471_report_antigravity(
                self.artifact, self.report_root
            ),
        )

    def test_live_lock_rejects_dead_owner_recovers_and_symlink_fails(self) -> None:
        """PID reuse and unsafe lock records must not authorize lock removal."""

        lock = reporting._report_lock_path_antigravity(self.report_root)
        lock.parent.mkdir(parents=True, exist_ok=True)
        current = reporting._process_start_identity_antigravity(os.getpid())
        active = reporting._lock_record_antigravity(os.getpid(), current)
        lock.write_bytes(canonical_json_bytes_exp471_antigravity(active))
        with self.assertRaisesRegex(ValueError, "active"):
            reporting.write_exp471_report_antigravity(
                self.artifact,
                self.report_root,
                worker_provenance=_worker_provenance_antigravity(),
            )
        self.assertEqual(
            lock.read_bytes(), canonical_json_bytes_exp471_antigravity(active)
        )

        lock.write_bytes(
            canonical_json_bytes_exp471_antigravity(
                reporting._lock_record_antigravity(
                    99_999_999, "dead-public-process-antigravity"
                )
            )
        )
        reporting.write_exp471_report_antigravity(
            self.artifact,
            self.report_root,
            worker_provenance=_worker_provenance_antigravity(),
        )
        self.assertFalse(os.path.lexists(lock))

        other_root = self.base / "other_report_antigravity"
        other_lock = reporting._report_lock_path_antigravity(other_root)
        target = self.base / "foreign_lock_antigravity.json"
        target.write_bytes(b"{}\n")
        other_lock.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "lock"):
            reporting.write_exp471_report_antigravity(
                self.artifact,
                other_root,
                worker_provenance=_worker_provenance_antigravity(),
            )


class _LifecycleRuntimeAntigravity:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.cache_complete = False
        self.evaluation_complete = False
        self.report_complete = False
        self.authority = type(
            "AuthorityAntigravity",
            (),
            {"authority_sha256": "a" * 64, "inventory_sha256": "b" * 64},
        )()
        self.session = type(
            "SessionAntigravity", (), {"authority": self.authority}
        )()
        self.cache = type(
            "CacheAntigravity",
            (),
            {"manifest_sha256": "c" * 64, "total_shards": 10},
        )()
        self.artifact = type(
            "ArtifactAntigravity", (), {"evaluation_sha256": "d" * 64}
        )()
        self.report_authority = type(
            "ReportAuthorityAntigravity",
            (),
            {"report_manifest_sha256": "e" * 64},
        )()

    def verify_source_antigravity(self):
        self.calls.append("verify-source")
        return self.session

    def open_cache_antigravity(self, session):
        self.calls.append("open-cache")
        if not self.cache_complete:
            raise FileNotFoundError("public cache absent")
        return self.cache

    def benchmark_prepare_antigravity(self, session):
        self.calls.append("benchmark-prepare")
        return {"selected_workers": 1, "byte_parity": True}

    def prepare_antigravity(self, session, progress):
        self.calls.append("prepare")
        self.cache_complete = True
        progress(
            {
                "durable_completed": 10,
                "total": 10,
                "failures": 0,
                "remaining": 0,
            }
        )
        return self.cache

    def open_evaluation_antigravity(self, session, cache):
        self.calls.append("open-evaluation")
        if not self.evaluation_complete:
            raise FileNotFoundError("public evaluation absent")
        return self.artifact

    def evaluate_antigravity(self, session, cache):
        self.calls.append("evaluate")
        self.evaluation_complete = True
        return self.artifact

    def report_antigravity(self, artifact):
        self.calls.append("report")
        if not self.report_complete:
            self.calls.append("render")
            self.report_complete = True
        return self.report_authority

    def status_antigravity(self):
        self.calls.append("status")
        return {
            "durable_completed": 10 if self.cache_complete else 0,
            "total": 10,
            "failures": 0,
            "remaining": 0 if self.cache_complete else 10,
            "verified_stage": "report" if self.report_complete else "source",
            "paths": {"root": "/public/literal/exp471_antigravity"},
            "worker_decision": {"selected_workers": 1},
            "latest_progress": None,
            "failure": None,
        }


class CliTestsAntigravity(unittest.TestCase):
    def test_complete_cache_score_reader_skips_source_score_preparation(self) -> None:
        """Reopening terminal cache must not rebuild Exp467 scores or calibration input."""

        from experiments.exp471_exp467_evaluation_recovery_antigravity import (
            cli_antigravity as cli,
        )

        session = object()
        reopened = {"schema": "public-reopened-score-input-antigravity.v1"}
        with mock.patch.object(cli.os.path, "lexists", return_value=True), \
             mock.patch.object(
                 cli,
                 "_reopened_score_bundle_antigravity",
                 return_value=reopened,
                 create=True,
             ) as reopen, \
             mock.patch.object(
                 cli,
                 "_production_score_bundle_antigravity",
                 side_effect=AssertionError("source score preparation repeated"),
             ) as prepare:
            self.assertIs(cli._score_input_antigravity(session), reopened)
        self.assertEqual(reopen.call_count, 1)
        self.assertEqual(prepare.call_count, 0)

    def test_report_lifecycle_order_and_exact_second_call_skip_counts(self) -> None:
        """Resume must not repeat preparation, labels, metrics, or rendering."""

        runtime = _LifecycleRuntimeAntigravity()
        for _ in range(2):
            stdout, stderr = io.StringIO(), io.StringIO()
            self.assertEqual(
                main_antigravity(
                    ["report"],
                    runtime=runtime,
                    stdout=stdout,
                    stderr=stderr,
                ),
                0,
            )
            self.assertEqual(stderr.getvalue(), "")
            self.assertEqual(json.loads(stdout.getvalue())["command"], "report")
        self.assertEqual(
            runtime.calls,
            [
                "verify-source",
                "open-cache",
                "prepare",
                "open-evaluation",
                "evaluate",
                "report",
                "render",
                "verify-source",
                "open-cache",
                "open-evaluation",
                "report",
            ],
        )
        self.assertEqual(runtime.calls.count("prepare"), 1)
        self.assertEqual(runtime.calls.count("evaluate"), 1)
        self.assertEqual(runtime.calls.count("render"), 1)

    def test_fixed_commands_route_literal_json_and_status_starts_no_work(self) -> None:
        """Status is observational and every accepted command is a fixed literal."""

        for command in (
            "verify-source",
            "benchmark-prepare",
            "prepare",
            "status",
        ):
            runtime = _LifecycleRuntimeAntigravity()
            if command == "prepare":
                runtime.cache_complete = False
            stdout, stderr = io.StringIO(), io.StringIO()
            self.assertEqual(
                main_antigravity(
                    [command],
                    runtime=runtime,
                    stdout=stdout,
                    stderr=stderr,
                ),
                0,
            )
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["command"], command)
            self.assertEqual(stderr.getvalue(), "")
            if command == "status":
                self.assertEqual(runtime.calls, ["status"])

        runtime = _LifecycleRuntimeAntigravity()
        stdout, stderr = io.StringIO(), io.StringIO()
        self.assertEqual(
            main_antigravity(
                ["report", "--report-root", "/foreign"],
                runtime=runtime,
                stdout=stdout,
                stderr=stderr,
            ),
            2,
        )
        self.assertEqual(stdout.getvalue(), "")
        error = json.loads(stderr.getvalue())
        self.assertEqual(error["error"], "invalid_arguments")
        self.assertNotIn("/foreign", stderr.getvalue())
        self.assertEqual(runtime.calls, [])

    def test_runtime_error_is_one_concise_json_without_traceback_or_paths(self) -> None:
        """Normal CLI errors must reveal neither traceback nor sensitive details."""

        runtime = _LifecycleRuntimeAntigravity()
        runtime.verify_source_antigravity = mock.Mock(
            side_effect=ValueError("private /secret/path public literal failure")
        )
        stdout, stderr = io.StringIO(), io.StringIO()
        self.assertEqual(
            main_antigravity(
                ["verify-source"],
                runtime=runtime,
                stdout=stdout,
                stderr=stderr,
            ),
            2,
        )
        self.assertEqual(stdout.getvalue(), "")
        lines = stderr.getvalue().splitlines()
        self.assertEqual(len(lines), 1)
        error = json.loads(lines[0])
        self.assertEqual(error["error"], "command_failed")
        self.assertEqual(error["command"], "verify-source")
        self.assertEqual(error["exception_type"], "ValueError")
        self.assertNotIn("secret", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
