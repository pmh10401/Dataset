"""Public-literal lifecycle tests for Exp471 reporting and its closed CLI."""

from __future__ import annotations

from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import runpy
import subprocess
import tempfile
import threading
import unittest
from unittest import mock

from scratch import (
    run_exp471_exp467_evaluation_recovery_antigravity as exp471_launcher,
)

from experiments.exp471_exp467_evaluation_recovery_antigravity import (
    cli_antigravity as cli,
)
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
    ACCELERATION_ASSESSMENT_ANTIGRAVITY,
    CPU_WORKERS_ANTIGRAVITY,
    INNER_NUMERICAL_THREADS_ANTIGRAVITY,
    LOGICAL_CPU_COUNT_ANTIGRAVITY,
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


def _worker_provenance_antigravity(
    requested_workers: int = 1, *, single_worker_seconds: float = 0.0
) -> dict[str, object]:
    cache_context = "66854b3754856512a8a99e47582dc2a18000a29f403e1964ab669a87804605fb"
    fallback = (
        "parallel worker request is one; canonical CPU single-worker retained"
        if requested_workers == 1
        else "parallel speedup is below the required 1.1x gate"
    )
    benchmark_bare = {
        "schema": "exp471-score-preparation-benchmark-antigravity.v1",
        "cache_context_sha256": cache_context,
        "detected_logical_cpus": LOGICAL_CPU_COUNT_ANTIGRAVITY,
        "default_cpu_workers": CPU_WORKERS_ANTIGRAVITY,
        "requested_parallel_workers": requested_workers,
        "selected_workers": 1,
        "inner_numerical_threads": INNER_NUMERICAL_THREADS_ANTIGRAVITY,
        "single_worker_seconds": single_worker_seconds,
        "parallel_seconds": 0.0,
        "speedup_threshold": 1.1,
        "speedup": 1.0,
        "byte_parity": True,
        "parallel_accepted": False,
        "parallel_error_type": None,
        "parallel_error_message": None,
        "fallback_reason": fallback,
        "acceleration": dict(ACCELERATION_ASSESSMENT_ANTIGRAVITY),
    }
    benchmark = {
        **benchmark_bare,
        "decision_sha256": sha256(
            canonical_json_bytes_exp471_antigravity(benchmark_bare)
        ).hexdigest(),
    }
    event_bare = {
        "schema": "exp471-score-preparation-execution-event-antigravity.v1",
        "cache_context_sha256": cache_context,
        "benchmark_decision_sha256": benchmark["decision_sha256"],
        "sequence": 0,
        "prior_execution_sha256": None,
        "event_type": "selection",
        "wave": "initial",
        "selected_workers": 1,
        "error_type": None,
        "error_message": None,
        "fallback_reason": "",
    }
    event = {
        **event_bare,
        "execution_sha256": sha256(
            canonical_json_bytes_exp471_antigravity(event_bare)
        ).hexdigest(),
    }
    return {
        "schema": "exp471-worker-provenance-antigravity.v1",
        "detected_logical_cpus": LOGICAL_CPU_COUNT_ANTIGRAVITY,
        "default_cpu_workers": CPU_WORKERS_ANTIGRAVITY,
        "inner_numerical_threads": INNER_NUMERICAL_THREADS_ANTIGRAVITY,
        "acceleration": dict(ACCELERATION_ASSESSMENT_ANTIGRAVITY),
        "benchmark_decision": benchmark,
        "execution_decision": {
            "schema": "exp471-score-preparation-execution-antigravity.v1",
            "cache_context_sha256": cache_context,
            "benchmark_decision_sha256": benchmark["decision_sha256"],
            "selected_workers": 1,
            "fallback_occurred": False,
            "events": [event],
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


def _rehash_report_leaf_antigravity(
    root: Path, relative: str, payload: bytes
) -> None:
    """Rehash one public leaf through both outer authority layers."""

    (root / relative).write_bytes(payload)
    artifact_map_path = root / reporting._ARTIFACT_MAP_NAME_ANTIGRAVITY
    artifact_map = json.loads(artifact_map_path.read_text(encoding="utf-8"))
    artifact_map["files"][relative] = sha256(payload).hexdigest()
    artifact_map_raw = canonical_json_bytes_exp471_antigravity(artifact_map)
    artifact_map_path.write_bytes(artifact_map_raw)
    marker_path = root / reporting._COMPLETE_MARKER_NAME_ANTIGRAVITY
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["report_file_sha256"] = artifact_map["files"]
    marker["report_manifest_sha256"] = sha256(artifact_map_raw).hexdigest()
    marker_path.write_bytes(canonical_json_bytes_exp471_antigravity(marker))


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

        status = _worker_provenance_antigravity()
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

    def test_live_pid_with_different_start_identity_recovers_as_pid_reuse(self) -> None:
        """A live reused PID must not preserve a lock from another process lifetime."""

        lock = reporting._report_lock_path_antigravity(self.report_root)
        lock.parent.mkdir(parents=True, exist_ok=True)
        actual = reporting._process_start_identity_antigravity(os.getpid())
        self.assertIsNotNone(actual)
        lock.write_bytes(
            canonical_json_bytes_exp471_antigravity(
                reporting._lock_record_antigravity(
                    os.getpid(), f"different-{actual}-public-antigravity"
                )
            )
        )
        authority = reporting.write_exp471_report_antigravity(
            self.artifact,
            self.report_root,
            worker_provenance=_worker_provenance_antigravity(),
        )
        self.assertRegex(authority.report_manifest_sha256, r"^[0-9a-f]{64}$")
        self.assertFalse(os.path.lexists(lock))

    def test_pre_rename_stage_replacement_is_never_published(self) -> None:
        """Replacing the verified staging name must not publish foreign bytes."""

        saved = self.base / "owned_stage_saved_antigravity"
        foreign = b"foreign-stage-public-literal-antigravity\n"

        def swap(stage_name: str) -> None:
            if stage_name != "before_rename":
                return
            stage = next(self.base.glob(".report_antigravity.stage_antigravity_*"))
            stage.rename(saved)
            stage.mkdir()
            (stage / "foreign_antigravity.txt").write_bytes(foreign)

        with self.assertRaisesRegex(ValueError, "staging|identity"):
            reporting.write_exp471_report_antigravity(
                self.artifact,
                self.report_root,
                worker_provenance=_worker_provenance_antigravity(),
                fault=swap,
            )
        self.assertFalse(os.path.lexists(self.report_root))
        replacement = next(
            self.base.glob(".report_antigravity.stage_antigravity_*")
        )
        self.assertEqual((replacement / "foreign_antigravity.txt").read_bytes(), foreign)
        self.assertTrue(saved.is_dir())

    def test_failed_stage_cleanup_preserves_namespace_replacement(self) -> None:
        """Failure cleanup must never recursively delete a replacement directory."""

        saved = self.base / "owned_failed_stage_saved_antigravity"
        foreign = b"foreign-cleanup-public-literal-antigravity\n"

        def swap_and_fail(stage_name: str) -> None:
            if stage_name != "before_rename":
                return
            stage = next(self.base.glob(".report_antigravity.stage_antigravity_*"))
            stage.rename(saved)
            stage.mkdir()
            (stage / "foreign_antigravity.txt").write_bytes(foreign)
            raise RuntimeError("public stage cleanup fault")

        with self.assertRaisesRegex(RuntimeError, "cleanup fault"):
            reporting.write_exp471_report_antigravity(
                self.artifact,
                self.report_root,
                worker_provenance=_worker_provenance_antigravity(),
                fault=swap_and_fail,
            )
        replacement = next(
            self.base.glob(".report_antigravity.stage_antigravity_*")
        )
        self.assertEqual((replacement / "foreign_antigravity.txt").read_bytes(), foreign)
        self.assertTrue(saved.is_dir())

    def test_lock_release_revalidates_identity_after_verified_read(self) -> None:
        """Normal release must not unlink a foreign lock pathname replacement."""

        token = reporting._acquire_report_lock_antigravity(self.report_root)
        owned = self.base / "owned_lock_saved_antigravity.json"
        foreign = b"foreign-lock-public-literal-antigravity\n"
        original = reporting._read_lock_antigravity

        def swap_after_read(path):
            result = original(path)
            path.rename(owned)
            path.write_bytes(foreign)
            return result

        with mock.patch.object(
            reporting, "_read_lock_antigravity", side_effect=swap_after_read
        ):
            with self.assertRaisesRegex(ValueError, "lock.*identity|changed"):
                reporting._release_report_lock_antigravity(token)
        self.assertEqual(token.path.read_bytes(), foreign)
        self.assertEqual(owned.read_bytes(), token.payload)

    def test_dead_lock_quarantine_cleanup_revalidates_moved_identity(self) -> None:
        """Dead-lock recovery must preserve a foreign quarantine replacement."""

        lock = reporting._report_lock_path_antigravity(self.report_root)
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.write_bytes(
            canonical_json_bytes_exp471_antigravity(
                reporting._lock_record_antigravity(
                    99_999_999, "dead-public-process-antigravity"
                )
            )
        )
        owned = self.base / "owned_dead_lock_saved_antigravity.json"
        foreign = b"foreign-dead-lock-public-literal-antigravity\n"
        original = reporting._read_lock_antigravity

        def swap_quarantine_after_read(path):
            result = original(path)
            if path.name.endswith(".dead_antigravity"):
                path.rename(owned)
                path.write_bytes(foreign)
            return result

        with mock.patch.object(
            reporting,
            "_read_lock_antigravity",
            side_effect=swap_quarantine_after_read,
        ):
            with self.assertRaisesRegex(ValueError, "dead lock.*identity|changed"):
                reporting._acquire_report_lock_antigravity(self.report_root)
        quarantine = next(self.base.glob(".*.dead_antigravity"))
        self.assertEqual(quarantine.read_bytes(), foreign)
        self.assertTrue(owned.is_file())

    def test_parent_replacement_fails_before_publication_or_cleanup(self) -> None:
        """A replaced parent must strand owned objects instead of mutating the new parent."""

        moved_parent = self.base.with_name(f"{self.base.name}_moved_antigravity")

        def restore_parent() -> None:
            if not moved_parent.exists():
                return
            if self.base.exists():
                self.base.rmdir()
            moved_parent.rename(self.base)

        self.addCleanup(restore_parent)

        def replace_parent(stage_name: str) -> None:
            if stage_name != "before_rename":
                return
            self.base.rename(moved_parent)
            self.base.mkdir()

        with self.assertRaisesRegex(ValueError, "parent.*changed|identity"):
            reporting.write_exp471_report_antigravity(
                self.artifact,
                self.report_root,
                worker_provenance=_worker_provenance_antigravity(),
                fault=replace_parent,
            )
        self.assertEqual(tuple(self.base.iterdir()), ())
        self.assertTrue(moved_parent.is_dir())

    def test_concurrent_identical_publishers_render_once_and_reuse_winner(self) -> None:
        """An identical concurrent loser must reopen the exact winner without rendering."""

        rendering = threading.Event()
        release = threading.Event()
        render_calls: list[str] = []

        def render(frozen, root):
            render_calls.append("render")
            rendering.set()
            self.assertTrue(release.wait(timeout=5))
            return reporting.write_exp450_report_antigravity(frozen, root)

        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(
                reporting.write_exp471_report_antigravity,
                self.artifact,
                self.report_root,
                worker_provenance=_worker_provenance_antigravity(),
                renderer=render,
            )
            self.assertTrue(rendering.wait(timeout=5))
            second = pool.submit(
                reporting.write_exp471_report_antigravity,
                self.artifact,
                self.report_root,
                worker_provenance=_worker_provenance_antigravity(),
                renderer=render,
            )
            release.set()
            first_authority = first.result(timeout=10)
            second_authority = second.result(timeout=10)
        self.assertEqual(first_authority, second_authority)
        self.assertEqual(render_calls, ["render"])

    def test_concurrent_different_provenance_rejects_loser_without_render(self) -> None:
        """A concurrent loser with different provenance must not reuse the winner."""

        rendering = threading.Event()
        release = threading.Event()
        render_calls: list[str] = []

        def render(frozen, root):
            render_calls.append("render")
            rendering.set()
            self.assertTrue(release.wait(timeout=5))
            return reporting.write_exp450_report_antigravity(frozen, root)

        different = _worker_provenance_antigravity(single_worker_seconds=0.125)
        self.assertNotEqual(different, _worker_provenance_antigravity())
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(
                reporting.write_exp471_report_antigravity,
                self.artifact,
                self.report_root,
                worker_provenance=_worker_provenance_antigravity(),
                renderer=render,
            )
            self.assertTrue(rendering.wait(timeout=5))
            second = pool.submit(
                reporting.write_exp471_report_antigravity,
                self.artifact,
                self.report_root,
                worker_provenance=different,
                renderer=render,
            )
            release.set()
            first.result(timeout=10)
            with self.assertRaisesRegex(ValueError, "provenance"):
                second.result(timeout=10)
        self.assertEqual(render_calls, ["render"])

    def test_rehashed_scientific_leaf_tampering_is_rejected(self) -> None:
        """Every frozen format must remain the exact Task 6 representation."""

        nested = "exp450_report_artifacts_antigravity"

        def json_mutation(relative: str, mutate) -> bytes:
            value = json.loads((root / relative).read_text(encoding="utf-8"))
            mutate(value)
            return canonical_json_bytes_exp471_antigravity(value)

        mutations = {
            f"{nested}/exp450_metric_tables_antigravity.json": lambda root: json_mutation(
                f"{nested}/exp450_metric_tables_antigravity.json",
                lambda value: value["method_order"].pop(),
            ),
            f"{nested}/exp450_curve_tables_antigravity.json": lambda root: json_mutation(
                f"{nested}/exp450_curve_tables_antigravity.json",
                lambda value: value["roc"].update(
                    {"x_name": "true_positive_rate", "y_name": "false_positive_rate"}
                ),
            ),
            f"{nested}/exp450_paired_differences_antigravity.json": lambda root: json_mutation(
                f"{nested}/exp450_paired_differences_antigravity.json",
                lambda value: value["paired_statistics"][0].update(
                    {"point_estimate": 0.125}
                ),
            ),
            f"{nested}/exp450_report_manifest_antigravity.json": lambda root: canonical_json_bytes_exp471_antigravity(
                {
                    **json.loads(
                        (
                            root
                            / nested
                            / "exp450_report_manifest_antigravity.json"
                        ).read_text(encoding="utf-8")
                    ),
                    "exp450_metric_tables_antigravity.json": "f" * 64,
                }
            ),
            f"{nested}/exp450_metric_tables_antigravity.csv": lambda root: (
                root / nested / "exp450_metric_tables_antigravity.csv"
            ).read_bytes()
            + b"foreign,metric,csv,public,literal\n",
            f"{nested}/exp450_curve_tables_antigravity.csv": lambda root: (
                root / nested / "exp450_curve_tables_antigravity.csv"
            ).read_bytes()
            + b"foreign,curve,csv,public,literal\n",
            f"{nested}/exp450_paired_differences_antigravity.csv": lambda root: (
                root / nested / "exp450_paired_differences_antigravity.csv"
            ).read_bytes()
            + b"foreign,paired,csv,public,literal\n",
            f"{nested}/exp450_curves_antigravity.svg": lambda root: (
                root / nested / "exp450_curves_antigravity.svg"
            ).read_bytes().replace(b"roc", b"bad", 1),
            f"{nested}/exp450_curves_antigravity.png": lambda root: (
                root / nested / "exp450_curves_antigravity.png"
            ).read_bytes()[:-1]
            + b"X",
            f"{nested}/exp450_paper_report_antigravity.html": lambda root: (
                root / nested / "exp450_paper_report_antigravity.html"
            ).read_bytes().replace(b"continuous", b"fabricated", 1),
        }
        for index, (relative, mutate) in enumerate(mutations.items()):
            with self.subTest(relative=relative):
                root = self.base / f"semantic_{index}_report_antigravity"
                reporting.write_exp471_report_antigravity(
                    self.artifact,
                    root,
                    worker_provenance=_worker_provenance_antigravity(),
                )
                _rehash_report_leaf_antigravity(root, relative, mutate(root))
                with self.assertRaisesRegex(ValueError, "frozen|semantic|leaf"):
                    reporting.read_exp471_report_antigravity(self.artifact, root)

    def test_rehashed_complete_provenance_tampering_is_rejected(self) -> None:
        """Every environment, worker, closure, axis, and digest field is authoritative."""

        mutations = {
            "worker": lambda value: value["worker_provenance"].update(
                {"detected_logical_cpus": 999}
            ),
            "interpreter": lambda value: value["interpreter"].update(
                {"version": "0.0-public-antigravity"}
            ),
            "dependency": lambda value: value["dependencies"].update(
                {"numpy": "0-public-antigravity"}
            ),
            "axes": lambda value: value["curve_axes"]["roc"].update(
                {"x": "true_positive_rate"}
            ),
            "evaluation": lambda value: value["evaluation_implementation"].update(
                {"scientific_contract": "foreign-public-antigravity"}
            ),
            "rendered": lambda value: value["rendered_leaf_sha256"].update(
                {next(iter(value["rendered_leaf_sha256"])): "f" * 64}
            ),
        }
        for index, (name, mutate) in enumerate(mutations.items()):
            with self.subTest(name=name):
                root = self.base / f"provenance_{index}_report_antigravity"
                reporting.write_exp471_report_antigravity(
                    self.artifact,
                    root,
                    worker_provenance=_worker_provenance_antigravity(),
                )
                relative = reporting._PROVENANCE_NAME_ANTIGRAVITY
                value = json.loads((root / relative).read_text(encoding="utf-8"))
                mutate(value)
                _rehash_report_leaf_antigravity(
                    root,
                    relative,
                    canonical_json_bytes_exp471_antigravity(value),
                )
                with self.assertRaisesRegex(ValueError, "provenance"):
                    reporting.read_exp471_report_antigravity(self.artifact, root)

    def test_rehashed_method_metric_and_coordinate_mutations_are_rejected(self) -> None:
        """Order, numeric metrics, and paired curve coordinates share one authority."""

        nested = "exp450_report_artifacts_antigravity"
        cases = (
            (
                "reordered-methods",
                f"{nested}/exp450_metric_tables_antigravity.json",
                lambda value: value["method_order"].reverse(),
            ),
            (
                "changed-metric",
                f"{nested}/exp450_metric_tables_antigravity.json",
                lambda value: value["macro"][METHOD_ORDER_ANTIGRAVITY[0]].update(
                    {"average_precision": 0.125}
                ),
            ),
            (
                "swapped-roc-coordinate",
                f"{nested}/exp450_curve_tables_antigravity.json",
                lambda value: value["roc"]["rows"][1].update(
                    {
                        "false_positive_rate": value["roc"]["rows"][1][
                            "true_positive_rate"
                        ],
                        "true_positive_rate": value["roc"]["rows"][1][
                            "false_positive_rate"
                        ],
                    }
                ),
            ),
        )
        for index, (name, relative, mutate) in enumerate(cases):
            with self.subTest(name=name):
                root = self.base / f"scientific_{index}_report_antigravity"
                reporting.write_exp471_report_antigravity(
                    self.artifact,
                    root,
                    worker_provenance=_worker_provenance_antigravity(),
                )
                value = json.loads((root / relative).read_text(encoding="utf-8"))
                mutate(value)
                _rehash_report_leaf_antigravity(
                    root,
                    relative,
                    canonical_json_bytes_exp471_antigravity(value),
                )
                with self.assertRaisesRegex(ValueError, "frozen|semantic|leaf"):
                    reporting.read_exp471_report_antigravity(self.artifact, root)

    def test_partial_worker_provenance_is_rejected_before_rendering(self) -> None:
        """A self-described worker mapping cannot replace Task 5 decisions."""

        with self.assertRaisesRegex(ValueError, "worker provenance"):
            reporting.write_exp471_report_antigravity(
                self.artifact,
                self.report_root,
                worker_provenance={
                    "benchmark_decision": {"selected_workers": 1},
                    "execution_decision": {"selected_workers": 1},
                },
            )
        self.assertFalse(os.path.lexists(self.report_root))


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
    def test_codemap_contains_every_tracked_exp471_operator_entrypoint(self) -> None:
        """A tracked Exp471 launcher must be fingerprinted, mapped, and linked to CLI."""

        repository = Path(__file__).resolve().parents[3]
        generator_path = repository / "scratch/generate_exp471_codemap_antigravity.py"
        generator = runpy.run_path(str(generator_path))
        paths = generator["tracked_existing_paths_antigravity"](repository)
        relative = {
            path.relative_to(repository).as_posix()
            for path in paths
        }
        tracked = subprocess.run(
            (
                "git",
                "-C",
                str(repository),
                "ls-files",
                "scratch/run_exp471*_antigravity.py",
            ),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        self.assertTrue(tracked)
        self.assertTrue(set(tracked).issubset(relative))
        callers = generator["ast_callers_antigravity"](paths)
        payload = generator["build_codemap_payload_antigravity"](paths, callers)
        launcher = Path(exp471_launcher.__file__).relative_to(repository).as_posix()
        cli_path = (
            "experiments/exp471_exp467_evaluation_recovery_antigravity/"
            "cli_antigravity.py"
        )
        nodes = {node["path"]: node for node in payload["nodes"]}
        self.assertIn(launcher, nodes)
        self.assertTrue(nodes[launcher]["effects"])
        self.assertIn(
            Path(__file__).relative_to(repository).as_posix(),
            nodes[launcher]["tests"],
        )
        self.assertIn(
            {"caller": launcher, "callee": cli_path, "kind": "ast_import"},
            payload["edges"],
        )
        self.assertTrue(
            any(
                flow.get("from") == launcher and flow.get("to") == "paper report manifest"
                for flow in payload["flows"]
            )
        )

    def test_progress_rejects_noncanonical_foreign_and_nonfinite_history(self) -> None:
        """Status must never return malformed history or an arbitrary secret field."""

        with tempfile.TemporaryDirectory(
            prefix="exp471_progress_schema_antigravity_"
        ) as directory:
            root = Path(directory).resolve()
            path = root / cli._PROGRESS_NAME_ANTIGRAVITY
            invalid = (
                b'{ "schema": "foreign", "secret": "/public/literal" }\n',
                b'{"schema":"foreign","secret":"public-literal-antigravity"}\n',
                b'{"schema":"exp471-progress-antigravity.v1","eta_seconds":NaN}\n',
                b'{"schema":"exp471-progress-antigravity.v1"',
            )
            with mock.patch.object(cli, "EXP471_ROOT_ANTIGRAVITY", root):
                for raw in invalid:
                    with self.subTest(raw=raw):
                        path.write_bytes(raw)
                        with self.assertRaisesRegex(
                            ValueError,
                            "progress.*malformed|noncanonical|schema|truncated",
                        ):
                            cli._latest_progress_antigravity()
                path.unlink()
                callback = cli._progress_callback_antigravity(
                    "score-cache", root / "cache_antigravity"
                )
                callback(
                    {
                        "durable_completed": 1,
                        "total": 10,
                        "status": "durable",
                    }
                )
                value = json.loads(path.read_text(encoding="utf-8"))
                value["secret"] = "public-literal-secret-antigravity"
                path.write_bytes(canonical_json_bytes_exp471_antigravity(value))
                with self.assertRaisesRegex(ValueError, "progress.*schema"):
                    cli._latest_progress_antigravity()

    def test_progress_rejects_hardlinks_symlinks_and_path_swap(self) -> None:
        """Append/read must bind the single-link log inode that occupies its name."""

        with tempfile.TemporaryDirectory(
            prefix="exp471_progress_identity_antigravity_"
        ) as directory:
            root = Path(directory).resolve()
            path = root / cli._PROGRESS_NAME_ANTIGRAVITY
            target = root / "foreign_progress_target_antigravity.jsonl"
            target.write_bytes(b"foreign-public-literal-antigravity\n")
            with mock.patch.object(cli, "EXP471_ROOT_ANTIGRAVITY", root):
                os.link(target, path)
                with self.assertRaisesRegex(ValueError, "progress log.*unsafe"):
                    cli._latest_progress_antigravity()
                callback = cli._progress_callback_antigravity(
                    "score-cache", root / "cache_antigravity"
                )
                with self.assertRaisesRegex(ValueError, "progress log.*unsafe"):
                    callback(
                        {
                            "durable_completed": 1,
                            "total": 10,
                            "status": "durable",
                        }
                    )
                path.unlink()
                path.symlink_to(target)
                with self.assertRaisesRegex(ValueError, "progress log.*unsafe"):
                    cli._latest_progress_antigravity()
                path.unlink()

                callback = cli._progress_callback_antigravity(
                    "score-cache", root / "cache_antigravity"
                )
                original_flock = cli.fcntl.flock
                saved = root / "owned_progress_saved_antigravity.jsonl"
                swapped = False

                def swap_after_open(descriptor, operation):
                    nonlocal swapped
                    result = original_flock(descriptor, operation)
                    if operation == cli.fcntl.LOCK_EX and not swapped:
                        swapped = True
                        path.rename(saved)
                        path.write_bytes(b"foreign-public-literal-antigravity\n")
                    return result

                with mock.patch.object(
                    cli.fcntl, "flock", side_effect=swap_after_open
                ):
                    with self.assertRaisesRegex(ValueError, "progress log.*changed"):
                        callback(
                            {
                                "durable_completed": 1,
                                "total": 10,
                                "status": "durable",
                            }
                        )
                self.assertEqual(
                    path.read_bytes(), b"foreign-public-literal-antigravity\n"
                )

    def test_progress_preserves_failure_attempt_history_and_unknown_eta(self) -> None:
        """A new invocation must continue cumulative attempts/failures truthfully."""

        with tempfile.TemporaryDirectory(
            prefix="exp471_progress_history_antigravity_"
        ) as directory:
            root = Path(directory).resolve()
            artifact = root / "cache_antigravity"
            with mock.patch.object(cli, "EXP471_ROOT_ANTIGRAVITY", root):
                first = cli._progress_callback_antigravity("score-cache", artifact)
                first(
                    {
                        "durable_completed": 0,
                        "total": 10,
                        "status": "failed",
                    }
                )
                first_event = cli._latest_progress_antigravity()
                self.assertEqual(first_event["attempts"], 1)
                self.assertEqual(first_event["failures"], 1)
                self.assertIsNone(first_event["eta_seconds"])

                resumed = cli._progress_callback_antigravity("score-cache", artifact)
                resumed(
                    {
                        "durable_completed": 1,
                        "total": 10,
                        "status": "durable",
                    }
                )
                latest = cli._latest_progress_antigravity()
                self.assertEqual(latest["attempts"], 2)
                self.assertEqual(latest["failures"], 1)
                self.assertEqual(latest["remaining"], 9)
                self.assertNotIn("secret", latest)

    def test_concurrent_progress_appends_are_canonical_and_cumulative(self) -> None:
        """Concurrent callbacks must produce whole ordered records without lost attempts."""

        with tempfile.TemporaryDirectory(
            prefix="exp471_progress_concurrent_antigravity_"
        ) as directory:
            root = Path(directory).resolve()
            artifact = root / "cache_antigravity"
            with mock.patch.object(cli, "EXP471_ROOT_ANTIGRAVITY", root):
                callbacks = [
                    cli._progress_callback_antigravity("score-cache", artifact)
                    for _ in range(8)
                ]
                with ThreadPoolExecutor(max_workers=8) as pool:
                    futures = [
                        pool.submit(
                            callback,
                            {
                                "durable_completed": 1,
                                "total": 10,
                                "status": "durable",
                            },
                        )
                        for callback in callbacks
                    ]
                    for future in futures:
                        future.result(timeout=10)
                path = root / cli._PROGRESS_NAME_ANTIGRAVITY
                records = [json.loads(line) for line in path.read_bytes().splitlines()]
                self.assertEqual(len(records), 8)
                self.assertEqual([row["attempts"] for row in records], list(range(1, 9)))
                self.assertTrue(
                    all(
                        canonical_json_bytes_exp471_antigravity(row).rstrip(b"\n")
                        == line
                        for row, line in zip(records, path.read_bytes().splitlines())
                    )
                )

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
