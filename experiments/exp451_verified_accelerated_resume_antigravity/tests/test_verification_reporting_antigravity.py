"""Post-completion gates and frozen Exp451 reporting definitions."""

from __future__ import annotations

from contextlib import contextmanager
from types import MappingProxyType
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import importlib.util
import io
from dataclasses import replace
from hashlib import sha256
import json
import subprocess
import sys
import unittest
from unittest.mock import patch


_VERIFICATION_CACHE_ANTIGRAVITY = None
_LIVE_VERIFICATION_FIXTURE_ANTIGRAVITY = None


def _live_verification_fixture_antigravity():
    """Build retained canonical checkpoint leaves for production-path gate tests."""
    global _LIVE_VERIFICATION_FIXTURE_ANTIGRAVITY
    if _LIVE_VERIFICATION_FIXTURE_ANTIGRAVITY is not None:
        return _LIVE_VERIFICATION_FIXTURE_ANTIGRAVITY
    from experiments.exp451_verified_accelerated_resume_antigravity import config_antigravity
    from experiments.exp451_verified_accelerated_resume_antigravity import verifier_antigravity as verifier
    from experiments.exp451_verified_accelerated_resume_antigravity.artifacts_antigravity import (
        SUCCESS_SCHEMA_ANTIGRAVITY,
        success_relative_path_antigravity,
    )
    from experiments.exp451_verified_accelerated_resume_antigravity.contracts_antigravity import (
        Exp451BackendDecision,
        Exp451MigrationEntry,
        Exp451MigrationManifest,
        Exp451RunContext,
        canonical_json_bytes,
    )
    from experiments.exp451_verified_accelerated_resume_antigravity.runner_antigravity import (
        _NATIVE_ORIGIN_ANTIGRAVITY,
        _NATIVE_SCHEMA_ANTIGRAVITY,
        _native_relative_path_antigravity,
    )

    temporary = TemporaryDirectory(prefix="exp451_full_fixture_antigravity_")
    root = Path(temporary.name).resolve()
    database = root / "fixture_database_antigravity.sqlite"
    database.write_bytes(b"fixture database")
    database_sha256 = sha256(database.read_bytes()).hexdigest()
    frozen = SimpleNamespace(database_path=database, database_sha256=database_sha256)
    datasets = [f"fixture_{index:04d}_antigravity" for index in range(1117)]
    tests = [
        {"dataset": name, "count": 600 if index == 1116 else 175}
        for index, name in enumerate(datasets)
    ]
    entries = [
        {"dataset": name, "train": {"count": 0}, "test": test}
        for name, test in zip(datasets, tests, strict=True)
    ]
    catalog = {
        "mode": "full",
        "database": {"sha256": database_sha256},
        "datasets": entries,
        "catalog_sha256": "b" * 64,
    }
    backend = Exp451BackendDecision("cpu", "fixture", False, False, 12, 11)
    context = Exp451RunContext(
        "full", root, database_sha256, catalog["catalog_sha256"],
        {"fixture_antigravity.py": "c" * 64}, backend, "d" * 64,
    )
    payload = {"sources": {}, "route": {}}
    payload_sha256 = sha256(canonical_json_bytes(payload)).hexdigest()
    migration_rows = []
    for condition, count in (("A", 1117), ("B", 1117), ("C", 1000), ("D", 1000)):
        for entry in entries[:count]:
            dataset = entry["dataset"]
            source_sha256 = sha256(f"source:{dataset}:{condition}".encode("utf-8")).hexdigest()
            envelope = {
                "schema": SUCCESS_SCHEMA_ANTIGRAVITY,
                "context_sha256": context.sha256,
                "source_context_sha256": config_antigravity.EXP450_SOURCE_CONTEXT_SHA256,
                "catalog_sha256": context.catalog_sha256,
                "condition_id": condition,
                "dataset": dataset,
                "raw_identity": entry,
                "backend": context.backend.artifact_record(),
                "execution_origin": "imported_exp450_mps",
                "payload": payload,
                "payload_sha256": payload_sha256,
                "source_envelope_sha256": source_sha256,
            }
            raw = canonical_json_bytes(envelope)
            relative = success_relative_path_antigravity(dataset, condition)
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            migration_rows.append(Exp451MigrationEntry(
                dataset, condition,
                f"source_antigravity/{condition}/{source_sha256}_antigravity.json",
                source_sha256, payload_sha256, relative, sha256(raw).hexdigest(),
            ))
    for condition in ("C", "D"):
        for entry in entries[1000:]:
            dataset = entry["dataset"]
            envelope = {
                "schema": _NATIVE_SCHEMA_ANTIGRAVITY,
                "context_sha256": context.sha256,
                "catalog_sha256": context.catalog_sha256,
                "condition_id": condition,
                "dataset": dataset,
                "raw_identity": entry,
                "backend": context.backend.artifact_record(),
                "execution_origin": _NATIVE_ORIGIN_ANTIGRAVITY,
                "payload": payload,
                "payload_sha256": payload_sha256,
            }
            path = root / _native_relative_path_antigravity(dataset, condition)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(canonical_json_bytes(envelope))
    manifest = Exp451MigrationManifest(
        config_antigravity.EXP450_SOURCE_CONTEXT_SHA256,
        tuple(migration_rows), context.sha256,
    )
    _LIVE_VERIFICATION_FIXTURE_ANTIGRAVITY = SimpleNamespace(
        temporary=temporary, verifier=verifier, root=root, frozen=frozen,
        catalog=catalog, context=context, manifest=manifest, datasets=tuple(datasets),
    )
    return _LIVE_VERIFICATION_FIXTURE_ANTIGRAVITY


@contextmanager
def _live_verification_environment_antigravity():
    fixture = _live_verification_fixture_antigravity()
    with (
        patch.object(fixture.verifier, "OFFICIAL_ROOT", fixture.root),
        patch.object(fixture.verifier, "FROZEN_EXP451_CONFIG", fixture.frozen),
        patch.object(fixture.verifier, "_typed_payload_antigravity"),
    ):
        yield fixture


def tearDownModule() -> None:
    global _LIVE_VERIFICATION_FIXTURE_ANTIGRAVITY
    if _LIVE_VERIFICATION_FIXTURE_ANTIGRAVITY is not None:
        _LIVE_VERIFICATION_FIXTURE_ANTIGRAVITY.temporary.cleanup()
        _LIVE_VERIFICATION_FIXTURE_ANTIGRAVITY = None


def _active_report_lock_owner_antigravity(path: Path) -> subprocess.Popen[str]:
    script = (
        "import fcntl, os, sys\n"
        "descriptor = os.open(sys.argv[1], os.O_CREAT | os.O_RDWR, 0o600)\n"
        "fcntl.flock(descriptor, fcntl.LOCK_EX)\n"
        "print('ready_antigravity', flush=True)\n"
        "sys.stdin.buffer.read(1)\n"
    )
    process = subprocess.Popen(
        (sys.executable, "-c", script, str(path)),
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    if process.stdout.readline().strip() != "ready_antigravity":
        stderr = process.stderr.read() if process.stderr is not None else ""
        process.kill()
        process.wait(timeout=5)
        raise AssertionError(f"report lock owner failed to start: {stderr}")
    return process


def _stop_report_lock_owner_antigravity(process: subprocess.Popen[str]) -> None:
    if process.poll() is None:
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is not None and not stream.closed:
            stream.close()


def _verification_antigravity(*, total: int = 4468, manifests: tuple[str, ...] = ("A", "B", "C", "D"), native_c: int = 117):
    """Build one real 4,468-leaf fixture and invoke the production verifier."""
    global _VERIFICATION_CACHE_ANTIGRAVITY
    if total == 4468 and manifests == ("A", "B", "C", "D") and native_c == 117 and _VERIFICATION_CACHE_ANTIGRAVITY is not None:
        return _VERIFICATION_CACHE_ANTIGRAVITY
    with _live_verification_environment_antigravity() as fixture:
        authority = fixture.verifier.verify_exp451_completion_antigravity(
            fixture.root, context=fixture.context, catalog=fixture.catalog,
            migration_manifest=fixture.manifest,
        )
    if total != 4468 or manifests != ("A", "B", "C", "D") or native_c != 117:
        return replace(authority, total_successes=total)
    _VERIFICATION_CACHE_ANTIGRAVITY = authority
    return authority


def _persisted_verification_antigravity():
    """Publish and reopen all four fixed-root leaves for the cached real authority."""
    from experiments.exp451_verified_accelerated_resume_antigravity.verifier_antigravity import (
        read_persisted_exp451_condition_manifests_antigravity,
        write_exp451_condition_manifests_antigravity,
    )
    verification = _verification_antigravity()
    with _live_verification_environment_antigravity() as fixture:
        write_exp451_condition_manifests_antigravity(fixture.root, verification)
        read_persisted_exp451_condition_manifests_antigravity(fixture.root, verification)
    return verification


class VerificationGateTests(unittest.TestCase):
    def test_direct_verifier_cannot_skip_persisted_manifest_reopen_before_labels(self) -> None:
        """A fresh verifier capability is not the same object whose four leaves reopened."""
        from experiments.exp451_verified_accelerated_resume_antigravity.config_antigravity import (
            EVALUATION_ROOT,
        )
        from experiments.exp451_verified_accelerated_resume_antigravity.evaluation_antigravity import (
            evaluate_exp451_antigravity,
        )
        from experiments.exp451_verified_accelerated_resume_antigravity.verifier_antigravity import (
            build_exp451_score_manifests_antigravity,
            read_persisted_exp451_condition_manifests_antigravity,
            write_exp451_condition_manifests_antigravity,
        )

        with _live_verification_environment_antigravity() as fixture:
            persisted = fixture.verifier.verify_exp451_completion_antigravity(
                fixture.root, context=fixture.context, catalog=fixture.catalog,
                migration_manifest=fixture.manifest,
            )
            write_exp451_condition_manifests_antigravity(fixture.root, persisted)
            read_persisted_exp451_condition_manifests_antigravity(fixture.root, persisted)
            direct = fixture.verifier.verify_exp451_completion_antigravity(
                fixture.root, context=fixture.context, catalog=fixture.catalog,
                migration_manifest=fixture.manifest,
            )
            with self.assertRaisesRegex(ValueError, "reopened persisted manifests"):
                build_exp451_score_manifests_antigravity(
                    fixture.root, verification=direct, context=fixture.context,
                    catalog=fixture.catalog, migration_manifest=fixture.manifest,
                    raw_datasets={},
                )
            called = []
            with self.assertRaisesRegex(ValueError, "reopened persisted manifests"):
                evaluate_exp451_antigravity(
                    EVALUATION_ROOT, verification=direct, score_manifests={},
                    label_loader=lambda _: called.append(True),
                )
        self.assertEqual(called, [])

    def test_evaluation_refuses_4467_successes_before_label_loader(self) -> None:
        """A partial coordinate set cannot reach the TEST-label boundary."""
        from experiments.exp451_verified_accelerated_resume_antigravity.evaluation_antigravity import (
            require_exp451_evaluation_ready_antigravity,
        )

        with self.assertRaisesRegex(ValueError, "authority"):
            require_exp451_evaluation_ready_antigravity(_verification_antigravity(total=4467))

    def test_evaluation_refuses_missing_condition_manifest(self) -> None:
        """All four independently frozen condition manifests are required."""
        from experiments.exp451_verified_accelerated_resume_antigravity.evaluation_antigravity import (
            require_exp451_evaluation_ready_antigravity,
        )

        with self.assertRaisesRegex(ValueError, "authority"):
            require_exp451_evaluation_ready_antigravity(
                _verification_antigravity(manifests=("A", "B", "C"))
            )

    def test_provenance_keeps_imported_and_native_counts_separate(self) -> None:
        """The evidence summary never collapses the two execution origins."""
        from experiments.exp451_verified_accelerated_resume_antigravity.verifier_antigravity import (
            provenance_summary_antigravity,
        )

        summary = provenance_summary_antigravity(_verification_antigravity())
        self.assertEqual(summary["imported_by_condition"], {"A": 1117, "B": 1117, "C": 1000, "D": 1000})
        self.assertEqual(summary["native_by_condition"], {"A": 0, "B": 0, "C": 117, "D": 117})
        self.assertEqual(summary["imported_total"], 4234)
        self.assertEqual(summary["native_total"], 234)

    def test_curve_axis_contracts_keep_roc_and_pr_coordinates_correct(self) -> None:
        """Paper figure coordinates are delegated to the frozen Exp450 definitions."""
        from experiments.exp451_verified_accelerated_resume_antigravity.evaluation_antigravity import (
            curve_tables_exp451_antigravity,
        )

        methods = ("A", "B", "C", "D", "knn_distance", "lof_novelty", "ocsvm_rbf", "isolation_forest", "deep_svdd_1d", "drocc_1d")
        scores = {method: {"fixture_antigravity": (0.1, 0.9)} for method in methods}
        curves = curve_tables_exp451_antigravity(
            {"fixture_antigravity": (0, 1)}, scores
        )
        self.assertEqual(curves["roc"]["x_name"], "false_positive_rate")
        self.assertEqual(curves["roc"]["y_name"], "true_positive_rate")
        self.assertEqual(curves["precision_recall"]["x_name"], "recall")
        self.assertEqual(curves["precision_recall"]["y_name"], "precision")
        self.assertEqual(
            {row["method"] for row in curves["roc"]["rows"]}, set(methods)
        )

    def test_four_verified_condition_manifests_publish_create_only(self) -> None:
        """The label gate is backed by four durable, byte-stable manifest leaves."""
        from experiments.exp451_verified_accelerated_resume_antigravity.verifier_antigravity import (
            write_exp451_condition_manifests_antigravity,
        )
        from experiments.exp451_verified_accelerated_resume_antigravity import verifier_antigravity

        with TemporaryDirectory(prefix="exp451_manifest_fixture_antigravity_") as temporary:
            root = Path(temporary).resolve()
            with patch.object(verifier_antigravity, "OFFICIAL_ROOT", root):
                first = write_exp451_condition_manifests_antigravity(root, _verification_antigravity())
                second = write_exp451_condition_manifests_antigravity(root, _verification_antigravity())
        self.assertEqual(tuple(first), ("A", "B", "C", "D"))
        self.assertEqual(first, second)

    def test_forged_public_dataclass_does_not_open_label_boundary(self) -> None:
        """Only an authority issued by checkpoint verification can cross the gate."""
        from experiments.exp451_verified_accelerated_resume_antigravity.evaluation_antigravity import (
            require_exp451_evaluation_ready_antigravity,
        )
        from experiments.exp451_verified_accelerated_resume_antigravity.verifier_antigravity import (
            Exp451VerificationAntigravity,
        )
        forged = Exp451VerificationAntigravity(
            4468, {"A": 1117, "B": 1117, "C": 1000, "D": 1000},
            {"A": 0, "B": 0, "C": 117, "D": 117}, MappingProxyType({}),
        )
        with self.assertRaisesRegex(ValueError, "authority"):
            require_exp451_evaluation_ready_antigravity(forged)

    def test_gate_rejects_one_coordinate_origin_distribution_drift(self) -> None:
        """C/D native counts are fixed at 117 each; total alone is insufficient."""
        from experiments.exp451_verified_accelerated_resume_antigravity.evaluation_antigravity import (
            require_exp451_evaluation_ready_antigravity,
        )
        with self.assertRaisesRegex(ValueError, "authority"):
            require_exp451_evaluation_ready_antigravity(_verification_antigravity(native_c=116))

    def test_cli_routes_verify_evaluate_and_report_commands(self) -> None:
        """Public documented commands no longer end at the former Task 6 placeholder."""
        script = Path(__file__).resolve().parents[3] / "scratch" / "run_exp451_verified_accelerated_resume_antigravity.py"
        specification = importlib.util.spec_from_file_location("exp451_task6_cli_antigravity", script)
        assert specification is not None and specification.loader is not None
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        for command, function in (("verify", "_verify_antigravity"), ("evaluate", "_evaluate_antigravity"), ("report", "_report_antigravity")):
            output = io.StringIO()
            with patch.object(module, function, return_value={"experiment": "Exp451", "command": command}), patch("sys.stdout", output):
                self.assertEqual(module.main([command, "--mode", "full"]), 0)
            self.assertIn(f'"command":"{command}"', output.getvalue())

    def test_evaluation_does_not_call_label_loader_when_completion_is_partial(self) -> None:
        """The public evaluator reaches no label callback when its first gate fails."""
        from experiments.exp451_verified_accelerated_resume_antigravity.evaluation_antigravity import (
            evaluate_exp451_antigravity,
        )
        from experiments.exp451_verified_accelerated_resume_antigravity.config_antigravity import EVALUATION_ROOT
        called = []
        with self.assertRaisesRegex(ValueError, "authority"):
            evaluate_exp451_antigravity(
                EVALUATION_ROOT, verification=_verification_antigravity(total=4467),
                score_manifests={}, label_loader=lambda _: called.append(True),
            )
        self.assertEqual(called, [])

    def test_tampered_manifest_and_wrong_database_are_rejected(self) -> None:
        """A capability alone cannot authorize altered manifest bytes or database identity."""
        from experiments.exp451_verified_accelerated_resume_antigravity.evaluation_antigravity import (
            require_exp451_evaluation_ready_antigravity,
        )
        verification = _persisted_verification_antigravity()
        altered = {key: dict(value) for key, value in verification.condition_manifests.items()}
        altered["A"]["origin_counts"] = dict(altered["A"]["origin_counts"])
        altered["A"]["origin_counts"]["imported_exp450_mps"] = 1116
        with self.assertRaisesRegex(ValueError, "authority"):
            require_exp451_evaluation_ready_antigravity(
                replace(verification, condition_manifests=MappingProxyType(altered))
            )
        with self.assertRaisesRegex(ValueError, "authority"):
            require_exp451_evaluation_ready_antigravity(
                replace(verification, database_sha256="f" * 64)
            )

    def test_economical_fixture_represents_all_4468_checkpoint_hashes_and_195900_test_rows(self) -> None:
        """The private fixture is full-population evidence, not a reduced production rule."""
        verification = _persisted_verification_antigravity()
        manifests = verification.condition_manifests
        self.assertEqual(sum(len(manifests[condition]["checkpoint_hashes"]) for condition in manifests), 4468)
        self.assertEqual(sum(item["count"] for item in manifests["A"]["test_identities"]), 195900)

    def test_score_manifest_tamper_and_wrong_root_stop_before_labels(self) -> None:
        """Changed score bytes and an unconfigured evaluation root cannot open labels."""
        from experiments.exp451_verified_accelerated_resume_antigravity.evaluation_antigravity import evaluate_exp451_antigravity
        verification = _persisted_verification_antigravity()
        called = []
        with self.assertRaisesRegex(ValueError, "root"):
            evaluate_exp451_antigravity(
                Path("/tmp/wrong_exp451_eval_antigravity"), verification=verification,
                score_manifests={}, label_loader=lambda _: called.append(True),
            )
        from experiments.exp451_verified_accelerated_resume_antigravity.config_antigravity import EVALUATION_ROOT
        tampered = {condition: {"score_rows": {}} for condition in ("A", "B", "C", "D")}
        with self.assertRaisesRegex(ValueError, "score manifest authority"):
            evaluate_exp451_antigravity(
                EVALUATION_ROOT, verification=verification, score_manifests=tampered,
                label_loader=lambda _: called.append(True),
            )
        self.assertEqual(called, [])

    def test_rehashed_fabricated_score_authority_cannot_open_labels(self) -> None:
        """Public score hashes are evidence, not an issuer for modified scores."""
        from experiments.exp451_verified_accelerated_resume_antigravity.evaluation_antigravity import evaluate_exp451_antigravity
        from experiments.exp451_verified_accelerated_resume_antigravity.config_antigravity import EVALUATION_ROOT
        from experiments.exp451_verified_accelerated_resume_antigravity.verifier_antigravity import (
            Exp451ScoreManifestsAntigravity, _score_manifests_digest_antigravity,
        )
        verification = _persisted_verification_antigravity()
        changed = MappingProxyType({condition: MappingProxyType({"score_rows": {}}) for condition in ("A", "B", "C", "D")})
        forged = Exp451ScoreManifestsAntigravity(
            changed, verification.inventory_sha256, verification.context_sha256,
            _score_manifests_digest_antigravity(changed),
        )
        called = []
        with self.assertRaisesRegex(ValueError, "score manifest authority"):
            evaluate_exp451_antigravity(
                EVALUATION_ROOT, verification=verification, score_manifests=forged,
                label_loader=lambda _: called.append(True),
            )
        self.assertEqual(called, [])

    def test_missing_persisted_manifests_and_fabricated_evaluation_are_rejected(self) -> None:
        """Neither absent manifest leaves nor a copied evaluation object can reach reporting."""
        from experiments.exp451_verified_accelerated_resume_antigravity import verifier_antigravity
        from experiments.exp451_verified_accelerated_resume_antigravity.verifier_antigravity import read_persisted_exp451_condition_manifests_antigravity
        verification = _verification_antigravity()
        with TemporaryDirectory(prefix="exp451_missing_manifest_antigravity_") as temporary:
            root = Path(temporary).resolve()
            with patch.object(verifier_antigravity, "OFFICIAL_ROOT", root):
                with self.assertRaises(ValueError):
                    read_persisted_exp451_condition_manifests_antigravity(root, verification)
        verification = _persisted_verification_antigravity()
        from experiments.exp451_verified_accelerated_resume_antigravity.evaluation_antigravity import Exp451EvaluationAntigravity
        from experiments.exp451_verified_accelerated_resume_antigravity.reporting_antigravity import write_exp451_report_antigravity
        from experiments.exp451_verified_accelerated_resume_antigravity.config_antigravity import REPORT_ROOT
        forged = Exp451EvaluationAntigravity(None, verification.inventory_sha256, verification.context_sha256)
        with self.assertRaisesRegex(ValueError, "evaluation authority"):
            write_exp451_report_antigravity(forged, REPORT_ROOT, verification=verification)

    def test_score_builder_rejects_rehashed_checkpoint_mutation_after_verification(self) -> None:
        """A canonical envelope with a recomputed payload hash cannot replace verified bytes."""
        from experiments.exp451_verified_accelerated_resume_antigravity.contracts_antigravity import (
            canonical_json_bytes,
        )
        from experiments.exp451_verified_accelerated_resume_antigravity.artifacts_antigravity import (
            success_relative_path_antigravity,
        )
        from experiments.exp451_verified_accelerated_resume_antigravity.verifier_antigravity import (
            build_exp451_score_manifests_antigravity,
            read_persisted_exp451_condition_manifests_antigravity,
            write_exp451_condition_manifests_antigravity,
        )

        with _live_verification_environment_antigravity() as fixture:
            verification = fixture.verifier.verify_exp451_completion_antigravity(
                fixture.root, context=fixture.context, catalog=fixture.catalog,
                migration_manifest=fixture.manifest,
            )
            write_exp451_condition_manifests_antigravity(fixture.root, verification)
            read_persisted_exp451_condition_manifests_antigravity(fixture.root, verification)
            checkpoint = fixture.root / success_relative_path_antigravity(
                fixture.datasets[0], "A",
            )
            original = checkpoint.read_bytes()
            mutated = json.loads(original)
            mutated["payload"] = {"sources": {}, "route": {"mutated_antigravity": True}}
            mutated["payload_sha256"] = sha256(
                canonical_json_bytes(mutated["payload"])
            ).hexdigest()
            checkpoint.write_bytes(canonical_json_bytes(mutated))
            try:
                with self.assertRaisesRegex(ValueError, "checkpoint differs from verified inventory"):
                    build_exp451_score_manifests_antigravity(
                        fixture.root, verification=verification, context=fixture.context,
                        catalog=fixture.catalog, migration_manifest=fixture.manifest,
                        raw_datasets={dataset: None for dataset in fixture.datasets},
                    )
            finally:
                checkpoint.write_bytes(original)

    def test_report_directory_publication_is_create_only_and_recovers_interruption(self) -> None:
        """A report marker is the only finalization step; no-clobber retries are stable."""
        from experiments.exp451_verified_accelerated_resume_antigravity.reporting_antigravity import (
            _publish_report_directory_antigravity,
        )
        with TemporaryDirectory(prefix="exp451_report_publish_antigravity_") as temporary:
            parent = Path(temporary).resolve()
            stage = parent / "stage_antigravity"; stage.mkdir()
            (stage / "one_antigravity.txt").write_bytes(b"one")
            target = parent / "target_antigravity"
            self.assertEqual(_publish_report_directory_antigravity(stage, target), target)
            retry = parent / "retry_antigravity"; retry.mkdir()
            (retry / "one_antigravity.txt").write_bytes(b"one")
            self.assertEqual(_publish_report_directory_antigravity(retry, target), target)
            conflict = parent / "conflict_antigravity"; conflict.mkdir()
            (conflict / "one_antigravity.txt").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "differs"):
                _publish_report_directory_antigravity(conflict, target)
            interrupted = parent / "interrupted_target_antigravity"; interrupted.mkdir()
            (interrupted / "one_antigravity.txt").write_bytes(b"partial")
            recovery = parent / "recovery_antigravity"; recovery.mkdir()
            (recovery / "one_antigravity.txt").write_bytes(b"one")
            with self.assertRaisesRegex(ValueError, "differs"):
                _publish_report_directory_antigravity(recovery, interrupted)
            (interrupted / ".exp451_report_in_progress_antigravity.json").write_bytes(
                b'{"destination":"interrupted_target_antigravity","schema":"exp451-report-publication-in-progress-antigravity.v1"}\n'
            )
            self.assertEqual(_publish_report_directory_antigravity(recovery, interrupted), interrupted)

    def test_report_publication_rejects_an_active_lock_owner(self) -> None:
        """A second publisher cannot remove or pass another live process reservation."""
        from experiments.exp451_verified_accelerated_resume_antigravity.reporting_antigravity import (
            _publish_report_directory_antigravity,
        )
        with TemporaryDirectory(prefix="exp451_active_report_lock_antigravity_") as temporary:
            parent = Path(temporary).resolve()
            stage = parent / "active_stage_antigravity"
            stage.mkdir()
            (stage / "one_antigravity.txt").write_bytes(b"one")
            target = parent / "active_target_antigravity"
            lock = parent / f".{target.name}.publish_lock_antigravity"
            owner = _active_report_lock_owner_antigravity(lock)
            try:
                with self.assertRaisesRegex(ValueError, "already in progress"):
                    _publish_report_directory_antigravity(stage, target)
                self.assertIsNone(owner.poll())
            finally:
                _stop_report_lock_owner_antigravity(owner)

    def test_report_publication_retries_after_lock_owner_process_dies(self) -> None:
        """Kernel-released ownership lets a later publisher recover a stale lock file."""
        from experiments.exp451_verified_accelerated_resume_antigravity.reporting_antigravity import (
            _publish_report_directory_antigravity,
        )
        with TemporaryDirectory(prefix="exp451_stale_report_lock_antigravity_") as temporary:
            parent = Path(temporary).resolve()
            stage = parent / "stale_stage_antigravity"
            stage.mkdir()
            (stage / "one_antigravity.txt").write_bytes(b"one")
            target = parent / "stale_target_antigravity"
            lock = parent / f".{target.name}.publish_lock_antigravity"
            owner = _active_report_lock_owner_antigravity(lock)
            owner.kill()
            owner.wait(timeout=5)
            try:
                self.assertEqual(_publish_report_directory_antigravity(stage, target), target)
            finally:
                _stop_report_lock_owner_antigravity(owner)
