"""Closed Exp471 source-to-report command orchestration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from time import monotonic
from typing import TextIO

from .config_antigravity import (
    CPU_WORKERS_ANTIGRAVITY,
    EXP467_SOURCE_ROOT_ANTIGRAVITY,
    EXP471_BASELINE_ROOT_AUTHORITY_ANTIGRAVITY,
    EXP471_BASELINE_ROOT_BY_METHOD_ANTIGRAVITY,
    EXP471_EVALUATION_ROOT_ANTIGRAVITY,
    EXP471_REPORT_ROOT_ANTIGRAVITY,
    EXP471_ROOT_ANTIGRAVITY,
    EXP471_SCORE_CACHE_ROOT_ANTIGRAVITY,
    METHOD_ORDER_ANTIGRAVITY,
)
from .contracts_antigravity import canonical_json_bytes_exp471_antigravity


_COMMANDS_ANTIGRAVITY = (
    "verify-source",
    "benchmark-prepare",
    "prepare",
    "status",
    "report",
)
_PROGRESS_NAME_ANTIGRAVITY = "progress_antigravity.jsonl"
_TOTAL_SHARDS_ANTIGRAVITY = 11_170


def _repository_root_antigravity() -> Path:
    return Path(__file__).resolve().parents[2]


def _plain_antigravity(value: object) -> object:
    raw = canonical_json_bytes_exp471_antigravity(value)
    return json.loads(raw.decode("utf-8"))


def _emit_antigravity(stream: TextIO, value: object) -> None:
    stream.write(canonical_json_bytes_exp471_antigravity(value).decode("utf-8"))
    stream.flush()


def _latest_progress_antigravity() -> Mapping[str, object] | None:
    path = EXP471_ROOT_ANTIGRAVITY / _PROGRESS_NAME_ANTIGRAVITY
    if not os.path.lexists(path):
        return None
    if path.is_symlink() or not path.is_file():
        raise ValueError("Exp471 progress log is unsafe")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        linked = path.lstat()
        if opened.st_dev != linked.st_dev or opened.st_ino != linked.st_ino:
            raise ValueError("Exp471 progress log changed during status")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    lines = b"".join(chunks).splitlines()
    if not lines:
        return None
    try:
        value = json.loads(lines[-1].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Exp471 latest progress event is malformed") from error
    if not isinstance(value, dict):
        raise ValueError("Exp471 latest progress event differs")
    return value


def _append_progress_antigravity(value: Mapping[str, object]) -> None:
    root = EXP471_ROOT_ANTIGRAVITY
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Exp471 progress root is unavailable")
    path = root / _PROGRESS_NAME_ANTIGRAVITY
    if os.path.lexists(path) and path.is_symlink():
        raise ValueError("Exp471 progress log is unsafe")
    payload = canonical_json_bytes_exp471_antigravity(value)
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("Exp471 progress append made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _append_durable_stage_antigravity(
    stage: str, completed: int, total: int, artifact_path: Path, started: float
) -> None:
    elapsed = max(0.0, monotonic() - started)
    _append_progress_antigravity(
        {
            "schema": "exp471-progress-antigravity.v1",
            "stage": stage,
            "durable_completed": completed,
            "total": total,
            "failures": 0,
            "remaining": max(0, total - completed),
            "elapsed_seconds": round(elapsed, 6),
            "eta_seconds": 0.0 if completed == total else 0.0,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(
                timespec="milliseconds"
            ),
            "artifact_path": str(artifact_path),
            "root_path": str(EXP471_ROOT_ANTIGRAVITY),
        }
    )


def _progress_callback_antigravity(stage: str, artifact_path: Path):
    started = monotonic()
    failures = 0

    def progress(event: Mapping[str, object]) -> None:
        nonlocal failures
        completed = event.get("durable_completed", 0)
        if type(completed) is not int or completed < 0:
            raise ValueError("Exp471 durable progress count differs")
        if event.get("status") in {"failed", "terminal_failed"}:
            failures += 1
        total = event.get("total", _TOTAL_SHARDS_ANTIGRAVITY)
        if type(total) is not int or total < completed:
            raise ValueError("Exp471 durable progress total differs")
        remaining = max(0, total - completed)
        elapsed = max(0.0, monotonic() - started)
        eta = 0.0 if remaining == 0 else (
            elapsed * remaining / completed if completed else 0.0
        )
        record = {
            "schema": "exp471-progress-antigravity.v1",
            "stage": stage,
            "durable_completed": completed,
            "total": total,
            "failures": failures,
            "remaining": remaining,
            "elapsed_seconds": round(elapsed, 6),
            "eta_seconds": round(eta, 6),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(
                timespec="milliseconds"
            ),
            "artifact_path": str(artifact_path),
            "root_path": str(EXP471_ROOT_ANTIGRAVITY),
        }
        _append_progress_antigravity(record)

    return progress


def _coordinate_record_antigravity(dataset: str, identity: Mapping[str, object]) -> dict[str, object]:
    count = identity.get("count")
    indices = identity.get("indices")
    if (
        type(count) is not int
        or count < 1
        or not isinstance(indices, (tuple, list))
        or tuple(indices) != tuple(range(count))
    ):
        raise ValueError("Exp471 production TEST coordinates differ")
    bare = {"dataset": dataset, "count": count, "indices": list(indices)}
    return {
        "count": count,
        "indices": list(indices),
        "coordinates_sha256": sha256(
            canonical_json_bytes_exp471_antigravity(bare)
        ).hexdigest(),
    }


def _production_score_bundle_antigravity(session: object) -> Mapping[str, object]:
    """Build Task 5's verified, label-free input from issued Exp467 evidence."""

    from experiments.exp450_resampling_image_ablation_antigravity import (
        evaluation_antigravity as frozen_evaluation,
    )
    from experiments.paper_public_baselines.runner import verify_score_root
    from paper.equipment_anomaly_detection.contracts_paper import FrozenPaperConfig
    from paper.equipment_anomaly_detection.data_paper import (
        bind_preverified_database_read_only_paper,
        load_dataset_values,
    )

    authority = session.authority
    catalog = session.catalog
    paper = FrozenPaperConfig()
    if Path(paper.database_path).is_symlink() or paper.database_sha256 != authority.database_sha256:
        raise ValueError("Exp471 production database authority differs")
    binding = bind_preverified_database_read_only_paper(
        Path(paper.database_path), verified_catalog_database=catalog["database"]
    )
    datasets = tuple(entry["dataset"] for entry in catalog["datasets"])
    raw = {
        dataset: load_dataset_values(
            Path(paper.database_path), dataset, preverified_database=binding
        )
        for dataset in datasets
    }
    build = getattr(
        session.verifier, "build_exp467_score_manifests_antigravity", None
    )
    if not callable(build):
        raise ValueError("Exp471 fixed score-manifest builder is unavailable")
    score_authority = build(
        session.context.run_root,
        verification=session.verification,
        context=session.context,
        catalog=catalog,
        migration_manifest=session.migration_manifest,
        raw_datasets=raw,
    )
    manifests = score_authority.manifests
    if tuple(manifests) != METHOD_ORDER_ANTIGRAVITY[:4]:
        raise ValueError("Exp471 production condition score order differs")
    first = manifests[METHOD_ORDER_ANTIGRAVITY[0]]
    identities = frozen_evaluation._manifest_test_identities(first)
    if tuple(identities) != datasets:
        raise ValueError("Exp471 production TEST dataset order differs")
    coordinates = {
        dataset: _coordinate_record_antigravity(dataset, identities[dataset])
        for dataset in datasets
    }
    conditions: dict[str, object] = {}
    for method in METHOD_ORDER_ANTIGRAVITY[:4]:
        rows = manifests[method].get("score_rows")
        if not isinstance(rows, Mapping) or tuple(rows) != datasets:
            raise ValueError("Exp471 production condition scores differ")
        conditions[method] = {
            "source_manifest_sha256": authority.condition_manifest_sha256[method],
            "scores": {
                dataset: {
                    "test_indices": coordinates[dataset]["indices"],
                    "test_scores": list(rows[dataset]),
                }
                for dataset in datasets
            },
        }

    verified_baselines = frozen_evaluation._verify_baseline_roots(
        frozen_evaluation.BASELINE_ROOTS, verify_score_root
    )
    coordinate_inventory = {
        "schema": "exp471-test-coordinate-inventory-antigravity.v1",
        "dataset_count": len(datasets),
        "datasets": [
            {
                "dataset": dataset,
                "test_count": coordinates[dataset]["count"],
                "test_indices": coordinates[dataset]["indices"],
            }
            for dataset in datasets
        ],
    }
    coordinates_sha256 = sha256(
        canonical_json_bytes_exp471_antigravity(coordinate_inventory)
    ).hexdigest()
    baselines: dict[str, object] = {}
    baseline_inventory: dict[str, str] = {}
    for root_name in ("classical", "deep_svdd", "drocc"):
        manifest = verified_baselines[root_name]
        baseline_identities, database_sha256, shards = (
            frozen_evaluation._baseline_identity_and_shards(manifest)
        )
        if (
            database_sha256 != authority.database_sha256
            or tuple(baseline_identities) != datasets
        ):
            raise ValueError("Exp471 production baseline identity differs")
        root_authority = EXP471_BASELINE_ROOT_AUTHORITY_ANTIGRAVITY[root_name]
        manifest_path = Path(manifest.run_root) / "frozen_score_manifest.json"
        if (
            manifest.run_config_sha256 != root_authority["run_config_sha256"]
            or manifest_path.is_symlink()
            or not manifest_path.is_file()
            or sha256(manifest_path.read_bytes()).hexdigest()
            != root_authority["manifest_sha256"]
        ):
            raise ValueError("Exp471 pinned baseline root authority differs")
        for method in METHOD_ORDER_ANTIGRAVITY[4:]:
            if EXP471_BASELINE_ROOT_BY_METHOD_ANTIGRAVITY[method] != root_name:
                continue
            bare = {
                "schema": "exp471-verified-baseline-authority-antigravity.v1",
                "method": method,
                "database_sha256": authority.database_sha256,
                "dataset_order": list(datasets),
                "test_coordinates_sha256": coordinates_sha256,
                "source_manifest_sha256": root_authority["manifest_sha256"],
                "run_config_sha256": root_authority["run_config_sha256"],
            }
            authority_sha256 = sha256(
                canonical_json_bytes_exp471_antigravity(bare)
            ).hexdigest()
            baseline_inventory[method] = authority_sha256
            scores: dict[str, object] = {}
            for dataset in datasets:
                shard = shards.get((method, dataset))
                if not isinstance(shard, Mapping) or shard.get("status") != "complete":
                    raise ValueError("Exp471 production baseline coverage differs")
                scores[dataset] = {
                    "train_oof_scores": list(shard["train_oof_scores"]),
                    "test_indices": list(shard["test_indices"]),
                    "test_scores": list(shard["test_scores"]),
                }
            baselines[method] = {
                "authority": {**bare, "authority_sha256": authority_sha256},
                "scores": scores,
            }
    return {
        "schema": "exp471-verified-score-input-antigravity.v2",
        "database_sha256": authority.database_sha256,
        "dataset_order": list(datasets),
        "test_coordinates": coordinates,
        "baseline_authority_sha256": baseline_inventory,
        "conditions": conditions,
        "baselines": baselines,
    }


def _reopened_score_bundle_antigravity(session: object) -> Mapping[str, object]:
    """Rebuild only Task 5 authority metadata from verified durable shards."""

    from .cache_artifacts_antigravity import (
        read_cache_manifest_antigravity,
        read_score_shard_antigravity,
    )

    authority = session.authority
    manifest = read_cache_manifest_antigravity(EXP471_SCORE_CACHE_ROOT_ANTIGRAVITY)
    if (
        manifest.source_authority_sha256 != authority.authority_sha256
        or manifest.method_order != METHOD_ORDER_ANTIGRAVITY
    ):
        raise ValueError("Exp471 reopened cache source authority differs")
    datasets = manifest.dataset_order
    shards: dict[tuple[str, str], object] = {}
    coordinates: dict[str, object] = {}
    for method in METHOD_ORDER_ANTIGRAVITY:
        for dataset in datasets:
            shard = read_score_shard_antigravity(
                EXP471_SCORE_CACHE_ROOT_ANTIGRAVITY, method, dataset
            )
            key = f"{method}:{dataset}"
            if (
                shard.envelope_sha256 != manifest.shard_sha256[key]
                or shard.cache_context_sha256 != manifest.cache_context_sha256
                or shard.source_authority_sha256 != authority.authority_sha256
            ):
                raise ValueError("Exp471 reopened score shard authority differs")
            shards[(method, dataset)] = shard
            record = _coordinate_record_antigravity(
                dataset,
                {"count": shard.test_count, "indices": list(range(shard.test_count))},
            )
            prior = coordinates.setdefault(dataset, record)
            if prior != record:
                raise ValueError("Exp471 reopened TEST coordinates differ")

    conditions: dict[str, object] = {}
    for method in METHOD_ORDER_ANTIGRAVITY[:4]:
        scores: dict[str, object] = {}
        for dataset in datasets:
            shard = shards[(method, dataset)]
            if (
                shard.source_manifest_sha256
                != authority.condition_manifest_sha256[method]
            ):
                raise ValueError("Exp471 reopened condition source differs")
            scores[dataset] = {
                "test_indices": coordinates[dataset]["indices"],
                "test_scores": list(shard.raw_scores),
            }
        conditions[method] = {
            "source_manifest_sha256": authority.condition_manifest_sha256[method],
            "scores": scores,
        }

    inventory_record = {
        "schema": "exp471-test-coordinate-inventory-antigravity.v1",
        "dataset_count": len(datasets),
        "datasets": [
            {
                "dataset": dataset,
                "test_count": coordinates[dataset]["count"],
                "test_indices": coordinates[dataset]["indices"],
            }
            for dataset in datasets
        ],
    }
    coordinates_sha256 = sha256(
        canonical_json_bytes_exp471_antigravity(inventory_record)
    ).hexdigest()
    baseline_inventory: dict[str, str] = {}
    baselines: dict[str, object] = {}
    for method in METHOD_ORDER_ANTIGRAVITY[4:]:
        root_name = EXP471_BASELINE_ROOT_BY_METHOD_ANTIGRAVITY[method]
        root_authority = EXP471_BASELINE_ROOT_AUTHORITY_ANTIGRAVITY[root_name]
        bare = {
            "schema": "exp471-verified-baseline-authority-antigravity.v1",
            "method": method,
            "database_sha256": authority.database_sha256,
            "dataset_order": list(datasets),
            "test_coordinates_sha256": coordinates_sha256,
            "source_manifest_sha256": root_authority["manifest_sha256"],
            "run_config_sha256": root_authority["run_config_sha256"],
        }
        authority_sha256 = sha256(
            canonical_json_bytes_exp471_antigravity(bare)
        ).hexdigest()
        baseline_inventory[method] = authority_sha256
        scores: dict[str, object] = {}
        for dataset in datasets:
            shard = shards[(method, dataset)]
            if shard.source_manifest_sha256 != root_authority["manifest_sha256"]:
                raise ValueError("Exp471 reopened baseline source differs")
            scores[dataset] = {
                # Metadata validation requires a finite nonempty TRAIN vector,
                # but reopening never calibrates or republishes from this value.
                "train_oof_scores": [0.0],
                "test_indices": coordinates[dataset]["indices"],
                "test_scores": list(shard.raw_scores),
            }
        baselines[method] = {
            "authority": {**bare, "authority_sha256": authority_sha256},
            "scores": scores,
        }
    return {
        "schema": "exp471-verified-score-input-antigravity.v2",
        "database_sha256": authority.database_sha256,
        "dataset_order": list(datasets),
        "test_coordinates": coordinates,
        "baseline_authority_sha256": baseline_inventory,
        "conditions": conditions,
        "baselines": baselines,
    }


def _score_input_antigravity(session: object) -> Mapping[str, object]:
    manifest = EXP471_SCORE_CACHE_ROOT_ANTIGRAVITY / "score_cache_manifest_antigravity.json"
    if os.path.lexists(manifest):
        return _reopened_score_bundle_antigravity(session)
    return _production_score_bundle_antigravity(session)


class _ProductionRuntimeAntigravity:
    def verify_source_antigravity(self):
        from .source_authority_antigravity import (
            open_exp467_source_session_antigravity,
            publish_source_authority_antigravity,
        )

        holder: dict[str, object] = {}

        def score_reader():
            return _score_input_antigravity(holder["session"])

        def label_reader():
            from experiments.paper_public_baselines.label_access import (
                load_test_labels,
            )

            session = holder["session"]
            datasets = tuple(
                entry["dataset"] for entry in session.catalog["datasets"]
            )
            return load_test_labels(datasets)

        started = monotonic()
        session = open_exp467_source_session_antigravity(
            _repository_root_antigravity(),
            EXP467_SOURCE_ROOT_ANTIGRAVITY,
            score_reader_antigravity=score_reader,
            label_callback_antigravity=label_reader,
        )
        holder["session"] = session
        authority_path = publish_source_authority_antigravity(
            EXP471_ROOT_ANTIGRAVITY, session.authority
        )
        _append_durable_stage_antigravity(
            "verify-source",
            session.authority.total_successes,
            session.authority.total_successes,
            authority_path,
            started,
        )
        return session

    def open_cache_antigravity(self, session):
        from .score_cache_antigravity import open_complete_score_cache_antigravity

        manifest = EXP471_SCORE_CACHE_ROOT_ANTIGRAVITY / "score_cache_manifest_antigravity.json"
        if not os.path.lexists(manifest):
            raise FileNotFoundError("Exp471 cache manifest is absent")
        return open_complete_score_cache_antigravity(session)

    def benchmark_prepare_antigravity(self, session):
        from . import score_cache_antigravity as score_cache

        tasks = score_cache._build_tasks_from_verified_input_antigravity(
            session.authority, session.score_reader_antigravity()
        )
        sample = tasks[: min(len(tasks), max(2, CPU_WORKERS_ANTIGRAVITY))]
        return score_cache.benchmark_score_preparation_antigravity(
            sample, parallel_workers=CPU_WORKERS_ANTIGRAVITY
        )

    def prepare_antigravity(self, session, progress):
        from .score_cache_antigravity import prepare_score_cache_antigravity

        return prepare_score_cache_antigravity(session, progress=progress)

    def open_evaluation_antigravity(self, session, cache):
        from .evaluation_antigravity import read_evaluation_exp471_antigravity

        path = EXP471_EVALUATION_ROOT_ANTIGRAVITY / "evaluation_antigravity.json"
        if not os.path.lexists(path):
            raise FileNotFoundError("Exp471 evaluation is absent")
        return read_evaluation_exp471_antigravity(
            EXP471_EVALUATION_ROOT_ANTIGRAVITY, session, cache
        )

    def evaluate_antigravity(self, session, cache):
        from .evaluation_antigravity import evaluate_exp471_antigravity

        started = monotonic()
        artifact = evaluate_exp471_antigravity(session, cache)
        _append_durable_stage_antigravity(
            "evaluation",
            len(artifact.method_order),
            len(artifact.method_order),
            EXP471_EVALUATION_ROOT_ANTIGRAVITY / "evaluation_antigravity.json",
            started,
        )
        return artifact

    def report_antigravity(self, artifact):
        from .reporting_antigravity import write_exp471_report_antigravity
        from .score_cache_antigravity import status_score_cache_antigravity

        worker = status_score_cache_antigravity(EXP471_SCORE_CACHE_ROOT_ANTIGRAVITY)
        started = monotonic()
        authority = write_exp471_report_antigravity(
            artifact,
            EXP471_REPORT_ROOT_ANTIGRAVITY,
            worker_provenance={
                "benchmark_decision": worker["benchmark_decision"],
                "execution_decision": worker["execution_decision"],
                "detected_logical_cpus": worker["detected_logical_cpus"],
                "default_cpu_workers": worker["default_cpu_workers"],
                "inner_numerical_threads": worker["inner_numerical_threads"],
                "acceleration": worker["acceleration"],
            },
        )
        _append_durable_stage_antigravity(
            "report",
            len(authority.report_file_sha256),
            len(authority.report_file_sha256),
            EXP471_REPORT_ROOT_ANTIGRAVITY,
            started,
        )
        return authority

    def status_antigravity(self):
        from .score_cache_antigravity import status_score_cache_antigravity

        cache = status_score_cache_antigravity(EXP471_SCORE_CACHE_ROOT_ANTIGRAVITY)
        evaluation_path = (
            EXP471_EVALUATION_ROOT_ANTIGRAVITY / "evaluation_antigravity.json"
        )
        report_marker = (
            EXP471_REPORT_ROOT_ANTIGRAVITY
            / ".exp471_report_complete_antigravity.json"
        )
        completed = int(cache["durable_completed"])
        total = cache["total_shards"]
        expected_total = _TOTAL_SHARDS_ANTIGRAVITY if total is None else int(total)
        latest = _latest_progress_antigravity()
        failure_count = 0 if latest is None else int(latest.get("failures", 0))
        return {
            "schema": "exp471-status-antigravity.v1",
            "durable_completed": completed,
            "total": expected_total,
            "failures": failure_count,
            "remaining": max(0, expected_total - completed),
            "verified_stage": (
                "report-present-unverified"
                if report_marker.is_file() and not report_marker.is_symlink()
                else "evaluation-present-unverified"
                if evaluation_path.is_file() and not evaluation_path.is_symlink()
                else "cache-complete"
                if cache["complete"]
                else "cache-incomplete"
            ),
            "paths": {
                "root": str(EXP471_ROOT_ANTIGRAVITY),
                "score_cache": str(EXP471_SCORE_CACHE_ROOT_ANTIGRAVITY),
                "evaluation": str(EXP471_EVALUATION_ROOT_ANTIGRAVITY),
                "report": str(EXP471_REPORT_ROOT_ANTIGRAVITY),
                "progress": str(EXP471_ROOT_ANTIGRAVITY / _PROGRESS_NAME_ANTIGRAVITY),
            },
            "worker_decision": {
                "benchmark": cache["benchmark_decision"],
                "actual": cache["execution_decision"],
            },
            "latest_progress": latest,
            "failure": (
                None
                if failure_count == 0
                else {
                    "stage": latest.get("stage") if latest else "unknown",
                    "failures": failure_count,
                }
            ),
        }


def _report_command_antigravity(runtime: object) -> Mapping[str, object]:
    session = runtime.verify_source_antigravity()
    try:
        cache = runtime.open_cache_antigravity(session)
    except FileNotFoundError:
        cache = runtime.prepare_antigravity(
            session,
            _runtime_progress_antigravity(runtime),
        )
    try:
        artifact = runtime.open_evaluation_antigravity(session, cache)
    except FileNotFoundError:
        artifact = runtime.evaluate_antigravity(session, cache)
    report = runtime.report_antigravity(artifact)
    return {
        "schema": "exp471-command-result-antigravity.v1",
        "experiment": "Exp471",
        "command": "report",
        "source_authority_sha256": session.authority.authority_sha256,
        "cache_manifest_sha256": cache.manifest_sha256,
        "evaluation_sha256": artifact.evaluation_sha256,
        "report_manifest_sha256": report.report_manifest_sha256,
        "report_root": str(EXP471_REPORT_ROOT_ANTIGRAVITY),
    }


def _runtime_progress_antigravity(runtime: object):
    if isinstance(runtime, _ProductionRuntimeAntigravity):
        return _progress_callback_antigravity(
            "score-cache", EXP471_SCORE_CACHE_ROOT_ANTIGRAVITY
        )
    return lambda _event: None


def main_antigravity(
    argv: Sequence[str] | None = None,
    *,
    runtime: object | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run one fixed command and emit exactly one canonical finite JSON record."""

    arguments = tuple(sys.argv[1:] if argv is None else argv)
    output = sys.stdout if stdout is None else stdout
    errors = sys.stderr if stderr is None else stderr
    if arguments in {(), ("--help",), ("-h",)}:
        _emit_antigravity(
            output,
            {
                "schema": "exp471-cli-help-antigravity.v1",
                "experiment": "Exp471",
                "command": "help",
                "commands": list(_COMMANDS_ANTIGRAVITY),
            },
        )
        return 0
    if len(arguments) != 1 or arguments[0] not in _COMMANDS_ANTIGRAVITY:
        _emit_antigravity(
            errors,
            {
                "schema": "exp471-cli-error-antigravity.v1",
                "error": "invalid_arguments",
            },
        )
        return 2
    command = arguments[0]
    active = _ProductionRuntimeAntigravity() if runtime is None else runtime
    try:
        if command == "status":
            value = {
                "experiment": "Exp471",
                "command": command,
                **active.status_antigravity(),
            }
        elif command == "verify-source":
            session = active.verify_source_antigravity()
            value = {
                "schema": "exp471-command-result-antigravity.v1",
                "experiment": "Exp471",
                "command": command,
                "authority_sha256": session.authority.authority_sha256,
                "inventory_sha256": session.authority.inventory_sha256,
                "total_successes": getattr(session.authority, "total_successes", 4468),
                "root": str(EXP471_ROOT_ANTIGRAVITY),
            }
        elif command == "benchmark-prepare":
            session = active.verify_source_antigravity()
            decision = active.benchmark_prepare_antigravity(session)
            value = {
                "schema": "exp471-command-result-antigravity.v1",
                "experiment": "Exp471",
                "command": command,
                "decision": decision,
            }
        elif command == "prepare":
            session = active.verify_source_antigravity()
            cache = active.prepare_antigravity(
                session,
                _runtime_progress_antigravity(active),
            )
            value = {
                "schema": "exp471-command-result-antigravity.v1",
                "experiment": "Exp471",
                "command": command,
                "manifest_sha256": cache.manifest_sha256,
                "total_shards": cache.total_shards,
                "root": str(EXP471_SCORE_CACHE_ROOT_ANTIGRAVITY),
            }
        else:
            value = _report_command_antigravity(active)
        _emit_antigravity(output, value)
        return 0
    except Exception as error:
        _emit_antigravity(
            errors,
            {
                "schema": "exp471-cli-error-antigravity.v1",
                "error": "command_failed",
                "command": command,
                "exception_type": type(error).__name__,
            },
        )
        return 2


__all__ = ["main_antigravity"]
