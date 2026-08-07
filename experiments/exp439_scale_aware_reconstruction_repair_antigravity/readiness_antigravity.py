"""One-shot clean-HEAD and backend readiness authority for Exp439."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Callable, Mapping

import numpy as np

from experiments.exp403_408.artifact_io import canonical_json_bytes, sha256_file
from .backend_antigravity import (
    BackendFixtureAntigravity,
    benchmark_candidate_antigravity,
    select_backend_antigravity,
)
from .config_antigravity import (
    FROZEN_CONFIG,
    RUN_PARENT_ANTIGRAVITY,
    frozen_config_payload_antigravity,
)
from .contracts_antigravity import BackendBatchEvidence, BackendDecision
from .aspe_antigravity import amplitude_sensitive_permutation_entropy_antigravity
from .inputs_antigravity import canonical_source_paths_antigravity, open_exp439_input_session
from .predecessor_antigravity import (
    classify_exp437_snapshot_antigravity,
    read_exp437_snapshot_antigravity,
)
from .predecessor_import_antigravity import (
    Exp438PrefixImportAntigravity,
    validate_exp438_prefix_antigravity,
)
from .scoring_antigravity import score_channel_cpu_antigravity


_SOURCE_NAMES_ANTIGRAVITY = (
    "database", "exp291_prediction", "exp291_manifest", "exp292_prediction", "exp292_manifest"
)
_EXP437_PARENT_ANTIGRAVITY = Path(
    "/Users/mac/Documents/Dataset/exp437_metadata_roundtrip_antigravity_20260807"
)


@dataclass(frozen=True)
class ReadinessDependenciesAntigravity:
    clean_head: Callable[[], str]
    sha256_file: Callable[[Path], str]
    exp437_terminal: Callable[[], Mapping[str, object]]
    input_session_factory: Callable[[str], object]
    assess_backends: Callable[[], tuple[BackendDecision, Mapping[str, object]]]
    focused_tests: Callable[[], Mapping[str, object]]
    exp438_prefix: Callable[[], object] = validate_exp438_prefix_antigravity

    def __post_init__(self) -> None:
        if any(not callable(getattr(self, name)) for name in self.__dataclass_fields__):
            raise ValueError("Exp439 readiness dependency is invalid")


def _digest_value_antigravity(value: object) -> str:
    return sha256(canonical_json_bytes(value)).hexdigest()


def _write_create_only_antigravity(path: Path, raw: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    parent = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


def _create_readiness_antigravity(
    *,
    run_parent: Path,
    source_paths: Mapping[str, Path],
    dependencies: ReadinessDependenciesAntigravity,
    expected_datasets: int,
) -> dict[str, object]:
    run_parent = Path(run_parent)
    if run_parent.exists() or run_parent.is_symlink():
        raise ValueError("Exp439 run parent already exists")
    if set(source_paths) != set(_SOURCE_NAMES_ANTIGRAVITY):
        raise ValueError("Exp439 readiness source paths are invalid")
    if type(expected_datasets) is not int or expected_datasets < 1 or expected_datasets > FROZEN_CONFIG.dataset_count:
        raise ValueError("Exp439 readiness population is invalid")
    commit = dependencies.clean_head()
    if type(commit) is not str or len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("Exp439 readiness clean HEAD is invalid")
    predecessor = dict(dependencies.exp437_terminal())
    try:
        predecessor_mode = classify_exp437_snapshot_antigravity(predecessor)
    except ValueError as error:
        raise ValueError("Exp437 execution is not safe for Exp439 readiness") from error
    prefix: Exp438PrefixImportAntigravity | None = None
    prefix_manifest: dict[str, object] | None = None
    if expected_datasets == FROZEN_CONFIG.dataset_count:
        candidate = dependencies.exp438_prefix()
        if (
            type(candidate) is not Exp438PrefixImportAntigravity
            or candidate.prediction_label_access_count != 0
        ):
            raise ValueError("Exp438 prefix is not safe for Exp439 readiness")
        prefix = candidate
        prefix_manifest = prefix.manifest_payload_antigravity()
    hashes: dict[str, str] = {}
    for name in _SOURCE_NAMES_ANTIGRAVITY:
        path = Path(source_paths[name])
        if not path.is_absolute() or not path.is_file() or path.is_symlink():
            raise ValueError(f"Exp439 readiness source is invalid: {name}")
        digest = dependencies.sha256_file(path)
        if type(digest) is not str or len(digest) != 64:
            raise ValueError("Exp439 readiness source digest is invalid")
        hashes[name] = digest
    if hashes["database"] != FROZEN_CONFIG.database_sha256:
        raise ValueError("Exp439 readiness database digest does not match")
    owned = dependencies.input_session_factory(commit)
    manager = owned if hasattr(owned, "__enter__") else None
    if manager is None:
        raise ValueError("Exp439 readiness input factory must return a context manager")
    with manager as session:
        context = getattr(session, "context", None)
        datasets = tuple(getattr(context, "datasets", ()))
        manifest = getattr(context, "test_identity_manifest", None)
        if (
            len(datasets) != expected_datasets
            or datasets != tuple(sorted(datasets))
            or len(set(datasets)) != len(datasets)
            or not isinstance(manifest, Mapping)
            or set(manifest) != set(datasets)
            or getattr(session, "label_access_count", None) != 0
        ):
            raise ValueError("Exp439 readiness input population is invalid")
        manifest_payload = {
            dataset: [[int(index), int(length)] for index, length in manifest[dataset]]
            for dataset in datasets
        }
    if prefix is not None and prefix.datasets != datasets:
        raise ValueError("Exp438 prefix dataset order does not match Exp439 readiness")
    decision, provenance_raw = dependencies.assess_backends()
    if type(decision) is not BackendDecision or not isinstance(provenance_raw, Mapping):
        raise ValueError("Exp439 readiness backend assessment is invalid")
    focused = dict(dependencies.focused_tests())
    if focused.get("returncode") != 0:
        raise ValueError("Exp439 readiness focused tests failed")
    payload: dict[str, object] = {
        "schema": "exp439.readiness.v1",
        "experiment_id": FROZEN_CONFIG.experiment_id,
        "run_identity": FROZEN_CONFIG.run_identity,
        "reviewed_commit": commit,
        "database_sha256": hashes["database"],
        "source_paths": {name: str(Path(source_paths[name]).resolve()) for name in _SOURCE_NAMES_ANTIGRAVITY},
        "source_sha256": hashes,
        "datasets": list(datasets),
        "dataset_order_sha256": _digest_value_antigravity(list(datasets)),
        "test_identity_manifest_sha256": _digest_value_antigravity(manifest_payload),
        "frozen_config": frozen_config_payload_antigravity(),
        "frozen_config_sha256": _digest_value_antigravity(frozen_config_payload_antigravity()),
        "backend_decision": asdict(decision),
        "backend_decision_sha256": _digest_value_antigravity(asdict(decision)),
        "backend_provenance": dict(provenance_raw),
        "focused_tests": focused,
        "exp437_execution": predecessor,
        "exp437_execution_mode": predecessor_mode,
        "concurrent_with_exp437": predecessor_mode == "concurrent",
        "test_values_opened": 0,
        "test_labels_opened": 0,
        "prediction_label_access_count": 0,
        "run_parent": str(run_parent.resolve()),
    }
    if prefix is not None and prefix_manifest is not None:
        payload["exp438_prefix_import"] = {
            "predecessor_root": str(prefix.source_root),
            "reviewed_commit": prefix.reviewed_commit,
            "database_sha256": prefix.database_sha256,
            "readiness_sha256": prefix.readiness_sha256,
            "context_sha256": prefix.context_sha256,
            "state_sha256": prefix.state_sha256,
            "progress_sha256": prefix.progress_sha256,
            "record_set_sha256": prefix.record_set_sha256,
            "record_count": len(prefix.records),
            "last_dataset": prefix.records[-1].dataset,
            "next_dataset": prefix.datasets[len(prefix.records)],
            "prediction_label_access_count": prefix.prediction_label_access_count,
        }
        payload["exp438_prefix_import_sha256"] = _digest_value_antigravity(
            prefix_manifest
        )
    run_parent.mkdir(mode=0o700)
    try:
        _write_create_only_antigravity(
            run_parent / "readiness_antigravity.json", canonical_json_bytes(payload)
        )
    except BaseException:
        try:
            run_parent.rmdir()
        except OSError:
            pass
        raise
    return json.loads(canonical_json_bytes(payload))


def _clean_head_antigravity() -> str:
    repository = Path(__file__).resolve().parents[2]
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=repository, text=True)
    if status:
        raise ValueError("Exp439 readiness requires a clean working tree")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()


def _exp437_terminal_antigravity() -> Mapping[str, object]:
    return read_exp437_snapshot_antigravity(_EXP437_PARENT_ANTIGRAVITY)


def _unavailable_evidence_antigravity(backend: str, reason: str):
    return tuple(
        BackendBatchEvidence(backend, batch, False, False, 1e30, 1.0, 1.0, 1.0, False, reason)
        for batch in FROZEN_CONFIG.batch_sizes
    )


def _assess_backends_antigravity() -> tuple[BackendDecision, Mapping[str, object]]:
    generator = np.random.default_rng(FROZEN_CONFIG.smoke_seed)
    references = generator.normal(size=(32, 16)).astype(np.float32)
    queries = generator.normal(size=(8, 16)).astype(np.float32)
    expected = np.asarray([
        score_channel_cpu_antigravity(query, references, FROZEN_CONFIG.nearest_k)
        for query in queries
    ], dtype=np.float32)
    thresholds = expected + np.asarray([(-1 if index % 2 else 1) * 1e-4 for index in range(expected.size)], dtype=np.float32)
    fixture = BackendFixtureAntigravity(expected, thresholds)

    def cpu_runner(fixture_value, batch_size):
        return np.asarray([
            score_channel_cpu_antigravity(query, references, FROZEN_CONFIG.nearest_k)
            for query in queries
        ], dtype=np.float32)

    evidence: list[BackendBatchEvidence] = []
    provenance: dict[str, object] = {}
    try:
        from .numba_antigravity import create_numba_aspe_engine_antigravity

        signals = generator.normal(size=(8, 128)).astype(np.float32)

        def cpu_aspe(fixture_value, batch_size):
            return np.asarray([
                amplitude_sensitive_permutation_entropy_antigravity(
                    signal, FROZEN_CONFIG.embedding_dimension, 1
                ).value
                for signal in signals
            ], dtype=np.float32)

        expected_aspe = cpu_aspe(None, 16)
        aspe_thresholds = expected_aspe + np.asarray(
            [(-1 if index % 2 else 1) * 1e-5 for index in range(expected_aspe.size)],
            dtype=np.float32,
        )
        aspe_fixture = BackendFixtureAntigravity(expected_aspe, aspe_thresholds)
        numba_engine = create_numba_aspe_engine_antigravity()
        compile_start = time.perf_counter()
        numba_engine.compute(signals[0], FROZEN_CONFIG.embedding_dimension, 1)
        compilation_seconds = time.perf_counter() - compile_start

        def numba_aspe(fixture_value, batch_size):
            return np.asarray([
                numba_engine.compute(signal, FROZEN_CONFIG.embedding_dimension, 1).value
                for signal in signals
            ], dtype=np.float32)

        measured = benchmark_candidate_antigravity(
            "numba", aspe_fixture, cpu_aspe, numba_aspe
        )
        amortization_runs = FROZEN_CONFIG.warmup_runs + FROZEN_CONFIG.timed_runs
        adjusted = []
        for item in measured:
            device_seconds = item.device_median_seconds + compilation_seconds / amortization_runs
            speedup = item.cpu_median_seconds / device_seconds
            passed = (
                item.finite
                and item.candidate_parity
                and item.maximum_absolute_error <= FROZEN_CONFIG.parity_tolerance
                and speedup >= FROZEN_CONFIG.minimum_speedup
            )
            adjusted.append(BackendBatchEvidence(
                backend="numba",
                batch_size=item.batch_size,
                finite=item.finite,
                candidate_parity=item.candidate_parity,
                maximum_absolute_error=item.maximum_absolute_error,
                cpu_median_seconds=item.cpu_median_seconds,
                device_median_seconds=device_seconds,
                speedup=speedup,
                passed=passed,
                reason=None if passed else "compile_amortized_speedup_below_1.1_or_parity_failed",
            ))
        evidence.extend(adjusted)
        provenance["numba"] = {
            "version": numba_engine.numba_version,
            "compilation_seconds": compilation_seconds,
            "compilation_amortization_runs": amortization_runs,
        }
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        provenance["numba"] = {"unavailable": True, "reason": str(error)}
        evidence.extend(_unavailable_evidence_antigravity("numba", str(error)))
    for backend in ("mlx", "mps"):
        try:
            if backend == "mlx":
                from .mlx_antigravity import create_mlx_dense_engine_antigravity
                engine = create_mlx_dense_engine_antigravity()
                provenance[backend] = {
                    "version": engine.mlx_version,
                    "mlx_metal_version": engine.mlx_metal_version,
                    "device": engine.device,
                }
            else:
                from .mps_antigravity import create_mps_dense_engine_antigravity
                engine = create_mps_dense_engine_antigravity()
                provenance[backend] = {"version": engine.pytorch_version, "device": engine.device}

            def device_runner(fixture_value, batch_size, engine=engine):
                return engine.score_batch(
                    queries, references, FROZEN_CONFIG.nearest_k, batch_size=batch_size
                )

            evidence.extend(
                benchmark_candidate_antigravity(
                    backend,
                    fixture,
                    cpu_runner,
                    device_runner,
                    synchronize=engine.synchronize,
                )
            )
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            provenance[backend] = {"unavailable": True, "reason": str(error)}
            evidence.extend(_unavailable_evidence_antigravity(backend, str(error)))
    decision = select_backend_antigravity(evidence)
    provenance["selected_backend"] = decision.selected_backend
    return decision, provenance


def _focused_tests_antigravity() -> Mapping[str, object]:
    repository = Path(__file__).resolve().parents[2]
    command = [
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        "experiments/exp439_scale_aware_reconstruction_repair_antigravity/tests",
        "-p",
        "test_*_antigravity.py",
    ]
    completed = subprocess.run(command, cwd=repository, text=True, capture_output=True, check=False)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout_sha256": sha256(completed.stdout.encode()).hexdigest(),
        "stderr_sha256": sha256(completed.stderr.encode()).hexdigest(),
    }


def create_readiness_antigravity() -> dict[str, object]:
    sources = canonical_source_paths_antigravity().to_payload()
    dependencies = ReadinessDependenciesAntigravity(
        clean_head=_clean_head_antigravity,
        sha256_file=sha256_file,
        exp437_terminal=_exp437_terminal_antigravity,
        input_session_factory=lambda commit: open_exp439_input_session(verified_code_commit=commit),
        assess_backends=_assess_backends_antigravity,
        focused_tests=_focused_tests_antigravity,
        exp438_prefix=validate_exp438_prefix_antigravity,
    )
    return _create_readiness_antigravity(
        run_parent=RUN_PARENT_ANTIGRAVITY,
        source_paths={name: Path(value) for name, value in sources.items()},
        dependencies=dependencies,
        expected_datasets=FROZEN_CONFIG.dataset_count,
    )


__all__ = [
    "ReadinessDependenciesAntigravity",
    "create_readiness_antigravity",
]
