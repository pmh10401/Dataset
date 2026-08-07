"""Fixed-path, label-free command firewall for Exp436.

The only mutable entrypoint before prediction is readiness.  It calculates
evidence while the run parent is absent, then publishes one canonical file by
renaming a same-parent staged directory.  Prediction commands never import the
evaluator; evaluation imports it lazily after publication is frozen.
"""
from __future__ import annotations

import argparse
import ctypes
from dataclasses import asdict, dataclass
import errno
from hashlib import sha256
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter
from typing import Callable, Mapping
from uuid import uuid4

from experiments.exp403_408.artifact_io import canonical_json_bytes

from .config_antigravity import FROZEN_CONFIG
from .execution_profile_antigravity import (
    EXP436_PROFILE,
    EXP437_PROFILE,
    ExecutionProfileAntigravity,
    frozen_config_for_profile_antigravity,
    validate_frozen_config_for_profile_antigravity,
    validated_profile_antigravity,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path("/Users/mac/Documents/Dataset")
DATABASE_PATH = DATA_ROOT / "univariate_ts.db"
RUN_PARENT = DATA_ROOT / "exp436_missing_safe_strict_consensus_antigravity_20260806"
FROZEN_OUTPUT = RUN_PARENT / "output_antigravity"
FROZEN_CHECKPOINT = RUN_PARENT / "checkpoint_antigravity"
FROZEN_READINESS = RUN_PARENT / "readiness_antigravity.json"
FROZEN_EVALUATION = RUN_PARENT / "evaluation_antigravity"
RUN_IDENTITY = FROZEN_CONFIG.run_identity
LANE_PATHS = (
    DATA_ROOT / "exp291_final_operational_routing_freeze/exp291_frozen_operational_routing.jsonl",
    DATA_ROOT / "exp291_final_operational_routing_freeze/exp291_prediction_manifest.json",
    DATA_ROOT / "exp292_standard_review_evidence_tiering/exp292_frozen_review_tiers.jsonl",
    DATA_ROOT / "exp292_standard_review_evidence_tiering/exp292_prediction_manifest.json",
)

_READINESS_KEYS = frozenset({
    "schema", "run_identity", "reviewed_commit", "database_path", "database_sha256",
    "exp291_prediction_path", "exp291_prediction_sha256", "exp291_manifest_path",
    "exp291_manifest_sha256", "exp292_prediction_path", "exp292_prediction_sha256",
    "exp292_manifest_path", "exp292_manifest_sha256", "datasets", "dataset_digest",
    "frozen_config", "frozen_config_sha256", "backend_decision", "backend_decision_sha256",
    "run_root", "checkpoint_root", "evaluation_root", "evaluation_root_absent",
    "test_values_opened", "test_labels_opened", "label_access_count", "focused_tests",
    "candidate_fixture", "cpu_topology", "runtime_versions",
})


@dataclass(frozen=True)
class _CliDependencies:
    """Injectable command edge for tests; production uses exact fixed roots."""
    output: Path
    checkpoint: Path
    readiness: Path
    current_head: Callable[[], str]
    readiness_call: Callable[[], Mapping[str, object]]
    execute: Callable[[str], Mapping[str, object]]
    inspect: Callable[[], Mapping[str, object]]
    evaluate: Callable[[], Mapping[str, object]]

    def __post_init__(self) -> None:
        for name in ("output", "checkpoint", "readiness"):
            object.__setattr__(self, name, Path(getattr(self, name)).absolute())
        for name in ("current_head", "readiness_call", "execute", "inspect", "evaluate"):
            if not callable(getattr(self, name)):
                raise ValueError(f"CLI dependency {name} is invalid")


def _fixed_paths(
    profile: ExecutionProfileAntigravity = EXP436_PROFILE,
) -> tuple[Path, Path, Path, Path, Path]:
    """Return the selected profile's fixed parent and four sibling artifacts.

    The module-level Exp436 names are retained solely as compatibility seams
    for existing tests and historical callers.  A non-default profile always
    resolves its roots directly from the immutable profile object.
    """

    selected = validated_profile_antigravity(profile)
    if selected == EXP436_PROFILE:
        return RUN_PARENT, FROZEN_OUTPUT, FROZEN_CHECKPOINT, FROZEN_READINESS, FROZEN_EVALUATION
    return selected.run_parent, selected.output, selected.checkpoint, selected.readiness, selected.evaluation


def build_parser(
    profile: ExecutionProfileAntigravity = EXP436_PROFILE,
) -> argparse.ArgumentParser:
    selected = validated_profile_antigravity(profile)
    parser = argparse.ArgumentParser(prog=selected.cli_program, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("readiness", "new", "resume", "inspect", "evaluate"):
        commands.add_parser(name, allow_abbrev=False)
    return parser


def _clean_head() -> str:
    head = subprocess.run(["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(REPOSITORY_ROOT), "status", "--porcelain"], check=True, capture_output=True, text=True).stdout
    return _validate_commit(head) if not dirty else (_ for _ in ()).throw(ValueError("clean Git worktree is required"))


def _validate_commit(value: object, label: str = "Git commit") -> str:
    if not isinstance(value, str) or len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{label} is invalid")
    return value


def _head(dependencies: _CliDependencies) -> str:
    return _validate_commit(dependencies.current_head(), "current Git commit")


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"input file is absent or invalid: {path}")
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is absent or invalid")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid") from error
    if not isinstance(value, dict) or canonical_json_bytes(value) != raw:
        raise ValueError(f"{label} is not canonical")
    return value


def _absolute(value: object, label: str) -> str:
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError(f"readiness {label} is invalid")
    return str(Path(value).resolve())


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"readiness {label} is invalid")
    return value


def _validate_readiness_payload(
    payload: Mapping[str, object],
    *,
    expected_commit: str,
    profile: ExecutionProfileAntigravity = EXP436_PROFILE,
) -> dict[str, object]:
    selected = validated_profile_antigravity(profile)
    config = frozen_config_for_profile_antigravity(selected)
    value = dict(payload)
    if set(value) != _READINESS_KEYS or value.get("schema") != "exp436.readiness-binding.v1":
        raise ValueError("readiness fields are invalid")
    if value.get("run_identity") != selected.run_identity or value.get("reviewed_commit") != _validate_commit(expected_commit, "readiness Git commit"):
        raise ValueError("readiness identity is invalid")
    roots = (_absolute(value["run_root"], "run_root"), _absolute(value["checkpoint_root"], "checkpoint_root"), _absolute(value["evaluation_root"], "evaluation_root"))
    if tuple(Path(item).name for item in roots) != ("output_antigravity", "checkpoint_antigravity", "evaluation_antigravity") or len({str(Path(item).parent) for item in roots}) != 1:
        raise ValueError("readiness root binding is invalid")
    for name in ("database", "exp291_prediction", "exp291_manifest", "exp292_prediction", "exp292_manifest"):
        _absolute(value[f"{name}_path"], f"{name}_path")
        _sha(value[f"{name}_sha256"], f"{name}_sha256")
    if value["database_sha256"] != config.database_sha256:
        raise ValueError("readiness database hash is invalid")
    if value["exp291_prediction_sha256"] != config.exp291_prediction_sha256 or value["exp291_manifest_sha256"] != config.exp291_manifest_sha256 or value["exp292_prediction_sha256"] != config.exp292_prediction_sha256 or value["exp292_manifest_sha256"] != config.exp292_manifest_sha256:
        raise ValueError("readiness lane hashes are invalid")
    if not isinstance(value["datasets"], list) or not value["datasets"] or sha256(canonical_json_bytes(value["datasets"])).hexdigest() != _sha(value["dataset_digest"], "dataset_digest"):
        raise ValueError("readiness dataset binding is invalid")
    frozen_config = json.loads(canonical_json_bytes(asdict(config)))
    if value["frozen_config"] != frozen_config or sha256(canonical_json_bytes(frozen_config)).hexdigest() != _sha(value["frozen_config_sha256"], "frozen_config_sha256"):
        raise ValueError("readiness frozen config is invalid")
    from .contracts_antigravity import BackendDecision
    decision = BackendDecision.from_payload(value["backend_decision"])
    if sha256(canonical_json_bytes(decision.to_payload())).hexdigest() != _sha(value["backend_decision_sha256"], "backend_decision_sha256"):
        raise ValueError("readiness backend decision is invalid")
    if value["evaluation_root_absent"] is not True or any(value[key] != 0 for key in ("test_values_opened", "test_labels_opened", "label_access_count")):
        raise ValueError("readiness label/evaluation evidence is invalid")
    focused, fixture, topology, versions = value["focused_tests"], value["candidate_fixture"], value["cpu_topology"], value["runtime_versions"]
    if not isinstance(focused, Mapping) or focused.get("returncode") != 0 or not isinstance(focused.get("command"), list):
        raise ValueError("readiness focused-test evidence is invalid")
    if not isinstance(fixture, Mapping) or set(fixture) != {"sha256", "candidate_indices"} or not isinstance(fixture["candidate_indices"], list):
        raise ValueError("readiness candidate fixture is invalid")
    _sha(fixture["sha256"], "candidate_fixture.sha256")
    expected_workers = max(1, (os.cpu_count() or 1) - 1)
    if not isinstance(topology, Mapping) or set(topology) != {"logical_cpus", "workers", "inner_threads"} or topology.get("logical_cpus") != (os.cpu_count() or 1) or topology.get("workers") != expected_workers or topology.get("inner_threads") != 1:
        raise ValueError("readiness CPU topology is invalid")
    if not isinstance(versions, Mapping) or set(versions) != {"python", "numpy", "torch", "mlx"} or not all(isinstance(item, str) and item for item in versions.values()):
        raise ValueError("readiness runtime versions are invalid")
    return value


def _write_create_only(path: Path, payload: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as handle:
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
    finally:
        os.close(fd)


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
    try: os.fsync(fd)
    finally: os.close(fd)


def _exclusive_rename(source: Path, destination: Path) -> None:
    """Darwin no-clobber directory rename for readiness ownership."""
    if sys.platform != "darwin":
        raise RuntimeError("exclusive directory rename is unsupported on this platform")
    try:
        renamex_np = ctypes.CDLL(None, use_errno=True).renamex_np
    except (AttributeError, OSError) as error:
        raise RuntimeError("Darwin renamex_np is unavailable") from error
    renamex_np.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
    renamex_np.restype = ctypes.c_int
    if renamex_np(os.fsencode(source), os.fsencode(destination), ctypes.c_uint(0x00000004)) != 0:
        code = ctypes.get_errno() or errno.EIO
        raise OSError(code, os.strerror(code), str(destination))


def _publish_readiness_atomically(readiness: Path, payload: Mapping[str, object]) -> None:
    """No-clobber creation of an absent run parent and its sole artifact."""
    readiness = Path(readiness)
    parent, container = readiness.parent, readiness.parent.parent
    if readiness.name != "readiness_antigravity.json" or parent.is_symlink() or parent.exists():
        raise ValueError("readiness requires an absent fixed run parent")
    if container.is_symlink() or not container.is_dir():
        raise ValueError("readiness parent container is absent or invalid")
    staging = container / f".{parent.name}.antigravity-staging-{uuid4().hex}"
    staging.mkdir(mode=0o700)
    published = False
    try:
        staged_readiness = staging / readiness.name
        raw = canonical_json_bytes(dict(payload))
        _write_create_only(staged_readiness, raw)
        if staged_readiness.read_bytes() != raw:
            raise ValueError("staged readiness verification failed")
        _fsync_directory(staging)
        # Darwin RENAME_EXCL refuses any concurrently created destination.
        _exclusive_rename(staging, parent)
        _fsync_directory(container)
        if (parent / readiness.name).read_bytes() != raw or tuple(item.name for item in parent.iterdir()) != (readiness.name,):
            raise ValueError("published readiness verification failed")
        published = True
    finally:
        if not published and staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)
            _fsync_directory(container)


def _runtime_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0]}
    for name in ("numpy", "torch", "mlx"):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = "unavailable"
    return versions


def _focused_test_evidence() -> dict[str, object]:
    command = [sys.executable, "-m", "unittest", "experiments.exp436_missing_safe_strict_consensus_antigravity.tests.test_reporting_antigravity", "experiments.exp436_missing_safe_strict_consensus_antigravity.tests.test_evaluation_antigravity", "experiments.exp436_missing_safe_strict_consensus_antigravity.tests.test_cli_antigravity"]
    started = perf_counter(); completed = subprocess.run(command, cwd=REPOSITORY_ROOT, capture_output=True, text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    return {"returncode": completed.returncode, "command": command[1:], "elapsed_seconds": perf_counter() - started, "stderr_sha256": sha256(completed.stderr.encode()).hexdigest()}


def _backend_decision_and_fixture(
    profile: ExecutionProfileAntigravity = EXP436_PROFILE,
) -> tuple[object, dict[str, object]]:
    selected = validated_profile_antigravity(profile)
    config = frozen_config_for_profile_antigravity(selected)
    import numpy as np
    from .backend_antigravity import select_backend
    from .features_antigravity import score_channel_cpu
    from .mlx_antigravity import MlxFixture, benchmark_mlx_adoption
    from .mps_antigravity import MpsFixture, benchmark_mps_adoption
    generator = np.random.default_rng(config.backend_smoke_seed)
    references = generator.normal(size=(32, 16)).astype(np.float32)
    query = generator.normal(size=16).astype(np.float32)
    # Task6's scalar fixture is the production dense-score primitive: exactly
    # one query vector, frozen k=3 references, and a finite threshold.
    threshold = float(np.float32(0.0))
    mlx_fixture, mps_fixture = MlxFixture(query, references, threshold), MpsFixture(query, references, threshold)
    decision = select_backend(benchmark_mlx_adoption((mlx_fixture,)), benchmark_mps_adoption((mps_fixture,)))
    cpu_score = float(score_channel_cpu(query, references, config.neighbor_count))
    candidates = [0] if cpu_score > threshold else []
    return decision, {"sha256": sha256(canonical_json_bytes({"query": query.tolist(), "references": references.tolist(), "threshold": threshold})).hexdigest(), "candidate_indices": candidates}


def _production_readiness(
    profile: ExecutionProfileAntigravity = EXP436_PROFILE,
) -> dict[str, object]:
    """Compute evidence before any run-parent side effect or label-capable call."""
    selected = validated_profile_antigravity(profile)
    _run_parent, output, checkpoint, readiness_path, evaluation = _fixed_paths(selected)
    config = frozen_config_for_profile_antigravity(selected)
    if any(path.exists() or path.is_symlink() for path in (_run_parent, output, checkpoint, evaluation, readiness_path)):
        raise ValueError("readiness requires every fixed root and parent to be absent")
    # Validate the shared scientific values against the immutable Exp436 base,
    # then validate the selected profile's two projected identity fields.
    FROZEN_CONFIG.validate()
    validate_frozen_config_for_profile_antigravity(config, selected)
    head = _clean_head()
    from .inputs_antigravity import DEFAULT_DATABASE, DEFAULT_EXP291_MANIFEST, DEFAULT_EXP291_PREDICTION, DEFAULT_EXP292_MANIFEST, DEFAULT_EXP292_PREDICTION, open_exp436_input_session
    with open_exp436_input_session(DEFAULT_DATABASE, DEFAULT_EXP291_PREDICTION, DEFAULT_EXP291_MANIFEST, DEFAULT_EXP292_PREDICTION, DEFAULT_EXP292_MANIFEST, verified_code_commit=head) as session:
        source = session.source_paths.to_payload(); datasets = list(session.context.datasets)
        if session.label_access_count != 0: raise ValueError("readiness input session exposed labels")
    decision, fixture = _backend_decision_and_fixture(selected)
    focused = _focused_test_evidence()
    if focused["returncode"] != 0: raise ValueError("focused tests failed during readiness")
    final_head = _clean_head()
    if final_head != head: raise ValueError("clean HEAD changed during readiness")
    config_payload = json.loads(canonical_json_bytes(asdict(config)))
    hashes = {"database": _sha256_file(DATABASE_PATH), "exp291_prediction": _sha256_file(LANE_PATHS[0]), "exp291_manifest": _sha256_file(LANE_PATHS[1]), "exp292_prediction": _sha256_file(LANE_PATHS[2]), "exp292_manifest": _sha256_file(LANE_PATHS[3])}
    if hashes["database"] != config.database_sha256: raise ValueError("database source hash changed")
    return {"schema": "exp436.readiness-binding.v1", "run_identity": selected.run_identity, "reviewed_commit": head,
        **{f"{name}_path": source[name] for name in hashes}, **{f"{name}_sha256": hashes[name] for name in hashes},
        "datasets": datasets, "dataset_digest": sha256(canonical_json_bytes(datasets)).hexdigest(),
        "frozen_config": config_payload, "frozen_config_sha256": sha256(canonical_json_bytes(config_payload)).hexdigest(),
        "backend_decision": decision.to_payload(), "backend_decision_sha256": sha256(canonical_json_bytes(decision.to_payload())).hexdigest(),
        "run_root": str(output.resolve()), "checkpoint_root": str(checkpoint.resolve()), "evaluation_root": str(evaluation.resolve()), "evaluation_root_absent": True,
        "test_values_opened": 0, "test_labels_opened": 0, "label_access_count": 0, "focused_tests": focused, "candidate_fixture": fixture,
        "cpu_topology": {"logical_cpus": os.cpu_count() or 1, "workers": max(1, (os.cpu_count() or 1) - 1), "inner_threads": 1}, "runtime_versions": _runtime_versions()}


def run_readiness(
    profile: ExecutionProfileAntigravity = EXP436_PROFILE,
) -> Mapping[str, object]:
    selected = validated_profile_antigravity(profile)
    _run_parent, _output, _checkpoint, readiness_path, _evaluation = _fixed_paths(selected)
    payload = _production_readiness(selected); _validate_readiness_payload(payload, expected_commit=payload["reviewed_commit"], profile=selected)
    if _clean_head() != payload["reviewed_commit"]: raise ValueError("clean HEAD changed before readiness publication")
    _publish_readiness_atomically(readiness_path, payload)
    return payload


def _preflight_roots(command: str, output: Path, checkpoint: Path, readiness: Path, evaluation: Path) -> None:
    if any(path.is_symlink() for path in (output, checkpoint, readiness, evaluation)):
        raise ValueError("fixed paths cannot be symlinks")
    if command == "readiness":
        if readiness.parent.exists(): raise ValueError("readiness requires an absent run parent")
    elif command == "new":
        if any(path.exists() for path in (output, checkpoint, evaluation)): raise ValueError("new roots must be absent")
        if not readiness.is_file(): raise ValueError("new readiness is absent")
    elif command in {"resume", "inspect", "evaluate"}:
        if not output.is_dir() or not checkpoint.is_dir(): raise ValueError("runtime output and checkpoint roots are absent or invalid")
        if command == "resume" and evaluation.exists(): raise ValueError("resume requires absent evaluation root")
    else: raise ValueError("unsupported Exp436 command")


def _resume_state(output: Path) -> str:
    return str(_read_json(Path(output) / "progress_antigravity.json", "runtime progress").get("state"))


_RESUME_ALLOWED_STATES = {
    EXP436_PROFILE: frozenset(("interrupted", "checkpoint_complete")),
    EXP437_PROFILE: frozenset(("interrupted",)),
}


def _validate_resume_state(
    state: object,
    profile: ExecutionProfileAntigravity = EXP436_PROFILE,
) -> str:
    selected = validated_profile_antigravity(profile)
    allowed = _RESUME_ALLOWED_STATES[selected]
    if not isinstance(state, str) or state not in allowed:
        raise ValueError(f"resume rejects runtime state: {state}")
    return state


def _summary_payload(summary: object, publication: object | None = None) -> dict[str, object]:
    payload = summary.to_payload() if hasattr(summary, "to_payload") else dict(summary)
    if publication is not None:
        manifest = publication.manifest
        payload.update({"state": "complete_review_only", "prediction_sha256": publication.prediction_sha256, "manifest_sha256": publication.manifest_sha256, "artifact_sha256": manifest.get("artifact_sha256")})
    return dict(payload)


def _input_factory(
    head: str,
    profile: ExecutionProfileAntigravity = EXP436_PROFILE,
):
    selected = validated_profile_antigravity(profile)
    from .inputs_antigravity import DEFAULT_DATABASE, DEFAULT_EXP291_MANIFEST, DEFAULT_EXP291_PREDICTION, DEFAULT_EXP292_MANIFEST, DEFAULT_EXP292_PREDICTION, open_exp436_input_session
    # ``selected`` is validated at the boundary even though the shared input
    # session intentionally keeps its scientific source arguments unchanged.
    return lambda: open_exp436_input_session(DEFAULT_DATABASE, DEFAULT_EXP291_PREDICTION, DEFAULT_EXP291_MANIFEST, DEFAULT_EXP292_PREDICTION, DEFAULT_EXP292_MANIFEST, verified_code_commit=head)


def _publish_after_complete(
    summary: object,
    profile: ExecutionProfileAntigravity = EXP436_PROFILE,
) -> Mapping[str, object]:
    selected = validated_profile_antigravity(profile)
    _run_parent, output, checkpoint, _readiness, _evaluation = _fixed_paths(selected)
    from .checkpoint_antigravity import Exp436CheckpointStore
    from .reporting_antigravity import inspect_publication, publish_exp436, PUBLICATION_DIRECTORY
    store = Exp436CheckpointStore.open_read_only(checkpoint, profile=selected)
    publication = inspect_publication(output, store, profile=selected) if (output / PUBLICATION_DIRECTORY).exists() else publish_exp436(store, output, profile=selected)
    return _summary_payload(summary, publication)


def _run_prediction(
    command: str,
    profile: ExecutionProfileAntigravity = EXP436_PROFILE,
) -> Mapping[str, object]:
    """Import only input/runtime after an exact readiness/head recheck."""
    selected = validated_profile_antigravity(profile)
    _run_parent, output, checkpoint, readiness_path, _evaluation = _fixed_paths(selected)
    head = _clean_head()
    if command == "new":
        readiness = _read_json(readiness_path, "readiness input")
        _validate_readiness_payload(readiness, expected_commit=head, profile=selected)
    from .runtime_antigravity import run_exp436_new, resume_exp436
    final_head = _clean_head()
    if final_head != head: raise ValueError("clean HEAD changed before input session")
    factory = _input_factory(final_head, selected)
    summary = run_exp436_new(readiness_path=readiness_path, run_root=output, checkpoint_root=checkpoint, reviewed_commit=final_head, input_session_factory=factory, profile=selected) if command == "new" else resume_exp436(readiness_path=readiness_path, run_root=output, checkpoint_root=checkpoint, reviewed_commit=final_head, input_session_factory=factory, profile=selected)
    return _publish_after_complete(summary, selected) if summary.status == "checkpoint_complete" else _summary_payload(summary)


def run_new(
    profile: ExecutionProfileAntigravity = EXP436_PROFILE,
) -> Mapping[str, object]:
    selected = validated_profile_antigravity(profile)
    _run_parent, output, checkpoint, readiness_path, evaluation = _fixed_paths(selected)
    _preflight_roots("new", output, checkpoint, readiness_path, evaluation)
    return _run_prediction("new", selected)


def run_resume(
    profile: ExecutionProfileAntigravity = EXP436_PROFILE,
) -> Mapping[str, object]:
    selected = validated_profile_antigravity(profile)
    _run_parent, output, checkpoint, readiness_path, evaluation = _fixed_paths(selected)
    _preflight_roots("resume", output, checkpoint, readiness_path, evaluation)
    _validate_resume_state(_resume_state(output), selected)
    return _run_prediction("resume", selected)


def run_inspect(
    profile: ExecutionProfileAntigravity = EXP436_PROFILE,
) -> Mapping[str, object]:
    selected = validated_profile_antigravity(profile)
    _run_parent, output, checkpoint, readiness_path, evaluation = _fixed_paths(selected)
    _preflight_roots("inspect", output, checkpoint, readiness_path, evaluation)
    from .reporting_antigravity import PUBLICATION_DIRECTORY, inspect_publication
    if (output / PUBLICATION_DIRECTORY).is_dir():
        from .checkpoint_antigravity import Exp436CheckpointStore
        publication = inspect_publication(output, Exp436CheckpointStore.open_read_only(checkpoint, profile=selected), profile=selected)
        manifest = publication.manifest
        return {"state": "complete_review_only", "run_identity": manifest.get("run_identity"),
                "completed_datasets": manifest.get("completed_datasets"), "candidate_count": manifest.get("candidate_count"),
                "prediction_sha256": publication.prediction_sha256, "manifest_sha256": publication.manifest_sha256}
    from .runtime_antigravity import inspect_exp436
    summary = inspect_exp436(readiness_path=readiness_path, run_root=output, checkpoint_root=checkpoint, reviewed_commit=_clean_head(), profile=selected)
    return _summary_payload(summary)


def run_evaluate(
    profile: ExecutionProfileAntigravity = EXP436_PROFILE,
) -> Mapping[str, object]:
    selected = validated_profile_antigravity(profile)
    _run_parent, output, _checkpoint, _readiness, evaluation = _fixed_paths(selected)
    evaluator = importlib.import_module("experiments.exp436_missing_safe_strict_consensus_antigravity.evaluation_antigravity")
    result = evaluator.evaluate_exp436(output, evaluation, profile=selected)
    return result.to_payload() if hasattr(result, "to_payload") else dict(result)


def _production_dependencies(
    profile: ExecutionProfileAntigravity = EXP436_PROFILE,
) -> _CliDependencies:
    selected = validated_profile_antigravity(profile)
    _run_parent, output, checkpoint, readiness_path, _evaluation = _fixed_paths(selected)
    return _CliDependencies(
        output,
        checkpoint,
        readiness_path,
        _clean_head,
        lambda: run_readiness(selected),
        lambda command: run_new(selected) if command == "new" else run_resume(selected),
        lambda: run_inspect(selected),
        lambda: run_evaluate(selected),
    )


def _main_with_dependencies(
    argv: list[str] | None,
    deps: _CliDependencies,
    *,
    profile: ExecutionProfileAntigravity = EXP436_PROFILE,
) -> int:
    """Private test seam; the exported CLI cannot accept dependency overrides."""
    selected = validated_profile_antigravity(profile)
    try: args = build_parser(selected).parse_args(argv)
    except SystemExit as error: return int(error.code) if isinstance(error.code, int) else 2
    try:
        evaluation = deps.output.parent / "evaluation_antigravity"
        _preflight_roots(args.command, deps.output, deps.checkpoint, deps.readiness, evaluation)
        if args.command == "readiness":
            payload = dict(deps.readiness_call()); verified = _validate_readiness_payload(payload, expected_commit=payload.get("reviewed_commit"), profile=selected)
            if deps.readiness.is_file() and _read_json(deps.readiness, "readiness artifact") != verified: raise ValueError("readiness artifact changed after action")
            result = verified
        elif args.command == "new":
            verified_head = _head(deps); _validate_readiness_payload(_read_json(deps.readiness, "readiness input"), expected_commit=verified_head, profile=selected)
            if _head(deps) != verified_head: raise ValueError("clean HEAD changed before execution")
            result = dict(deps.execute("new"))
        elif args.command == "resume":
            _validate_resume_state(_resume_state(deps.output), selected)
            result = dict(deps.execute("resume"))
        elif args.command == "inspect": result = dict(deps.inspect())
        else: result = dict(deps.evaluate())
        print(json.dumps(result, sort_keys=True))
        if args.command == "readiness": return 0
        if args.command == "evaluate": return 0 if result.get("state") in {"evaluated", "complete", "success"} else 1
        return 0 if result.get("state") in {"complete_review_only", "checkpoint_complete", "complete"} else 1
    except (OSError, ValueError) as error:
        print(f"{args.command} failed: {type(error).__name__}: {error}", file=sys.stderr); return 2


def main_for_profile_antigravity(
    profile: ExecutionProfileAntigravity,
    argv: list[str] | None = None,
) -> int:
    selected = validated_profile_antigravity(profile)
    return _main_with_dependencies(
        argv,
        _production_dependencies(selected),
        profile=selected,
    )


def main(argv: list[str] | None = None) -> int:
    """Existing Exp436-only command entrypoint."""

    return main_for_profile_antigravity(EXP436_PROFILE, argv)


__all__ = ["DATABASE_PATH", "FROZEN_CHECKPOINT", "FROZEN_EVALUATION", "FROZEN_OUTPUT", "FROZEN_READINESS", "LANE_PATHS", "RUN_IDENTITY", "build_parser", "main", "main_for_profile_antigravity", "run_evaluate", "run_inspect", "run_new", "run_readiness", "run_resume"]


def _revalidate_readiness_for_execution(
    readiness: Path,
    *,
    validated_commit: str | None,
    current_head: str,
    profile: ExecutionProfileAntigravity = EXP436_PROFILE,
) -> dict[str, object]:
    if validated_commit is None or validated_commit != current_head: raise ValueError("readiness commit changed before execution")
    return _validate_readiness_payload(_read_json(Path(readiness), "readiness input"), expected_commit=current_head, profile=profile)
