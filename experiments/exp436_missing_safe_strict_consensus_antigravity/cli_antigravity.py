"""Fixed-path, label-free command firewall for Exp436.

The only mutable entrypoint before prediction is readiness.  It calculates
evidence while the run parent is absent, then publishes one canonical file by
renaming a same-parent staged directory.  Prediction commands never import the
evaluator; evaluation imports it lazily after publication is frozen.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_exp436_missing_safe_strict_consensus_antigravity.py", allow_abbrev=False)
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


def _validate_readiness_payload(payload: Mapping[str, object], *, expected_commit: str) -> dict[str, object]:
    value = dict(payload)
    if set(value) != _READINESS_KEYS or value.get("schema") != "exp436.readiness-binding.v1":
        raise ValueError("readiness fields are invalid")
    if value.get("run_identity") != RUN_IDENTITY or value.get("reviewed_commit") != _validate_commit(expected_commit, "readiness Git commit"):
        raise ValueError("readiness identity is invalid")
    roots = (_absolute(value["run_root"], "run_root"), _absolute(value["checkpoint_root"], "checkpoint_root"), _absolute(value["evaluation_root"], "evaluation_root"))
    if tuple(Path(item).name for item in roots) != ("output_antigravity", "checkpoint_antigravity", "evaluation_antigravity") or len({str(Path(item).parent) for item in roots}) != 1:
        raise ValueError("readiness root binding is invalid")
    for name in ("database", "exp291_prediction", "exp291_manifest", "exp292_prediction", "exp292_manifest"):
        _absolute(value[f"{name}_path"], f"{name}_path")
        _sha(value[f"{name}_sha256"], f"{name}_sha256")
    if value["database_sha256"] != FROZEN_CONFIG.database_sha256:
        raise ValueError("readiness database hash is invalid")
    if value["exp291_prediction_sha256"] != FROZEN_CONFIG.exp291_prediction_sha256 or value["exp291_manifest_sha256"] != FROZEN_CONFIG.exp291_manifest_sha256 or value["exp292_prediction_sha256"] != FROZEN_CONFIG.exp292_prediction_sha256 or value["exp292_manifest_sha256"] != FROZEN_CONFIG.exp292_manifest_sha256:
        raise ValueError("readiness lane hashes are invalid")
    if not isinstance(value["datasets"], list) or not value["datasets"] or sha256(canonical_json_bytes(value["datasets"])).hexdigest() != _sha(value["dataset_digest"], "dataset_digest"):
        raise ValueError("readiness dataset binding is invalid")
    config = json.loads(canonical_json_bytes(asdict(FROZEN_CONFIG)))
    if value["frozen_config"] != config or sha256(canonical_json_bytes(config)).hexdigest() != _sha(value["frozen_config_sha256"], "frozen_config_sha256"):
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
        # POSIX directory rename is exclusive when destination is absent and
        # fails if a concurrent owner created a non-empty destination.
        os.rename(staging, parent)
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


def _backend_decision_and_fixture() -> tuple[object, dict[str, object]]:
    import numpy as np
    from .backend_antigravity import select_backend
    from .mlx_antigravity import MlxFixture, benchmark_mlx_adoption
    from .mps_antigravity import MpsFixture, benchmark_mps_adoption
    generator = np.random.default_rng(FROZEN_CONFIG.backend_smoke_seed)
    references = generator.normal(size=(32, 16)).astype(np.float32); queries = generator.normal(size=(1, 16)).astype(np.float32)
    mlx_fixture, mps_fixture = MlxFixture(queries, references), MpsFixture(queries, references)
    decision = select_backend(benchmark_mlx_adoption((mlx_fixture,)), benchmark_mps_adoption((mps_fixture,)))
    return decision, {"sha256": sha256(canonical_json_bytes({"queries": queries.tolist(), "references": references.tolist()})).hexdigest(), "candidate_indices": []}


def _production_readiness() -> dict[str, object]:
    """Compute evidence before any run-parent side effect or label-capable call."""
    if any(path.exists() or path.is_symlink() for path in (RUN_PARENT, FROZEN_OUTPUT, FROZEN_CHECKPOINT, FROZEN_EVALUATION, FROZEN_READINESS)):
        raise ValueError("readiness requires every fixed root and parent to be absent")
    FROZEN_CONFIG.validate(); head = _clean_head()
    from .inputs_antigravity import DEFAULT_DATABASE, DEFAULT_EXP291_MANIFEST, DEFAULT_EXP291_PREDICTION, DEFAULT_EXP292_MANIFEST, DEFAULT_EXP292_PREDICTION, open_exp436_input_session
    with open_exp436_input_session(DEFAULT_DATABASE, DEFAULT_EXP291_PREDICTION, DEFAULT_EXP291_MANIFEST, DEFAULT_EXP292_PREDICTION, DEFAULT_EXP292_MANIFEST, verified_code_commit=head) as session:
        source = session.source_paths.to_payload(); datasets = list(session.context.datasets)
        if session.label_access_count != 0: raise ValueError("readiness input session exposed labels")
    decision, fixture = _backend_decision_and_fixture()
    focused = _focused_test_evidence()
    if focused["returncode"] != 0: raise ValueError("focused tests failed during readiness")
    final_head = _clean_head()
    if final_head != head: raise ValueError("clean HEAD changed during readiness")
    config = json.loads(canonical_json_bytes(asdict(FROZEN_CONFIG)))
    hashes = {"database": _sha256_file(DATABASE_PATH), "exp291_prediction": _sha256_file(LANE_PATHS[0]), "exp291_manifest": _sha256_file(LANE_PATHS[1]), "exp292_prediction": _sha256_file(LANE_PATHS[2]), "exp292_manifest": _sha256_file(LANE_PATHS[3])}
    if hashes["database"] != FROZEN_CONFIG.database_sha256: raise ValueError("database source hash changed")
    return {"schema": "exp436.readiness-binding.v1", "run_identity": RUN_IDENTITY, "reviewed_commit": head,
        **{f"{name}_path": source[name] for name in hashes}, **{f"{name}_sha256": hashes[name] for name in hashes},
        "datasets": datasets, "dataset_digest": sha256(canonical_json_bytes(datasets)).hexdigest(),
        "frozen_config": config, "frozen_config_sha256": sha256(canonical_json_bytes(config)).hexdigest(),
        "backend_decision": decision.to_payload(), "backend_decision_sha256": sha256(canonical_json_bytes(decision.to_payload())).hexdigest(),
        "run_root": str(FROZEN_OUTPUT.resolve()), "checkpoint_root": str(FROZEN_CHECKPOINT.resolve()), "evaluation_root": str(FROZEN_EVALUATION.resolve()), "evaluation_root_absent": True,
        "test_values_opened": 0, "test_labels_opened": 0, "label_access_count": 0, "focused_tests": focused, "candidate_fixture": fixture,
        "cpu_topology": {"logical_cpus": os.cpu_count() or 1, "workers": max(1, (os.cpu_count() or 1) - 1), "inner_threads": 1}, "runtime_versions": _runtime_versions()}


def run_readiness() -> Mapping[str, object]:
    payload = _production_readiness(); _validate_readiness_payload(payload, expected_commit=payload["reviewed_commit"])
    if _clean_head() != payload["reviewed_commit"]: raise ValueError("clean HEAD changed before readiness publication")
    _publish_readiness_atomically(FROZEN_READINESS, payload)
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


def _summary_payload(summary: object, publication: object | None = None) -> dict[str, object]:
    payload = summary.to_payload() if hasattr(summary, "to_payload") else dict(summary)
    if publication is not None:
        manifest = publication.manifest
        payload.update({"state": "complete_review_only", "prediction_sha256": publication.prediction_sha256, "manifest_sha256": publication.manifest_sha256, "artifact_sha256": manifest.get("artifact_sha256")})
    return dict(payload)


def _input_factory(head: str):
    from .inputs_antigravity import DEFAULT_DATABASE, DEFAULT_EXP291_MANIFEST, DEFAULT_EXP291_PREDICTION, DEFAULT_EXP292_MANIFEST, DEFAULT_EXP292_PREDICTION, open_exp436_input_session
    return lambda: open_exp436_input_session(DEFAULT_DATABASE, DEFAULT_EXP291_PREDICTION, DEFAULT_EXP291_MANIFEST, DEFAULT_EXP292_PREDICTION, DEFAULT_EXP292_MANIFEST, verified_code_commit=head)


def _publish_after_complete(summary: object) -> Mapping[str, object]:
    from .checkpoint_antigravity import Exp436CheckpointStore
    from .reporting_antigravity import inspect_publication, publish_exp436, PUBLICATION_DIRECTORY
    store = Exp436CheckpointStore.open_read_only(FROZEN_CHECKPOINT)
    publication = inspect_publication(FROZEN_OUTPUT, store) if (FROZEN_OUTPUT / PUBLICATION_DIRECTORY).exists() else publish_exp436(store, FROZEN_OUTPUT)
    return _summary_payload(summary, publication)


def _run_prediction(command: str) -> Mapping[str, object]:
    """Import only input/runtime after an exact readiness/head recheck."""
    head = _clean_head()
    if command == "new":
        readiness = _read_json(FROZEN_READINESS, "readiness input")
        _validate_readiness_payload(readiness, expected_commit=head)
    from .runtime_antigravity import run_exp436_new, resume_exp436
    final_head = _clean_head()
    if final_head != head: raise ValueError("clean HEAD changed before input session")
    factory = _input_factory(final_head)
    summary = run_exp436_new(readiness_path=FROZEN_READINESS, run_root=FROZEN_OUTPUT, checkpoint_root=FROZEN_CHECKPOINT, reviewed_commit=final_head, input_session_factory=factory) if command == "new" else resume_exp436(readiness_path=FROZEN_READINESS, run_root=FROZEN_OUTPUT, checkpoint_root=FROZEN_CHECKPOINT, reviewed_commit=final_head, input_session_factory=factory)
    return _publish_after_complete(summary) if summary.status == "checkpoint_complete" else _summary_payload(summary)


def run_new() -> Mapping[str, object]:
    _preflight_roots("new", FROZEN_OUTPUT, FROZEN_CHECKPOINT, FROZEN_READINESS, FROZEN_EVALUATION)
    return _run_prediction("new")


def run_resume() -> Mapping[str, object]:
    _preflight_roots("resume", FROZEN_OUTPUT, FROZEN_CHECKPOINT, FROZEN_READINESS, FROZEN_EVALUATION)
    state = _resume_state(FROZEN_OUTPUT)
    if state not in {"interrupted", "checkpoint_complete"}:
        raise ValueError(f"resume rejects runtime state: {state}")
    return _run_prediction("resume")


def run_inspect() -> Mapping[str, object]:
    _preflight_roots("inspect", FROZEN_OUTPUT, FROZEN_CHECKPOINT, FROZEN_READINESS, FROZEN_EVALUATION)
    from .runtime_antigravity import inspect_exp436
    summary = inspect_exp436(readiness_path=FROZEN_READINESS, run_root=FROZEN_OUTPUT, checkpoint_root=FROZEN_CHECKPOINT, reviewed_commit=_clean_head())
    return _summary_payload(summary)


def run_evaluate() -> Mapping[str, object]:
    evaluator = importlib.import_module("experiments.exp436_missing_safe_strict_consensus_antigravity.evaluation_antigravity")
    result = evaluator.evaluate_exp436(FROZEN_OUTPUT, FROZEN_EVALUATION)
    return result.to_payload() if hasattr(result, "to_payload") else dict(result)


def _production_dependencies() -> _CliDependencies:
    return _CliDependencies(FROZEN_OUTPUT, FROZEN_CHECKPOINT, FROZEN_READINESS, _clean_head, run_readiness,
                            lambda command: run_new() if command == "new" else run_resume(), run_inspect, run_evaluate)


def main(argv: list[str] | None = None, *, _dependencies: _CliDependencies | None = None) -> int:
    deps = _production_dependencies() if _dependencies is None else _dependencies
    try: args = build_parser().parse_args(argv)
    except SystemExit as error: return int(error.code) if isinstance(error.code, int) else 2
    try:
        evaluation = deps.output.parent / "evaluation_antigravity"
        _preflight_roots(args.command, deps.output, deps.checkpoint, deps.readiness, evaluation)
        if args.command == "readiness":
            payload = dict(deps.readiness_call()); verified = _validate_readiness_payload(payload, expected_commit=payload.get("reviewed_commit"))
            if deps.readiness.is_file() and _read_json(deps.readiness, "readiness artifact") != verified: raise ValueError("readiness artifact changed after action")
            result = verified
        elif args.command == "new":
            verified_head = _head(deps); _validate_readiness_payload(_read_json(deps.readiness, "readiness input"), expected_commit=verified_head)
            if _head(deps) != verified_head: raise ValueError("clean HEAD changed before execution")
            result = dict(deps.execute("new"))
        elif args.command == "resume":
            state = _resume_state(deps.output)
            if state not in {"interrupted", "checkpoint_complete"}: raise ValueError(f"resume rejects runtime state: {state}")
            result = dict(deps.execute("resume"))
        elif args.command == "inspect": result = dict(deps.inspect())
        else: result = dict(deps.evaluate())
        print(json.dumps(result, sort_keys=True))
        if args.command == "evaluate": return 0 if result.get("state") in {"evaluated", "complete", "success"} else 1
        return 0 if result.get("state") in {"complete_review_only", "checkpoint_complete", "complete"} else 1
    except (OSError, ValueError) as error:
        print(f"{args.command} failed: {type(error).__name__}: {error}", file=sys.stderr); return 2


__all__ = ["DATABASE_PATH", "FROZEN_CHECKPOINT", "FROZEN_EVALUATION", "FROZEN_OUTPUT", "FROZEN_READINESS", "LANE_PATHS", "RUN_IDENTITY", "_CliDependencies", "_publish_readiness_atomically", "_revalidate_readiness_for_execution", "_validate_readiness_payload", "build_parser", "main", "run_evaluate", "run_inspect", "run_new", "run_readiness", "run_resume"]


def _revalidate_readiness_for_execution(readiness: Path, *, validated_commit: str | None, current_head: str) -> dict[str, object]:
    if validated_commit is None or validated_commit != current_head: raise ValueError("readiness commit changed before execution")
    return _validate_readiness_payload(_read_json(Path(readiness), "readiness input"), expected_commit=current_head)
