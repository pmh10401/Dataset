"""Fixed-path five-mode boundary for Exp435.

Prediction modes never import the retrospective evaluator.  Readiness binds a
clean commit and a backend decision; ``new`` rechecks that binding immediately
before opening the descriptor-bound input session.  All paths are constants
from the preregistration and alternate output arguments are intentionally not
accepted.
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
import subprocess
import sys
from time import perf_counter
from typing import Callable, Mapping

from experiments.exp403_408.artifact_io import canonical_json_bytes

from .config_antigravity import FROZEN_CONFIG
from .reporting_antigravity import (
    MANIFEST_FILE,
    PREDICTION_FILE,
    FrozenPublication,
    _backend_from_context,
    _fsync_directory,
    _read_json,
    _write_new,
    inspect_publication,
    publish_exp435,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path("/Users/mac/Documents/Dataset")
DATABASE_PATH = DATA_ROOT / "univariate_ts.db"
RUN_PARENT = DATA_ROOT / "exp435_train_only_consensus_recovery_antigravity_20260806"
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
    "schema", "state", "experiment_id", "run_identity", "git_commit",
    "config_sha256", "input_sha256", "database_sha256", "lane_artifact_sha256",
    "backend", "backend_decision", "runtime_versions", "scientific_parity",
    "output_root_absent", "checkpoint_root_absent", "evaluation_root_absent",
    "test_values_opened", "test_labels_opened", "label_access_count",
    "focused_tests",
})
_RUNTIME_FILES = frozenset({
    "progress_antigravity.json", "events_antigravity.jsonl",
    "parent_antigravity.log", "run_antigravity.pid",
})


@dataclass(frozen=True)
class _CliDependencies:
    """Injectable command effects used by boundary tests."""

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
            value = Path(getattr(self, name)).absolute()
            object.__setattr__(self, name, value)
        for name in ("current_head", "readiness_call", "execute", "inspect", "evaluate"):
            if not callable(getattr(self, name)):
                raise ValueError(f"CLI dependency {name} is invalid")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_exp435_train_only_consensus_recovery_antigravity.py",
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("readiness", "new", "resume", "inspect", "evaluate"):
        commands.add_parser(name, allow_abbrev=False)
    return parser


def _clean_head() -> str:
    head = subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), "status", "--porcelain"],
        check=True, capture_output=True, text=True,
    ).stdout
    if len(head) != 40 or any(character not in "0123456789abcdef" for character in head):
        raise ValueError("current Git commit is invalid")
    if dirty:
        raise ValueError("clean Git worktree is required")
    return head


def _head(dependencies: _CliDependencies) -> str:
    value = dependencies.current_head()
    if not isinstance(value, str) or len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("current Git commit is invalid")
    return value


def _validate_commit(value: object, label: str = "Git commit") -> str:
    if not isinstance(value, str) or len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} is invalid")
    return value


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"input file is absent or invalid: {path}")
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_versions() -> dict[str, str]:
    result = {"python": sys.version.split()[0]}
    for name in ("numpy", "numba", "torch", "mlx"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "unavailable"
    return dict(sorted(result.items()))


def _focused_test_evidence() -> dict[str, object]:
    """Run the small publication/CLI contract suite without creating bytecode."""

    command = [
        sys.executable,
        "-m",
        "unittest",
        "experiments.exp435_train_only_consensus_recovery_antigravity.tests.test_reporting_antigravity",
        "experiments.exp435_train_only_consensus_recovery_antigravity.tests.test_cli_antigravity",
    ]
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    started = perf_counter()
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    elapsed = perf_counter() - started
    return {
        "command": command[1:],
        "returncode": completed.returncode,
        "elapsed_seconds": elapsed,
        "stderr_sha256": sha256(completed.stderr.encode("utf-8")).hexdigest(),
    }


def _preflight_roots(command: str, output: Path, checkpoint: Path, readiness: Path, evaluation: Path) -> None:
    for path in (output, checkpoint, readiness, evaluation):
        if path.is_symlink():
            raise ValueError("fixed runtime paths cannot be symlinks")
    if command == "readiness":
        if output.exists() or checkpoint.exists() or readiness.exists() or evaluation.exists():
            raise ValueError("readiness requires all fixed roots to be absent")
        parent = readiness.parent
        if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
            raise ValueError("readiness parent is absent or invalid")
        if parent.exists() and any(parent.iterdir()):
            raise ValueError("readiness parent contains unknown existing content")
        if not parent.exists() and (parent.parent.is_symlink() or not parent.parent.is_dir()):
            raise ValueError("readiness parent container is absent or invalid")
        return
    if command == "new":
        if output.exists() or checkpoint.exists() or evaluation.exists():
            raise ValueError("new output, checkpoint, and evaluation roots must be absent")
        if readiness.is_symlink() or not readiness.is_file():
            raise ValueError("new readiness input is absent or invalid")
        return
    if command in {"resume", "inspect", "evaluate"}:
        if not output.is_dir() or not checkpoint.is_dir():
            raise ValueError("runtime output and checkpoint roots are absent or invalid")
        return
    raise ValueError("unsupported Exp435 command")


def _validate_sha(value: object, label: str, *, length: int = 64) -> str:
    if not isinstance(value, str) or len(value) != length or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"readiness {label} is invalid")
    return value


def _validate_readiness_payload(payload: Mapping[str, object], *, expected_commit: str) -> dict[str, object]:
    expected_commit = _validate_commit(expected_commit, "readiness Git commit")
    value = dict(payload)
    if set(value) != _READINESS_KEYS:
        raise ValueError("readiness fields are invalid")
    exact = {
        "schema": "exp435.readiness.v1",
        "state": "ready",
        "experiment_id": FROZEN_CONFIG.experiment_id,
        "run_identity": RUN_IDENTITY,
        "git_commit": expected_commit,
        "database_sha256": FROZEN_CONFIG.database_sha256,
        "output_root_absent": True,
        "checkpoint_root_absent": True,
        "evaluation_root_absent": True,
        "test_values_opened": 0,
        "test_labels_opened": 0,
        "label_access_count": 0,
        "scientific_parity": True,
    }
    for name, expected in exact.items():
        if value.get(name) != expected:
            raise ValueError(f"readiness {name} is invalid")
    _validate_sha(value["config_sha256"], "config_sha256")
    input_hashes = value.get("input_sha256")
    if not isinstance(input_hashes, Mapping) or "database" not in input_hashes:
        raise ValueError("readiness input hashes are invalid")
    for name, digest in input_hashes.items():
        _validate_sha(digest, f"input_sha256.{name}")
    if value.get("database_sha256") != FROZEN_CONFIG.database_sha256:
        raise ValueError("readiness database hash does not match the frozen DB")
    lanes = value.get("lane_artifact_sha256")
    if not isinstance(lanes, Mapping) or not lanes:
        raise ValueError("readiness lane hashes are invalid")
    for name, digest in lanes.items():
        _validate_sha(digest, f"lane_artifact_sha256.{name}")
    expected_lanes = {
        "exp291_prediction": FROZEN_CONFIG.exp291_prediction_sha256,
        "exp291_manifest": FROZEN_CONFIG.exp291_manifest_sha256,
        "exp292_prediction": FROZEN_CONFIG.exp292_prediction_sha256,
        "exp292_manifest": FROZEN_CONFIG.exp292_manifest_sha256,
    }
    if dict(lanes) != expected_lanes:
        raise ValueError("readiness lane hashes do not match the frozen inputs")
    if dict(input_hashes) != {"database": FROZEN_CONFIG.database_sha256, **expected_lanes}:
        raise ValueError("readiness input hashes do not match the frozen inputs")
    backend = value.get("backend")
    if backend not in {"cpu", "mlx"}:
        raise ValueError("readiness backend is invalid")
    decision = value.get("backend_decision")
    if not isinstance(decision, Mapping) or decision.get("backend") != backend:
        raise ValueError("readiness backend decision is invalid")
    if not isinstance(value.get("runtime_versions"), Mapping):
        raise ValueError("readiness runtime versions are invalid")
    tests = value.get("focused_tests")
    if not isinstance(tests, Mapping) or tests.get("returncode") != 0:
        raise ValueError("readiness focused-test evidence is invalid")
    return value


def _revalidate_readiness_for_execution(
    readiness: Path,
    *,
    validated_commit: str | None,
    current_head: str,
) -> dict[str, object]:
    if validated_commit is None or validated_commit != current_head:
        raise ValueError("readiness commit changed before execution")
    if readiness.is_symlink() or not readiness.is_file():
        raise ValueError("readiness input is absent or invalid")
    payload = _read_json(readiness, "readiness input")
    verified = _validate_readiness_payload(payload, expected_commit=current_head)
    if verified.get("git_commit") != validated_commit:
        raise ValueError("readiness commit changed before execution")
    return verified


def _backend_from_readiness(payload: Mapping[str, object]):
    return _backend_from_context({"backend_decision": payload.get("backend_decision")})


def _synthetic_mlx_decision():
    import numpy as np

    from .mlx_antigravity import MlxFixture, benchmark_mlx_adoption

    generator = np.random.default_rng(FROZEN_CONFIG.mlx_smoke_seed)
    references = generator.normal(size=(32, 16)).astype(np.float32)
    # Production scores exactly one TEST query row for one channel per call.
    queries = generator.normal(size=(1, 16)).astype(np.float32)
    return benchmark_mlx_adoption((MlxFixture(queries, references),))


def _ensure_run_parent(parent: Path) -> None:
    """Create the fixed run parent once, without accepting unknown content."""

    parent = Path(parent)
    container = parent.parent
    if container.is_symlink() or not container.is_dir():
        raise ValueError("readiness parent container is absent or invalid")
    if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
        raise ValueError("readiness parent is absent or invalid")
    if not parent.exists():
        try:
            parent.mkdir(mode=0o700)
            _fsync_directory(container)
        except FileExistsError:
            # Another writer won creation; validate its contents below rather
            # than following or replacing anything it created.
            pass
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError("readiness parent is absent or invalid")
    if any(parent.iterdir()):
        raise ValueError("readiness parent contains unknown existing content")


def _production_readiness() -> Mapping[str, object]:
    _ensure_run_parent(RUN_PARENT)
    FROZEN_CONFIG.validate()
    head = _clean_head()
    if FROZEN_OUTPUT.exists() or FROZEN_CHECKPOINT.exists() or FROZEN_READINESS.exists() or FROZEN_EVALUATION.exists():
        raise ValueError("all fixed Exp435 roots must be absent for readiness")
    started = perf_counter()
    database_hash = _sha256_file(DATABASE_PATH)
    lane_hashes = {
        "exp291_prediction": _sha256_file(LANE_PATHS[0]),
        "exp291_manifest": _sha256_file(LANE_PATHS[1]),
        "exp292_prediction": _sha256_file(LANE_PATHS[2]),
        "exp292_manifest": _sha256_file(LANE_PATHS[3]),
    }
    decision = _synthetic_mlx_decision()
    config_hash = sha256(canonical_json_bytes(asdict(FROZEN_CONFIG))).hexdigest()
    elapsed = perf_counter() - started
    final_head = _clean_head()
    if final_head != head:
        raise ValueError("clean HEAD changed during readiness")
    payload: dict[str, object] = {
        "schema": "exp435.readiness.v1",
        "state": "ready",
        "experiment_id": FROZEN_CONFIG.experiment_id,
        "run_identity": RUN_IDENTITY,
        "git_commit": head,
        "config_sha256": config_hash,
        "input_sha256": {"database": database_hash, **lane_hashes},
        "database_sha256": database_hash,
        "lane_artifact_sha256": lane_hashes,
        "backend": decision.backend,
        "backend_decision": decision.to_dict(),
        "runtime_versions": _runtime_versions(),
        "scientific_parity": True,
        "output_root_absent": True,
        "checkpoint_root_absent": True,
        "evaluation_root_absent": True,
        "test_values_opened": 0,
        "test_labels_opened": 0,
        "label_access_count": 0,
        "focused_tests": _focused_test_evidence(),
    }
    return payload


def run_readiness() -> Mapping[str, object]:
    payload = dict(_production_readiness())
    bound_commit = _validate_commit(payload.get("git_commit"), "readiness Git commit")
    _validate_readiness_payload(payload, expected_commit=bound_commit)
    final_head = _clean_head()
    if final_head != bound_commit:
        raise ValueError("clean HEAD changed before readiness write")
    _write_new(FROZEN_READINESS, canonical_json_bytes(payload))
    return payload


def _summary_payload(summary: object, publication: FrozenPublication | None = None) -> dict[str, object]:
    payload = summary.to_payload() if hasattr(summary, "to_payload") else dict(summary)  # type: ignore[arg-type]
    if publication is not None:
        payload.update({
            "state": "complete_review_only",
            "prediction_path": str(publication.publication / PREDICTION_FILE),
            "manifest_path": str(publication.publication / MANIFEST_FILE),
            "prediction_sha256": publication.prediction_sha256,
            "manifest_sha256": publication.manifest_sha256,
            "artifact_sha256": publication.artifact_sha256,
        })
    return dict(payload)


def _publication_summary(publication: FrozenPublication) -> dict[str, object]:
    """Return a scalar CLI result from an already verified publication."""

    manifest = publication.manifest
    completed = manifest.get("completed_datasets")
    failed = manifest.get("failed_datasets")
    candidate_count = manifest.get("candidate_count")
    if not all(type(value) is int for value in (completed, failed, candidate_count)):
        raise ValueError("publication summary accounting is invalid")
    return {
        "run_identity": manifest.get("run_identity"),
        "state": "complete_review_only",
        "total_datasets": completed,
        "completed_datasets": completed,
        "failed_datasets": failed,
        "candidate_count": candidate_count,
        "prediction_path": str(publication.publication / PREDICTION_FILE),
        "manifest_path": str(publication.publication / MANIFEST_FILE),
        "prediction_sha256": publication.prediction_sha256,
        "manifest_sha256": publication.manifest_sha256,
        "artifact_sha256": publication.artifact_sha256,
    }


def _run_prediction(command: str) -> Mapping[str, object]:
    if command not in {"new", "resume"}:
        raise ValueError("prediction command is invalid")
    from .checkpoint_antigravity import Exp435CheckpointStore
    head = _clean_head()
    readiness_payload: Mapping[str, object] | None = None
    if command == "new":
        # Retain the readiness-validated clean HEAD and read it again directly
        # before opening the database-bound session.  The exact final value is
        # passed into the descriptor-bound input session below.
        readiness_payload = _revalidate_readiness_for_execution(
            FROZEN_READINESS,
            validated_commit=head,
            current_head=head,
        )
        backend = _backend_from_readiness(readiness_payload)
        # Import every input/runtime dependency before the final HEAD read.
        # Import hooks and module initialization are part of the race window;
        # a commit change during that work must prevent the input session.
        from .inputs_antigravity import (
            DEFAULT_DATABASE,
            DEFAULT_EXP291_MANIFEST,
            DEFAULT_EXP291_PREDICTION,
            DEFAULT_EXP292_MANIFEST,
            DEFAULT_EXP292_PREDICTION,
            open_exp435_input_session,
        )
        from .runtime_antigravity import _runtime_context, run_exp435_new

        final_head = _clean_head()
        if final_head != head or final_head != readiness_payload.get("git_commit"):
            raise ValueError("clean HEAD changed before input session")
        head = final_head
    else:
        stored_context = Exp435CheckpointStore.read_context_projection(FROZEN_CHECKPOINT)
        if stored_context.get("run_identity") != RUN_IDENTITY:
            raise ValueError("resume run identity does not match frozen identity")
        stored_commit = stored_context.get("code_commit")
        if stored_commit != head:
            raise ValueError("resume code commit does not match current clean HEAD")
        backend = _backend_from_context(stored_context)
        publication_root = FROZEN_OUTPUT / "publication_antigravity"
        if publication_root.exists() or publication_root.is_symlink():
            store = Exp435CheckpointStore.open_read_only(FROZEN_CHECKPOINT)
            publication = inspect_publication(FROZEN_OUTPUT, store)
            return _publication_summary(publication)
        durable_progress = _read_json(
            FROZEN_OUTPUT / "progress_antigravity.json",
            "runtime progress",
        )
        if durable_progress.get("state") == "failed":
            raise ValueError("failed Exp435 runtime cannot resume")
        from .runtime_antigravity import inspect_exp435, resume_exp435

        # A completed checkpoint can be published/read without reopening the
        # DB.  This also makes repeated resume safe after a publication race.
        try:
            summary = inspect_exp435(
                run_root=FROZEN_OUTPUT,
                checkpoint_root=FROZEN_CHECKPOINT,
                expected_context=stored_context,
            )
        except (OSError, ValueError):
            summary = None
        if summary is not None and summary.status == "checkpoint_complete":
            store = Exp435CheckpointStore.open_read_only(FROZEN_CHECKPOINT)
            publication = publish_exp435(store, FROZEN_OUTPUT)
            return _summary_payload(summary, publication)
        if summary is not None and summary.status == "failed":
            raise ValueError("failed Exp435 runtime cannot resume")
        from .inputs_antigravity import (
            DEFAULT_DATABASE,
            DEFAULT_EXP291_MANIFEST,
            DEFAULT_EXP291_PREDICTION,
            DEFAULT_EXP292_MANIFEST,
            DEFAULT_EXP292_PREDICTION,
            open_exp435_input_session,
        )
        from .runtime_antigravity import _runtime_context, resume_exp435

        # The unfinished-resume path has now imported the runtime and input
        # boundaries.  Re-read HEAD immediately before opening the session and
        # bind that exact final value to both the checkpoint and session.
        final_head = _clean_head()
        if final_head != head or final_head != stored_commit:
            raise ValueError("clean HEAD changed before resume input session")
        head = final_head

    with open_exp435_input_session(
        DEFAULT_DATABASE,
        DEFAULT_EXP291_PREDICTION,
        DEFAULT_EXP291_MANIFEST,
        DEFAULT_EXP292_PREDICTION,
        DEFAULT_EXP292_MANIFEST,
        verified_code_commit=head,
    ) as session:
        runtime_context = _runtime_context(session, backend)
        if command == "new":
            summary = run_exp435_new(
                run_root=FROZEN_OUTPUT,
                checkpoint_root=FROZEN_CHECKPOINT,
                input_session=session,
                backend_decision=backend,
            )
        else:
            summary = resume_exp435(
                run_root=FROZEN_OUTPUT,
                checkpoint_root=FROZEN_CHECKPOINT,
                input_session=session,
                backend_decision=backend,
            )
    if summary.status != "checkpoint_complete":
        return _summary_payload(summary)
    store = Exp435CheckpointStore.open_read_only(
        FROZEN_CHECKPOINT,
        expected_context=runtime_context,
    )
    publication = publish_exp435(store, FROZEN_OUTPUT)
    return _summary_payload(summary, publication)


def run_new() -> Mapping[str, object]:
    return _run_prediction("new")


def run_resume() -> Mapping[str, object]:
    return _run_prediction("resume")


def _production_inspect() -> Mapping[str, object]:
    from .checkpoint_antigravity import Exp435CheckpointStore

    publication_root = FROZEN_OUTPUT / "publication_antigravity"
    if publication_root.exists() or publication_root.is_symlink():
        store = Exp435CheckpointStore.open_read_only(FROZEN_CHECKPOINT)
        return _publication_summary(inspect_publication(FROZEN_OUTPUT, store))

    from .runtime_antigravity import inspect_exp435

    stored_context = Exp435CheckpointStore.read_context_projection(FROZEN_CHECKPOINT)
    summary = inspect_exp435(
        run_root=FROZEN_OUTPUT,
        checkpoint_root=FROZEN_CHECKPOINT,
        expected_context=stored_context,
    )
    payload = _summary_payload(summary)
    publication = FROZEN_OUTPUT / "publication_antigravity"
    if summary.status == "checkpoint_complete" and publication.exists():
        verified = inspect_publication(
            FROZEN_OUTPUT,
            Exp435CheckpointStore.open_read_only(FROZEN_CHECKPOINT),
        )
        payload = _summary_payload(summary, verified)
    return payload


def run_inspect() -> Mapping[str, object]:
    return _production_inspect()


def run_evaluate() -> Mapping[str, object]:
    """Lazy evaluation boundary; prediction modes never import this module."""

    module_name = "experiments.exp435_train_only_consensus_recovery_antigravity." + "evaluation_antigravity"
    evaluator = importlib.import_module(module_name)
    function = getattr(evaluator, "evaluate_exp435", None)
    if not callable(function):
        function = getattr(evaluator, "run_evaluation", None)
    if not callable(function):
        raise ValueError("Exp435 evaluation entrypoint is absent")
    result = function(FROZEN_OUTPUT, FROZEN_EVALUATION)
    if isinstance(result, Mapping):
        return dict(result)
    if hasattr(result, "to_payload"):
        return dict(result.to_payload())
    raise ValueError("Exp435 evaluation result is invalid")


def _production_dependencies() -> _CliDependencies:
    return _CliDependencies(
        output=FROZEN_OUTPUT,
        checkpoint=FROZEN_CHECKPOINT,
        readiness=FROZEN_READINESS,
        current_head=_clean_head,
        readiness_call=run_readiness,
        execute=lambda command: run_new() if command == "new" else run_resume(),
        inspect=run_inspect,
        evaluate=run_evaluate,
    )


def main(
    argv: list[str] | None = None,
    *,
    _dependencies: _CliDependencies | None = None,
) -> int:
    dependencies = _production_dependencies() if _dependencies is None else _dependencies
    if not isinstance(dependencies, _CliDependencies):
        raise ValueError("CLI dependencies are invalid")
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as error:
        # Keep the callable CLI contract numeric even when argparse rejects an
        # alternate path or an unknown mode.  Direct parser users still get
        # argparse's normal SystemExit behavior.
        return int(error.code) if isinstance(error.code, int) else 2
    try:
        _preflight_roots(
            args.command,
            dependencies.output,
            dependencies.checkpoint,
            dependencies.readiness,
            dependencies.output.parent / "evaluation_antigravity",
        )
        if args.command == "readiness":
            payload = dict(dependencies.readiness_call())
            bound_commit = _validate_commit(payload.get("git_commit"), "readiness Git commit")
            verified = _validate_readiness_payload(payload, expected_commit=bound_commit)
            # The injected/production action is the sole writer.  Re-read the
            # artifact and require byte identity with the canonical returned
            # payload; the CLI must never perform a second write.
            if dependencies.readiness.is_symlink() or not dependencies.readiness.is_file():
                raise ValueError("readiness action did not create its artifact")
            artifact_raw = dependencies.readiness.read_bytes()
            canonical = canonical_json_bytes(verified)
            if artifact_raw != canonical:
                raise ValueError("readiness artifact is not byte-identical to the action result")
            reread = _read_json(dependencies.readiness, "readiness artifact")
            if reread != verified:
                raise ValueError("readiness artifact changed after the action")
            print(json.dumps(verified, sort_keys=True))
            return 0
        if args.command == "new":
            readiness = _read_json(dependencies.readiness, "readiness input")
            validated_head = _head(dependencies)
            validated = _validate_readiness_payload(readiness, expected_commit=validated_head)
            # Read a final clean HEAD immediately before execution and require
            # it to equal both the earlier validation and readiness commit.
            final_head = _head(dependencies)
            if final_head != validated_head or final_head != validated.get("git_commit"):
                raise ValueError("clean HEAD changed before execution")
            result = dict(dependencies.execute("new"))
        elif args.command == "resume":
            result = dict(dependencies.execute("resume"))
        elif args.command == "inspect":
            result = dict(dependencies.inspect())
        elif args.command == "evaluate":
            result = dict(dependencies.evaluate())
        else:  # pragma: no cover - argparse choices are exhaustive
            raise ValueError("unsupported Exp435 command")
        print(json.dumps(result, sort_keys=True))
        if args.command == "inspect":
            return 0 if result.get("state") in {"complete_review_only", "complete"} else 1
        if args.command in {"new", "resume"}:
            return 0 if result.get("state") in {"complete_review_only", "complete"} else 1
        if args.command == "evaluate":
            return 0 if result.get("state") in {"evaluated", "success", "complete"} else 1
        return 1
    except (OSError, ValueError) as error:
        print(f"{args.command} failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


__all__ = [
    "DATABASE_PATH",
    "FROZEN_CHECKPOINT",
    "FROZEN_EVALUATION",
    "FROZEN_OUTPUT",
    "FROZEN_READINESS",
    "LANE_PATHS",
    "RUN_IDENTITY",
    "_CliDependencies",
    "_revalidate_readiness_for_execution",
    "build_parser",
    "main",
    "run_evaluate",
    "run_inspect",
    "run_new",
    "run_readiness",
    "run_resume",
]
