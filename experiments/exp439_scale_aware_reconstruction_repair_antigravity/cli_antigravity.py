"""Fixed five-command CLI for the isolated Exp439 lifecycle."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import partial
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from typing import Callable, Mapping

from experiments.exp403_408.artifact_io import canonical_json_bytes, sha256_file
from .config_antigravity import (
    CHECKPOINT_ROOT_ANTIGRAVITY,
    EVALUATION_ROOT_ANTIGRAVITY,
    FROZEN_CONFIG,
    OUTPUT_ROOT_ANTIGRAVITY,
    READINESS_PATH_ANTIGRAVITY,
    RUN_PARENT_ANTIGRAVITY,
    frozen_config_payload_antigravity,
)
from .inputs_antigravity import open_exp439_input_session
from .inputs_antigravity import canonical_source_paths_antigravity
from .reporting_antigravity import (
    PUBLICATION_DIRECTORY_ANTIGRAVITY,
    inspect_publication_antigravity,
    publish_exp439_antigravity,
)
from .runtime_antigravity import (
    _decision_from_payload_antigravity,
    inspect_exp439_antigravity,
    resume_exp439_antigravity,
    run_exp439_new_antigravity,
)


@dataclass(frozen=True)
class CliDependenciesAntigravity:
    readiness: Callable[[], object]
    new: Callable[[], object]
    resume: Callable[[], object]
    inspect: Callable[[], object]
    evaluate: Callable[[], object]

    def __post_init__(self) -> None:
        if any(not callable(getattr(self, name)) for name in self.__dataclass_fields__):
            raise ValueError("Exp439 CLI dependency is invalid")


def build_parser_antigravity() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_exp439_scale_aware_reconstruction_repair_antigravity.py",
        description="Frozen label-free Exp439 research lifecycle",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("readiness", "new", "resume", "inspect", "evaluate"):
        commands.add_parser(command)
    return parser


def main_with_dependencies_antigravity(
    argv: list[str] | None,
    dependencies: CliDependenciesAntigravity,
) -> int:
    if type(dependencies) is not CliDependenciesAntigravity:
        raise ValueError("Exp439 CLI dependencies are invalid")
    command = build_parser_antigravity().parse_args(argv).command
    getattr(dependencies, command)()
    return 0


def _validate_exp437_replay_antigravity(
    readiness: Mapping[str, object], current: Mapping[str, object]
) -> str:
    from .predecessor_antigravity import (
        classify_exp437_snapshot_antigravity,
        revalidate_exp437_snapshot_antigravity,
    )

    bound = readiness.get("exp437_execution")
    if not isinstance(bound, Mapping) or not isinstance(current, Mapping):
        raise ValueError("Exp437 readiness execution snapshot is invalid")
    bound_mode = classify_exp437_snapshot_antigravity(bound)
    if (
        readiness.get("exp437_execution_mode") != bound_mode
        or readiness.get("concurrent_with_exp437") is not (bound_mode == "concurrent")
    ):
        raise ValueError("Exp437 readiness execution mode is invalid")
    return revalidate_exp437_snapshot_antigravity(bound, current)


def _read_readiness_antigravity() -> dict[str, object]:
    try:
        raw = READINESS_PATH_ANTIGRAVITY.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Exp439 readiness artifact is invalid") from error
    if (
        not isinstance(payload, dict)
        or canonical_json_bytes(payload) != raw
        or payload.get("schema") != "exp439.readiness.v1"
        or payload.get("experiment_id") != FROZEN_CONFIG.experiment_id
        or payload.get("run_identity") != FROZEN_CONFIG.run_identity
        or payload.get("database_sha256") != FROZEN_CONFIG.database_sha256
        or not isinstance(payload.get("backend_decision"), dict)
    ):
        raise ValueError("Exp439 readiness binding is invalid")
    repository = Path(__file__).resolve().parents[2]
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repository, text=True):
        raise ValueError("Exp439 execution requires the readiness clean HEAD")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    paths = payload.get("source_paths")
    hashes = payload.get("source_sha256")
    from .readiness_antigravity import _exp437_terminal_antigravity

    current_exp437 = dict(_exp437_terminal_antigravity())
    try:
        _validate_exp437_replay_antigravity(payload, current_exp437)
    except ValueError as error:
        raise ValueError("Exp439 readiness replay validation failed") from error
    expected_paths = canonical_source_paths_antigravity().to_payload()
    if (
        head != payload.get("reviewed_commit")
        or payload.get("run_parent") != str(RUN_PARENT_ANTIGRAVITY.resolve())
        or not isinstance(paths, dict)
        or not isinstance(hashes, dict)
        or set(paths) != set(hashes)
        or paths != expected_paths
        or any(sha256_file(Path(paths[name])) != hashes[name] for name in paths)
        or not isinstance(payload.get("datasets"), list)
        or len(payload["datasets"]) != FROZEN_CONFIG.dataset_count
        or payload.get("dataset_order_sha256") != sha256(canonical_json_bytes(payload.get("datasets"))).hexdigest()
        or payload.get("frozen_config") != frozen_config_payload_antigravity()
        or payload.get("frozen_config_sha256") != sha256(canonical_json_bytes(payload.get("frozen_config"))).hexdigest()
        or payload.get("backend_decision_sha256") != sha256(canonical_json_bytes(payload.get("backend_decision"))).hexdigest()
        or not isinstance(payload.get("focused_tests"), dict)
        or payload["focused_tests"].get("returncode") != 0
        or payload.get("test_values_opened") != 0
        or payload.get("test_labels_opened") != 0
        or payload.get("prediction_label_access_count") != 0
    ):
        raise ValueError("Exp439 readiness replay validation failed")
    return payload


def _input_factory_antigravity(commit: str):
    return partial(open_exp439_input_session, verified_code_commit=commit)


def _publish_after_run_antigravity() -> object:
    return publish_exp439_antigravity(
        OUTPUT_ROOT_ANTIGRAVITY,
        CHECKPOINT_ROOT_ANTIGRAVITY,
    )


def run_new_command_antigravity() -> object:
    readiness = _read_readiness_antigravity()
    summary = run_exp439_new_antigravity(
        output_root=OUTPUT_ROOT_ANTIGRAVITY,
        checkpoint_root=CHECKPOINT_ROOT_ANTIGRAVITY,
        reviewed_commit=str(readiness["reviewed_commit"]),
        decision=_decision_from_payload_antigravity(readiness["backend_decision"]),
        input_session_factory=_input_factory_antigravity(str(readiness["reviewed_commit"])),
        readiness_payload=readiness,
    )
    _publish_after_run_antigravity()
    return summary


def run_resume_command_antigravity() -> object:
    readiness = _read_readiness_antigravity()
    summary = resume_exp439_antigravity(
        output_root=OUTPUT_ROOT_ANTIGRAVITY,
        checkpoint_root=CHECKPOINT_ROOT_ANTIGRAVITY,
        reviewed_commit=str(readiness["reviewed_commit"]),
        decision=_decision_from_payload_antigravity(readiness["backend_decision"]),
        input_session_factory=_input_factory_antigravity(str(readiness["reviewed_commit"])),
        readiness_payload=readiness,
    )
    _publish_after_run_antigravity()
    return summary


def run_inspect_command_antigravity() -> object:
    summary = inspect_exp439_antigravity(
        OUTPUT_ROOT_ANTIGRAVITY, CHECKPOINT_ROOT_ANTIGRAVITY
    )
    publication = RUN_PARENT_ANTIGRAVITY / PUBLICATION_DIRECTORY_ANTIGRAVITY
    if publication.exists():
        inspect_publication_antigravity(
            publication, OUTPUT_ROOT_ANTIGRAVITY, CHECKPOINT_ROOT_ANTIGRAVITY
        )
    return summary


def run_evaluate_command_antigravity() -> object:
    # This import is deliberately behind the explicit evaluate command.
    from .evaluation_antigravity import evaluate_exp439_antigravity

    return evaluate_exp439_antigravity(
        publication_root=RUN_PARENT_ANTIGRAVITY / PUBLICATION_DIRECTORY_ANTIGRAVITY,
        output_root=OUTPUT_ROOT_ANTIGRAVITY,
        checkpoint_root=CHECKPOINT_ROOT_ANTIGRAVITY,
        evaluation_root=EVALUATION_ROOT_ANTIGRAVITY,
    )


def run_readiness_command_antigravity() -> object:
    from .readiness_antigravity import create_readiness_antigravity

    return create_readiness_antigravity()


def production_dependencies_antigravity() -> CliDependenciesAntigravity:
    return CliDependenciesAntigravity(
        readiness=run_readiness_command_antigravity,
        new=run_new_command_antigravity,
        resume=run_resume_command_antigravity,
        inspect=run_inspect_command_antigravity,
        evaluate=run_evaluate_command_antigravity,
    )


def main_antigravity(argv: list[str] | None = None) -> int:
    return main_with_dependencies_antigravity(argv, production_dependencies_antigravity())


__all__ = [
    "CliDependenciesAntigravity",
    "build_parser_antigravity",
    "main_antigravity",
    "main_with_dependencies_antigravity",
]
