#!/usr/bin/env python3
"""Official, all-population, label-free Exp417 one-shot CLI."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.exp403_408.artifact_io import canonical_json_bytes
from experiments.exp417_causal_multihorizon_shadow.backend import (
    FROZEN_TASK7_BACKEND_AUDIT_SHA256,
)
from experiments.exp417_causal_multihorizon_shadow.checkpoints import (
    Exp417CheckpointProvenance,
    Exp417CheckpointStore,
)
from experiments.exp417_causal_multihorizon_shadow.config import FROZEN_CONFIG
from experiments.exp417_causal_multihorizon_shadow.inputs import (
    assert_label_free_sql_contract,
    freeze_lane_inputs,
    freeze_run_context,
    verify_input_snapshots,
)
from experiments.exp417_causal_multihorizon_shadow.publication import publish_exp417
from experiments.exp417_causal_multihorizon_shadow.reporting import (
    TrustedCheckpointProvenance,
    render_exp417_payloads,
    verify_exp417_payloads,
)
from experiments.exp417_causal_multihorizon_shadow.runner import (
    RunnerResources,
    RunnerRuntime,
    resume_or_run_population,
)


def _existing_file(value: str) -> Path:
    path = Path(value).expanduser().resolve(strict=False)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"file does not exist: {path}")
    return path


def _checkpoint_dir(value: str) -> Path:
    path = Path(value).expanduser().resolve(strict=False)
    if path.exists() and not path.is_dir():
        raise argparse.ArgumentTypeError("checkpoint path must be a directory or new path")
    if not path.exists() and not path.parent.is_dir():
        raise argparse.ArgumentTypeError("checkpoint parent must exist")
    return path


def _new_output_dir(value: str) -> Path:
    path = Path(value).expanduser().resolve(strict=False)
    if path.exists():
        raise argparse.ArgumentTypeError("output directory must not exist")
    if not path.parent.is_dir():
        raise argparse.ArgumentTypeError("output parent must exist")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen all-population Exp417 official label-free workflow.",
        allow_abbrev=False,
    )
    parser.add_argument("--database", type=_existing_file, required=True, metavar="PATH")
    parser.add_argument("--exp291-prediction", type=_existing_file, required=True, metavar="PATH")
    parser.add_argument("--exp291-manifest", type=_existing_file, required=True, metavar="PATH")
    parser.add_argument("--exp292-prediction", type=_existing_file, required=True, metavar="PATH")
    parser.add_argument("--exp292-manifest", type=_existing_file, required=True, metavar="PATH")
    parser.add_argument("--checkpoint-dir", type=_checkpoint_dir, required=True, metavar="NEW_OR_MATCHING_PATH")
    parser.add_argument("--output-dir", type=_new_output_dir, required=True, metavar="NEW_PATH")
    return parser


def require_clean_git(repository: Path = REPO_ROOT) -> str:
    """Return HEAD only when the exact worktree has no tracked or untracked edits."""
    status = subprocess.run(
        ["git", "-C", str(repository), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise ValueError("Exp417 official execution requires a clean Git worktree")
    return subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@dataclass(frozen=True)
class OfficialStages:
    """Narrow stage boundary used to enforce fail-stop ordering."""

    clean_git: Callable[[], Any]
    label_guard: Callable[[], Any]
    freeze_inputs: Callable[[str], Any]
    open_checkpoint: Callable[[Any], Any]
    run: Callable[[Any, Any], Any]
    revalidate: Callable[[], Any]
    render: Callable[[Any, Any], Mapping[str, bytes]]
    verify: Callable[[Mapping[str, bytes], Any], Any]
    publish: Callable[[Mapping[str, bytes], Any], Any]


def execute_official(stages: OfficialStages) -> int:
    """Execute the exact official order; exceptions stop all later stages."""
    if not isinstance(stages, OfficialStages):
        raise ValueError("official stages must be typed")
    verified_head = stages.clean_git()
    if not isinstance(verified_head, str) or not verified_head:
        raise ValueError("clean Git stage must return the verified HEAD")
    stages.label_guard()
    context = stages.freeze_inputs(verified_head)
    checkpoints = stages.open_checkpoint(context)
    summary = stages.run(context, checkpoints)
    stages.revalidate()
    payloads = stages.render(summary, checkpoints)
    stages.verify(payloads, checkpoints)
    stages.publish(payloads, checkpoints)
    return 0


def _backend_audit_path(database: Path) -> Path:
    return database.parent / "exp417_causal_multihorizon_shadow" / "task7_backend_audit.json"


def _source_digest(*relative_paths: str) -> str:
    digest = sha256()
    for relative in relative_paths:
        path = REPO_ROOT / relative
        payload = path.read_bytes()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _provenance(context) -> Exp417CheckpointProvenance:
    return Exp417CheckpointProvenance(
        git_commit=context.code_commit,
        database_sha256=context.database_sha256,
        exp291_prediction_sha256=context.lanes.exp291_prediction_sha256,
        exp291_manifest_sha256=context.lanes.exp291_manifest_sha256,
        exp292_prediction_sha256=context.lanes.exp292_prediction_sha256,
        exp292_manifest_sha256=context.lanes.exp292_manifest_sha256,
        frozen_config_sha256=sha256(
            canonical_json_bytes(asdict(FROZEN_CONFIG))
        ).hexdigest(),
        preprocessing_contract_sha256=_source_digest(
            "experiments/exp417_causal_multihorizon_shadow/preprocessing.py"
        ),
        fold_contract_sha256=_source_digest(
            "experiments/exp417_causal_multihorizon_shadow/preprocessing.py",
            "experiments/exp417_causal_multihorizon_shadow/model.py",
        ),
        model_contract_sha256=_source_digest(
            "experiments/exp417_causal_multihorizon_shadow/model.py",
            "experiments/exp417_causal_multihorizon_shadow/scoring.py",
            "experiments/exp417_causal_multihorizon_shadow/qualification.py",
            "experiments/exp417_causal_multihorizon_shadow/runner.py",
        ),
        model_seeds=FROZEN_CONFIG.model_seeds,
        perturbation_seeds=FROZEN_CONFIG.perturbation_seeds,
        runtime_versions=context.runtime_versions,
        backend_contract_sha256=FROZEN_TASK7_BACKEND_AUDIT_SHA256,
    )


def _build_stages(args: argparse.Namespace) -> OfficialStages:
    paths = (
        args.database,
        args.exp291_prediction,
        args.exp291_manifest,
        args.exp292_prediction,
        args.exp292_manifest,
    )
    state: dict[str, Any] = {}

    def freeze_inputs(verified_head: str):
        lanes = freeze_lane_inputs(*paths[1:])
        context = freeze_run_context(
            paths[0], lanes, verified_code_commit=verified_head
        )
        state["resources"] = RunnerResources(
            *paths, backend_audit_path=_backend_audit_path(paths[0])
        )
        state["verified_head"] = verified_head
        state["provenance"] = _provenance(context)
        return context

    def open_checkpoint(context):
        return Exp417CheckpointStore(
            args.checkpoint_dir, state["provenance"], context.datasets
        )

    def run(context, checkpoints):
        runtime = RunnerRuntime.production(state["resources"])
        return resume_or_run_population(context, checkpoints, runtime=runtime)

    def revalidate() -> None:
        if require_clean_git(REPO_ROOT) != state["verified_head"]:
            raise ValueError("Exp417 verified Git HEAD changed before publication")
        verify_input_snapshots(*paths)

    def render(summary, checkpoints):
        trusted = TrustedCheckpointProvenance.from_store(checkpoints)
        return render_exp417_payloads(
            summary, state["provenance"], trusted
        )

    def verify(payloads, checkpoints) -> None:
        verify_exp417_payloads(
            payloads,
            trusted_checkpoint=TrustedCheckpointProvenance.from_store(checkpoints),
        )

    def publish(payloads, checkpoints) -> None:
        publish_exp417(
            payloads,
            args.output_dir,
            checkpoints=checkpoints,
            pre_publish_verify=revalidate,
        )

    return OfficialStages(
        clean_git=require_clean_git,
        label_guard=assert_label_free_sql_contract,
        freeze_inputs=freeze_inputs,
        open_checkpoint=open_checkpoint,
        run=run,
        revalidate=revalidate,
        render=render,
        verify=verify,
        publish=publish,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _require_disjoint_checkpoint_output(args.checkpoint_dir, args.output_dir)
    return execute_official(_build_stages(args))


def _require_disjoint_checkpoint_output(
    checkpoint_dir: Path, output_dir: Path
) -> None:
    checkpoint = Path(checkpoint_dir).resolve(strict=False)
    output = Path(output_dir).resolve(strict=False)
    if (
        checkpoint == output
        or checkpoint in output.parents
        or output in checkpoint.parents
    ):
        raise ValueError(
            "Exp417 checkpoint and output paths must have disjoint ancestry"
        )


if __name__ == "__main__":
    raise SystemExit(main())
