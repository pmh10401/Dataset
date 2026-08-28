"""Closed, parameter-free Exp472 command surface.

The controller derives its typed runtime from frozen public authority.  The
scientific scorer is label-free, while the TEST-label loader is imported only
after both independent authority gates, so read-only commands cannot discover
or open it by accident.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import TextIO

from .config_antigravity import (
    EXP471_REUSE_ROOT_ANTIGRAVITY,
    EXP472_EVALUATION_ROOT_ANTIGRAVITY,
    EXP472_OFFICIAL_ROOT_ANTIGRAVITY,
    EXP472_REPORT_ROOT_ANTIGRAVITY,
    EXP472_ROOT_ANTIGRAVITY,
    EXP472_SMOKE_ROOT_ANTIGRAVITY,
    FROZEN_EXP472_CONFIG,
    assess_exp472_backend_antigravity,
)
from .contracts_antigravity import canonical_json_bytes_exp472_antigravity
from .contracts_antigravity import (
    Exp472BackendDecisionAntigravity,
    Exp472RunContextAntigravity,
)
from .artifacts_antigravity import (
    Exp472CatalogEntryAntigravity,
    catalog_sha256_exp472_antigravity,
    verify_manifest_exp472_antigravity,
    verify_score_shard_exp472_antigravity,
)
from .authority_antigravity import verify_exp471_reuse_exp472_antigravity
from .evaluation_antigravity import evaluate_exp472_antigravity
from .reporting_antigravity import write_exp472_report_antigravity
from .runner_antigravity import run_score_wave_exp472_antigravity
from .scoring_antigravity import (
    compose_final_score_exp472_antigravity,
    score_dataset_condition_e_exp472_antigravity,
)


_COMMANDS_ANTIGRAVITY = ("readiness", "smoke", "run", "status", "verify", "evaluate", "report")


def _score_one_from_raw_antigravity(raw_by_dataset: Mapping[str, object], backend: object, dataset: str) -> object:
    """Pickle-safe CPU worker entry for the frozen, label-free condition E."""
    values = raw_by_dataset[dataset]
    result = score_dataset_condition_e_exp472_antigravity(values, backend)
    return compose_final_score_exp472_antigravity(values, result, backend)


def _code_map_lock_sha256_antigravity() -> str:
    path = Path(__file__).resolve().parents[2] / "docs/codemap/codemap_exp472_antigravity.lock"
    return sha256(path.read_bytes()).hexdigest()


def _code_closure_antigravity() -> Mapping[str, str]:
    root = Path(__file__).resolve().parents[2]
    names = (
        "experiments/exp472_makima_upsampling_ablation_antigravity/config_antigravity.py",
        "experiments/exp472_makima_upsampling_ablation_antigravity/contracts_antigravity.py",
        "experiments/exp472_makima_upsampling_ablation_antigravity/transforms_antigravity.py",
        "experiments/exp472_makima_upsampling_ablation_antigravity/scoring_antigravity.py",
        "experiments/exp472_makima_upsampling_ablation_antigravity/artifacts_antigravity.py",
        "experiments/exp472_makima_upsampling_ablation_antigravity/runner_antigravity.py",
    )
    return {name: sha256((root / name).read_bytes()).hexdigest() for name in names}


def _config_sha256_antigravity() -> str:
    value = {"experiment": FROZEN_EXP472_CONFIG.experiment_id, "database_sha256": FROZEN_EXP472_CONFIG.database_sha256, "dataset_count": FROZEN_EXP472_CONFIG.dataset_count, "test_row_count": FROZEN_EXP472_CONFIG.test_row_count, "condition": {"id": "E", "downsampling": "polyphase", "upsampling": "makima", "image_mode": "area"}, "policy": FROZEN_EXP472_CONFIG.policy_id, "threshold": FROZEN_EXP472_CONFIG.score_threshold, "bootstrap_seed": FROZEN_EXP472_CONFIG.bootstrap_seed}
    return sha256(canonical_json_bytes_exp472_antigravity(value)).hexdigest()


def _read_scored_context_candidate_antigravity(
    root: Path, catalog: tuple[Exp472CatalogEntryAntigravity, ...],
) -> Exp472RunContextAntigravity:
    """Read one untrusted canonical E shard only to reconstruct its context.

    The candidate is not authority: immediately after construction every
    public Task4 manifest/shard verifier must authenticate its exact digest,
    catalog, closure, and backend before it can reach evaluation or reporting.
    """
    if Path(root) != EXP472_OFFICIAL_ROOT_ANTIGRAVITY or len(catalog) != 1117:
        raise ValueError("Exp472 scored context requires the official full E root")
    token = sha256(catalog[0].dataset.encode("utf-8")).hexdigest()
    paths = (
        Path(root),
        Path(root) / "score_shards_antigravity",
        Path(root) / "score_shards_antigravity" / f"{token}_antigravity",
    )
    for path in paths:
        try:
            details = path.lstat()
        except FileNotFoundError as error:
            raise ValueError("Exp472 scored context is missing") from error
        if path.is_symlink() or not path.is_dir():
            raise ValueError("Exp472 scored context is unsafe")
    shard_path = paths[-1] / "score_shard_antigravity.json"
    try:
        details = shard_path.lstat()
    except FileNotFoundError as error:
        raise ValueError("Exp472 scored context is missing") from error
    if shard_path.is_symlink() or not shard_path.is_file() or details.st_nlink != 1:
        raise ValueError("Exp472 scored context is unsafe")
    raw = shard_path.read_bytes()
    try:
        record = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Exp472 scored context is invalid") from error
    if not isinstance(record, dict) or raw != canonical_json_bytes_exp472_antigravity(record):
        raise ValueError("Exp472 scored context is noncanonical")
    backend_record = record.get("backend")
    closure = record.get("code_closure")
    if (
        record.get("schema") != "exp472-score-shard-antigravity.v1"
        or record.get("dataset") != catalog[0].dataset
        or record.get("database_sha256") != FROZEN_EXP472_CONFIG.database_sha256
        or record.get("catalog_sha256") != catalog_sha256_exp472_antigravity(catalog)
        or not isinstance(backend_record, Mapping)
        or not isinstance(closure, Mapping)
    ):
        raise ValueError("Exp472 scored context identity differs")
    try:
        backend = Exp472BackendDecisionAntigravity(**dict(backend_record))
        context = Exp472RunContextAntigravity(
            "full", EXP472_OFFICIAL_ROOT_ANTIGRAVITY,
            FROZEN_EXP472_CONFIG.database_sha256,
            catalog_sha256_exp472_antigravity(catalog), closure, backend,
            _config_sha256_antigravity(),
        )
    except (TypeError, ValueError) as error:
        raise ValueError("Exp472 scored context differs") from error
    if record.get("context_sha256") != context.sha256:
        raise ValueError("Exp472 scored context digest differs")
    return context


def _emit_antigravity(stream: TextIO, value: object) -> None:
    stream.write(canonical_json_bytes_exp472_antigravity(value).decode("utf-8"))
    stream.flush()


def _paths_antigravity() -> dict[str, str]:
    return {
        "root": str(EXP472_ROOT_ANTIGRAVITY),
        "smoke": str(EXP472_SMOKE_ROOT_ANTIGRAVITY),
        "official": str(EXP472_OFFICIAL_ROOT_ANTIGRAVITY),
        "evaluation": str(EXP472_EVALUATION_ROOT_ANTIGRAVITY),
        "report": str(EXP472_REPORT_ROOT_ANTIGRAVITY),
        "progress": str(EXP472_OFFICIAL_ROOT_ANTIGRAVITY / "progress_antigravity.jsonl"),
    }


def _status_antigravity() -> dict[str, object]:
    """Produce a values-only observation without creating or importing data paths."""
    progress = EXP472_OFFICIAL_ROOT_ANTIGRAVITY / "progress_antigravity.jsonl"
    completed = failed = remaining = None
    eta_seconds = None
    if progress.is_file() and not progress.is_symlink():
        try:
            lines = progress.read_text(encoding="utf-8").splitlines()
            if lines:
                last = json.loads(lines[-1])
                if isinstance(last, dict):
                    completed = last.get("completed"); failed = last.get("failed")
                    remaining = last.get("remaining")
                    elapsed = last.get("elapsed_seconds")
                    if all(type(value) in (int, float) and value >= 0 for value in (completed, remaining, elapsed)) and completed:
                        eta_seconds = float(elapsed) / float(completed) * float(remaining)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
            # Status reports a non-authoritative unreadable log rather than
            # attempting repair or a hidden restart.
            pass
    return {"experiment": "Exp472", "paths": _paths_antigravity(), "scored": completed, "failed": failed, "remaining": remaining, "eta_seconds": eta_seconds, "read_only": True}


@dataclass(frozen=True, slots=True)
class Exp472ControllerRuntimeAntigravity:
    """Typed controller adapter for Task4 E and Task5 Exp471 authorities.

    It deliberately contains data-free paths, a frozen run context, exact
    catalog, and a lazy label loader.  There are no completion booleans or
    arbitrary callback maps: authority is reopened on every verify/evaluate/
    report operation through the public Task4/Task5 readers.
    """

    context: Exp472RunContextAntigravity
    catalog: tuple[Exp472CatalogEntryAntigravity, ...]
    exp471_root: Path
    code_map_lock_sha256: str
    report_root: Path = EXP472_REPORT_ROOT_ANTIGRAVITY

    def __post_init__(self) -> None:
        if not isinstance(self.context, Exp472RunContextAntigravity) or (self.context.mode, self.context.run_root) not in {("smoke", EXP472_SMOKE_ROOT_ANTIGRAVITY), ("full", EXP472_OFFICIAL_ROOT_ANTIGRAVITY)}:
            raise ValueError("Exp472 controller requires a frozen E context")
        entries = tuple(self.catalog)
        expected = 23 if self.context.mode == "smoke" else 1117
        if len(entries) != expected or not all(isinstance(entry, Exp472CatalogEntryAntigravity) for entry in entries):
            raise ValueError(f"Exp472 controller requires exactly {expected} typed catalog entries")
        if catalog_sha256_exp472_antigravity(entries) != self.context.catalog_sha256:
            raise ValueError("Exp472 controller catalog differs from the frozen E context")
        if Path(self.exp471_root) != EXP471_REUSE_ROOT_ANTIGRAVITY:
            raise ValueError("Exp472 controller reuse boundary differs")
        if type(self.code_map_lock_sha256) is not str or len(self.code_map_lock_sha256) != 64:
            raise ValueError("Exp472 controller code-map digest differs")
        if not Path(self.report_root).is_absolute() or "antigravity" not in Path(self.report_root).name:
            raise ValueError("Exp472 controller report root differs")
        object.__setattr__(self, "catalog", entries)
        object.__setattr__(self, "exp471_root", Path(self.exp471_root))
        object.__setattr__(self, "report_root", Path(self.report_root))

    def verify_exp471_antigravity(self) -> object:
        return verify_exp471_reuse_exp472_antigravity(self.exp471_root)

    def verify_e_antigravity(self) -> tuple[Mapping[str, object], tuple[Mapping[str, object], ...]]:
        manifest = verify_manifest_exp472_antigravity(self.context.run_root, self.context, self.catalog)
        shards = tuple(
            verify_score_shard_exp472_antigravity(self.context.run_root, self.context, entry)
            for entry in self.catalog
        )
        expected = 23 if self.context.mode == "smoke" else 1117
        if len(shards) != expected:
            raise ValueError(f"Exp472 controller requires all {expected} verified E score shards")
        return manifest, shards

    def evaluate_antigravity(self) -> object:
        # The implementation owns the label firewall: it verifies Exp471 and
        # all E shard authorities before this loader is ever invoked.
        # Importing the exact TEST-label module is deliberately inside the
        # post-authority evaluation call, never CLI import/status/verify.
        from paper.equipment_anomaly_detection.label_access_paper import load_test_labels
        return evaluate_exp472_antigravity(
            self.context.run_root, self.exp471_root,
            lambda order: load_test_labels(FROZEN_EXP472_CONFIG.database_path, order),
            context=self.context, catalog=self.catalog,
        )

    def worker_record_antigravity(self) -> Mapping[str, object]:
        return self.context.backend.artifact_record()

    def run_e_antigravity(self) -> Mapping[str, object]:
        """Run only condition E, using the issued source capability label-free."""
        from experiments.exp471_exp467_evaluation_recovery_antigravity.config_antigravity import EXP467_SOURCE_ROOT_ANTIGRAVITY
        from experiments.exp471_exp467_evaluation_recovery_antigravity.source_authority_antigravity import (
            _bind_captured_values_database_antigravity,
            _load_captured_dataset_values_antigravity,
            _require_captured_values_api_antigravity,
            open_exp467_source_session_antigravity,
        )
        repository = Path(__file__).resolve().parents[2]
        session = open_exp467_source_session_antigravity(repository, EXP467_SOURCE_ROOT_ANTIGRAVITY)
        api = _require_captured_values_api_antigravity(session)
        paper = api.database_config_antigravity
        binding = _bind_captured_values_database_antigravity(session, Path(paper.database_path), config=paper, verified_catalog_database=session.catalog["database"])
        raw = {entry.dataset: _load_captured_dataset_values_antigravity(session, Path(paper.database_path), entry.dataset, preverified_database=binding) for entry in self.catalog}
        score_one = partial(_score_one_from_raw_antigravity, raw, self.context.backend)
        return run_score_wave_exp472_antigravity(self.context.run_root, self.context, self.catalog, score_one=score_one)


def _runtime_antigravity(
    runtime: object | None, *, for_scoring: bool = False,
) -> Exp472ControllerRuntimeAntigravity:
    if runtime is None:
        return _build_runtime_antigravity("full", for_scoring=for_scoring)
    if not isinstance(runtime, Exp472ControllerRuntimeAntigravity):
        raise ValueError("Exp472 command requires a typed controller runtime; completion flags are not authority")
    return runtime


def _build_runtime_antigravity(
    mode: str, *, for_scoring: bool = False,
) -> Exp472ControllerRuntimeAntigravity:
    """Build the executable official controller from frozen public authority."""
    if mode not in {"smoke", "full"}:
        raise ValueError("Exp472 runtime mode differs")
    exp471 = verify_exp471_reuse_exp472_antigravity(EXP471_REUSE_ROOT_ANTIGRAVITY)
    names = FROZEN_EXP472_CONFIG.smoke_datasets if mode == "smoke" else exp471.dataset_order
    catalog = tuple(Exp472CatalogEntryAntigravity(name, exp471.coordinates[name]) for name in names)
    run_root = EXP472_SMOKE_ROOT_ANTIGRAVITY if mode == "smoke" else EXP472_OFFICIAL_ROOT_ANTIGRAVITY
    if mode == "full" and not for_scoring:
        context = _read_scored_context_candidate_antigravity(run_root, catalog)
    else:
        context = Exp472RunContextAntigravity(
            mode, run_root, FROZEN_EXP472_CONFIG.database_sha256,
            catalog_sha256_exp472_antigravity(catalog),
            _code_closure_antigravity(), assess_exp472_backend_antigravity(),
            _config_sha256_antigravity(),
        )
    return Exp472ControllerRuntimeAntigravity(context, catalog, EXP471_REUSE_ROOT_ANTIGRAVITY, _code_map_lock_sha256_antigravity())


def _verified_authorities_antigravity(runtime: Exp472ControllerRuntimeAntigravity) -> tuple[object, Mapping[str, object], tuple[Mapping[str, object], ...]]:
    """Reopen Exp471 first, then the full E manifest and every E shard."""
    exp471 = runtime.verify_exp471_antigravity()
    manifest, shards = runtime.verify_e_antigravity()
    return exp471, manifest, shards


def _readiness_antigravity() -> dict[str, object]:
    backend = assess_exp472_backend_antigravity().artifact_record()
    return {"experiment": "Exp472", "database_sha256": FROZEN_EXP472_CONFIG.database_sha256, "dataset_count": 1117, "test_row_count": 195900, "paths": _paths_antigravity(), "backend": backend, "labels_opened": False}


def main_antigravity(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None, runtime: Exp472ControllerRuntimeAntigravity | None = None) -> int:
    """Run exactly one closed command; no root/parameter override is accepted."""
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 1 or arguments[0] not in _COMMANDS_ANTIGRAVITY:
        raise ValueError("Exp472 command must be one of readiness, smoke, run, status, verify, evaluate, report")
    stream = sys.stdout if stdout is None else stdout
    command = arguments[0]
    if command == "status":
        _emit_antigravity(stream, _status_antigravity()); return 0
    if command == "readiness":
        _emit_antigravity(stream, _readiness_antigravity()); return 0
    if command == "smoke":
        typed_smoke = runtime if isinstance(runtime, Exp472ControllerRuntimeAntigravity) else _build_runtime_antigravity("smoke")
        if typed_smoke.context.mode != "smoke":
            raise ValueError("Exp472 smoke requires the frozen smoke E context")
        result = typed_smoke.run_e_antigravity()
        _emit_antigravity(stream, {"experiment": "Exp472", "command": "smoke", "result": result}); return 0
    if command == "run":
        typed_run = _runtime_antigravity(runtime, for_scoring=True)
        if typed_run.context.mode != "full":
            raise ValueError("Exp472 run requires the frozen full E context")
        result = typed_run.run_e_antigravity()
        _emit_antigravity(stream, {"experiment": "Exp472", "command": "run", "result": result}); return 0
    typed = _runtime_antigravity(runtime)
    if command in {"evaluate", "report"} and typed.context.mode != "full":
        raise ValueError("Exp472 evaluation and report require the frozen full E context")
    if command == "verify":
        exp471, manifest, _shards = _verified_authorities_antigravity(typed)
        _emit_antigravity(stream, {"experiment": "Exp472", "command": "verify", "exp471_authority": type(exp471).__name__, "e_manifest_sha256": manifest["manifest_sha256"], "read_only": True}); return 0
    if command == "evaluate":
        _verified_authorities_antigravity(typed)
        evaluation = typed.evaluate_antigravity()
        _emit_antigravity(stream, {"experiment": "Exp472", "command": "evaluate", "evaluation_sha256": getattr(evaluation, "artifact_record", lambda: {})().get("evaluation_sha256")}); return 0
    # Report deliberately does not dispatch score generation.  It can only
    # consume completed, independently reopened authorities before evaluation.
    _exp471, manifest, shards = _verified_authorities_antigravity(typed)
    evaluation = typed.evaluate_antigravity()
    shape = tuple({"dataset": shard["dataset"], "upsampled_rows": shard["upsampled_rows"], **dict(shard["diagnostic"])} for shard in shards)
    authority = write_exp472_report_antigravity(evaluation, typed.report_root, typed.worker_record_antigravity(), e_manifest_sha256=str(manifest["manifest_sha256"]), code_map_lock_sha256=typed.code_map_lock_sha256, shape_diagnostics=shape)
    _emit_antigravity(stream, {"experiment": "Exp472", "command": "report", "report_manifest_sha256": authority.report_manifest_sha256}); return 0


__all__ = ["Exp472ControllerRuntimeAntigravity", "main_antigravity"]
