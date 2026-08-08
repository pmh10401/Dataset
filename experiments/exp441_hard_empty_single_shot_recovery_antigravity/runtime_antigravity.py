"""Immutable, label-free Exp441 attempt allocation and source-to-selection runtime."""
from __future__ import annotations

from hashlib import sha256
import os
from pathlib import Path
import signal
import subprocess
from typing import Mapping

from .artifacts_antigravity import (
    canonical_json_bytes_antigravity, parse_canonical_json_bytes_antigravity,
    pin_exact_regular_inventory_antigravity, read_pinned_regular_bytes_antigravity,
    write_create_only_antigravity,
)
from .config_antigravity import FROZEN_CONFIG, governance_payload_antigravity
from .contracts_antigravity import (
    CandidateEvidenceAntigravity, CompletedAttemptAntigravity, FrozenSourceSnapshotAntigravity,
    RankedCandidateAntigravity, SelectionResultAntigravity,
    terminal_attempt_authority_sha256_from_bytes_antigravity,
    validate_production_selection_antigravity,
)
from .progress_antigravity import ProgressWriterAntigravity
from .selector_antigravity import select_hard_empty_single_shot_antigravity
from .sources_antigravity import (
    _load_prefix_antigravity, _verify_frozen_sources_with_observer_antigravity,
    verify_frozen_sources_antigravity,
)


_RUN_ROOT_ANTIGRAVITY = Path("/Users/mac/Documents/Dataset/exp441_hard_empty_single_shot_recovery_antigravity_20260808")
_TERMINAL_ATTEMPT_AUTHORITY_FILENAMES_ANTIGRAVITY = (
    "source_snapshot_antigravity.json", "selection_antigravity.json", "progress_antigravity.json",
    "pid_antigravity", "events_antigravity.jsonl", "text_antigravity.log",
)


class _InterruptedAttemptAntigravity(RuntimeError):
    pass


def terminal_attempt_authority_sha256_antigravity(attempt: Path) -> str:
    """Hash one grouped, fully revalidated six-file terminal authority snapshot."""
    with pin_exact_regular_inventory_antigravity(attempt, _TERMINAL_ATTEMPT_AUTHORITY_FILENAMES_ANTIGRAVITY) as pinned:
        digest = terminal_attempt_authority_sha256_from_bytes_antigravity(pinned.payloads)
        pinned.revalidate_antigravity()
        return digest


def _ensure_clean_reviewed_commit_antigravity(reviewed_commit: str) -> None:
    if not isinstance(reviewed_commit, str) or len(reviewed_commit) != 40:
        raise ValueError("reviewed commit is invalid")
    root = Path(__file__).resolve().parents[2]
    actual = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True).stdout
    if actual != reviewed_commit or dirty:
        raise ValueError("reviewed commit is not the current clean HEAD")


def allocate_attempt_antigravity(run_root: Path = _RUN_ROOT_ANTIGRAVITY) -> Path:
    """Create the smallest missing immutable attempt directory without a resume path."""
    root = Path(run_root)
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("run root is invalid")
    attempts = root / "attempts_antigravity"
    attempts.mkdir(mode=0o700, exist_ok=True)
    if attempts.is_symlink() or not attempts.is_dir():
        raise ValueError("attempt root is invalid")
    present: set[int] = set()
    for entry in attempts.iterdir():
        if not entry.name.startswith("attempt_") or not entry.name.endswith("_antigravity"):
            raise ValueError("attempt inventory is invalid")
        middle = entry.name[len("attempt_"):-len("_antigravity")]
        if len(middle) != 4 or not middle.isdigit() or entry.is_symlink() or not entry.is_dir():
            raise ValueError("attempt inventory is invalid")
        present.add(int(middle))
    sequence = 0
    while sequence in present:
        sequence += 1
    attempt = attempts / f"attempt_{sequence:04d}_antigravity"
    try:
        attempt.mkdir(mode=0o700)
    except FileExistsError as error:
        raise ValueError("attempt allocation raced") from error
    return attempt


def _snapshot_payload_antigravity(snapshot: FrozenSourceSnapshotAntigravity) -> dict[str, object]:
    ranking_evidence = [
        {"dataset": item.dataset, "instance_index": item.instance_index,
         "second_excess_ratio": item.second_excess_ratio, "second_empirical_rank": item.second_empirical_rank}
        for item in snapshot.provisional_candidates
    ]
    digest_map = dict(snapshot.record_digest_by_dataset)
    return {
        "context_dataset_count": len(snapshot.context_datasets),
        "record_digest_count": len(digest_map),
        "record_digest_map_identity_sha256": sha256(canonical_json_bytes_antigravity(digest_map)).hexdigest(),
        "hard_coordinate_count": len(snapshot.hard_coordinates),
        "hard_coordinate_identity_sha256": snapshot.hard_coordinate_identity_sha256,
        "provisional_candidate_count": len(snapshot.provisional_candidates),
        "ranking_evidence": ranking_evidence,
        "source_hashes": {
            "context": snapshot.context_sha256, "state": snapshot.state_sha256,
            "retained_inventory": snapshot.retained_inventory_sha256,
            "publication_prediction": snapshot.publication_prediction_sha256,
            "publication_manifest": snapshot.publication_manifest_sha256,
            "publication_artifact_map": snapshot.publication_artifact_map_sha256,
            "hard_prediction": snapshot.hard_prediction_sha256, "hard_manifest": snapshot.hard_manifest_sha256,
            "hard_coordinate_identity": snapshot.hard_coordinate_identity_sha256,
        },
        "label_access_count": snapshot.label_access_count,
    }


def _selection_payload_antigravity(selection: SelectionResultAntigravity, *, reviewed_commit: str, source_snapshot_sha256: str) -> dict[str, object]:
    payload = selection.to_payload()
    payload.update({"reviewed_commit": reviewed_commit, "source_snapshot_sha256": source_snapshot_sha256})
    payload.update(governance_payload_antigravity())
    return payload


def _failure_payload_antigravity(*, sequence: int, reviewed_commit: str, status: str, error: str) -> dict[str, object]:
    payload: dict[str, object] = {"attempt_sequence": sequence, "reviewed_commit": reviewed_commit, "status": status, "error": error}
    payload.update(governance_payload_antigravity())
    return payload


def _validate_governance_payload_antigravity(payload: Mapping[str, object]) -> None:
    """Reject incomplete, aliased, nested, or truthy-substituted governance."""
    expected = governance_payload_antigravity()
    if not isinstance(payload, Mapping) or set(payload) != set(expected):
        raise ValueError("governance payload keys are invalid")
    for name, value in expected.items():
        if type(payload[name]) is not type(value) or payload[name] != value:
            raise ValueError("governance payload value is invalid")


def _validate_terminal_artifact_governance_antigravity(payload: Mapping[str, object]) -> None:
    """Validate flat governance in a real terminal artifact without accepting aliases."""
    if not isinstance(payload, Mapping):
        raise ValueError("terminal artifact is invalid")
    expected = governance_payload_antigravity()
    if any(name not in payload for name in expected):
        raise ValueError("terminal artifact governance is invalid")
    _validate_governance_payload_antigravity({name: payload[name] for name in expected})
    aliases = {"governance", "governance_payload", "review_status"}
    if any(name in payload for name in aliases) or any(name.endswith("_alias") for name in payload):
        raise ValueError("terminal artifact governance alias is invalid")


def _validate_selection_artifact_antigravity(payload: Mapping[str, object]) -> None:
    expected = set(SelectionResultAntigravity.__dataclass_fields__) | {"reviewed_commit", "source_snapshot_sha256"} | set(governance_payload_antigravity())
    if (set(payload) != expected or not _is_lower_hex_antigravity(payload["reviewed_commit"], 40) or
            not _is_lower_hex_antigravity(payload["source_snapshot_sha256"], 64)):
        raise ValueError("selection artifact schema is invalid")
    selection = _selection_from_payload_antigravity(payload)
    if selection.to_payload() != {name: payload[name] for name in SelectionResultAntigravity.__dataclass_fields__}:
        raise ValueError("selection artifact content is invalid")
    _validate_terminal_artifact_governance_antigravity(payload)


def _validate_failure_artifact_antigravity(payload: Mapping[str, object]) -> None:
    expected = {"attempt_sequence", "reviewed_commit", "status", "error"} | set(governance_payload_antigravity())
    if (set(payload) != expected or type(payload["attempt_sequence"]) is not int or payload["attempt_sequence"] < 0 or
            not _is_lower_hex_antigravity(payload["reviewed_commit"], 40) or
            payload["status"] not in {"failed", "interrupted"} or not isinstance(payload["error"], str) or not payload["error"]):
        raise ValueError("failure artifact schema is invalid")
    _validate_terminal_artifact_governance_antigravity(payload)


def _is_lower_hex_antigravity(value: object, length: int) -> bool:
    return isinstance(value, str) and len(value) == length and all(character in "0123456789abcdef" for character in value)


def _selection_from_payload_antigravity(payload: Mapping[str, object]) -> SelectionResultAntigravity:
    """Reopen all nested selection fields through their exact immutable contracts."""
    def candidates(name: str) -> tuple[CandidateEvidenceAntigravity, ...]:
        rows = payload[name]
        if not isinstance(rows, list):
            raise ValueError("selection artifact candidate population is invalid")
        return tuple(CandidateEvidenceAntigravity.from_payload(row) for row in rows)

    def ranked(name: str) -> tuple[RankedCandidateAntigravity, ...]:
        rows = payload[name]
        if not isinstance(rows, list):
            raise ValueError("selection artifact ranked population is invalid")
        return tuple(RankedCandidateAntigravity.from_payload(row) for row in rows)

    coordinates = payload["hard_coordinates"]
    if not isinstance(coordinates, list) or any(not isinstance(item, list) or len(item) != 2 for item in coordinates):
        raise ValueError("selection artifact Hard coordinates are invalid")
    return SelectionResultAntigravity(
        provisional_candidates=candidates("provisional_candidates"),
        outside_hard_candidates=candidates("outside_hard_candidates"),
        eligible_candidates=candidates("eligible_candidates"),
        winners=ranked("winners"), selected=ranked("selected"),
        hard_coordinates=frozenset((item[0], item[1]) for item in coordinates),
    )


def _verify_with_observer_antigravity(observer):
    return _verify_frozen_sources_with_observer_antigravity(observer)


def _reverify_sources_after_artifacts_antigravity() -> FrozenSourceSnapshotAntigravity:
    """Replay the immutable source authority after create-only attempt artifacts exist."""
    return verify_frozen_sources_antigravity()


def _best_effort_failure_antigravity(writer: ProgressWriterAntigravity | None, attempt: Path, sequence: int, reviewed_commit: str, status: str, error: BaseException) -> None:
    """Keep the original failure primary while independently attempting each evidence channel."""
    message = str(error) or type(error).__name__
    if writer is not None:
        try: writer.terminal_antigravity(stage=status, latest_error=message)
        except BaseException: pass
    try:
        payload = _failure_payload_antigravity(sequence=sequence, reviewed_commit=reviewed_commit, status=status, error=message)
        _validate_failure_artifact_antigravity(payload)
        path = attempt / "failure_antigravity.json"
        write_create_only_antigravity(path, canonical_json_bytes_antigravity(payload))
        parsed = parse_canonical_json_bytes_antigravity(read_pinned_regular_bytes_antigravity(path)[0])
        _validate_failure_artifact_antigravity(parsed)  # type: ignore[arg-type]
    except BaseException:
        pass


def run_exp441_new_antigravity(*, reviewed_commit: str) -> CompletedAttemptAntigravity:
    """Run one immutable label-free source replay and record selection evidence."""
    attempt = allocate_attempt_antigravity(_RUN_ROOT_ANTIGRAVITY)
    sequence = int(attempt.name.split("_")[1])
    writer: ProgressWriterAntigravity | None = None
    interrupted = {"requested": False}
    prior_handler = signal.getsignal(signal.SIGTERM)

    def _on_sigterm(_signum: int, _frame: object) -> None:
        interrupted["requested"] = True
        raise _InterruptedAttemptAntigravity("SIGTERM received")

    installed = False
    try:
        writer = ProgressWriterAntigravity(attempt, total=FROZEN_CONFIG.dataset_count)
        signal.signal(signal.SIGTERM, _on_sigterm)
        installed = True
        _ensure_clean_reviewed_commit_antigravity(reviewed_commit)
        context_datasets, _digests, _hard, _hashes, _watched = _load_prefix_antigravity()
        active_datasets = list(context_datasets)

        def observer(event: Mapping[str, object]) -> None:
            dataset = event.get("dataset")
            sequence_value = event.get("sequence")
            test_rows, raw_length = event.get("test_row_count"), event.get("raw_length")
            if (not isinstance(dataset, str) or type(sequence_value) is not int or type(test_rows) is not int or test_rows < 0 or
                    (raw_length is not None and (type(raw_length) is not int or raw_length < 1))):
                raise ValueError("source observer event is invalid")
            if sequence_value != writer.completed_count_antigravity or active_datasets[0] != dataset:
                raise ValueError("source observer accepted out of context order")
            if writer.completed_count_antigravity == 0:
                writer.begin_record_antigravity(
                    dataset=dataset, record_name=f"{sequence_value:04d}_record_antigravity.json",
                    active_datasets=tuple(active_datasets),
                )
            active_datasets.pop(0)
            next_dataset = active_datasets[0] if active_datasets else None
            writer.accept_record_antigravity(
                dataset=dataset,
                record_name=f"{sequence_value:04d}_record_antigravity.json",
                active_datasets=tuple(active_datasets),
                next_dataset=next_dataset,
                next_record_name=(f"{sequence_value + 1:04d}_record_antigravity.json" if next_dataset else None),
                test_row_count=test_rows, raw_length=raw_length,
            )

        snapshot = _verify_with_observer_antigravity(observer)
        writer.source_replay_complete_antigravity()
        if not isinstance(snapshot, FrozenSourceSnapshotAntigravity) or snapshot.label_access_count != 0:
            raise ValueError("frozen source snapshot is invalid")
        snapshot_payload = _snapshot_payload_antigravity(snapshot)
        snapshot_bytes = canonical_json_bytes_antigravity(snapshot_payload)
        snapshot_sha256 = sha256(snapshot_bytes).hexdigest()
        selection = validate_production_selection_antigravity(select_hard_empty_single_shot_antigravity(snapshot))
        writer.selection_counts_antigravity(provisional=len(selection.provisional_candidates), outside_hard=len(selection.outside_hard_candidates), eligible=len(selection.eligible_candidates), selected=len(selection.selected))
        selection_payload = _selection_payload_antigravity(selection, reviewed_commit=reviewed_commit, source_snapshot_sha256=snapshot_sha256)
        _validate_selection_artifact_antigravity(selection_payload)
        selection_bytes = canonical_json_bytes_antigravity(selection_payload)
        selection_sha256 = sha256(selection_bytes).hexdigest()
        write_create_only_antigravity(attempt / "source_snapshot_antigravity.json", snapshot_bytes)
        write_create_only_antigravity(attempt / "selection_antigravity.json", selection_bytes)
        parsed_selection = parse_canonical_json_bytes_antigravity(read_pinned_regular_bytes_antigravity(attempt / "selection_antigravity.json")[0])
        _validate_selection_artifact_antigravity(parsed_selection)  # type: ignore[arg-type]
        reverified = _reverify_sources_after_artifacts_antigravity()
        if (not isinstance(reverified, FrozenSourceSnapshotAntigravity) or
                _snapshot_payload_antigravity(reverified) != snapshot_payload or
                reverified.label_access_count != 0):
            raise ValueError("frozen sources changed after attempt artifact creation")
        _ensure_clean_reviewed_commit_antigravity(reviewed_commit)
        writer.terminal_antigravity(stage="complete", latest_error=None)
        authority_sha256 = terminal_attempt_authority_sha256_antigravity(attempt)
        return CompletedAttemptAntigravity(sequence, reviewed_commit, snapshot_sha256, selection, selection_sha256, authority_sha256)
    except _InterruptedAttemptAntigravity as error:
        _best_effort_failure_antigravity(writer, attempt, sequence, reviewed_commit, "interrupted", error)
        raise
    except BaseException as error:
        _best_effort_failure_antigravity(writer, attempt, sequence, reviewed_commit, "failed", error)
        raise
    finally:
        if installed:
            signal.signal(signal.SIGTERM, prior_handler)


__all__ = ["allocate_attempt_antigravity", "run_exp441_new_antigravity", "terminal_attempt_authority_sha256_antigravity"]
