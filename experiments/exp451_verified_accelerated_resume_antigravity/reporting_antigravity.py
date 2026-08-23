"""Exp451 report wrapper preserving frozen Exp450 tables and curve rendering."""

from __future__ import annotations

from dataclasses import dataclass
import fcntl
import os
from pathlib import Path
import secrets
import shutil
import stat
import tempfile
from hashlib import sha256

from experiments.exp450_resampling_image_ablation_antigravity.reporting_antigravity import (
    Exp450ReportArtifactAntigravity,
    write_exp450_report_antigravity,
)

from .contracts_antigravity import canonical_json_bytes
from .artifacts_antigravity import _checked_target_path_antigravity, _publish_create_only_antigravity
from .config_antigravity import REPORT_ROOT
from .evaluation_antigravity import Exp451EvaluationAntigravity, _validate_evaluation_antigravity, require_exp451_evaluation_ready_antigravity
from .verifier_antigravity import Exp451VerificationAntigravity, _validate_verification_authority_antigravity, provenance_summary_antigravity


@dataclass(frozen=True, slots=True)
class Exp451ReportArtifactAntigravity:
    """Frozen Exp450 figure/table leaves plus an Exp451 origin-provenance leaf."""

    frozen_report: Exp450ReportArtifactAntigravity
    provenance_path: Path


def _dir_fsync_antigravity(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _acquire_report_publication_lock_antigravity(destination: Path) -> int:
    """Take a process-owned lock that the kernel releases after any process death."""
    parent = destination.parent
    relative = f".{destination.name}.publish_lock_antigravity"
    lock = _checked_target_path_antigravity(parent, relative, create=True)
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lock, flags, 0o600)
    except OSError as error:
        raise ValueError("Exp451 report publication lock is unsafe") from error
    try:
        opened = os.fstat(descriptor)
        linked = lock.lstat()
        if (
            not stat.S_ISREG(opened.st_mode)
            or not stat.S_ISREG(linked.st_mode)
            or opened.st_dev != linked.st_dev
            or opened.st_ino != linked.st_ino
            or opened.st_nlink != 1
        ):
            raise ValueError("Exp451 report publication lock is unsafe")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Exp451 report publication is already in progress") from None
        owner = canonical_json_bytes({
            "schema": "exp451-report-publication-lock-antigravity.v1",
            "destination": destination.name,
            "pid": os.getpid(),
            "nonce": secrets.token_hex(16),
        })
        os.ftruncate(descriptor, 0)
        offset = 0
        while offset < len(owner):
            written = os.write(descriptor, owner[offset:])
            if written <= 0:
                raise OSError("Exp451 report publication lock write made no progress")
            offset += written
        os.fsync(descriptor)
        _dir_fsync_antigravity(parent)
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _release_report_publication_lock_antigravity(descriptor: int) -> None:
    """Release without unlinking a path that another process may already have opened."""
    try:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _publish_report_directory_antigravity(stage: Path, destination: Path) -> Path:
    """Publish staged leaves under a process-owned lock and completion marker.

    Portable ``rename`` can replace an existing directory, so it is not a
    create-only primitive.  A nonblocking kernel lock serializes publishers and
    is released automatically after process death.  Individual leaves use
    hard-link create-only publication and the marker is written last.  A
    directory lacking that marker is an interrupted staging attempt, never a
    valid final report, and is removed only by its lock owner before a retry.
    """
    parent = destination.parent
    _checked_target_path_antigravity(parent, "report_guard_antigravity.json", create=True)
    if stage.is_symlink() or not stage.is_dir():
        raise ValueError("Exp451 report stage is unsafe")
    expected = {path.name: path.read_bytes() for path in stage.iterdir() if path.is_file()}
    if len(expected) != len(list(stage.iterdir())):
        raise ValueError("Exp451 report stage contains an unsafe leaf")
    marker_name = ".exp451_report_complete_antigravity.json"
    progress_name = ".exp451_report_in_progress_antigravity.json"
    marker_data = canonical_json_bytes({
        "schema": "exp451-report-publication-antigravity.v1",
        "leaves_sha256": sha256(canonical_json_bytes({
            name: sha256(data).hexdigest() for name, data in sorted(expected.items())
        })).hexdigest(),
    })
    progress_data = canonical_json_bytes({
        "schema": "exp451-report-publication-in-progress-antigravity.v1",
        "destination": destination.name,
    })
    lock_descriptor = _acquire_report_publication_lock_antigravity(destination)
    try:
        current = destination.lstat() if destination.exists() or destination.is_symlink() else None
        if current is not None:
            if destination.is_symlink() or not destination.is_dir():
                raise ValueError("Exp451 report destination is unsafe")
            marker = destination / marker_name
            if marker.exists() and marker.is_file() and marker.read_bytes() == marker_data:
                actual = {path.name: path.read_bytes() for path in destination.iterdir() if path.is_file() and path.name not in {marker_name, progress_name}}
                if (
                    expected != actual or (destination / progress_name).read_bytes() != progress_data
                    or len(actual) + 2 != len(list(destination.iterdir()))
                ):
                    raise ValueError("Exp451 report destination already differs")
                return destination
            # Only an incomplete directory made by this protocol can be cleaned.
            progress = destination / progress_name
            if (
                marker.exists() or not progress.is_file() or progress.read_bytes() != progress_data
                or any(path.is_symlink() or not path.is_file() for path in destination.iterdir())
            ):
                raise ValueError("Exp451 report destination already differs")
            shutil.rmtree(destination)
            _dir_fsync_antigravity(parent)
        destination.mkdir()
        _publish_create_only_antigravity(destination / progress_name, progress_data)
        for name, data in expected.items():
            _publish_create_only_antigravity(destination / name, data)
        _dir_fsync_antigravity(destination)
        _publish_create_only_antigravity(destination / marker_name, marker_data)
        _dir_fsync_antigravity(destination)
        _dir_fsync_antigravity(parent)
        return destination
    finally:
        _release_report_publication_lock_antigravity(lock_descriptor)


def write_exp451_report_antigravity(
    evaluation: Exp451EvaluationAntigravity, root: Path, *, verification: Exp451VerificationAntigravity,
) -> Exp451ReportArtifactAntigravity:
    """Write unchanged comparison figures only after the 4,468-result gate closes."""
    require_exp451_evaluation_ready_antigravity(verification)
    verification = _validate_verification_authority_antigravity(verification)
    evaluation = _validate_evaluation_antigravity(evaluation, verification)
    if Path(root) != REPORT_ROOT or Path(root).is_symlink():
        raise ValueError("Exp451 report root differs from the frozen configured root")
    report_root = Path(root)
    stage_parent = report_root.parent
    stage_root = Path(tempfile.mkdtemp(prefix=".exp451_report_stage_antigravity_", dir=stage_parent))
    try:
        frozen_stage = write_exp450_report_antigravity(evaluation.frozen_evaluation, stage_root)
        summary = provenance_summary_antigravity(verification)
        payload = {
        "schema": "exp451-report-provenance-antigravity.v1",
        "experiment": "Exp451",
        "frozen_metric_definition": "exp450_resampling_image_ablation_antigravity",
        "method_order": list(evaluation.frozen_evaluation.metric_tables.method_order),
        "curve_axes": {
            "roc": {"x": "false_positive_rate", "y": "true_positive_rate"},
            "precision_recall": {"x": "recall", "y": "precision"},
        },
            **summary,
        }
        data = canonical_json_bytes(payload)
        stage_provenance = _checked_target_path_antigravity(
            frozen_stage.root, "exp451_report_provenance_antigravity.json", create=True
        )
        _publish_create_only_antigravity(stage_provenance, data)
        _dir_fsync_antigravity(frozen_stage.root)
        final_directory = _publish_report_directory_antigravity(
            frozen_stage.root, report_root / "exp450_report_artifacts_antigravity"
        )
        frozen = Exp450ReportArtifactAntigravity(
            root=final_directory,
            metric_csv_path=final_directory / frozen_stage.metric_csv_path.name,
            metric_json_path=final_directory / frozen_stage.metric_json_path.name,
            curve_csv_path=final_directory / frozen_stage.curve_csv_path.name,
            curve_json_path=final_directory / frozen_stage.curve_json_path.name,
            paired_csv_path=final_directory / frozen_stage.paired_csv_path.name,
            paired_json_path=final_directory / frozen_stage.paired_json_path.name,
            png_path=final_directory / frozen_stage.png_path.name,
            svg_path=final_directory / frozen_stage.svg_path.name,
            html_path=final_directory / frozen_stage.html_path.name,
            manifest_path=final_directory / frozen_stage.manifest_path.name,
        )
        return Exp451ReportArtifactAntigravity(frozen, final_directory / stage_provenance.name)
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)


__all__ = ["Exp451ReportArtifactAntigravity", "write_exp451_report_antigravity"]
