"""Exp451 report wrapper preserving frozen Exp450 tables and curve rendering."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import errno
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


def _rename_report_directory_no_replace_antigravity(
    source: Path, destination: Path,
) -> None:
    """Atomically publish one sibling directory without replacing any entry."""
    source = Path(source)
    destination = Path(destination)
    if source.parent != destination.parent:
        raise ValueError("Exp451 report publication stage is not a sibling")
    try:
        renamex_np = ctypes.CDLL(None, use_errno=True).renamex_np
    except AttributeError as error:
        raise OSError(
            errno.ENOTSUP, "Exp451 atomic no-replace rename is unavailable"
        ) from error
    renamex_np.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
    renamex_np.restype = ctypes.c_int
    if renamex_np(
        os.fsencode(source), os.fsencode(destination), ctypes.c_uint(0x00000004)
    ) == 0:
        return
    code = ctypes.get_errno()
    if code in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(code, "Exp451 report destination already exists")
    raise OSError(code, "Exp451 atomic no-replace report publication failed")


def _publish_report_directory_antigravity(stage: Path, destination: Path) -> Path:
    """Publish a complete sibling through one locked, atomic no-replace rename."""
    parent = destination.parent
    _checked_target_path_antigravity(parent, "report_guard_antigravity.json", create=True)
    if stage.is_symlink() or not stage.is_dir():
        raise ValueError("Exp451 report stage is unsafe")
    marker_name = ".exp451_report_complete_antigravity.json"
    entries = list(stage.iterdir())
    expected: dict[str, bytes] = {}
    for path in entries:
        details = path.lstat()
        if path.name == marker_name or path.is_symlink() or not stat.S_ISREG(details.st_mode):
            raise ValueError("Exp451 report stage contains an unsafe leaf")
        expected[path.name] = path.read_bytes()
    marker_data = canonical_json_bytes({
        "schema": "exp451-report-publication-antigravity.v1",
        "leaves_sha256": sha256(canonical_json_bytes({
            name: sha256(data).hexdigest() for name, data in sorted(expected.items())
        })).hexdigest(),
    })
    lock_descriptor = _acquire_report_publication_lock_antigravity(destination)
    try:
        current = destination.lstat() if destination.exists() or destination.is_symlink() else None
        if current is not None:
            if destination.is_symlink() or not destination.is_dir():
                raise ValueError("Exp451 report destination is unsafe")
            marker = destination / marker_name
            if marker.exists() and marker.is_file() and marker.read_bytes() == marker_data:
                destination_entries = list(destination.iterdir())
                if any(
                    path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode)
                    for path in destination_entries
                ):
                    raise ValueError("Exp451 report destination already differs")
                actual = {
                    path.name: path.read_bytes()
                    for path in destination_entries
                    if path.name != marker_name
                }
                if (
                    expected != actual or len(actual) + 1 != len(destination_entries)
                ):
                    raise ValueError("Exp451 report destination already differs")
                return destination
            # A markerless or differently marked destination may be foreign.
            # It is never deleted or adopted by this publication protocol.
            raise ValueError("Exp451 report destination already differs")

        candidate = Path(tempfile.mkdtemp(
            prefix=f".{destination.name}.publish_stage_antigravity_", dir=parent
        ))
        published = False
        try:
            for name, data in expected.items():
                _publish_create_only_antigravity(candidate / name, data)
            _publish_create_only_antigravity(candidate / marker_name, marker_data)
            _dir_fsync_antigravity(candidate)
            try:
                _rename_report_directory_no_replace_antigravity(candidate, destination)
            except FileExistsError as error:
                raise ValueError("Exp451 report destination already differs") from error
            published = True
            _dir_fsync_antigravity(parent)
            return destination
        finally:
            if not published and candidate.exists():
                shutil.rmtree(candidate)
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
