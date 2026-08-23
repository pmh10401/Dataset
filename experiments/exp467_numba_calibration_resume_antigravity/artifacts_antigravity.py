"""Create-only Exp467 checkpoint envelopes used by the verified migration."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import stat
import tempfile

from .contracts_antigravity import canonical_json_bytes


SUCCESS_SCHEMA_ANTIGRAVITY = "exp467-dataset-condition-success-antigravity.v1"
SUCCESS_NAME_ANTIGRAVITY = "dataset_condition_success_antigravity.json"


def _safe_relative_path_antigravity(value: object) -> Path:
    if type(value) is not str or not value or value.startswith(("/", "\\")):
        raise ValueError("Exp467 artifact relative path is invalid")
    path = Path(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("Exp467 artifact relative path is invalid")
    return path


def _sha256_antigravity(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(item in "0123456789abcdef" for item in value)


def _canonical_read_antigravity(path: Path) -> dict[str, object]:
    details = path.lstat() if path.exists() or path.is_symlink() else None
    if details is None or stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
        raise ValueError("Exp467 artifact is missing or invalid")
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Exp467 artifact JSON is invalid") from error
    if not isinstance(value, dict) or raw != canonical_json_bytes(value):
        raise ValueError("Exp467 artifact is noncanonical")
    return value


def _checked_target_path_antigravity(
    root: Path, relative_path: str, *, create: bool
) -> Path:
    root = Path(root)
    if not root.is_absolute():
        raise ValueError("Exp467 artifact root must be absolute")
    relative = _safe_relative_path_antigravity(relative_path)
    if create:
        root.mkdir(parents=True, exist_ok=True)
    try:
        root_details = root.lstat()
    except FileNotFoundError as error:
        raise ValueError("Exp467 artifact root is missing") from error
    if stat.S_ISLNK(root_details.st_mode) or not stat.S_ISDIR(root_details.st_mode):
        raise ValueError("Exp467 artifact root is unsafe")
    current = root
    for component in relative.parts[:-1]:
        current = current / component
        if create:
            try:
                current.mkdir()
            except FileExistsError:
                pass
        try:
            details = current.lstat()
        except FileNotFoundError as error:
            raise ValueError("Exp467 artifact parent is missing") from error
        if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
            raise ValueError("Exp467 artifact parent is unsafe")
    return root / relative


def _validate_success_envelope_antigravity(value: object) -> dict[str, object]:
    required = {
        "schema", "context_sha256", "source_context_sha256", "catalog_sha256",
        "condition_id", "dataset", "raw_identity", "backend", "execution_origin",
        "payload", "payload_sha256", "source_envelope_sha256",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Exp467 success envelope schema differs")
    if value["schema"] != SUCCESS_SCHEMA_ANTIGRAVITY:
        raise ValueError("Exp467 success envelope schema differs")
    if not all(_sha256_antigravity(value[key]) for key in (
        "context_sha256", "source_context_sha256", "catalog_sha256", "payload_sha256", "source_envelope_sha256",
    )):
        raise ValueError("Exp467 success envelope digest differs")
    if value["condition_id"] not in {"A", "B", "C", "D"} or type(value["dataset"]) is not str or not value["dataset"]:
        raise ValueError("Exp467 success envelope coordinate differs")
    if value["execution_origin"] != "imported_exp450_mps":
        raise ValueError("Exp467 success envelope origin differs")
    if not isinstance(value["raw_identity"], dict) or value["raw_identity"].get("dataset") != value["dataset"]:
        raise ValueError("Exp467 success envelope raw identity differs")
    if not isinstance(value["backend"], dict) or not isinstance(value["payload"], dict):
        raise ValueError("Exp467 success envelope payload differs")
    if sha256(canonical_json_bytes(value["payload"])).hexdigest() != value["payload_sha256"]:
        raise ValueError("Exp467 success envelope payload digest differs")
    return value


def success_relative_path_antigravity(dataset: str, condition_id: str) -> str:
    """Return the fixed root-contained filename for one coordinate."""
    if type(dataset) is not str or not dataset or condition_id not in {"A", "B", "C", "D"}:
        raise ValueError("Exp467 artifact coordinate is invalid")
    token = sha256(dataset.encode("utf-8")).hexdigest()
    return (
        f"checkpoints_antigravity/{condition_id}_antigravity/"
        f"{token}_antigravity/{SUCCESS_NAME_ANTIGRAVITY}"
    )


def _publish_create_only_antigravity(destination: Path, data: bytes) -> None:
    """Publish durable bytes by hard-linking a same-directory fsynced temporary file."""
    existing = destination.lstat() if destination.exists() or destination.is_symlink() else None
    if existing is not None:
        if stat.S_ISLNK(existing.st_mode) or not stat.S_ISREG(existing.st_mode):
            raise ValueError("Exp467 existing artifact is invalid")
        if destination.read_bytes() != data:
            raise ValueError("Exp467 existing create-only artifact differs")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".exp467_publish_antigravity_", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            existing = destination.lstat()
            if stat.S_ISLNK(existing.st_mode) or not stat.S_ISREG(existing.st_mode) or destination.read_bytes() != data:
                raise ValueError("Exp467 existing create-only artifact differs") from None
        finally:
            temporary.unlink(missing_ok=True)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def write_exp467_dataset_condition_success_antigravity(
    root: Path, envelope: dict[str, object], relative_path: str | None = None
) -> Path:
    """Create or byte-identically reuse one imported Exp467 success envelope."""
    checked = _validate_success_envelope_antigravity(envelope)
    relative = relative_path or success_relative_path_antigravity(
        checked["dataset"], checked["condition_id"]
    )
    destination = _checked_target_path_antigravity(Path(root), relative, create=True)
    _publish_create_only_antigravity(destination, canonical_json_bytes(checked))
    return destination


def read_exp467_dataset_condition_success_antigravity(root: Path, relative_path: str) -> dict[str, object]:
    """Read one canonical imported envelope through the same containment checks."""
    return _validate_success_envelope_antigravity(
        _canonical_read_antigravity(
            _checked_target_path_antigravity(Path(root), relative_path, create=False)
        )
    )


__all__ = [
    "SUCCESS_NAME_ANTIGRAVITY", "SUCCESS_SCHEMA_ANTIGRAVITY",
    "read_exp467_dataset_condition_success_antigravity", "success_relative_path_antigravity",
    "write_exp467_dataset_condition_success_antigravity",
]
