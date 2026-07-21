from __future__ import annotations

import hashlib
from pathlib import Path

from .config import DB_PATH, FROZEN_DATABASE_SHA256


class DatabaseContractError(ValueError):
    """Raised when the frozen Exp266 SQLite database does not match its pin."""


def database_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise DatabaseContractError(f"cannot read frozen database: {path}") from exc
    return digest.hexdigest()


def verify_database_sha256(
    path: Path = DB_PATH,
    expected_sha256: str = FROZEN_DATABASE_SHA256,
) -> str:
    actual = database_sha256(path)
    if actual != expected_sha256:
        raise DatabaseContractError(
            f"database SHA256 mismatch: expected={expected_sha256} actual={actual}"
        )
    return actual
