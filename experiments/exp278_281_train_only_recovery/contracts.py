from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path


EXP272_ROOT = Path("/Users/minho/Documents/Dataset/exp267_273_zero_support_fn_recovery")
OUTPUT_ROOT = Path("/Users/minho/Documents/Dataset/exp278_281_train_only_recovery")
ALPHA = 0.005
RISK_TARGETS = (0.005, 0.01, 0.02)
BOOTSTRAP_SEEDS = tuple(range(20260722, 20260727))
MIN_CONTINUOUS_GROUPS = 2
VARIANTS = {
    "risk_001_stability_080": (0.01, 0.80),
    "risk_002_stability_080": (0.02, 0.80),
    "risk_002_stability_060": (0.02, 0.60),
}
CONTINUOUS_GROUPS = frozenset({
    "scattering", "signature",
    "rff_forecast+tcn_seed_intersection+tcn_seed_median",
})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def candidate_rows() -> tuple[list[dict[str, object]], tuple[str, ...]]:
    path = EXP272_ROOT / "exp272_frozen_predictions.jsonl"
    rows = read_jsonl(path)
    datasets = tuple(sorted(str(row["dataset"]) for row in rows))
    if len(rows) != 1117 or tuple(str(row["dataset"]) for row in rows) != datasets:
        raise ValueError("Exp272 population/order mismatch")
    return rows, datasets
