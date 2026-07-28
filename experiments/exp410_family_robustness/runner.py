"""Synthetic smoke and separately gated full orchestration for Exp410."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from hashlib import sha256
import importlib.metadata
import json
import math
import platform
from pathlib import Path
import sqlite3
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np

from experiments.exp403_408.artifact_io import canonical_json_bytes, sha256_file

from . import profiles as profile_module
from .aggregation import aggregate_family_robustness
from .config import FROZEN_CONFIG, Exp410FrozenConfig
from .contracts import AuditResult
from .inputs import load_verified_inputs, recompute_dataset_metrics
from .profiles import database_snapshot, scan_dataset_profiles
from .publication import publish_audit


_REPO_ROOT = Path(__file__).resolve().parents[2]
_FULL_CPU_REASON = (
    "SQLite streaming, exact byte hashing, grouped counters, and a "
    "128-family bootstrap are not tensor-heavy; CPU/NumPy is canonical."
)


@dataclass(frozen=True)
class _SyntheticFixture:
    input_dir: Path
    db_path: Path
    exp291_path: Path
    exp292_path: Path
    config: Exp410FrozenConfig
    datasets: tuple[str, ...]


def run_smoke(output_dir: Path) -> Mapping[str, str]:
    """Run the full component chain against a deterministic 32-dataset fixture."""
    output = Path(output_dir)
    if not output.parent.is_dir():
        raise FileNotFoundError("Exp410 smoke output parent does not exist")
    if output.exists():
        raise FileExistsError("Exp410 smoke output directory must not exist")
    with TemporaryDirectory(
        dir=output.parent,
        prefix=".exp410-synthetic-fixture.",
    ) as raw:
        fixture = _write_synthetic_fixture(Path(raw))
        inputs = load_verified_inputs(
            fixture.input_dir,
            fixture.exp291_path,
            fixture.exp292_path,
            config=fixture.config,
        )
        metrics = recompute_dataset_metrics(inputs, config=fixture.config)
        profiles = profile_module.scan_dataset_profiles(
            fixture.db_path,
            fixture.datasets,
            config=fixture.config,
        )
        result = aggregate_family_robustness(
            metrics,
            profiles,
            config=fixture.config,
        )
        result = replace(
            result,
            summary={
                **dict(result.summary),
                "smoke": True,
                "smoke_dataset_count": 32,
                "full_audit_executed": False,
            },
        )
        provenance = _build_provenance(
            inputs=dict(inputs.input_sha256),
            db_path=fixture.db_path,
            config=fixture.config,
            mode="representative_smoke",
            git_commit=_git_commit(allow_dirty=True),
        )
        return publish_audit(
            result,
            output,
            provenance,
            config=fixture.config,
        )


def run_full_audit(
    input_dir: Path,
    db_path: Path,
    output_dir: Path,
) -> Mapping[str, str]:
    """Run the real read-only audit after an external explicit authorization."""
    starting_commit = _require_clean_git()
    database = Path(db_path)
    data_root = database.parent
    exp291_path = (
        data_root
        / "exp291_final_operational_routing_freeze"
        / "exp291_frozen_operational_routing.jsonl"
    )
    exp292_path = (
        data_root
        / "exp292_standard_review_evidence_tiering"
        / "exp292_frozen_review_tiers.jsonl"
    )
    inputs = load_verified_inputs(
        Path(input_dir),
        exp291_path,
        exp292_path,
        config=FROZEN_CONFIG,
    )
    metrics = recompute_dataset_metrics(inputs, config=FROZEN_CONFIG)
    datasets = tuple(sorted(inputs.labels_by_dataset))
    profiles = scan_dataset_profiles(
        database,
        datasets,
        config=FROZEN_CONFIG,
    )
    result = aggregate_family_robustness(
        metrics,
        profiles,
        config=FROZEN_CONFIG,
    )
    ending_commit = _require_clean_git()
    if ending_commit != starting_commit:
        raise ValueError("Exp410 Git commit changed during full audit")
    provenance = _build_provenance(
        inputs=dict(inputs.input_sha256),
        db_path=database,
        config=FROZEN_CONFIG,
        mode="full_audit",
        git_commit=starting_commit,
    )
    return publish_audit(
        result,
        Path(output_dir),
        provenance,
        config=FROZEN_CONFIG,
    )


def _write_synthetic_fixture(root: Path) -> _SyntheticFixture:
    input_dir = root / "publication"
    input_dir.mkdir()
    db_path = root / "fixture.db"
    specifications = _synthetic_specifications()
    datasets = tuple(spec.dataset for spec in specifications)

    labels_path = input_dir / "exp403_408_frozen_test_labels.csv"
    prediction_path = input_dir / "exp404_hard_confidence.jsonl"
    prediction_manifest_path = input_dir / "exp404_prediction_manifest.json"
    exp291_path = root / "exp291.jsonl"
    exp292_path = root / "exp292.jsonl"

    labels_lines = ["dataset,candidate_index,is_true_anomaly\n"]
    prediction_rows: list[dict[str, object]] = []
    lane_rows: list[dict[str, object]] = []
    frozen_hard: dict[str, list[int]] = {}
    f1_values: list[float] = []
    total_hard = total_tp = total_fp = total_fn = 0

    connection = sqlite3.connect(db_path)
    try:
        connection.executescript(
            """
            CREATE TABLE datasets (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE NOT NULL
            );
            CREATE TABLE instances (
                id INTEGER PRIMARY KEY,
                dataset_id INTEGER NOT NULL,
                split TEXT NOT NULL,
                instance_index INTEGER NOT NULL,
                label TEXT NOT NULL,
                values_blob BLOB NOT NULL,
                labels_blob BLOB
            );
            """
        )
        for dataset_id, spec in enumerate(specifications, start=1):
            connection.execute(
                "INSERT INTO datasets(id, name) VALUES (?, ?)",
                (dataset_id, spec.dataset),
            )
            train_blobs = [
                _fixture_blob(dataset_id, index)
                for index in range(spec.train_count)
            ]
            connection.executemany(
                """
                INSERT INTO instances(
                    dataset_id, split, instance_index, label, values_blob
                ) VALUES (?, 'TRAIN', ?, 'normal', ?)
                """,
                (
                    (dataset_id, index, blob)
                    for index, blob in enumerate(train_blobs)
                ),
            )

            labels = [1] * spec.anomaly_count + [0] * spec.normal_count
            test_blobs = [
                _fixture_blob(dataset_id, 10_000 + index)
                for index in range(spec.anomaly_count)
            ]
            test_blobs.extend(train_blobs[: spec.overlap_count])
            test_blobs.extend(
                _fixture_blob(dataset_id, 20_000 + index)
                for index in range(spec.normal_count - spec.overlap_count)
            )
            connection.executemany(
                """
                INSERT INTO instances(
                    dataset_id, split, instance_index, label, values_blob
                ) VALUES (?, 'TEST', ?, ?, ?)
                """,
                (
                    (
                        dataset_id,
                        index,
                        "anomaly" if label else "normal",
                        test_blobs[index],
                    )
                    for index, label in enumerate(labels)
                ),
            )
            labels_lines.extend(
                f"{spec.dataset},{index},{label}\n"
                for index, label in enumerate(labels)
            )
            hard = list(spec.hard_indices)
            frozen_hard[spec.dataset] = hard
            prediction_rows.extend(
                {"dataset": spec.dataset, "candidate_index": index}
                for index in hard
            )
            lane_rows.append(
                {
                    "dataset": spec.dataset,
                    "automatic_hard_indices": hard,
                    "standard_review_indices": [],
                    "priority_review_indices": [],
                }
            )
            tp = sum(labels[index] for index in hard)
            fp = len(hard) - tp
            fn = spec.anomaly_count - tp
            total_hard += len(hard)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            denominator = 2 * tp + fp + fn
            f1_values.append(2 * tp / denominator if denominator else 0.0)
        connection.commit()
    finally:
        connection.close()

    labels_path.write_text("".join(labels_lines), encoding="utf-8")
    _write_jsonl(prediction_path, prediction_rows)
    _write_json(
        prediction_manifest_path,
        {
            "hard_confidence_file": prediction_path.name,
            "artifact_hashes": {"prediction": sha256_file(prediction_path)},
            "frozen_hard_lanes": frozen_hard,
        },
    )
    _write_jsonl(exp291_path, lane_rows)
    _write_jsonl(exp292_path, lane_rows)

    input_manifest_path = input_dir / "exp403_408_evaluation_input_manifest.json"
    _write_json(
        input_manifest_path,
        {
            "artifact_sha256": {
                prediction_manifest_path.name: sha256_file(prediction_manifest_path),
                prediction_path.name: sha256_file(prediction_path),
            }
        },
    )
    _write_json(
        input_dir / "exp403_408_evaluation_publication_manifest.json",
        {
            "artifact_sha256": {
                input_manifest_path.name: sha256_file(input_manifest_path),
                labels_path.name: sha256_file(labels_path),
            }
        },
    )
    mean_f1 = math.fsum(f1_values) / len(f1_values)
    config = replace(
        FROZEN_CONFIG,
        expected_datasets=32,
        expected_families=4,
        expected_family_size_buckets=(1, 1, 1, 1),
        expected_hard=total_hard,
        expected_tp=total_tp,
        expected_fp=total_fp,
        expected_fn=total_fn,
        expected_dataset_macro_f1=mean_f1,
        display_dataset_macro_f1=f"{mean_f1:.6f}",
        prediction_sha256=sha256_file(prediction_path),
        frozen_labels_sha256=sha256_file(labels_path),
        database_sha256=sha256_file(db_path),
        exp291_sha256=sha256_file(exp291_path),
        exp292_sha256=sha256_file(exp292_path),
    )
    return _SyntheticFixture(
        input_dir=input_dir,
        db_path=db_path,
        exp291_path=exp291_path,
        exp292_path=exp292_path,
        config=config,
        datasets=datasets,
    )


@dataclass(frozen=True)
class _SyntheticSpec:
    dataset: str
    train_count: int
    anomaly_count: int
    normal_count: int
    overlap_count: int
    hard_indices: tuple[int, ...]


def _synthetic_specifications() -> tuple[_SyntheticSpec, ...]:
    result: list[_SyntheticSpec] = []
    global_index = 0
    for family, size in (
        ("Singleton", 1),
        ("Small", 3),
        ("Medium", 7),
        ("Large", 21),
    ):
        for suffix in range(size):
            bucket = global_index % 3
            anomaly_count = (1, 5, 11)[bucket]
            hard = (
                ()
                if global_index == 0
                else (0, anomaly_count)
                if global_index == 1
                else (0,)
            )
            result.append(
                _SyntheticSpec(
                    dataset=f"{family}_normal_{suffix}",
                    train_count=(10, 30, 60)[bucket],
                    anomaly_count=anomaly_count,
                    normal_count=4,
                    overlap_count=(1, 3, 4)[bucket],
                    hard_indices=hard,
                )
            )
            global_index += 1
    return tuple(result)


def _fixture_blob(dataset_id: int, index: int) -> bytes:
    return np.asarray(
        (float(dataset_id), float(index), 0.25),
        dtype="<f4",
    ).tobytes()


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_bytes(
        b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    )


def _build_provenance(
    *,
    inputs: Mapping[str, str],
    db_path: Path,
    config: Exp410FrozenConfig,
    mode: str,
    git_commit: str,
) -> dict[str, object]:
    code_hashes = {
        path.name: sha256_file(path)
        for path in sorted(Path(__file__).parent.glob("*.py"))
    }
    script = _REPO_ROOT / "scratch" / "run_exp410_family_robustness.py"
    if script.is_file():
        code_hashes[str(script.relative_to(_REPO_ROOT))] = sha256_file(script)
    config_sha256 = sha256(canonical_json_bytes(config.to_payload())).hexdigest()
    code_sha256 = sha256(canonical_json_bytes(code_hashes)).hexdigest()
    runtime = _runtime_payload(mode)
    return {
        "inputs": {
            **dict(sorted(inputs.items())),
            "database_snapshot": dict(database_snapshot(db_path)),
        },
        "code": {
            "git_commit": git_commit,
            "config_sha256": config_sha256,
            "code_sha256": code_sha256,
            "artifact_sha256": dict(sorted(code_hashes.items())),
        },
        "runtime": runtime,
    }


def _runtime_payload(mode: str) -> dict[str, object]:
    try:
        import torch

        pytorch_version: str | None = str(torch.__version__)
        mps_built: bool | None = bool(torch.backends.mps.is_built())
        mps_available: bool | None = bool(torch.backends.mps.is_available())
    except (ImportError, AttributeError):
        pytorch_version = None
        mps_built = None
        mps_available = None
    try:
        mlx_version: str | None = importlib.metadata.version("mlx")
    except importlib.metadata.PackageNotFoundError:
        mlx_version = None
    return {
        "mode": mode,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pytorch_version": pytorch_version,
        "pytorch_mps_built": mps_built,
        "pytorch_mps_available": mps_available,
        "mlx_version": mlx_version,
        "platform": platform.platform(),
        "canonical_path": "cpu_sqlite_hashlib_numpy",
        "canonical_path_reason": _FULL_CPU_REASON,
        "tensor_eligible": False,
        "accelerated_path_canonical": False,
    }


def _require_clean_git() -> str:
    status = subprocess.run(
        ["git", "-C", str(_REPO_ROOT), "status", "--porcelain"],
        text=True,
        capture_output=True,
        check=False,
    )
    if status.returncode != 0:
        raise ValueError(f"Exp410 cannot inspect Git worktree: {status.stderr.strip()}")
    if status.stdout.strip():
        raise ValueError("Exp410 full audit requires a clean Git worktree")
    return _git_commit(allow_dirty=False)


def _git_commit(*, allow_dirty: bool) -> str:
    result = subprocess.run(
        ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        if allow_dirty:
            return "unavailable"
        raise ValueError("Exp410 cannot resolve Git commit")
    return result.stdout.strip()
