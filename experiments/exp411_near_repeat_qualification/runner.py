"""Representative smoke and separately authorized Exp411 full qualification."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from hashlib import sha256
import json
import platform
from pathlib import Path
import subprocess

import numpy as np

from experiments.exp403_408.artifact_io import sha256_file

from .backends import probe_backends
from .config import FROZEN_CONFIG, Exp411FrozenConfig
from .contracts import TrainRun
from .inputs import database_snapshot, load_train_runs
from .publication import publish_payloads
from .qualification import aggregate_qualification, qualify_dataset
from .reporting import render_payloads


_REPO_ROOT = Path(__file__).resolve().parents[2]


def load_population_manifest(
    path: Path,
    *,
    config: Exp411FrozenConfig = FROZEN_CONFIG,
) -> tuple[str, ...]:
    source = Path(path)
    if sha256_file(source) != config.population_manifest_sha256:
        raise ValueError("Exp411 population manifest SHA256 mismatch")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Exp411 population manifest is invalid") from error
    lanes = payload.get("frozen_hard_lanes") if isinstance(payload, dict) else None
    if not isinstance(lanes, dict):
        raise ValueError("Exp411 population manifest lacks frozen_hard_lanes")
    datasets = tuple(sorted(str(dataset) for dataset in lanes))
    families = {_family_from_dataset(dataset) for dataset in datasets}
    if (
        len(datasets) != config.expected_datasets
        or len(set(datasets)) != len(datasets)
        or len(families) != config.expected_families
    ):
        raise ValueError("Exp411 population manifest universe mismatch")
    return datasets


def run_representative_smoke(output_dir: Path) -> Mapping[str, str]:
    output = Path(output_dir)
    if not output.parent.is_dir():
        raise FileNotFoundError("Exp411 output parent does not exist")
    if output.exists():
        raise FileExistsError("Exp411 output directory must not exist")
    rows = []
    for dataset_index, (family_size, family_index) in enumerate(
        (size, index) for size in (1, 3, 7, 21) for index in range(8)
    ):
        family = f"SmokeFamily{family_size}_{family_index}"
        dataset = f"{family}_normal_0"
        rows.append(
            qualify_dataset(
                dataset=dataset,
                family=family,
                runs=_synthetic_runs(dataset, dataset_index),
                config=FROZEN_CONFIG,
            )
        )
    aggregate = aggregate_qualification(
        rows,
        bootstrap_iterations=20,
        config=FROZEN_CONFIG,
    )
    provenance = {
        "mode": "representative_smoke",
        "git_commit": _git_commit(allow_dirty=True),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "synthetic_dataset_count": 32,
        "contains_real_dataset_rows": False,
    }
    backend = {
        "canonical_backend": "numpy_cpu",
        "mlx_status": "excluded_by_user_decision",
        "probes": {
            name: asdict(value) if hasattr(value, "__dataclass_fields__") else value
            for name, value in probe_backends().items()
        },
    }
    payloads = render_payloads(
        rows,
        aggregate,
        backend,
        provenance,
        full_qualification_executed=False,
        config=FROZEN_CONFIG,
    )
    return publish_payloads(
        payloads,
        output,
        provenance,
        config=FROZEN_CONFIG,
    )


def run_full_qualification(
    db_path: Path,
    population_manifest: Path,
    output_dir: Path,
) -> Mapping[str, str]:
    """Run the real 1,117-dataset qualification only after external approval."""
    starting_commit = _require_clean_git()
    datasets = load_population_manifest(population_manifest, config=FROZEN_CONFIG)
    pinned_snapshot = database_snapshot(db_path)
    runs_by_dataset = load_train_runs(db_path, datasets, config=FROZEN_CONFIG)
    rows = [
        qualify_dataset(
            dataset=dataset,
            family=_family_from_dataset(dataset),
            runs=runs_by_dataset[dataset],
            config=FROZEN_CONFIG,
        )
        for dataset in datasets
    ]
    aggregate = aggregate_qualification(rows, config=FROZEN_CONFIG)
    if _require_clean_git() != starting_commit:
        raise ValueError("Exp411 Git commit changed during full qualification")
    provenance = {
        "mode": "full_qualification",
        "git_commit": starting_commit,
        "database_snapshot": dict(pinned_snapshot),
        "population_manifest_sha256": sha256_file(population_manifest),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }
    backend = {
        "canonical_backend": "numpy_cpu",
        "mlx_status": "excluded_by_user_decision",
        "probes": {
            name: asdict(value) if hasattr(value, "__dataclass_fields__") else value
            for name, value in probe_backends().items()
        },
    }
    payloads = render_payloads(
        rows,
        aggregate,
        backend,
        provenance,
        full_qualification_executed=True,
        config=FROZEN_CONFIG,
    )
    return publish_payloads(
        payloads,
        Path(output_dir),
        provenance,
        config=FROZEN_CONFIG,
        pre_publish_verify=lambda: _require_full_state(
            db_path,
            pinned_snapshot,
            starting_commit,
        ),
    )


def _synthetic_runs(dataset: str, dataset_index: int) -> tuple[TrainRun, ...]:
    x = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    output = []
    for identity in range(6):
        values = (
            np.sin(x * (1.0 + identity * 0.07))
            + 0.23 * np.cos(x * (2.0 + (dataset_index % 3) * 0.1))
            + identity * 0.037
            + dataset_index * 0.001
        ).astype("<f4")
        payload = values.tobytes()
        output.append(
            TrainRun(
                dataset=dataset,
                instance_index=identity,
                values=tuple(float(value) for value in values),
                values_sha256=sha256(payload).hexdigest(),
            )
        )
    return tuple(output)


def _family_from_dataset(dataset: str) -> str:
    if dataset.count("_normal_") != 1:
        raise ValueError("Exp411 dataset must contain one _normal_ marker")
    family, suffix = dataset.split("_normal_", 1)
    if not family or not (
        suffix.isdecimal() or (suffix.startswith("-") and suffix[1:].isdecimal())
    ):
        raise ValueError("Exp411 dataset family and numeric class suffix are required")
    return family


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
        raise ValueError("Exp411 cannot resolve Git commit")
    return result.stdout.strip()


def _require_clean_git() -> str:
    status = subprocess.run(
        ["git", "-C", str(_REPO_ROOT), "status", "--porcelain"],
        text=True,
        capture_output=True,
        check=False,
    )
    if status.returncode != 0:
        raise ValueError("Exp411 cannot inspect Git worktree")
    if status.stdout.strip():
        raise ValueError("Exp411 full qualification requires a clean Git worktree")
    return _git_commit(allow_dirty=False)


def _require_full_state(
    db_path: Path,
    expected_snapshot: Mapping[str, str],
    expected_commit: str,
) -> None:
    if dict(database_snapshot(db_path)) != dict(expected_snapshot):
        raise ValueError("Exp411 SQLite snapshot changed before publication")
    if _require_clean_git() != expected_commit:
        raise ValueError("Exp411 Git commit changed before publication")


__all__ = [
    "load_population_manifest",
    "run_full_qualification",
    "run_representative_smoke",
]
