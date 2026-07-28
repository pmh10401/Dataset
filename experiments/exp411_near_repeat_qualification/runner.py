"""Representative smoke and separately authorized Exp411 full qualification."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from hashlib import sha256
import json
import platform
from pathlib import Path
import subprocess
from time import perf_counter

import numpy as np

from experiments.exp403_408.artifact_io import canonical_json_bytes, sha256_file

from .backends import assess_backend, pairwise_rms, probe_backends
from .config import FROZEN_CONFIG, Exp411FrozenConfig
from .contracts import TrainRun
from .inputs import database_snapshot, iter_train_runs
from .publication import publish_payloads
from .perturbations import perturbation_contract_provenance
from .qualification import aggregate_qualification, qualify_dataset
from .reporting import render_payloads


_REPO_ROOT = Path(__file__).resolve().parents[2]


def load_population_manifest(
    path: Path,
    *,
    config: Exp411FrozenConfig = FROZEN_CONFIG,
) -> tuple[str, ...]:
    config.validate_research_parameters()
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
    FROZEN_CONFIG.validate()
    output = Path(output_dir)
    if not output.parent.is_dir():
        raise FileNotFoundError("Exp411 output parent does not exist")
    if output.exists():
        raise FileExistsError("Exp411 output directory must not exist")
    rows, baseline_seconds = _build_smoke_rows("numpy_cpu")
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
        "perturbation_contract": perturbation_contract_provenance(),
        "smoke_coverage": {
            "family_sizes": [1, 3, 7, 21],
            "identity_regimes": [
                "abstain_lt5",
                "logo_5_19",
                "five_fold_20_plus",
            ],
        },
    }
    backend = _build_backend_assessment(rows, aggregate, baseline_seconds)
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
    FROZEN_CONFIG.validate()
    starting_commit = _require_clean_git()
    datasets = load_population_manifest(population_manifest, config=FROZEN_CONFIG)
    pinned_snapshot = database_snapshot(db_path)
    rows = []
    for dataset, runs in iter_train_runs(
        db_path,
        datasets,
        config=FROZEN_CONFIG,
    ):
        rows.append(qualify_dataset(
            dataset=dataset,
            family=_family_from_dataset(dataset),
            runs=runs,
            config=FROZEN_CONFIG,
        ))
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
        "perturbation_contract": perturbation_contract_provenance(),
    }
    smoke_rows, baseline_seconds = _build_smoke_rows("numpy_cpu")
    smoke_aggregate = aggregate_qualification(
        smoke_rows,
        bootstrap_iterations=20,
        config=FROZEN_CONFIG,
    )
    backend = _build_backend_assessment(
        smoke_rows,
        smoke_aggregate,
        baseline_seconds,
    )
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


def _smoke_specifications() -> tuple[tuple[str, str, int, int], ...]:
    result = []
    for dataset_index, (family_size, member_index) in enumerate(
        (size, index)
        for size in (1, 3, 7, 21)
        for index in range(size)
    ):
        family = f"SmokeFamily{family_size}"
        dataset = f"{family}_normal_{member_index}"
        identity_count = (4, 6, 20)[dataset_index % 3]
        result.append((dataset, family, dataset_index, identity_count))
    return tuple(result)


def _build_smoke_rows(
    backend: str,
) -> tuple[list[object], float]:
    started = perf_counter()
    rows = [
        qualify_dataset(
            dataset=dataset,
            family=family,
            runs=_synthetic_runs(dataset, dataset_index, identity_count),
            backend=backend,
            config=FROZEN_CONFIG,
        )
        for dataset, family, dataset_index, identity_count
        in _smoke_specifications()
    ]
    return rows, perf_counter() - started


def _synthetic_runs(
    dataset: str,
    dataset_index: int,
    identity_count: int,
) -> tuple[TrainRun, ...]:
    x = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    output = []
    for identity in range(identity_count):
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


def _build_backend_assessment(
    baseline_rows: list[object],
    baseline_aggregate: Mapping[str, object],
    baseline_seconds: float,
) -> dict[str, object]:
    probes = probe_backends()
    baseline_hash = _tag_hash(baseline_rows)
    rng = np.random.default_rng(411)
    fit = rng.normal(size=(64, 96))
    query = rng.normal(size=(32, 96))
    expected_distances = pairwise_rms(fit, query, backend="numpy_cpu")
    assessments: dict[str, object] = {}
    timing: dict[str, object] = {"numpy_cpu_stage_seconds": baseline_seconds}
    for name in ("numba_cpu", "torch_cpu"):
        probe = probes[name]
        if not getattr(probe, "available", False):
            assessments[name] = {
                "status": "unavailable",
                "reason": getattr(probe, "reason", None),
            }
            continue
        if name == "numba_cpu":
            cold_started = perf_counter()
            pairwise_rms(fit, query, backend=name)
            cold_seconds = perf_counter() - cold_started
            warm_started = perf_counter()
            observed_distances = pairwise_rms(fit, query, backend=name)
            warm_seconds = perf_counter() - warm_started
            timing["numba_cold_seconds"] = cold_seconds
            timing["numba_warm_seconds"] = warm_seconds
        else:
            observed_distances = pairwise_rms(fit, query, backend=name)
        candidate_rows, candidate_seconds = _build_smoke_rows(name)
        candidate_aggregate = aggregate_qualification(
            candidate_rows,
            bootstrap_iterations=20,
            config=FROZEN_CONFIG,
        )
        assessment = assess_backend(
            name=name,
            max_abs_error=float(
                np.max(np.abs(expected_distances - observed_distances))
            ),
            tag_identity=_tag_hash(candidate_rows) == baseline_hash,
            selected_quantile_identity=(
                candidate_aggregate["selected_quantile"]
                == baseline_aggregate["selected_quantile"]
            ),
            cpu_seconds=baseline_seconds,
            backend_seconds=candidate_seconds,
        )
        assessments[name] = {
            **asdict(assessment),
            "stage_scope": "32-dataset representative synthetic qualification",
        }
    return {
        "canonical_backend": "numpy_cpu",
        "accelerated_backend_adopted": False,
        "fallback_reason": (
            "full-population accelerated tag parity is not established; "
            "NumPy CPU remains fail-closed canonical"
        ),
        "baseline_tag_sha256": baseline_hash,
        "canonical_tag_sha256": baseline_hash,
        "selected_quantile_identity": True,
        "timing": timing,
        "assessments": assessments,
        "probes": {
            name: asdict(value) if hasattr(value, "__dataclass_fields__") else value
            for name, value in probes.items()
        },
        "mlx_status": "excluded_by_user_decision",
    }


def _tag_hash(rows: list[object]) -> str:
    payload = [
        asdict(row) if hasattr(row, "__dataclass_fields__") else row
        for row in rows
    ]
    return sha256(canonical_json_bytes(payload)).hexdigest()


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
