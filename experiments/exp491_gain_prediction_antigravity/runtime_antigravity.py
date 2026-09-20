"""Shared hashes, atomic JSON, binding, cache, workers, and acceleration assessment."""
from __future__ import annotations

import os

THREAD_ENV_ANTIGRAVITY = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
)


def configure_inner_threads_antigravity() -> None:
    for key in THREAD_ENV_ANTIGRAVITY:
        os.environ[key] = "1"


configure_inner_threads_antigravity()

import json
import math
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform
import sqlite3
import subprocess
import sys
from time import perf_counter
from typing import Mapping, Sequence

import numpy as np

from experiments.exp479_missing_score_recovery_antigravity.config_antigravity import (
    DATABASE_EXPECTED_SHA256_ANTIGRAVITY,
    DATABASE_PATH_ANTIGRAVITY,
    METHODS_ANTIGRAVITY,
    SEED_ANTIGRAVITY,
)
from experiments.exp480_family_balanced_evidence_routing_antigravity.evidence_antigravity import (
    _cached_source_sha256_antigravity as file_sha256_antigravity,
    _source_identity_antigravity as file_identity_antigravity,
)
from experiments.exp481_strict_evidence_residual_correction_antigravity.run_antigravity import (
    json_bytes_antigravity,
    write_json_atomic_antigravity,
    _write_text_atomic_antigravity as write_text_atomic_antigravity,
)
from experiments.exp484_distance_qualification_antigravity.run_antigravity import (
    load_frozen_sources_antigravity,
)


ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[2]
PREREG_PATH_ANTIGRAVITY = (
    ROOT_ANTIGRAVITY
    / "docs/superpowers/specs/2026-09-16-exp489-491-preregistration-antigravity.html"
)
EXP488_CACHE_PATH_ANTIGRAVITY = (
    ROOT_ANTIGRAVITY
    / "experiments/exp488_catch22_train_descriptors_antigravity/descriptor_cache_antigravity.json"
)
EXPECTED_REPRESENTATIVE_FAMILIES_ANTIGRAVITY = 125
EXPECTED_EXP491_DATASETS_ANTIGRAVITY = 1117
EXPECTED_EXP491_FAMILIES_ANTIGRAVITY = 128
MIN_N_TRAIN_ANTIGRAVITY = 5.0
EXCLUDED_FIVEFOLD_FAMILIES_ANTIGRAVITY = (
    "PigAirwayPressure",
    "PigArtPressure",
    "PigCVP",
)
FROZEN_SCORER_PATHS_ANTIGRAVITY = (
    ROOT_ANTIGRAVITY
    / "experiments/exp449_residual_fn_method_qualification_antigravity/scoring_antigravity.py",
    ROOT_ANTIGRAVITY
    / "experiments/exp449_residual_fn_method_qualification_antigravity/representations_antigravity.py",
    ROOT_ANTIGRAVITY
    / "experiments/exp449_residual_fn_method_qualification_antigravity/local_gap_antigravity.py",
)
FROZEN_PREPROCESS_PATHS_ANTIGRAVITY = (
    ROOT_ANTIGRAVITY
    / "experiments/exp479_missing_score_recovery_antigravity/preprocessing_antigravity.py",
    ROOT_ANTIGRAVITY
    / "experiments/exp479_missing_score_recovery_antigravity/evaluation_antigravity.py",
)
FROZEN_EVIDENCE_PATHS_ANTIGRAVITY = (
    ROOT_ANTIGRAVITY / "experiments/exp484_distance_qualification_antigravity/run_antigravity.py",
    ROOT_ANTIGRAVITY
    / "experiments/exp488_catch22_train_descriptors_antigravity/features_antigravity.py",
    ROOT_ANTIGRAVITY
    / "experiments/exp488_catch22_train_descriptors_antigravity/run_antigravity.py",
)
PROTOCOL_ANTIGRAVITY = {
    "seed": SEED_ANTIGRAVITY,
    "folds": 5,
    "max_train_reference": 1000,
    "query_chunk_size": 128,
    "requested_kernels": {
        "minirocket_local_gap": 4096,
        "multirocket_local_gap": 1024,
        "hydra_local_gap": 8,
        "hydra_multirocket_local_gap": 1024,
    },
    "methods": list(METHODS_ANTIGRAVITY),
    "robust_scaler": [10, 90],
    "local_gap_neighbors": 3,
    "lof": {
        "novelty": True,
        "n_neighbors": "min(3, nref-1)",
        "metric": "minkowski",
        "p": 2,
        "contamination": "auto",
        "n_jobs": 1,
        "anomaly_score": "-score_samples",
    },
    "behavior_b_fraction": 0.8,
    "behavior_b_salt": 104729,
    "ridge_alpha": 10.0,
    "ridge_solver": "cholesky",
}


def cpu_worker_policy_antigravity() -> dict[str, int]:
    detected = os.cpu_count() or 1
    workers = max(1, int(detected) - 1)
    return {
        "logical_cpu_count": int(detected),
        "workers": int(workers),
        "inner_threads": 1,
    }


def package_version_antigravity(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def sha256_bytes_antigravity(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def sha256_json_antigravity(payload: object) -> str:
    return sha256_bytes_antigravity(json_bytes_antigravity(payload))


def protocol_sha256_antigravity() -> str:
    return sha256_json_antigravity(PROTOCOL_ANTIGRAVITY)


def code_sha256_antigravity(paths: Sequence[Path]) -> dict[str, str]:
    mapping = {}
    for path in paths:
        target = Path(path)
        try:
            key = str(target.resolve().relative_to(ROOT_ANTIGRAVITY))
        except ValueError:
            key = str(target)
        mapping[key] = file_sha256_antigravity(target)
    return mapping


def refuse_official_overwrite_antigravity(path: Path) -> None:
    if Path(path).exists():
        raise RuntimeError("official result already exists; verify it instead of rerunning")


def require_matching_smoke_antigravity(path: Path, binding: Mapping[str, object]) -> Mapping[str, object]:
    target = Path(path)
    if not target.exists():
        raise RuntimeError("matching smoke is required before official evaluation")
    smoke = json.loads(target.read_text(encoding="utf-8"))
    if smoke.get("state") != "smoke_nonofficial" or smoke.get("binding") != binding:
        raise RuntimeError("matching smoke is required before official evaluation")
    return smoke


def assert_unbound_wal_antigravity(database: Path) -> None:
    wal = Path(str(database) + "-wal")
    if wal.exists() and wal.stat().st_size:
        raise RuntimeError("raw database has an unbound WAL")


def readonly_sqlite_antigravity(database: Path) -> sqlite3.Connection:
    uri = Path(database).resolve().as_uri() + "?mode=ro"
    return sqlite3.connect(uri, uri=True)


def validate_database_antigravity(database: Path, *, frozen: bool) -> None:
    if not Path(database).is_file():
        raise RuntimeError(f"raw database does not exist: {database}")
    assert_unbound_wal_antigravity(database)
    if frozen and file_sha256_antigravity(database) != DATABASE_EXPECTED_SHA256_ANTIGRAVITY:
        raise RuntimeError("frozen raw database SHA changed")


def representative_rows_antigravity(
    rows: Sequence[object],
    *,
    expected_families: int | None = None,
    expected_excluded: tuple[str, ...] | None = None,
    min_n_train: float = MIN_N_TRAIN_ANTIGRAVITY,
) -> tuple[tuple[object, ...], tuple[str, ...]]:
    """Select lex-first dataset per family among rows with n_train >= 5; never fill excluded families."""
    if type(min_n_train) is not float and type(min_n_train) is not int:
        raise ValueError("min_n_train must be numeric")
    all_families = {str(row.source_family) for row in rows}
    eligible = [row for row in rows if float(row.features[1]) >= float(min_n_train)]
    chosen: dict[str, object] = {}
    for row in eligible:
        family = str(row.source_family)
        current = chosen.get(family)
        if current is None or str(row.dataset) < str(current.dataset):
            chosen[family] = row
    excluded = tuple(sorted(all_families - set(chosen)))
    ordered = tuple(chosen[name] for name in sorted(chosen))
    if expected_families is not None and len(ordered) != int(expected_families):
        raise RuntimeError(
            f"expected {expected_families} representative families, got {len(ordered)}; excluded={excluded}"
        )
    if expected_excluded is not None and excluded != tuple(expected_excluded):
        raise RuntimeError(f"unexpected excluded families: {excluded}")
    return ordered, excluded


def dataset_stem_antigravity(dataset: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in dataset)
    return f"{safe}_antigravity"


def jsonable_antigravity(value: object) -> object:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, int) and type(value) is not bool:
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("non-finite JSON number")
        return number
    if isinstance(value, np.ndarray):
        return jsonable_antigravity(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): jsonable_antigravity(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable_antigravity(item) for item in value]
    raise TypeError(f"unsupported JSON type: {type(value)!r}")


def write_cache_antigravity(path: Path, binding: Mapping[str, object], records: Mapping[str, object]) -> None:
    payload = {
        "binding": dict(binding),
        "records": dict(records),
        "records_sha256": sha256_json_antigravity(records),
    }
    write_json_atomic_antigravity(path, jsonable_antigravity(payload))


def read_cache_antigravity(path: Path, binding: Mapping[str, object]) -> dict[str, object]:
    target = Path(path)
    if not target.exists():
        return {}
    payload = json.loads(target.read_text(encoding="utf-8"))
    if payload.get("binding") != binding:
        raise ValueError("descriptor cache binding or checksum mismatch")
    records = payload.get("records")
    if not isinstance(records, dict):
        raise ValueError("descriptor cache records are invalid")
    if payload.get("records_sha256") != sha256_json_antigravity(records):
        raise ValueError("descriptor cache binding or checksum mismatch")
    return records


def _probe_subprocess_antigravity(code: str, timeout: float = 30.0) -> dict[str, object]:
    started = perf_counter()
    try:
        process = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if process.returncode != 0:
            result: dict[str, object] = {
                "available": False,
                "exit_code": process.returncode,
                "stderr_tail": (process.stderr or "")[-500:],
            }
        else:
            result = json.loads(process.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError) as error:
        result = {"available": False, "probe_failed": True, "error": type(error).__name__}
    result["probe_seconds"] = perf_counter() - started
    return result


def assess_acceleration_antigravity() -> dict[str, object]:
    """Record eligibility; do not promote unproven GPU or Numba paths."""
    configure_inner_threads_antigravity()
    versions = {
        "numpy": package_version_antigravity("numpy"),
        "scipy": package_version_antigravity("scipy"),
        "scikit-learn": package_version_antigravity("scikit-learn"),
        "aeon": package_version_antigravity("aeon"),
        "numba": package_version_antigravity("numba"),
        "torch": package_version_antigravity("torch"),
        "mlx": package_version_antigravity("mlx"),
        "mlx-metal": package_version_antigravity("mlx-metal"),
        "python": platform.python_version(),
        "architecture": platform.machine(),
    }
    pytorch_probe = _probe_subprocess_antigravity(
        "import json,torch\n"
        "r=dict(available=True,version=torch.__version__,"
        "mps_available=bool(torch.backends.mps.is_available()),cpu_float64=True)\n"
        "try:\n"
        "    a=torch.tensor([1.,2.],device='mps',dtype=torch.float64)\n"
        "    r['mps_float64_finite']=bool(torch.isfinite(a.sum()).item())\n"
        "    torch.mps.synchronize()\n"
        "except (TypeError,RuntimeError) as e:\n"
        "    r.update(mps_float64_finite=False,fallback_reason=str(e))\n"
        "print(json.dumps(r))"
    )
    mlx_probe = _probe_subprocess_antigravity(
        "import json,time\n"
        "r=dict(available=False)\n"
        "try:\n"
        "    import mlx.core as mx\n"
        "    r.update(available=True,metal_available=bool(mx.metal.is_available()),"
        "default_device=str(mx.default_device()))\n"
        "    started=time.perf_counter()\n"
        "    mx.set_default_device(mx.cpu)\n"
        "    a=mx.array([[1.,0.],[0.,1.]],dtype=mx.float64)\n"
        "    b=mx.array([1.,2.],dtype=mx.float64)\n"
        "    x=mx.linalg.solve(a,b)\n"
        "    mx.eval(x)\n"
        "    r['cpu_solve_seconds']=time.perf_counter()-started\n"
        "    r['cpu_float64_finite']=bool(mx.isfinite(mx.sum(x)).item())\n"
        "except Exception as e:\n"
        "    r.update(available=False,fallback_reason=f'{type(e).__name__}: {e}')\n"
        "print(json.dumps(r))"
    )
    rng = np.random.default_rng(20260721)
    vector = rng.normal(size=4096)
    other = rng.normal(size=4096)
    from scipy.stats import spearmanr

    started = perf_counter()
    np.quantile(vector, [0.25, 0.5, 0.75, 0.95, 0.99], method="linear")
    spearmanr(vector, other)
    quantile_warmup_seconds = perf_counter() - started
    started = perf_counter()
    np.quantile(vector, [0.25, 0.5, 0.75, 0.95, 0.99], method="linear")
    spearmanr(vector, other)
    quantile_measured_seconds = perf_counter() - started

    from sklearn.linear_model import Ridge
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.preprocessing import RobustScaler

    features = rng.normal(size=(128, 32))
    started = perf_counter()
    scaler = RobustScaler(quantile_range=(10, 90)).fit(features[:80])
    scaled = scaler.transform(features)
    LocalOutlierFactor(
        n_neighbors=3, novelty=True, metric="minkowski", p=2, contamination="auto", n_jobs=1
    ).fit(scaled[:80])
    lof_warmup_seconds = perf_counter() - started
    started = perf_counter()
    scaler = RobustScaler(quantile_range=(10, 90)).fit(features[:80])
    scaled = scaler.transform(features)
    model = LocalOutlierFactor(
        n_neighbors=3, novelty=True, metric="minkowski", p=2, contamination="auto", n_jobs=1
    )
    model.fit(scaled[:80])
    scores = -model.score_samples(scaled[80:])
    lof_measured_seconds = perf_counter() - started
    if not np.isfinite(scores).all():
        raise RuntimeError("LOF microbenchmark produced non-finite scores")

    design = rng.normal(size=(256, 24))
    targets = rng.normal(size=(256, 6))
    weights = np.repeat(1.0 / 32.0, 256)
    started = perf_counter()
    Ridge(alpha=10.0, fit_intercept=True, solver="cholesky").fit(design, targets, sample_weight=weights)
    ridge_warmup_seconds = perf_counter() - started
    started = perf_counter()
    fitted = Ridge(alpha=10.0, fit_intercept=True, solver="cholesky").fit(
        design, targets, sample_weight=weights
    )
    predicted = fitted.predict(design[:16])
    ridge_measured_seconds = perf_counter() - started
    if not np.isfinite(predicted).all():
        raise RuntimeError("Ridge microbenchmark produced non-finite predictions")

    workers = cpu_worker_policy_antigravity()
    return {
        "schema": "exp489_491_acceleration_v1",
        "versions": versions,
        "device": platform.machine(),
        "cpu_policy": workers,
        "pytorch": {
            "tensor_eligible_stages": ["exp491_dense_ridge"],
            "selected": False,
            "probe": pytorch_probe,
            "fallback_reason": (
                "official_aeon_sklearn_cpu_remain_canonical_except_where_a_stage_is_tensor_eligible; "
                "dense_ridge_is_tensor_eligible_under_torch_but_not_promoted_without_float64_full_stage_parity_and_1.1x"
            ),
        },
        "mlx": {
            "tensor_eligible_stages": ["exp491_dense_ridge"],
            "selected": False,
            "probe": mlx_probe,
            "full_stage_seconds": None,
            "candidate_index_parity": None,
            "numerical_parity": None,
            "speedup": None,
            "fallback_reason": (
                "dense_ridge_not_promoted_without_float64_full_stage_parity_and_1.1x"
            ),
        },
        "mps": {
            "selected": False,
            "available": bool(pytorch_probe.get("mps_available")),
            "fallback_reason": "mps_not_canonical_for_frozen_aeon_sklearn_cpu_algorithms",
        },
        "stages": {
            "exp489_minirocket_lof": {
                "tensor_eligible": False,
                "pytorch_tensor_eligible": False,
                "canonical": "aeon_minirocket_sklearn_lof_localgap_cpu",
                "selected_backend": "sklearn_aeon_cpu",
                "warmup_seconds": lof_warmup_seconds,
                "measured_stage_seconds": lof_measured_seconds,
                "exact_full_population_stage_benchmarked": False,
                "numba": {
                    "selected": False,
                    "fallback_reason": (
                        "official_aeon_MiniRocket_and_sklearn_LocalOutlierFactor_have_no_supported_public_numba_rewrite; "
                        "retain_existing_cpu_implementations"
                    ),
                },
                "fallback_reason": (
                    "official_aeon_MiniRocket_and_sklearn_LocalOutlierFactor_remain_canonical; "
                    "do_not_rewrite_third_party_cpu_transformer"
                ),
            },
            "exp490_rank_quantile": {
                "tensor_eligible": True,
                "pytorch_tensor_eligible": True,
                "canonical": "numpy_scipy_blas",
                "selected_backend": "numpy_scipy",
                "warmup_seconds": quantile_warmup_seconds,
                "measured_stage_seconds": quantile_measured_seconds,
                "exact_full_population_stage_benchmarked": False,
                "numba": {
                    "selected": False,
                    "fallback_reason": (
                        "public_numba_compiler_path_for_scipy_spearmanr_and_numpy_quantile_orchestration "
                        "is_unsupported_or_has_no_measured_speedup; retain_existing_numpy_scipy_blas"
                    ),
                },
                "fallback_reason": (
                    "public_numba_compiler_path_for_scipy_spearmanr_and_numpy_quantile_orchestration "
                    "is_unsupported_or_has_no_measured_speedup; retain_existing_numpy_scipy_blas"
                ),
            },
            "exp491_ridge": {
                "tensor_eligible": True,
                "pytorch_tensor_eligible": True,
                "mlx_tensor_eligible": True,
                "canonical": "sklearn_ridge_cholesky_cpu",
                "selected_backend": "sklearn_cpu",
                "warmup_seconds": ridge_warmup_seconds,
                "measured_stage_seconds": ridge_measured_seconds,
                "exact_full_population_stage_benchmarked": False,
                "numba": {
                    "selected": False,
                    "fallback_reason": (
                        "sklearn_cholesky_ridge_retained; no_measured_numba_full_stage_float64_parity_and_1.1x"
                    ),
                },
                "mlx_selected": False,
                "pytorch_selected": False,
                "fallback_reason": (
                    "dense_ridge_acceleration_not_canonical_without_float64_full_stage_parity_and_1.1x; "
                    "not_promoted"
                ),
            },
        },
        "concurrent_gpu_jobs": 0,
    }


def preregistration_sha256_antigravity() -> str:
    if not PREREG_PATH_ANTIGRAVITY.exists():
        raise RuntimeError(
            "preregistration document is required: "
            "docs/superpowers/specs/2026-09-16-exp489-491-preregistration-antigravity.html"
        )
    return file_sha256_antigravity(PREREG_PATH_ANTIGRAVITY)


def source_binding_antigravity(
    *,
    extra: Mapping[str, object] | None = None,
    code_paths: Sequence[Path] = (),
    database: Path | None = None,
    source_sha: Mapping[str, str] | None = None,
) -> dict[str, object]:
    sources = dict(source_sha or {})
    if database is not None:
        path = Path(database)
        if not path.exists():
            raise RuntimeError(f"raw database does not exist: {path}")
        sources["raw_database"] = file_sha256_antigravity(path)
    binding = {
        "schema": "exp489_491_binding_v1",
        "seed": SEED_ANTIGRAVITY,
        "protocol": dict(PROTOCOL_ANTIGRAVITY),
        "protocol_sha256": protocol_sha256_antigravity(),
        "code_sha256": code_sha256_antigravity(tuple(code_paths)),
        "sources": sources,
        "preregistration_sha256": preregistration_sha256_antigravity(),
        "versions": {
            "aeon": package_version_antigravity("aeon"),
            "numpy": package_version_antigravity("numpy"),
            "scipy": package_version_antigravity("scipy"),
            "scikit-learn": package_version_antigravity("scikit-learn"),
            "numba": package_version_antigravity("numba"),
            "python": platform.python_version(),
        },
    }
    if extra:
        binding.update(dict(extra))
    return jsonable_antigravity(binding)


__all__ = [
    "DATABASE_EXPECTED_SHA256_ANTIGRAVITY",
    "DATABASE_PATH_ANTIGRAVITY",
    "EXCLUDED_FIVEFOLD_FAMILIES_ANTIGRAVITY",
    "EXPECTED_EXP491_DATASETS_ANTIGRAVITY",
    "EXPECTED_EXP491_FAMILIES_ANTIGRAVITY",
    "EXPECTED_REPRESENTATIVE_FAMILIES_ANTIGRAVITY",
    "EXP488_CACHE_PATH_ANTIGRAVITY",
    "FROZEN_EVIDENCE_PATHS_ANTIGRAVITY",
    "FROZEN_PREPROCESS_PATHS_ANTIGRAVITY",
    "FROZEN_SCORER_PATHS_ANTIGRAVITY",
    "METHODS_ANTIGRAVITY",
    "MIN_N_TRAIN_ANTIGRAVITY",
    "PREREG_PATH_ANTIGRAVITY",
    "PROTOCOL_ANTIGRAVITY",
    "ROOT_ANTIGRAVITY",
    "SEED_ANTIGRAVITY",
    "assess_acceleration_antigravity",
    "assert_unbound_wal_antigravity",
    "code_sha256_antigravity",
    "configure_inner_threads_antigravity",
    "cpu_worker_policy_antigravity",
    "dataset_stem_antigravity",
    "file_identity_antigravity",
    "file_sha256_antigravity",
    "json_bytes_antigravity",
    "jsonable_antigravity",
    "load_frozen_sources_antigravity",
    "package_version_antigravity",
    "preregistration_sha256_antigravity",
    "protocol_sha256_antigravity",
    "read_cache_antigravity",
    "readonly_sqlite_antigravity",
    "refuse_official_overwrite_antigravity",
    "representative_rows_antigravity",
    "require_matching_smoke_antigravity",
    "sha256_json_antigravity",
    "source_binding_antigravity",
    "write_cache_antigravity",
    "write_json_atomic_antigravity",
    "write_text_atomic_antigravity",
]
