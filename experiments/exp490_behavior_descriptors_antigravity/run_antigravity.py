"""Smoke and official normal-TRAIN behavioral descriptors plus 24D vs 42D meta comparison."""
from __future__ import annotations

import os

for _key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ[_key] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from experiments.exp488_catch22_train_descriptors_antigravity.features_antigravity import FEATURE_KEYS
from experiments.exp488_catch22_train_descriptors_antigravity.run_antigravity import (
    RAW_SHA,
    identity as exp488_identity,
    read_cache as read_exp488_cache,
)
from experiments.exp490_behavior_descriptors_antigravity.behavior_descriptors_antigravity import (
    BEHAVIOR_KEYS_ANTIGRAVITY,
    extract_behavior_antigravity,
    prepare_normal_train_matrix_antigravity,
)
from experiments.exp491_gain_prediction_antigravity.gain_prediction_antigravity import (
    evaluate_meta_antigravity,
)
from experiments.exp491_gain_prediction_antigravity.runtime_antigravity import (
    DATABASE_PATH_ANTIGRAVITY,
    EXCLUDED_FIVEFOLD_FAMILIES_ANTIGRAVITY,
    EXPECTED_REPRESENTATIVE_FAMILIES_ANTIGRAVITY,
    EXP488_CACHE_PATH_ANTIGRAVITY,
    FROZEN_EVIDENCE_PATHS_ANTIGRAVITY,
    FROZEN_PREPROCESS_PATHS_ANTIGRAVITY,
    FROZEN_SCORER_PATHS_ANTIGRAVITY,
    ROOT_ANTIGRAVITY,
    SEED_ANTIGRAVITY,
    assess_acceleration_antigravity,
    assert_unbound_wal_antigravity,
    cpu_worker_policy_antigravity,
    file_identity_antigravity,
    file_sha256_antigravity,
    jsonable_antigravity,
    load_frozen_sources_antigravity,
    read_cache_antigravity,
    refuse_official_overwrite_antigravity,
    representative_rows_antigravity,
    require_matching_smoke_antigravity,
    sha256_json_antigravity,
    source_binding_antigravity,
    validate_database_antigravity,
    write_cache_antigravity,
    write_json_atomic_antigravity,
    write_text_atomic_antigravity,
)


PACKAGE_ANTIGRAVITY = Path(__file__).resolve().parent
CODE_PATHS_ANTIGRAVITY = (
    Path(__file__),
    PACKAGE_ANTIGRAVITY / "behavior_descriptors_antigravity.py",
    ROOT_ANTIGRAVITY / "experiments/exp491_gain_prediction_antigravity/runtime_antigravity.py",
    ROOT_ANTIGRAVITY / "experiments/exp491_gain_prediction_antigravity/gain_prediction_antigravity.py",
    *FROZEN_SCORER_PATHS_ANTIGRAVITY,
    *FROZEN_PREPROCESS_PATHS_ANTIGRAVITY,
    *FROZEN_EVIDENCE_PATHS_ANTIGRAVITY,
)


def _binding_antigravity(database, source_sha, descriptor_sha):
    return source_binding_antigravity(
        code_paths=CODE_PATHS_ANTIGRAVITY,
        database=database,
        source_sha=source_sha,
        extra={"experiment": "exp490_behavior_descriptors", "extraction_split": "normal_TRAIN_only",
               "exp488_descriptor_cache_sha256": descriptor_sha},
    )


def extract_dataset_worker_antigravity(payload: dict) -> tuple[str, dict, float]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    started = perf_counter()
    train, provenance = prepare_normal_train_matrix_antigravity(payload["database"], payload["dataset"])
    extracted = extract_behavior_antigravity(train, seed=SEED_ANTIGRAVITY)
    record = {
        "values": list(extracted["values"]),
        "keys": list(extracted["keys"]),
        "diagnostics": extracted["diagnostics"],
        **provenance,
    }
    return payload["dataset"], jsonable_antigravity(record), perf_counter() - started


def _load_24d_antigravity(path: Path, rows, database, source_sha) -> dict[str, list]:
    if not Path(database).exists():
        raise RuntimeError(f"raw database does not exist: {database}")
    actual = file_sha256_antigravity(Path(database))
    if actual != RAW_SHA:
        raise RuntimeError("raw database SHA changed relative to the frozen Exp488 identity")
    binding = exp488_identity(database, source_sha)
    if binding["sources"].get("raw_database") != actual:
        raise RuntimeError("Exp488 cache binding raw SHA does not match the current database")
    records = read_exp488_cache(path, binding, rows)
    if len(records) != len(rows):
        raise RuntimeError(f"Exp488 descriptor cache is incomplete: {len(records)}/{len(rows)}")
    values = {}
    for row in rows:
        record = records[row.dataset]
        payload = record["values"]
        if len(payload) != len(FEATURE_KEYS):
            raise ValueError(f"Exp488 width mismatch for {row.dataset}")
        values[row.dataset] = list(payload)
    return values


def validate_behavior_cache_antigravity(records, representatives, *, complete=False):
    expected = {row.dataset: row for row in representatives}
    if not isinstance(records, dict) or not set(records) <= set(expected):
        raise ValueError("behavior cache contains an unexpected dataset")
    if complete and set(records) != set(expected):
        raise ValueError("behavior cache population is incomplete")
    for dataset, record in records.items():
        values = record.get("values", [])
        membership = record.get("train_membership_sha256", "")
        if (record.get("keys") != list(BEHAVIOR_KEYS_ANTIGRAVITY) or len(values) != 18
                or any(type(v) not in (int, float) or not np.isfinite(v) for v in values)
                or record.get("finite_rows") != int(expected[dataset].features[1])
                or record.get("total_rows", -1) < record["finite_rows"]
                or record.get("diagnostics", {}).get("n_train") != record["finite_rows"]
                or len(membership) != 64 or any(c not in "0123456789abcdef" for c in membership)):
            raise ValueError(f"invalid behavior cache record: {dataset}")


def render_report_antigravity(result: dict, manifest: dict) -> str:
    comparison = result.get("comparison") or {}
    summary_24 = ((comparison.get("features_24") or {}).get("summary") or {})
    summary_42 = ((comparison.get("features_42") or {}).get("summary") or {})
    return f"""<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Exp490 결과</title>
<style>body{{font:16px/1.7 system-ui;max-width:1100px;margin:32px auto;padding:0 20px;color:#172033}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccd3dc;padding:8px;text-align:left}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f5f8;padding:16px}}</style>
<h1>Exp490: 정상 TRAIN 행동 기술자</h1>
<p>네 개의 고정 Exp449 표현을 원래 5폴드로 맞추고, A는 원래 참조, B는 A 참조의 80% 결정적 표본입니다. 표현 시드는 바꾸지 않았습니다. 추출은 정상 TRAIN만 사용합니다. 비교는 같은 {result.get('n_rows', 0)}개 행에서 기존 24차원과 24+18차원을 Exp491 프로토콜로 돌린 역사적 감독 메타 개발이며, 깨끗한 검증이 아닙니다.</p>
<p>제외 family: {escape(', '.join(result.get('excluded_families') or ()))}. 상수 상관을 0으로 둔 횟수는 데이터셋마다 diagnostics에 있습니다.</p>
<h2>24차원 대 42차원</h2>
<pre>{escape(json.dumps({'features_24': summary_24, 'features_42': summary_42}, ensure_ascii=False, indent=2))}</pre>
<h2>제한</h2>
<p>합성 이상 주장을 하지 않습니다. 라우팅 승격이나 운영 게이트 변경이 아닙니다. 역사적 TEST 파생 점수는 개발용 메타 라벨입니다.</p>
<h2>재현</h2>
<pre>PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m experiments.exp490_behavior_descriptors_antigravity.run_antigravity smoke
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m experiments.exp490_behavior_descriptors_antigravity.run_antigravity official</pre>
<pre>{escape(json.dumps(manifest, ensure_ascii=False, indent=2))}</pre>
</html>"""


def _synthetic_train_antigravity():
    rng = np.random.default_rng(SEED_ANTIGRAVITY)
    rows = rng.normal(size=(10, 24)).astype(np.float32)
    rows += np.linspace(0.0, 1.0, 10, dtype=np.float32)[:, None]
    return rows


def run(
    mode: str,
    database=DATABASE_PATH_ANTIGRAVITY,
    output=PACKAGE_ANTIGRAVITY,
    rows=None,
    descriptor_cache=EXP488_CACHE_PATH_ANTIGRAVITY,
):
    if mode not in ("smoke", "official"):
        raise ValueError("mode must be smoke or official")
    started, started_at = perf_counter(), datetime.now(timezone.utc).isoformat()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    official_path = output / "results_antigravity.json"
    if mode == "official":
        refuse_official_overwrite_antigravity(official_path)
    database = Path(database)
    workers = cpu_worker_policy_antigravity()
    if rows is None:
        evidence_rows, _control, source_sha = load_frozen_sources_antigravity()
    else:
        evidence_rows, source_sha = rows, {}
    validate_database_antigravity(database, frozen=rows is None)
    representatives, excluded = representative_rows_antigravity(
        evidence_rows,
        expected_families=EXPECTED_REPRESENTATIVE_FAMILIES_ANTIGRAVITY if rows is None else None,
        expected_excluded=EXCLUDED_FIVEFOLD_FAMILIES_ANTIGRAVITY if rows is None else None,
    )
    if not representatives:
        raise ValueError("no eligible representative datasets")
    if rows is None:
        assert_unbound_wal_antigravity(database)
        before = file_identity_antigravity(database)
    else:
        before = None
    descriptor_sha = file_sha256_antigravity(Path(descriptor_cache))
    values_24_all = _load_24d_antigravity(Path(descriptor_cache), evidence_rows, database, source_sha)
    def current_binding():
        current_sources = load_frozen_sources_antigravity()[2] if rows is None else source_sha
        return _binding_antigravity(database, current_sources, file_sha256_antigravity(Path(descriptor_cache)))
    binding = _binding_antigravity(database, source_sha, descriptor_sha)
    if current_binding() != binding:
        raise RuntimeError("source or descriptor cache changed while loading inputs")
    cache_path = output / "descriptor_cache_antigravity.json"
    if mode == "smoke":
        acceleration = assess_acceleration_antigravity()
        train = _synthetic_train_antigravity()
        first = extract_behavior_antigravity(train, seed=SEED_ANTIGRAVITY)
        second = extract_behavior_antigravity(train, seed=SEED_ANTIGRAVITY)
        if first["values"] != second["values"]:
            raise RuntimeError("synthetic behavior extraction is not deterministic")
        if first["diagnostics"]["representation_fits"] != 40:
            raise RuntimeError("each fold must fit A and B encoders with the same seed")
        repeated = None
        records = {}
        if representatives:
            dataset, record, first_seconds = extract_dataset_worker_antigravity(
                {"database": str(database), "dataset": representatives[0].dataset}
            )
            _, repeated_record, warm_seconds = extract_dataset_worker_antigravity(
                {"database": str(database), "dataset": representatives[0].dataset}
            )
            if record != repeated_record:
                raise RuntimeError("representative behavior extraction is not repeatable")
            records[dataset] = record
            repeated = {
                "dataset": dataset,
                "first_seconds": first_seconds,
                "warm_seconds": warm_seconds,
                "constant_correlation_count": record["diagnostics"]["constant_correlation_count"],
            }
        write_cache_antigravity(output / "smoke_descriptor_cache_antigravity.json", binding, records)
        if before is not None and before != file_identity_antigravity(database):
            raise RuntimeError("raw database changed during smoke")
        if current_binding() != binding:
            raise RuntimeError("source or implementation changed during smoke")
        result = {
            "state": "smoke_nonofficial",
            "binding": binding,
            "acceleration": acceleration,
            "synthetic_values": list(first["values"]),
            "synthetic_diagnostics": first["diagnostics"],
            "representative_repeat": repeated,
            "excluded_families": list(excluded),
            "extraction_split": "normal_TRAIN_only",
            "comparison_policy": "historical_supervised_meta_development_not_pristine_validation",
            "cpu_policy": workers,
            "elapsed_seconds": perf_counter() - started,
        }
        write_json_atomic_antigravity(output / "smoke_results_antigravity.json", jsonable_antigravity(result))
        write_text_atomic_antigravity(
            output / "smoke_report_antigravity.html",
            render_report_antigravity(
                {"excluded_families": list(excluded), "n_rows": 0, "comparison": {}},
                {"mode": "smoke", "binding": binding, "acceleration": acceleration},
            ),
        )
        print(json.dumps({"state": result["state"], "output": str(output)}), flush=True)
        return result

    require_matching_smoke_antigravity(output / "smoke_results_antigravity.json", binding)
    records = read_cache_antigravity(cache_path, binding) if cache_path.exists() else {}
    validate_behavior_cache_antigravity(records, representatives)
    resumed_datasets = len(records)
    pending = [row for row in representatives if row.dataset not in records]
    errors = []
    timings = {}
    with ProcessPoolExecutor(max_workers=workers["workers"]) as pool:
        futures = {
            pool.submit(
                extract_dataset_worker_antigravity,
                {"database": str(database), "dataset": row.dataset},
            ): row.dataset
            for row in pending
        }
        for future in as_completed(futures):
            dataset = futures[future]
            try:
                name, record, seconds = future.result()
                records[name] = record
                timings[name] = seconds
            except Exception as error:
                errors.append({"dataset": dataset, "error": f"{type(error).__name__}: {error}"})
            write_cache_antigravity(cache_path, binding, records)
    if errors or len(records) != len(representatives):
        write_json_atomic_antigravity(output / "failures_antigravity.json", {"errors": errors})
        raise RuntimeError(
            f"partial Exp490 extraction: completed {len(records)}/{len(representatives)}, failed {len(errors)}"
        )
    validate_behavior_cache_antigravity(records, representatives, complete=True)
    values_18 = {dataset: record["values"] for dataset, record in records.items()}
    values_24 = {row.dataset: values_24_all[row.dataset] for row in representatives}
    values_42 = {
        dataset: list(values_24[dataset]) + list(values_18[dataset]) for dataset in values_18
    }
    keys_42 = tuple(FEATURE_KEYS) + BEHAVIOR_KEYS_ANTIGRAVITY
    comparison = {
        "features_24": evaluate_meta_antigravity(representatives, values_24, FEATURE_KEYS),
        "features_42": evaluate_meta_antigravity(representatives, values_42, keys_42),
        "same_rows": True,
        "same_family_exclusions": True,
        "label_policy": "historical_supervised_meta_development; extraction is normal TRAIN only",
    }
    result = {
        "schema": "exp490_behavior_descriptors_v1",
        "state": "official_complete",
        "binding": binding,
        "extraction_split": "normal_TRAIN_only",
        "excluded_families": list(excluded),
        "n_rows": len(representatives),
        "descriptor_cache_sha256": file_sha256_antigravity(cache_path),
        "comparison": comparison,
    }
    manifest = {
        "schema": "exp490_run_manifest_v1",
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": perf_counter() - started,
        "cpu_policy": workers,
        "extraction_workers": workers["workers"],
        "meta_evaluation_workers": 1,
        "resumed_datasets": resumed_datasets,
        "completed_datasets": len(records),
        "expected_datasets": len(representatives),
        "failed_datasets": 0,
        "excluded_families": list(excluded),
        "dataset_seconds_this_invocation": timings,
        "result_sha256": sha256_json_antigravity(result),
        "code_sha256": binding["code_sha256"],
        "preregistration_sha256": binding["preregistration_sha256"],
        "acceleration": json.loads((output / "smoke_results_antigravity.json").read_text(encoding="utf-8"))[
            "acceleration"
        ],
    }
    if before is not None and before != file_identity_antigravity(database):
        raise RuntimeError("raw database changed during execution")
    if current_binding() != binding:
        raise RuntimeError("source or implementation changed during execution")
    write_json_atomic_antigravity(output / "run_manifest_antigravity.json", jsonable_antigravity(manifest))
    write_text_atomic_antigravity(output / "report_antigravity.html", render_report_antigravity(result, manifest))
    write_json_atomic_antigravity(official_path, jsonable_antigravity(result))
    print(json.dumps({"state": result["state"], "output": str(output)}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "official"))
    parser.add_argument("--database", type=Path, default=DATABASE_PATH_ANTIGRAVITY)
    parser.add_argument("--output", type=Path, default=PACKAGE_ANTIGRAVITY)
    parser.add_argument("--descriptor-cache", type=Path, default=EXP488_CACHE_PATH_ANTIGRAVITY)
    args = parser.parse_args()
    run(args.mode, args.database, args.output, descriptor_cache=args.descriptor_cache)
