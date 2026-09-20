"""Smoke and official family-held-out Ridge probe of cached descriptors."""
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
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
from time import perf_counter

from experiments.exp480_family_balanced_evidence_routing_antigravity.contracts_antigravity import (
    CandidateMetricAntigravity,
    EvidenceRowAntigravity,
)
from experiments.exp488_catch22_train_descriptors_antigravity.features_antigravity import FEATURE_KEYS
from experiments.exp488_catch22_train_descriptors_antigravity.run_antigravity import (
    RAW_SHA,
    identity as exp488_identity,
    read_cache as read_exp488_cache,
)
from experiments.exp491_gain_prediction_antigravity.gain_prediction_antigravity import (
    TARGET_KEYS_ANTIGRAVITY,
    evaluate_meta_antigravity,
)
from experiments.exp491_gain_prediction_antigravity.runtime_antigravity import (
    DATABASE_PATH_ANTIGRAVITY,
    EXPECTED_EXP491_DATASETS_ANTIGRAVITY,
    EXPECTED_EXP491_FAMILIES_ANTIGRAVITY,
    EXP488_CACHE_PATH_ANTIGRAVITY,
    FROZEN_EVIDENCE_PATHS_ANTIGRAVITY,
    METHODS_ANTIGRAVITY,
    ROOT_ANTIGRAVITY,
    SEED_ANTIGRAVITY,
    assess_acceleration_antigravity,
    cpu_worker_policy_antigravity,
    file_sha256_antigravity,
    jsonable_antigravity,
    load_frozen_sources_antigravity,
    refuse_official_overwrite_antigravity,
    require_matching_smoke_antigravity,
    sha256_json_antigravity,
    source_binding_antigravity,
    validate_database_antigravity,
    write_json_atomic_antigravity,
    write_text_atomic_antigravity,
)


PACKAGE_ANTIGRAVITY = Path(__file__).resolve().parent
CODE_PATHS_ANTIGRAVITY = (
    Path(__file__),
    PACKAGE_ANTIGRAVITY / "gain_prediction_antigravity.py",
    PACKAGE_ANTIGRAVITY / "runtime_antigravity.py",
    *FROZEN_EVIDENCE_PATHS_ANTIGRAVITY,
)


def _binding_antigravity(database, source_sha, descriptor_sha):
    return source_binding_antigravity(
        code_paths=CODE_PATHS_ANTIGRAVITY,
        database=database,
        source_sha=source_sha,
        extra={"experiment": "exp491_gain_prediction", "descriptor_cache_sha256": descriptor_sha},
    )


def load_exp488_descriptors_antigravity(path: Path, rows, database, source_sha) -> dict[str, list]:
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
        payload = records[row.dataset]["values"]
        if len(payload) != len(FEATURE_KEYS):
            raise ValueError(f"Exp488 width mismatch for {row.dataset}")
        values[row.dataset] = list(payload)
    return values


def synthetic_evidence_rows_antigravity(n_families: int = 6, per_family: int = 2):
    rows = []
    for family_index in range(n_families):
        for member in range(per_family):
            scores = []
            for method_index, method in enumerate(METHODS_ANTIGRAVITY):
                auc = 0.55 + 0.04 * method_index + 0.01 * family_index - 0.005 * member
                zf = max(0, 2 + method_index - (family_index % 3) + member)
                scores.append(CandidateMetricAntigravity(method, float(auc), int(zf)))
            rows.append(
                EvidenceRowAntigravity(
                    dataset=f"F{family_index:03}_normal_{member}",
                    source_family=f"F{family_index:03}",
                    proposal=METHODS_ANTIGRAVITY[0],
                    features=(64.0, 20.0 + member, 0.4, 0.1, 2.0, 0.8, 0.3),
                    scores=tuple(scores),
                )
            )
    return tuple(rows)


def synthetic_descriptors_antigravity(rows, width: int = 24):
    values = {}
    for index, row in enumerate(rows):
        vector = [float(row.features[0]), float(row.features[1])]
        vector.extend(float((index + column) % 7) for column in range(width - 2))
        if index == 0:
            vector[5] = None
        values[row.dataset] = vector
    return values


def render_report_antigravity(result: dict, manifest: dict) -> str:
    summary = result.get("evaluation", {}).get("summary", {})
    return f"""<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Exp491 결과</title>
<style>body{{font:16px/1.7 system-ui;max-width:1100px;margin:32px auto;padding:0 20px;color:#172033}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f5f8;padding:16px}}</style>
<h1>Exp491: 캐시된 이득 예측</h1>
<p>고정 Ridge α=10, 출력 6개(대안−MiniRocket AUC 차이 3개와 ZF 차이 부호 3개). 소스 family 전체를 한 폴드씩 빼며, 결측 중앙값·표준화·Ridge는 donor에만 맞춥니다. 행 가중치는 1/가족행수라서 가족 합은 1입니다. 상수 비교는 같은 가중 평균입니다. 중앙값과 표준편차는 가중하지 않은 donor 행에 맞춥니다.</p>
<p>행 {result.get('n_rows', 0)}개, family {result.get('n_families', 0)}개. 질의 목표는 예측 뒤에만 평가에 사용했습니다. 양쪽 클래스가 있을 때만 양-이득 AUROC를 보고합니다.</p>
<pre>{escape(json.dumps(summary, ensure_ascii=False, indent=2))}</pre>
<h2>제한</h2>
<p>실패한 선형 탐침이 모든 모형을 기각하지는 않습니다. 하이퍼파라미터 탐색, 지역 게이트 변경, 결과로 고른 이웃, 라우팅 승격은 없습니다. 역사적 TEST 파생 점수는 개발용 메타 라벨이며 깨끗한 검증이 아닙니다.</p>
<h2>재현</h2>
<pre>PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m experiments.exp491_gain_prediction_antigravity.run_antigravity smoke
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m experiments.exp491_gain_prediction_antigravity.run_antigravity official</pre>
<pre>{escape(json.dumps(manifest, ensure_ascii=False, indent=2))}</pre>
</html>"""


def run(
    mode: str,
    database=DATABASE_PATH_ANTIGRAVITY,
    output=PACKAGE_ANTIGRAVITY,
    descriptor_cache=EXP488_CACHE_PATH_ANTIGRAVITY,
    rows=None,
    descriptor_values=None,
):
    if mode not in ("smoke", "official"):
        raise ValueError("mode must be smoke or official")
    started, started_at = perf_counter(), datetime.now(timezone.utc).isoformat()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    official_path = output / "results_antigravity.json"
    if mode == "official":
        refuse_official_overwrite_antigravity(official_path)
    workers = cpu_worker_policy_antigravity()
    evidence_rows, _control, source_sha = load_frozen_sources_antigravity() if rows is None else (rows, None, {})
    if rows is None and descriptor_values is not None:
        raise ValueError("injected descriptors require injected fixture rows")
    if rows is None and (
        len(evidence_rows) != EXPECTED_EXP491_DATASETS_ANTIGRAVITY
        or len({row.source_family for row in evidence_rows}) != EXPECTED_EXP491_FAMILIES_ANTIGRAVITY
    ):
        raise RuntimeError("Exp491 evidence population changed")
    database = Path(database)
    validate_database_antigravity(database, frozen=rows is None)
    descriptor_sha = (file_sha256_antigravity(Path(descriptor_cache)) if descriptor_values is None
                      else sha256_json_antigravity(descriptor_values))
    values = descriptor_values if descriptor_values is not None else load_exp488_descriptors_antigravity(
        Path(descriptor_cache), evidence_rows, database, source_sha
    )
    def current_binding():
        current_sources = load_frozen_sources_antigravity()[2] if rows is None else source_sha
        descriptor_sha = (file_sha256_antigravity(Path(descriptor_cache)) if descriptor_values is None
                          else sha256_json_antigravity(descriptor_values))
        return _binding_antigravity(database, current_sources, descriptor_sha)
    binding = _binding_antigravity(database, source_sha, descriptor_sha)
    if current_binding() != binding:
        raise RuntimeError("source or descriptor cache changed while loading inputs")
    if mode == "smoke":
        acceleration = assess_acceleration_antigravity()
        smoke_rows = rows if rows is not None else synthetic_evidence_rows_antigravity()
        smoke_values = descriptor_values if descriptor_values is not None else synthetic_descriptors_antigravity(smoke_rows)
        first = evaluate_meta_antigravity(smoke_rows, smoke_values, FEATURE_KEYS[: len(next(iter(smoke_values.values())))])
        second = evaluate_meta_antigravity(smoke_rows, smoke_values, FEATURE_KEYS[: len(next(iter(smoke_values.values())))])
        if first != second:
            raise RuntimeError("synthetic meta evaluation is not deterministic")
        subset = smoke_rows[:4]
        subset_values = {row.dataset: smoke_values[row.dataset] for row in subset}
        subset_eval = evaluate_meta_antigravity(
            subset, subset_values, FEATURE_KEYS[: len(next(iter(smoke_values.values())))]
        )
        if current_binding() != binding:
            raise RuntimeError("source or implementation changed during smoke")
        result = {
            "state": "smoke_nonofficial",
            "binding": binding,
            "acceleration": acceleration,
            "evaluation": first,
            "n_rows": first["n_rows"],
            "n_families": first["n_families"],
            "fixed_subset": subset_eval,
            "target_keys": list(TARGET_KEYS_ANTIGRAVITY),
            "cpu_policy": workers,
            "elapsed_seconds": perf_counter() - started,
        }
        write_json_atomic_antigravity(output / "smoke_results_antigravity.json", jsonable_antigravity(result))
        write_text_atomic_antigravity(
            output / "smoke_report_antigravity.html",
            render_report_antigravity(result, {"mode": "smoke", "acceleration": acceleration}),
        )
        print(json.dumps({"state": result["state"], "output": str(output)}), flush=True)
        return result

    require_matching_smoke_antigravity(output / "smoke_results_antigravity.json", binding)
    evaluation = evaluate_meta_antigravity(evidence_rows, values, FEATURE_KEYS)
    if evaluation["n_rows"] != len(evidence_rows) or evaluation["n_families"] != len(
        {row.source_family for row in evidence_rows}
    ):
        raise RuntimeError("meta evaluation population changed")
    result = {
        "schema": "exp491_gain_prediction_v1",
        "state": "official_complete",
        "binding": binding,
        "n_rows": evaluation["n_rows"],
        "n_families": evaluation["n_families"],
        "evaluation": evaluation,
        "label_policy": evaluation["label_policy"],
    }
    manifest = {
        "schema": "exp491_run_manifest_v1",
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": perf_counter() - started,
        "cpu_policy": workers,
        "evaluation_workers": 1,
        "execution_policy": "serial_small_cached_ridge_stage; cpu_policy records available parallel capacity",
        "completed_folds": evaluation["n_families"],
        "expected_folds": evaluation["n_families"],
        "completed_rows": evaluation["n_rows"],
        "expected_rows": EXPECTED_EXP491_DATASETS_ANTIGRAVITY if rows is None else len(evidence_rows),
        "failed_folds": 0,
        "result_sha256": sha256_json_antigravity(result),
        "code_sha256": binding["code_sha256"],
        "preregistration_sha256": binding["preregistration_sha256"],
        "acceleration": json.loads((output / "smoke_results_antigravity.json").read_text(encoding="utf-8"))[
            "acceleration"
        ],
    }
    if current_binding() != binding:
        raise RuntimeError("source or implementation changed during evaluation")
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
