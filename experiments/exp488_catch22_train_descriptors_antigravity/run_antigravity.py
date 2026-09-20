"""Run the single preregistered Exp488 arm; never overwrite official results."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
from html import escape
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

# Must precede NumPy/Numba imports, including imports in spawned workers.
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np
from aeon.transformations.collection.feature_based import _catch22
from aeon.utils.numba import general, stats
from experiments.exp480_family_balanced_evidence_routing_antigravity.evidence_antigravity import (
    _cached_source_sha256_antigravity as file_sha,
    _source_identity_antigravity as file_identity,
)
from experiments.exp481_strict_evidence_residual_correction_antigravity.acceleration_antigravity import _fixed_evidence_rows_antigravity
from experiments.exp481_strict_evidence_residual_correction_antigravity.evaluation_antigravity import evaluate_nested_out_of_family_antigravity
from experiments.exp481_strict_evidence_residual_correction_antigravity.run_antigravity import json_bytes_antigravity as encode, write_json_atomic_antigravity as write_json, _write_text_atomic_antigravity as write_text
from experiments.exp484_distance_qualification_antigravity.run_antigravity import load_frozen_sources_antigravity
from . import features_antigravity
from .features_antigravity import FEATURE_KEYS, CATCH22_CONFIG, load_training_rows, describe_rows

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
DATABASE = Path("/Users/mac/Documents/Dataset/univariate_ts.db")
RAW_SHA = "8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c"
EXP484_SHA = "a5a097f8289477ca67e4e2cd9bdc7edffaa852b435a1f92fef338a0823dec415"
PREREG = ROOT / "docs/superpowers/specs/2026-09-16-exp488-catch22-train-descriptors-antigravity.html"
CODE_PATHS = (Path(__file__), Path(features_antigravity.__file__),
              ROOT / "experiments/exp481_strict_evidence_residual_correction_antigravity/policy_antigravity.py",
              ROOT / "experiments/exp481_strict_evidence_residual_correction_antigravity/evaluation_antigravity.py")


def identity(database, source_sha):
    if version("aeon") != "1.5.0":
        raise RuntimeError("Exp488 requires the preregistered aeon 1.5.0")
    wal = Path(str(database) + "-wal")
    if wal.exists() and wal.stat().st_size:
        raise RuntimeError("raw database has an unbound WAL")
    raw_sha = file_sha(database)
    exp484 = ROOT / "experiments/exp484_distance_qualification_antigravity/results_antigravity.json"
    if raw_sha != RAW_SHA or file_sha(exp484) != EXP484_SHA:
        raise RuntimeError("frozen source SHA changed")
    return dict(schema="exp488_descriptor_binding_v1", sources={**source_sha, "raw_database": raw_sha, "exp484": EXP484_SHA},
                feature_keys=list(FEATURE_KEYS), catch22_config=CATCH22_CONFIG,
                versions={key: version(key) for key in ("aeon", "numpy", "numba", "scipy", "scikit-learn", "llvmlite")},
                python=platform.python_version(), architecture=platform.machine(),
                implementation_sha256={Path(m.__file__).name: file_sha(Path(m.__file__)) for m in (features_antigravity, _catch22, general, stats)},
                code_sha256={str(path.relative_to(ROOT)): file_sha(path) for path in CODE_PATHS},
                preregistration_sha256=file_sha(PREREG))


def validate_records(records, rows):
    expected = {row.dataset: row for row in rows}
    if not isinstance(records, dict) or not set(records) <= set(expected):
        raise ValueError("descriptor cache population mismatch")
    for dataset, record in records.items():
        values = record.get("values", [])
        if (len(values) != 24 or values[:2] != list(expected[dataset].features[:2])
                or any(v is not None and (type(v) is not float or not np.isfinite(v)) for v in values)):
            raise ValueError("descriptor cache values mismatch")
        counts = record.get("feature_valid_counts", [])
        finite, eligible, total = (record.get(key) for key in ("finite_rows", "eligible_rows", "total_rows"))
        membership = record.get("train_membership_sha256", "")
        if (any(type(v) is not int for v in (finite, eligible, total))
                or not 0 <= eligible <= finite <= total or finite != values[1]
                or len(counts) != 22 or any(type(v) is not int or not 0 <= v <= eligible for v in counts)
                or any((n == 0) != (v is None) for n, v in zip(counts, values[2:], strict=True))
                or len(membership) != 64 or any(c not in "0123456789abcdef" for c in membership)):
            raise ValueError("descriptor cache provenance mismatch")


def read_cache(path, binding, rows):
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    if (payload.get("binding") != binding
            or payload.get("records_sha256") != sha256(encode(payload.get("records"))).hexdigest()):
        raise ValueError("descriptor cache binding or checksum mismatch")
    validate_records(payload["records"], rows)
    return payload["records"]


def save_cache(path, binding, records):
    write_json(path, dict(binding=binding, records=records, records_sha256=sha256(encode(records)).hexdigest()))


def extract_one(database, dataset, frozen):
    started = perf_counter()
    rows, membership = load_training_rows(database, dataset)
    record = describe_rows(rows, frozen)
    record["train_membership_sha256"] = membership
    return dataset, record, perf_counter() - started


def extract_all(database, rows, binding, cache_path, workers):
    records = read_cache(cache_path, binding, rows)
    missing = [row for row in rows if row.dataset not in records]
    timing = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(extract_one, database, r.dataset, r.features[:2]) for r in missing]
        for index, future in enumerate(as_completed(futures), 1):
            dataset, record, seconds = future.result()
            validate_records({dataset: record}, rows)
            records[dataset], timing[dataset] = record, seconds
            if index % 25 == 0 or index == len(missing):
                save_cache(cache_path, binding, records)
                print(f"descriptors {len(records)}/{len(rows)}", flush=True)
    return records, timing


def assess_acceleration():
    """Diagnostic parity cannot promote a backend without the full-stage gate."""
    rng = np.random.default_rng(488)
    prototype = _fixed_evidence_rows_antigravity()[0]
    rows = tuple(replace(prototype, dataset=f"F{i:03}_normal_0", source_family=f"F{i:03}") for i in range(34))
    values = {r.dataset: (*r.features[:2], *rng.normal(size=22)) for r in rows}
    timings, results = {}, {}
    for key, backend in (("numpy_seconds", "numpy_reference"), ("numba_first_seconds", "numba_cpu"), ("numba_warm_seconds", "numba_cpu")):
        start = perf_counter()
        results[key] = evaluate_nested_out_of_family_antigravity(rows, residual_policy_kwargs={"feature_keys": FEATURE_KEYS, "descriptor_values": values, "backend": backend})
        timings[key] = perf_counter() - start
    parity = results["numpy_seconds"] == results["numba_first_seconds"] == results["numba_warm_seconds"]
    # Probe GPU support in a child: a missing Metal device can abort its process.
    probes = {
        "pytorch": """import json,torch
r=dict(version=torch.__version__,cpu_float64=True,mps_available=torch.backends.mps.is_available())
try:
    a=torch.tensor([1.,2.],device='mps',dtype=torch.float64)
    r['mps_float64_finite']=bool(torch.isfinite(a.sum()).item())
    torch.mps.synchronize()
except (TypeError,RuntimeError) as e:
    r.update(mps_float64_finite=False,fallback_reason=str(e))
print(json.dumps(r))""",
        "mlx": """import json,mlx.core as mx
r=dict(metal_available=mx.metal.is_available())
try:
    mx.set_default_device(mx.gpu)
    a=mx.array([1.,2.],dtype=mx.float64)
    s=mx.sum(a)
    mx.eval(s)
    r['gpu_float64_finite']=bool(mx.isfinite(s).item())
except (ValueError,RuntimeError) as e:
    r.update(gpu_float64_finite=False,fallback_reason=str(e))
print(json.dumps(r))""",
    }
    availability = {}
    for name, code in probes.items():
        start = perf_counter()
        try:
            process = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
            availability[name] = json.loads(process.stdout) if process.returncode == 0 else {"available": False, "exit_code": process.returncode}
        except (subprocess.TimeoutExpired, json.JSONDecodeError):
            availability[name] = {"available": False, "probe_failed": True}
        availability[name]["probe_seconds"] = perf_counter() - start
    return dict(selected_feature_backend="official_aeon_numpy_numba_cpu", selected_routing_backend="numpy_reference",
                versions={key: version(key) for key in ("torch", "mlx", "mlx-metal", "numba", "numpy")},
                device=platform.machine(), availability=availability, seed=488, tolerance=dict(rtol=1e-12, atol=1e-12),
                numba=dict(**timings, nested_decision_parity=parity, diagnostic_families=34, k=31, inner_min_changed_families=20,
                           exact_full_population_stage_benchmarked=False, cold_jit_verified=False,
                           fallback_reason="full_population_including_compilation_not_benchmarked"),
                preprocessing_aggregation=dict(tensor_eligible=True, selected="numpy", alternative_benchmark_seconds=None,
                    fallback_reason="short_vector_normalization_and_22_column_medians_surround_official_numba_transform; no_measured_full_stage_gain"),
                pytorch_mlx_mps=dict(exact_catch22_tensor_eligible=False, distance_tensor_eligible=True,
                    full_stage_seconds=None, candidate_index_parity=None, numerical_parity=None, speedup=None, selected=False,
                    fallback_reason="official_Catch22_has_no_tensor_backend; preserve_float64_reference_and_require_full_stage_parity_and_speed_before_promotion"))


def report(result, manifest):
    evaluation = result["evaluation"]
    summary, changed = evaluation["summaries"]["exp481"], evaluation["summaries"]["changed"]
    gates = "".join(f"<tr><td>{escape(key)}</td><td>{escape(str(value['actual']))}</td><td>{escape(value['required'])}</td><td>{'통과' if value['pass'] else '실패'}</td></tr>"
                    for key, value in evaluation["gates"].items() if isinstance(value, dict) and "pass" in value)
    reasons = {}
    for decision in evaluation["decisions"]:
        reasons[decision["reason"]] = reasons.get(decision["reason"], 0) + 1
    return f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Exp488 결과</title>
<style>body{{font:16px/1.7 system-ui;max-width:1000px;margin:32px auto;padding:0 20px;color:#172033}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccd3dc;padding:8px;text-align:left}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f5f8;padding:16px}}</style>
<h1>Exp488: 정상 TRAIN Catch22 특징</h1><p>공식 판정: <strong>{'기준 통과' if evaluation['gates']['overall_pass'] else '기준 실패'}</strong>. {evaluation['dataset_count']:,}개 데이터셋, {evaluation['family_count']}개 family. 변경 {changed['count']}건, ZF {summary['zf_total']}, 평균 AUC {summary['dataset_mean_auc']:.10f}.</p>
<p>Exp484의 거리 교체 이후, residual policy 입력만 width/n_train + Catch22 중앙값 22개로 교체했다. 기존 row-KNN, 네 점수원, K=31, local/inner gate와 후보 순서는 유지했다. 원본 정상 TRAIN만 읽고 각 폴드의 donor만으로 결측 대체와 표준화를 수행했다. 새 점수 학습이나 TEST 평가는 실행하지 않았다.</p>
<table><tr><th>기준</th><th>실제</th><th>요구</th><th>판정</th></tr>{gates}</table>
<h2>검증·출처</h2><p>기존 7차원 입력 경로 동등성, 34 family 중첩 게이트, TRAIN 선택, 가변 길이, 결측과 상수열, query 성능 격리 및 캐시 무결성은 연결된 테스트로 검증한다. 이 보고서는 테스트 통과 횟수를 대신하지 않는다.</p>
<pre>{escape(json.dumps(evaluation['leakage'], ensure_ascii=False, indent=2))}</pre>
<h2>판정 사유와 제한</h2><pre>{escape(json.dumps(reasons, ensure_ascii=False, indent=2))}</pre><p>이 결과는 기존 동결 TRAIN 근거의 중첩 평가다. 새로운 family에서의 일반화나 서비스 승격을 의미하지 않는다. 결과에 맞춰 이번 특징이나 기준을 다시 선택하지 않았다. JSON의 exp481 요약 키는 재사용한 evaluator의 역사적 이름이며 해당 값은 Exp488 평가다.</p>
<h2>계산·재현</h2><p>공식 aeon 1.5.0 CPU/Numba 특징 추출, NumPy 라우팅. 특징 worker {manifest['descriptor_workers']}개, 내부 스레드 1개. 실행 {manifest['elapsed_seconds']:.1f}초.</p><pre>{escape(json.dumps(manifest, ensure_ascii=False, indent=2))}</pre>
<p>저장소 루트에서 실행한다. 이미 공식 결과가 있으면 덮어쓰기를 거부한다. 캐시는 원본과 특징 구현의 해시가 같을 때만 재사용한다.</p><pre>PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=/private/tmp/exp488_numba_antigravity .venv/bin/python -m experiments.exp488_catch22_train_descriptors_antigravity.run_antigravity smoke
PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=/private/tmp/exp488_numba_antigravity .venv/bin/python -m experiments.exp488_catch22_train_descriptors_antigravity.run_antigravity official
PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=/private/tmp/exp488_numba_antigravity .venv/bin/python -m pytest -q -p no:cacheprovider --import-mode=importlib experiments/exp488_catch22_train_descriptors_antigravity/test_exp488_antigravity.py experiments/exp481_strict_evidence_residual_correction_antigravity/tests experiments/exp484_distance_qualification_antigravity/tests</pre>
<p><a href="../../docs/superpowers/specs/2026-09-16-exp488-catch22-train-descriptors-antigravity.html">사전등록</a> · <a href="results_antigravity.json">결과 JSON</a> · <a href="run_manifest_antigravity.json">실행 manifest</a> · <a href="test_exp488_antigravity.py">검증 코드</a></p></html>'''


def run(mode, database=DATABASE, output=OUTPUT):
    started, started_at = perf_counter(), datetime.now(timezone.utc).isoformat()
    output = Path(output)
    official_path = output / "results_antigravity.json"
    if mode == "official" and official_path.exists():
        raise RuntimeError("official result already exists; verify it instead of rerunning")
    rows, control, sources = load_frozen_sources_antigravity()
    before = file_identity(database)
    binding = identity(database, sources)
    workers = max(1, (os.cpu_count() or 1) - 1)
    cache_path = output / "descriptor_cache_antigravity.json"
    if mode == "smoke":
        acceleration = assess_acceleration()
        if not acceleration["numba"]["nested_decision_parity"]:
            raise RuntimeError("diagnostic NumPy/Numba parity failed")
        records = read_cache(cache_path, binding, rows)
        samples = [min((r for r in rows if lo <= r.features[0] < hi), key=lambda r: (r.features[1], r.dataset)) for lo, hi in ((0, 64), (128, 300), (1000, 4001))]
        smoke = []
        for row in samples:
            _, record, first_seconds = extract_one(database, row.dataset, row.features[:2])
            _, repeated, warm_seconds = extract_one(database, row.dataset, row.features[:2])
            if record != repeated:
                raise RuntimeError("descriptor repeatability failed")
            records[row.dataset] = record
            smoke.append(dict(dataset=row.dataset, width=row.features[0], n_train=row.features[1], first_seconds=first_seconds, warm_seconds=warm_seconds, exact_parity=True))
        validate_records(records, rows)
        if before != file_identity(database) or identity(database, sources) != binding:
            raise RuntimeError("source or implementation changed during smoke")
        save_cache(cache_path, binding, records)
        result = dict(state="smoke_nonofficial", binding=binding, acceleration=acceleration, samples=smoke,
                      descriptor_workers=workers, elapsed_seconds=perf_counter() - started)
        write_json(output / "smoke_results_antigravity.json", result)
    else:
        smoke = json.loads((output / "smoke_results_antigravity.json").read_text())
        if smoke.get("state") != "smoke_nonofficial" or smoke.get("binding") != binding:
            raise RuntimeError("matching smoke is required before official evaluation")
        feature_started = perf_counter()
        records, dataset_timings = extract_all(database, rows, binding, cache_path, workers)
        feature_seconds = perf_counter() - feature_started
        values = {dataset: tuple(np.nan if v is None else v for v in record["values"]) for dataset, record in records.items()}
        if len(values) != len(rows):
            raise RuntimeError("incomplete descriptors")
        print("descriptors frozen; starting nested evaluation (128 outer families)", flush=True)
        evaluation_started = perf_counter()
        evaluation = evaluate_nested_out_of_family_antigravity(rows, residual_policy_kwargs={"feature_keys": FEATURE_KEYS, "descriptor_values": values})
        evaluation_seconds = perf_counter() - evaluation_started
        if evaluation["summaries"]["row_knn"] != control["evaluation"]["summaries"]["row_knn"]:
            raise RuntimeError("original row-KNN control changed")
        expected_audit = dict(family_overlap_count=0, outer_family_seen_by_inner_count=0, route_input_key_count=24,
                              dataset_name_feature_count=0, held_out_score_feature_count=0, held_out_scores_read_after_routing=True)
        if any(evaluation["leakage"].get(key) != value for key, value in expected_audit.items()):
            raise RuntimeError("evaluation isolation failed")
        if (evaluation["dataset_count"], evaluation["family_count"]) != (1117, 128):
            raise RuntimeError("evaluation population changed")
        evaluation["gates"]["retire_registered_descriptor_space"] = evaluation["gates"].pop("retire_seven_feature_space")
        result = dict(schema="exp488_qualification_v1", state="train_nested_complete", binding=binding,
                      descriptor_cache_sha256=file_sha(cache_path), evaluation=evaluation)
        manifest = dict(schema="exp488_run_manifest_v1", started_at_utc=started_at, completed_at_utc=datetime.now(timezone.utc).isoformat(),
                        elapsed_seconds=perf_counter() - started, feature_seconds=feature_seconds, evaluation_seconds=evaluation_seconds,
                        logical_cpu_count=os.cpu_count(), descriptor_workers=workers, evaluation_workers=1, inner_threads=1,
                        descriptor_dataset_seconds=dataset_timings, acceleration=smoke["acceleration"], python=sys.version,
                        code_sha256=binding["code_sha256"],
                        result_sha256=sha256(encode(result)).hexdigest())
        if before != file_identity(database) or identity(database, sources) != binding:
            raise RuntimeError("source or implementation changed during execution")
        write_json(official_path, result)
        write_json(output / "run_manifest_antigravity.json", manifest)
        write_text(output / "report_antigravity.html", report(result, manifest))
    if before != file_identity(database):
        raise RuntimeError("raw database changed during execution")
    print(json.dumps(dict(state=result["state"], output=str(output), elapsed_seconds=perf_counter() - started)), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "official"))
    parser.add_argument("--database", type=Path, default=DATABASE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(args.mode, args.database, args.output)
