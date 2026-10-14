"""Official aeon ROCKAD vs frozen service, full read-only database catalog."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import csv
import gzip
from hashlib import sha256
from html import escape
import importlib.metadata as metadata
import inspect
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from aeon.anomaly_detection.collection import ROCKAD
from numba import set_num_threads
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
SERVICE_ROOT = Path(os.environ.get("ROCKAD_SERVICE_ROOT", str(ROOT)))
sys.path.insert(0, str(SERVICE_ROOT))
from services.time_series_anomaly_diagnosis.anomaly_detection import ALPHA, FIXED_SEEDS, empirical_upper_pvalues, run_anomaly_detection
from services.time_series_anomaly_diagnosis.dataset_repository import get_dataset_catalog, load_dataset_values, prepare_dataset_values, load_test_labels, open_source_database
from services.time_series_anomaly_diagnosis.model_resources import load_model_resources
from services.time_series_anomaly_diagnosis.score_sources import ScoreResult, crossfit_plan, score_method

RESOURCES = SERVICE_ROOT / "services/time_series_anomaly_diagnosis/resources"
DEFAULT_DB = Path("/Users/mac/Documents/Dataset/univariate_ts.db")
DEFAULT_OUT = ROOT / "results/aeon_rockad_full_20261014_antigravity"
DEFAULT_DASHBOARD = ROOT / "docs/ROCKAD_FULL_COMPARISON_antigravity.html"
PINNED = {"aeon": "1.4.0", "scikit-learn": "1.7.2", "numpy": "2.3.5", "numba": "0.63.1", "torch": "2.13.0"}


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    value = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, payload):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)


def write_score(path, payload):
    path = Path(path)
    raw = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
    compressed = gzip.compress(raw, mtime=0)
    with path.open("xb") as stream:
        stream.write(compressed)


def read_score(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def full_catalog(rows):
    names = [r["dataset_name"] for r in rows]
    if len(names) != len(set(names)):
        raise ValueError("duplicate catalog dataset")
    return sorted(rows, key=lambda r: r["dataset_name"])


def score_inventory(folder, protocol):
    inventory = []
    expected = {f"score_{i:04d}_antigravity.json.gz" for i in range(len(protocol["catalog"]))}
    if {p.name for p in folder.glob("score_*_antigravity.json.gz")} != expected:
        raise ValueError("complete catalog score coverage is required before TEST labels")
    for index, row in enumerate(protocol["catalog"]):
        path = folder / f"score_{index:04d}_antigravity.json.gz"
        if read_score(path)["dataset"] != row["dataset_name"]:
            raise ValueError("score identity does not match frozen catalog")
        inventory.append({"path": path.name, "sha256": digest(path)})
    return inventory


def rockad_score(train, test, method, seed):
    if method != "aeon_rockad":
        raise ValueError("invalid method")
    train, test = np.asarray(train, dtype=np.float32), np.asarray(test, dtype=np.float32)
    folds = crossfit_plan(len(train), seed=seed)
    if any(len(fit) < 5 for fit, _ in folds):
        raise ValueError("five-neighbor crossfit requires at least seven normal TRAIN rows")
    oof = np.empty(len(train), dtype=float)
    for fit, query in folds:
        model = ROCKAD(random_state=seed, n_jobs=1).fit(train[fit])
        oof[query] = model.predict(train[query])
    model = ROCKAD(random_state=seed, n_jobs=1).fit(train)
    scores = np.asarray(model.predict(test), dtype=float)
    chunks = np.concatenate([model.predict(test[start:start+128]) for start in range(0, len(test), 128)])
    if not np.allclose(scores, chunks, rtol=0, atol=1e-10):
        raise ValueError("batch score consistency failed")
    p, q = empirical_upper_pvalues(oof, scores), empirical_upper_pvalues(oof, chunks)
    for cutoff in (ALPHA, max(ALPHA, 1/(len(train)+1))):
        if not np.array_equal(p <= cutoff+1e-15, q <= cutoff+1e-15):
            raise ValueError("batch decision consistency failed")
    return ScoreResult(oof, scores, 200, seed)


def score_dataset(entry, database):
    set_num_threads(1)
    name = entry["dataset_name"]
    item = {"dataset": name, "family": name.rsplit("_normal_", 1)[0], "catalog": entry,
            "started_at": now(), "models": {}, "n_train": None, "n_test": None}
    if entry["train_count"] < 5:
        for model in ("service", "rockad"):
            item["models"][model] = {"status": "unavailable", "reason": "fewer than five total TRAIN rows; finite normal rows cannot reach k=5"}
        item["finished_at"] = now()
        return item
    try:
        with open_source_database(Path(database)) as connection:
            prepared = (load_dataset_values(connection, name, entry["availability"])
                        if entry["analysis_available"] else prepare_dataset_values(connection, name))
        item.update(n_train=len(prepared.train_values), n_test=len(prepared.test_values), target_length=prepared.target_length,
                    test_indices=list(prepared.test_indices),
                    prepared_train_sha256=sha256(prepared.train_values.tobytes()).hexdigest(),
                    prepared_test_sha256=sha256(prepared.test_values.tobytes()).hexdigest())
    except Exception as error:
        item["preparation_error"] = f"{type(error).__name__}: {error}"
        item["finished_at"] = now()
        return item
    for key, method, scorer in (("service", entry["method_name"], score_method), ("rockad", "aeon_rockad", rockad_score)):
        if key == "service" and not entry["analysis_available"]:
            item["models"][key] = {"status": "unavailable", "reason": entry["availability"]}
            continue
        begin = time.perf_counter()
        seeds = []
        def collect(train, test, method_name, seed):
            result = scorer(train, test, method_name, seed)
            seeds.append({"seed": seed, "train_oof": result.train_oof.tolist(), "test_scores": result.test_scores.tolist()})
            return result
        try:
            if key == "rockad" and len(prepared.train_values) < 7:
                # k=5 is unchanged; ranking-only rows never enter automatic-decision totals.
                for seed in FIXED_SEEDS:
                    fitted = ROCKAD(n_jobs=1, random_state=seed).fit(np.asarray(prepared.train_values, dtype=np.float32))
                    scores = np.asarray(fitted.predict(np.asarray(prepared.test_values, dtype=np.float32)), dtype=float)
                    if not np.isfinite(scores).all():
                        raise ValueError("non-finite raw ROCKAD scores")
                    seeds.append({"seed": seed, "test_scores": scores.tolist()})
                median = np.median([s["test_scores"] for s in seeds], axis=0)
                rows = [{"series_index": index, "score": float(score)} for index, score in zip(prepared.test_indices, median, strict=True)]
                item["models"][key] = {"status": "ranking_only", "reason": "TRAIN 5/6: full-fit ranking is available, common k=5 crossfit is not", "rows": rows, "seed_results": seeds}
            else:
                detection = run_anomaly_detection(prepared, method, scorer=collect)
                item["models"][key] = {"status": "ok", "rows": list(detection.rows), "seed_results": seeds,
                                        "warnings": list(detection.resolution_limited_warning_indices)}
        except Exception as error:
            item["models"][key] = {"status": "error", "reason": f"{type(error).__name__}: {error}", "completed_seeds": len(seeds)}
        item["models"][key].update(method=method, seconds=time.perf_counter()-begin)
    item["finished_at"] = now()
    return item


def paired_summary(rows, policy):
    paired = [r for r in rows if all(r.get("models", {}).get(m, {}).get(policy) is not None for m in ("service", "rockad"))]
    result = {"paired_datasets": len(paired), "paired_families": len({r["family"] for r in paired})}
    for method in ("service", "rockad"):
        totals = {k: sum(r["models"][method][policy][k] for r in paired)
                  for k in ("true_positives", "false_positives", "true_negatives", "false_negatives")}
        for metric in ("auc", "ap"):
            values = [r["models"][method][policy][metric] for r in paired if r["models"][method][policy].get(metric) is not None]
            groups = defaultdict(list)
            for row in paired:
                value = row["models"][method][policy].get(metric)
                if value is not None:
                    groups[row["family"]].append(value)
            totals["dataset_mean_"+metric] = float(np.mean(values)) if values else None
            totals["family_mean_"+metric] = float(np.mean([np.mean(v) for v in groups.values()])) if groups else None
            totals[metric+"_datasets"] = len(values)
        result[method] = totals
    return result


def evaluate_rows(rows, labels, policy):
    y = np.array([labels[r["series_index"]] for r in rows], dtype=int)
    scores = np.array([r["score"] for r in rows], dtype=float)
    if not len(y) or not set(y).issubset({0, 1}) or not np.isfinite(scores).all():
        raise ValueError("invalid evaluated labels or scores")
    ranking = {"auc": float(roc_auc_score(y, scores)) if len(set(y)) == 2 else None,
               "ap": float(average_precision_score(y, scores)) if len(set(y)) == 2 else None}
    if policy == "ranking_only":
        return ranking
    mask = np.array([r["automatic_anomaly"] if policy == "service_policy" else r["p_value"] <= ALPHA+1e-15 for r in rows], dtype=bool)
    return dict(ranking, true_positives=int(np.sum(mask & (y==1))), false_positives=int(np.sum(mask & (y==0))),
                true_negatives=int(np.sum(~mask & (y==0))), false_negatives=int(np.sum(~mask & (y==1))))


def evaluate(folder, database):
    protocol = json.loads((folder / "protocol_antigravity.json").read_text())
    freeze_path = folder / "score_freeze_antigravity.json"
    freeze = json.loads(freeze_path.read_text())
    if freeze["protocol_sha256"] != digest(folder / "protocol_antigravity.json") or freeze["score_files"] != score_inventory(folder, protocol):
        raise ValueError("frozen inputs changed")
    if protocol["database_sha256"] != digest(database):
        raise ValueError("source DB changed before label evaluation")
    results = []
    evaluated_start = now()
    with open_source_database(database) as connection:
        for frozen in freeze["score_files"]:
            raw = read_score(folder / frozen["path"])
            row = {k: v for k, v in raw.items() if k not in ("models", "test_indices")}
            row["models"] = {}
            scored = {m: v for m, v in raw["models"].items() if v["status"] in ("ok", "ranking_only")}
            try:
                labels = load_test_labels(connection, raw["dataset"], tuple(raw["test_indices"])) if scored else {}
                if labels:
                    row["test_class_counts"] = dict(Counter(str(v) for v in labels.values()))
                for method, model in raw["models"].items():
                    details = {k: v for k, v in model.items() if k not in ("rows", "seed_results", "warnings")}
                    if method in scored:
                        details["ranking"] = evaluate_rows(model["rows"], labels, "ranking_only")
                        if model["status"] == "ok":
                            for policy in ("service_policy", "strict_alpha_policy"):
                                details[policy] = evaluate_rows(model["rows"], labels, policy)
                            details["warning_count"] = len(model["warnings"])
                    row["models"][method] = details
            except Exception as error:
                row["evaluation_error"] = f"{type(error).__name__}: {error}"
                row["models"] = {m: {"status": v["status"], "reason": v.get("reason")} for m, v in raw["models"].items()}
            results.append(row)
    payload = {"evaluation_started_at": evaluated_start, "evaluated_at": now(), "score_freeze_sha256": digest(freeze_path),
               "catalog_datasets": len(protocol["catalog"]), "rows": results,
               "service_policy": paired_summary(results, "service_policy"), "strict_alpha_policy": paired_summary(results, "strict_alpha_policy")}
    write_json(folder / "evaluation_antigravity.json", payload)
    fields = ["dataset", "family", "n_train", "n_test", "method", "status", "reason", "auc", "ap", "TP", "FP", "TN", "FN", "strict_TP", "strict_FP", "strict_FN", "seconds"]
    with (folder / "dataset_metrics_antigravity.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in results:
            for method in ("service", "rockad"):
                model = row["models"].get(method, {"status": "preparation_error", "reason": row.get("preparation_error")})
                policy, strict, ranking = model.get("service_policy", {}), model.get("strict_alpha_policy", {}), model.get("ranking", {})
                writer.writerow(dict(dataset=row["dataset"], family=row["family"], n_train=row["n_train"], n_test=row["n_test"], method=method,
                    status=model["status"], reason=row.get("evaluation_error") or model.get("reason"), auc=ranking.get("auc"), ap=ranking.get("ap"),
                    TP=policy.get("true_positives"), FP=policy.get("false_positives"), TN=policy.get("true_negatives"), FN=policy.get("false_negatives"),
                    strict_TP=strict.get("true_positives"), strict_FP=strict.get("false_positives"), strict_FN=strict.get("false_negatives"), seconds=model.get("seconds")))
    return payload


def dashboard(path, completed, total, results=None, stage="scoring", last=""):
    def text(value):
        return "—" if value is None else (f"{value:.4f}" if isinstance(value, float) else str(value))
    summary = results["service_policy"] if results else None
    table = ""
    if summary:
        for key, label in (("false_negatives", "미탐 FN"), ("false_positives", "오탐 FP"), ("true_positives", "정탐 TP"),
                           ("dataset_mean_auc", "데이터셋 평균 AUROC"), ("family_mean_auc", "가족 균형 AUROC"), ("dataset_mean_ap", "데이터셋 평균 AP"), ("family_mean_ap", "가족 균형 AP")):
            table += f"<tr><th>{label}</th><td>{text(summary['service'][key])}</td><td>{text(summary['rockad'][key])}</td></tr>"
        table = f"<h2>같은 {summary['paired_datasets']}개 · {summary['paired_families']}가족의 공통 서비스 판정 비교</h2><table><tr><th>지표</th><th>현재 서비스</th><th>aeon ROCKAD</th></tr>{table}</table>"
        strict = results["strict_alpha_policy"]
        table += "<p>엄격한 p ≤ 0.005 판정: " + "; ".join(f"{m}: TP {strict[m]['true_positives']}, FN {strict[m]['false_negatives']}, FP {strict[m]['false_positives']}" for m in ("service", "rockad")) + ".</p>"
    body = ""
    if results:
        for row in results["rows"]:
            cells = [escape(row["dataset"]), f"{text(row['n_train'])} / {text(row['n_test'])}"]
            for method in ("service", "rockad"):
                model = row["models"].get(method, {})
                ranked, policy = model.get("ranking", {}), model.get("service_policy", {})
                cells += [escape(model.get("status", "preparation_error")), f"{text(ranked.get('auc'))} / {text(ranked.get('ap'))}", f"{text(policy.get('false_negatives'))} / {text(policy.get('false_positives'))}"]
            cells.append(escape(str(row.get("evaluation_error") or row.get("preparation_error") or row["models"].get("rockad", {}).get("reason", ""))))
            body += "<tr>" + "".join(f"<td>{v}</td>" for v in cells) + "</tr>"
    finished = stage in ("verified", "uploaded")
    checks = [("全 DB 목록·설정 고정", True), ("전체 점수 처리 및 라벨 읽기 전 고정", completed==total),
              ("정답 라벨 평가·결과 집계", results is not None), ("계산·해시·반복 실행 검증", finished), ("GitHub 결과 업로드", stage=="uploaded")]
    checkboxes = "".join(f'<label><input type="checkbox" disabled {"checked" if done else ""}> {escape(label)}</label>' for label, done in checks)
    refresh = '<meta http-equiv="refresh" content="15">' if not finished else ""
    html = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">{refresh}
<title>aeon ROCKAD 전체 데이터셋 비교</title><style>body{{font:16px/1.7 system-ui;max-width:1150px;margin:32px auto;padding:0 20px;color:#1d1d1f;word-break:keep-all}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}section{{margin:24px 0}}label{{display:block}}.wide{{overflow-x:auto}}td:first-child{{overflow-wrap:anywhere}}code{{overflow-wrap:anywhere}}.state{{padding:16px;background:#f5f5f7;border-radius:12px}}</style>
<h1>aeon ROCKAD 전체 데이터셋 비교</h1><p class="state">{completed:,} / {total:,} 처리 · 단계: {escape(stage)} · {escape(last)}</p>{checkboxes}
<section>{table}<p>실패·평가 불가·ranking_only를 0점으로 바꾸거나 양쪽 성능 합계에 넣지 않습니다. ranking_only는 원시 점수의 순위만 평가하고 자동 판정은 평가하지 않은 경우입니다.</p></section>
<section><h2>범위와 비교 조건</h2><p>DB 1,119개 전체를 결과 목록에 포함합니다. 서비스 등록 모델은 1,117개이며, 서비스의 기존 모델 선택과 정상 TRAIN·전처리를 유지합니다. 등록되지 않은 CornellWhaleChallenge는 기존 값 기반 정상 TRAIN 전처리로 ROCKAD를 단독 평가합니다. TRAIN 5~6개는 ROCKAD 기본 이웃 5개를 유지한 전체 학습 점수만 평가합니다. 5개 미만은 평가 불가입니다.</p>
<p>aeon 1.4.0의 <code>aeon.anomaly_detection.collection.ROCKAD</code>, 커널 100·모델 10·이웃 5·power transform 사용·normalise=False. 두 초기값 20260721/20261721, 같은 5겹 교차 검증, 점수 중앙값 및 동일 서비스 판정 규칙을 적용합니다. 작은 표본의 자동 이상은 경고가 붙는 후보이지 엄격한 p ≤ 0.005 통과가 아닙니다.</p>
<p>공식 CPU/Numba 구현을 사용합니다. MPS/MLX 하드웨어는 가용하지만 공식 aeon ROCKAD GPU 경로와 현재 실행 환경의 변환된 Core ML 모델은 없으므로 다른 구현으로 바꾸지 않았습니다. 12개 논리 CPU 중 11개 작업자를 사용하며 결과를 보고 설정이나 임계값을 바꾸지 않습니다.</p>
<p>이것은 현행 등록 모델과 고정 ROCKAD 기본 설정의 비교입니다. 서비스가 과거 연구의 모델 선택을 거쳤으므로 신규 미학습 라우팅의 독립 일반화 성능이라고 해석하지 않습니다. 같은 원본 이름으로 묶은 가족 균형 평균도 별도 제공합니다.</p></section>
<section><h2>전체 데이터셋별 결과</h2><p>순위 칸: AUROC / AP. 판정 칸: FN / FP. 좁은 화면에서는 표를 가로로 스크롤합니다.</p><div class="wide"><table id="dataset-results"><tr><th>데이터셋</th><th>실제 TRAIN / TEST</th><th>서비스 상태</th><th>서비스 순위</th><th>서비스 판정</th><th>ROCKAD 상태</th><th>ROCKAD 순위</th><th>ROCKAD 판정</th><th>제약·실패</th></tr>{body}</table></div></section>
<section><h2>재현 자료</h2><p>결과 폴더: <code>results/aeon_rockad_full_20261014_antigravity/</code>. 프로토콜·gzip 점수 체크포인트·점수 고정 해시·평가 JSON·데이터셋별 CSV·검증 결과를 보존합니다. 원본 시계열과 DB는 결과 업로드에 포함하지 않습니다.</p><p>실행: <code>python experiments/aeon_rockad_full_comparison_antigravity/run_full_antigravity.py run --database /path/univariate_ts.db</code>. 의존성 버전은 프로토콜을 따릅니다. 중단 후 같은 명령을 사용하면 완료된 체크포인트를 보존하고 미처리 항목만 이어갑니다.</p>
<p><a href="https://www.aeon-toolkit.org/en/v1.4.0/api_reference/auto_generated/aeon.anomaly_detection.collection.ROCKAD.html">aeon 1.4.0 공식 ROCKAD 문서</a></p></section></html>'''
    temp = Path(str(path)+".tmp")
    temp.write_text(html.replace("全 DB", "전체 DB"), encoding="utf-8")
    os.replace(temp, path)


def selfcheck():
    set_num_threads(1)
    for package, pinned in PINNED.items():
        if metadata.version(package) != pinned:
            raise ValueError(f"version mismatch: {package}")
    rng = np.random.default_rng(20261014)
    train, test = rng.normal(size=(24, 64)), rng.normal(size=(13, 64))
    first = rockad_score(train, test, "aeon_rockad", FIXED_SEEDS[0])
    repeat = rockad_score(train, test, "aeon_rockad", FIXED_SEEDS[0])
    if not np.array_equal(first.train_oof, repeat.train_oof) or not np.array_equal(first.test_scores, repeat.test_scores):
        raise ValueError("same-seed repeat mismatch")
    for method in ("minirocket_local_gap", "multirocket_local_gap", "hydra_local_gap"):
        score_method(train, test, method, FIXED_SEEDS[0])
    return {"status": "passed", "finite_repeat_batch_decisions": True}


def run(args):
    folder, database, report = Path(args.output), Path(args.database), Path(args.dashboard)
    folder.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    checked = selfcheck()
    with open_source_database(database) as connection:
        catalog = full_catalog(get_dataset_catalog(connection, load_model_resources(RESOURCES)))
    if len(catalog) != 1119:
        raise ValueError("expected the approved 1119-dataset database catalog")
    workers = max(1, (os.cpu_count() or 1)-1)
    manifest_path = folder / "protocol_antigravity.json"
    provenance = {"database_sha256": digest(database), "runner_sha256": digest(__file__),
                  "service_source_hashes": {str(p.relative_to(SERVICE_ROOT)): digest(p) for p in (SERVICE_ROOT/"services/time_series_anomaly_diagnosis").glob("*.py")},
                  "resource_hashes": {p.name: digest(p) for p in RESOURCES.glob("*.json")},
                  "aeon_rockad_source_sha256": digest(inspect.getfile(ROCKAD)),
                  "versions": {p: metadata.version(p) for p in (*PINNED, "scipy", "mlx")}}
    if manifest_path.exists():
        protocol = json.loads(manifest_path.read_text())
        if any(protocol[k] != v for k, v in provenance.items()) or protocol["catalog"] != catalog:
            raise ValueError("resume provenance differs; do not mix runs")
    else:
        protocol = dict(provenance, schema="aeon_rockad_full_v1", created_at=now(), catalog=catalog, workers=workers,
                        cpu_count=os.cpu_count(), seeds=list(FIXED_SEEDS), folds=5, alpha=ALPHA, rockad_params=ROCKAD().get_params(),
                        selfcheck=checked, backend="official_aeon_CPU_Numba", python=sys.version,
                        accelerator_assessment="MPS/MLX hardware available in preceding exact-method probe; aeon ROCKAD has no device/backend option; official Numba Rocket and sklearn CPU power transform/KDTree retained; no converted Core ML model or official MLX ROCKAD. No algorithm rewrite.",
                        numba_assessment="official aeon Numba kernels and unchanged service Numba preparation/p-values reused; no new hot numerical kernel", batch_atol=1e-10)
        write_json(manifest_path, protocol)
    pending, complete = [], 0
    for index, entry in enumerate(catalog):
        path = folder / f"score_{index:04d}_antigravity.json.gz"
        if path.exists():
            if read_score(path)["dataset"] != entry["dataset_name"]:
                raise ValueError("existing score identity mismatch")
            complete += 1
        else:
            pending.append((index, entry))
    dashboard(report, complete, len(catalog))
    # Start expensive cases first; no cap is imposed on ROCKAD training references.
    pending.sort(key=lambda pair: pair[1]["train_count"]*pair[1]["series_length"], reverse=True)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        jobs = {executor.submit(score_dataset, entry, str(database)): index for index, entry in pending}
        for future in as_completed(jobs):
            index = jobs[future]
            item = future.result()
            write_score(folder / f"score_{index:04d}_antigravity.json.gz", item)
            complete += 1
            if complete % 20 == 0 or complete == len(catalog):
                print(f"{complete}/{len(catalog)} {item['dataset']} " + str({m:v['status'] for m,v in item['models'].items()}), flush=True)
                dashboard(report, complete, len(catalog), last=item["dataset"])
    freeze_path = folder / "score_freeze_antigravity.json"
    inventory = score_inventory(folder, protocol)
    if digest(database) != protocol["database_sha256"]:
        raise ValueError("source DB changed while scoring")
    if not freeze_path.exists():
        write_json(freeze_path, {"frozen_at": now(), "protocol_sha256": digest(manifest_path), "score_files": inventory,
                                 "current_invocation_seconds": time.perf_counter()-started})
    evaluation_path = folder / "evaluation_antigravity.json"
    results = json.loads(evaluation_path.read_text()) if evaluation_path.exists() else evaluate(folder, database)
    dashboard(report, complete, len(catalog), results, stage="evaluated")
    print(json.dumps({"catalog": len(catalog), "service_policy": results["service_policy"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "selfcheck"))
    parser.add_argument("--database", default=str(DEFAULT_DB))
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    parser.add_argument("--dashboard", default=str(DEFAULT_DASHBOARD))
    arguments = parser.parse_args()
    print(json.dumps(selfcheck())) if arguments.command == "selfcheck" else run(arguments)
