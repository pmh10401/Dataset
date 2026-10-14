"""Independent result verification and atomic, publication-ready full-catalog outputs."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import io
import json
import os
from pathlib import Path
from statistics import mean
import time

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import run_full_antigravity as run


def atomic_text(path, text):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path, payload):
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    atomic_text(path, text)


def exclude_cornell_and_freeze(folder, database):
    """Complete the explicitly amended scope without changing any existing score."""
    protocol_path = folder / "protocol_antigravity.json"
    protocol = json.loads(protocol_path.read_text())
    names = [r["dataset_name"] for r in protocol["catalog"]]
    index = names.index("CornellWhaleChallenge")
    excluded_path = folder / f"score_{index:04d}_antigravity.json.gz"
    expected = {f"score_{i:04d}_antigravity.json.gz" for i in range(len(names)) if i != index}
    present = {p.name for p in folder.glob("score_*_antigravity.json.gz")}
    if present != expected or (folder / "score_freeze_antigravity.json").exists():
        raise ValueError("only Cornell may be pending; an existing freeze must not be amended")
    if run.digest(database) != protocol["database_sha256"] or run.digest(run.__file__) != protocol["runner_sha256"]:
        raise ValueError("source DB or executed runner changed")
    for position, entry in enumerate(protocol["catalog"]):
        if position != index and run.read_score(folder / f"score_{position:04d}_antigravity.json.gz")["dataset"] != entry["dataset_name"]:
            raise ValueError("existing score identity mismatch")
    amended_at = run.now()
    reason = "user explicitly excluded CornellWhaleChallenge; incomplete computation stopped before score freeze and TEST-label evaluation"
    amendment_path = folder / "scope_amendment_antigravity.json"
    run.write_json(amendment_path, {"amended_at": amended_at, "excluded_datasets": [names[index]],
        "effective_datasets": len(names)-1, "reason": reason, "protocol_sha256": run.digest(protocol_path), "test_labels_read": False})
    run.write_score(excluded_path, {"dataset": names[index], "family": names[index], "catalog": protocol["catalog"][index],
        "started_at": amended_at, "finished_at": amended_at, "n_train": None, "n_test": None,
        "models": {m: {"status": "excluded", "reason": reason} for m in ("service", "rockad")}})
    run.write_json(folder / "score_freeze_antigravity.json", {"frozen_at": run.now(),
        "protocol_sha256": run.digest(protocol_path), "score_files": run.score_inventory(folder, protocol),
        "scope_amendment_sha256": run.digest(amendment_path), "excluded_datasets": [names[index]], "effective_datasets": len(names)-1})


def ranking_summary(rows):
    paired = [r for r in rows if all(r.get("models", {}).get(m, {}).get("ranking") is not None for m in ("service", "rockad"))]
    result = {"paired_datasets": len(paired), "paired_families": len({r["family"] for r in paired})}
    for model in ("service", "rockad"):
        values = {}
        for metric in ("auc", "ap"):
            grouped = defaultdict(list)
            numbers = []
            for row in paired:
                value = row["models"][model]["ranking"][metric]
                if value is not None:
                    numbers.append(value)
                    grouped[row["family"]].append(value)
            values["dataset_mean_"+metric] = mean(numbers) if numbers else None
            values["family_mean_"+metric] = mean(mean(g) for g in grouped.values()) if grouped else None
            values[metric+"_datasets"] = len(numbers)
        result[model] = values
    return result


def validate_snapshot(folder, database):
    freeze = json.loads((folder / "score_freeze_antigravity.json").read_text())
    protocol_path = folder / "protocol_antigravity.json"
    protocol = json.loads(protocol_path.read_text())
    if freeze["protocol_sha256"] != run.digest(protocol_path) or freeze["score_files"] != run.score_inventory(folder, protocol):
        raise ValueError("frozen catalog or score hashes changed")
    if freeze.get("scope_amendment_sha256"):
        amendment_path = folder / "scope_amendment_antigravity.json"
        amendment = json.loads(amendment_path.read_text())
        if run.digest(amendment_path) != freeze["scope_amendment_sha256"] or amendment["protocol_sha256"] != run.digest(protocol_path):
            raise ValueError("user scope amendment changed")
        if amendment["excluded_datasets"] != freeze["excluded_datasets"] or amendment["effective_datasets"] != freeze["effective_datasets"]:
            raise ValueError("exclusion scope differs from freeze")
    if run.digest(database) != protocol["database_sha256"] or run.digest(run.__file__) != protocol["runner_sha256"]:
        raise ValueError("source DB or executed scoring runner changed")
    for path, expected in protocol["service_source_hashes"].items():
        if run.digest(run.SERVICE_ROOT / path) != expected:
            raise ValueError("frozen service source changed")
    for name, expected in protocol["resource_hashes"].items():
        if run.digest(run.RESOURCES / name) != expected:
            raise ValueError("frozen resource changed")
    if any(run.metadata.version(name) != version for name, version in protocol["versions"].items()):
        raise ValueError("execution environment versions changed")
    if run.digest(run.inspect.getfile(run.ROCKAD)) != protocol["aeon_rockad_source_sha256"]:
        raise ValueError("official ROCKAD source changed")
    evaluation = json.loads((folder / "evaluation_antigravity.json").read_text())
    if evaluation["score_freeze_sha256"] != run.digest(folder / "score_freeze_antigravity.json"):
        raise ValueError("evaluation refers to another freeze")
    if freeze["frozen_at"] >= evaluation["evaluation_started_at"]:
        raise ValueError("evaluation started before score freeze")
    if [r["dataset"] for r in evaluation["rows"]] != [r["dataset_name"] for r in protocol["catalog"]]:
        raise ValueError("evaluation lost or reordered catalog entries")
    return protocol, freeze, evaluation


def raw_batch_replay(job):
    entry, database, expected = job
    run.set_num_threads(1)
    with run.open_source_database(Path(database)) as connection:
        prepared = run.load_dataset_values(connection, entry["dataset_name"], entry["availability"])
    train, test = prepared.train_values.astype(np.float32), prepared.test_values.astype(np.float32)
    for seed, saved in zip(run.FIXED_SEEDS, expected, strict=True):
        fitted = run.ROCKAD(n_jobs=1, random_state=seed).fit(train)
        whole = np.asarray(fitted.predict(test))
        batched = np.concatenate([fitted.predict(test[i:i+17]) for i in range(0, len(test), 17)])
        if not np.array_equal(whole, saved["test_scores"]) or not np.allclose(whole, batched, rtol=0, atol=1e-10):
            raise ValueError("ranking-only repeat or real split-batch check failed")
    return entry["dataset_name"]


def verify(folder, database):
    protocol, freeze, evaluation = validate_snapshot(folder, database)
    raw_jobs, checked = [], 0
    # Independent sorted-search p-values are not an alternative official score backend.
    rng = np.random.default_rng(20261014)
    reference, query = rng.normal(size=1000), rng.normal(size=5000)
    begin = time.perf_counter()
    accelerated = run.empirical_upper_pvalues(reference, query)
    numba_seconds = time.perf_counter()-begin
    begin = time.perf_counter()
    independent = (1 + len(reference) - np.searchsorted(np.sort(reference), query, side="left"))/(len(reference)+1)
    numpy_seconds = time.perf_counter()-begin
    if not np.array_equal(accelerated, independent):
        raise ValueError("independent p-value formula differs")
    with run.open_source_database(database) as connection:
        for position, (frozen, evaluated) in enumerate(zip(freeze["score_files"], evaluation["rows"], strict=True)):
            raw = run.read_score(folder / frozen["path"])
            if any(evaluated[field] != raw[field] for field in ("dataset", "family", "n_train", "n_test")):
                raise ValueError("evaluation population identity or dimensions changed")
            successful = {m: v for m, v in raw["models"].items() if v["status"] in ("ok", "ranking_only")}
            if not successful:
                continue
            if evaluated.get("evaluation_error"):
                # Do not turn unsupported labels into a successful scored evaluation.
                continue
            labels = run.load_test_labels(connection, raw["dataset"], tuple(raw["test_indices"]))
            for method, model in successful.items():
                if tuple(s["seed"] for s in model["seed_results"]) != run.FIXED_SEEDS:
                    raise ValueError("seed alignment changed")
                median = np.median([s["test_scores"] for s in model["seed_results"]], axis=0)
                by_index = {r["series_index"]: r for r in model["rows"]}
                if len(by_index) != raw["n_test"] or set(by_index) != set(raw["test_indices"]):
                    raise ValueError("retained TEST identity mismatch")
                rows = [by_index[i] for i in raw["test_indices"]]
                if not np.isfinite(median).all() or not np.array_equal(median, [r["score"] for r in rows]):
                    raise ValueError("median scores are not finite or aligned")
                y = np.array([labels[i] for i in raw["test_indices"]])
                ranked = evaluated["models"][method]["ranking"]
                if len(set(y)) == 2:
                    for field, expected in (("auc", roc_auc_score(y, median)), ("ap", average_precision_score(y, median))):
                        if abs(ranked[field]-expected) > 1e-12:
                            raise ValueError("ranking metric differs")
                elif ranked != {"auc": None, "ap": None}:
                    raise ValueError("undefined ranking metric was treated as a number")
                if model["status"] == "ranking_only":
                    raw_jobs.append((protocol["catalog"][position], str(database), model["seed_results"]))
                else:
                    train = np.median([s["train_oof"] for s in model["seed_results"]], axis=0)
                    if len(train) != raw["n_train"] or not np.isfinite(train).all():
                        raise ValueError("OOF alignment mismatch")
                    p = (1 + len(train) - np.searchsorted(np.sort(train), median, side="left"))/(len(train)+1)
                    if not np.array_equal(p, [r["p_value"] for r in rows]):
                        raise ValueError("independent p-values differ")
                    prediction = p <= max(run.ALPHA, 1/(len(train)+1))+1e-15
                    if not np.array_equal(prediction, [r["automatic_anomaly"] for r in rows]):
                        raise ValueError("sample-size decision policy differs")
                    for policy, mask in (("service_policy", prediction), ("strict_alpha_policy", p <= run.ALPHA+1e-15)):
                        counts = Counter((int(label), bool(flag)) for label, flag in zip(y, mask, strict=True))
                        expected = {"true_positives": counts[1, True], "false_positives": counts[0, True],
                                    "true_negatives": counts[0, False], "false_negatives": counts[1, False]}
                        actual = evaluated["models"][method][policy]
                        if any(expected[k] != actual[k] for k in expected):
                            raise ValueError("independent confusion counts differ")
                checked += 1
    with ProcessPoolExecutor(max_workers=protocol["workers"]) as executor:
        raw_replayed = list(executor.map(raw_batch_replay, raw_jobs))
    trace_position = next(i for i, r in enumerate(protocol["catalog"]) if r["dataset_name"] == "Trace_normal_2")
    original = run.read_score(folder / freeze["score_files"][trace_position]["path"])
    repeated = run.score_dataset(protocol["catalog"][trace_position], str(database))
    for method in ("service", "rockad"):
        if repeated["models"][method].get("rows") != original["models"][method].get("rows") or repeated["models"][method].get("seed_results") != original["models"][method].get("seed_results"):
            raise ValueError("Trace actual two-model repeat differs")
    result = dict(evaluation, effective_datasets=freeze.get("effective_datasets", len(protocol["catalog"])),
                  excluded_datasets=freeze.get("excluded_datasets", []),
                  ranking_comparison=ranking_summary(evaluation["rows"]),
                  comparison_notes={"same_prepared_inputs": True, "baseline_internal_reference_cap": 1000,
                    "rockad_reference_cap": None, "equal_compute_budget_claimed": False,
                    "TRAIN_database_labels": "all 170937 stored TRAIN labels are encoded normal 0; original class identities are not changed"},
                  verification={"verified_at": run.now(), "status": "passed", "checked_model_outputs": checked,
                    "ranking_only_real_split_batch_and_repeat": raw_replayed, "Trace_normal_2_repeat": "both models, both seeds, exact",
                    "independent_verifier_backend": "NumPy sorted search; official p-values remain Numba",
                    "numba_synthetic_seconds_including_first_call": numba_seconds, "numpy_synthetic_seconds": numpy_seconds},
                  coverage={method: dict(Counter(r["models"].get(method, {}).get("status", "preparation_error") for r in evaluation["rows"])) for method in ("service", "rockad")})
    atomic_json(folder / "publication_results_antigravity.json", result)
    stream = io.StringIO(newline="")
    fields = ["dataset", "family", "n_train", "n_test", "method", "status", "auc", "ap", "TP", "FP", "TN", "FN", "strict_TP", "strict_FP", "strict_FN", "reason"]
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in result["rows"]:
        for method in ("service", "rockad"):
            model = row["models"].get(method, {})
            ranked, policy, strict = model.get("ranking", {}), model.get("service_policy", {}), model.get("strict_alpha_policy", {})
            writer.writerow(dict(dataset=row["dataset"], family=row["family"], n_train=row["n_train"], n_test=row["n_test"], method=method,
                status=model.get("status", "preparation_error"), auc=ranked.get("auc"), ap=ranked.get("ap"),
                TP=policy.get("true_positives"), FP=policy.get("false_positives"), TN=policy.get("true_negatives"), FN=policy.get("false_negatives"),
                strict_TP=strict.get("true_positives"), strict_FP=strict.get("false_positives"), strict_FN=strict.get("false_negatives"),
                reason=row.get("preparation_error") or row.get("evaluation_error") or model.get("reason")))
    atomic_text(folder / "publication_metrics_antigravity.csv", stream.getvalue())
    run.dashboard(run.DEFAULT_DASHBOARD, len(protocol["catalog"]), len(protocol["catalog"]), result, "verified")
    ranks = result["ranking_comparison"]
    rank_rows = "".join(f"<tr><th>{label}</th><td>{ranks['service'][key]:.4f}</td><td>{ranks['rockad'][key]:.4f}</td></tr>"
                        for key, label in (("dataset_mean_auc", "데이터셋 평균 AUROC"), ("family_mean_auc", "가족 균형 AUROC"),
                                           ("dataset_mean_ap", "데이터셋 평균 AP"), ("family_mean_ap", "가족 균형 AP"))
                        if ranks["service"][key] is not None and ranks["rockad"][key] is not None)
    extra = f"<section><h2>자동 판정과 분리한 전체 순위 비교: {ranks['paired_datasets']}개</h2><p>TRAIN 5~6개에서 순위만 평가한 경우도 포함합니다. 실제 이상을 탐지했다고 주장하거나 FN/FP 합계에 넣지는 않습니다.</p><table><tr><th>지표</th><th>서비스</th><th>ROCKAD</th></tr>{rank_rows}</table><h3>전체 결과 상태</h3><pre>{run.escape(json.dumps(result['coverage'],ensure_ascii=False,indent=2))}</pre></section>"
    html = run.DEFAULT_DASHBOARD.read_text()
    if result["excluded_datasets"]:
        html = html.replace("1,119 / 1,119 처리", "1,118 / 1,118 대상 처리 · 1개 사용자 제외")
        html = html.replace("등록되지 않은 CornellWhaleChallenge는 기존 값 기반 정상 TRAIN 전처리로 ROCKAD를 단독 평가합니다.",
                            "CornellWhaleChallenge는 사용자 요청으로 계산을 중단하고 성능 집계에서 제외했습니다. 실제 실행 대상은 1,118개이며, 제외 기록은 원래 DB 목록과 함께 보존합니다.")
        html = html.replace("python experiments/aeon_rockad_full_comparison_antigravity/run_full_antigravity.py run --database /path/univariate_ts.db",
                            "python experiments/aeon_rockad_full_comparison_antigravity/finalize_antigravity.py --exclude-cornell --database /path/univariate_ts.db (점수 처리 후 최초 고정 시에만); 이후 검증은 --exclude-cornell 없이 실행")
    html = html.replace("<section><h2>범위와 비교 조건</h2>", extra+"<section><h2>범위와 비교 조건</h2>")
    html = html.replace("중단 후 같은 명령을 사용하면 완료된 체크포인트를 보존하고 미처리 항목만 이어갑니다.",
                        "점수 파일과 중간 평가 파일은 최종 결과로 간주하지 않습니다. 업로드 전에는 finalize_antigravity.py로 전체 해시·평가 연결·실제 반복 실행을 검증합니다. 체크포인트가 손상되면 자동 재개하지 않고 중단합니다.")
    html = html.replace("이것은 현행 등록 모델과 고정 ROCKAD 기본 설정의 비교입니다.",
                        "두 방법에 같은 전처리된 정상 입력을 제공합니다. 서비스 내부 참조 상한 1,000개와 ROCKAD의 전체 TRAIN 사용은 각 알고리즘의 설정 그대로 유지하며, 같은 계산 예산이라고 주장하지 않습니다. 이것은 현행 등록 모델과 고정 ROCKAD 기본 설정의 비교입니다.")
    atomic_text(run.DEFAULT_DASHBOARD, html)
    print(json.dumps({"coverage": result["coverage"], "service_policy": result["service_policy"], "ranking_comparison": result["ranking_comparison"]}, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", default=str(run.DEFAULT_DB))
    parser.add_argument("--output", default=str(run.DEFAULT_OUT))
    parser.add_argument("--exclude-cornell", action="store_true", help="apply the user-approved scope amendment once, then freeze and evaluate")
    arguments = parser.parse_args()
    if arguments.exclude_cornell:
        exclude_cornell_and_freeze(Path(arguments.output), Path(arguments.database))
        run.evaluate(Path(arguments.output), Path(arguments.database))
    verify(Path(arguments.output), Path(arguments.database))
