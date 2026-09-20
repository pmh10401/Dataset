"""Smoke and official paired MiniRocket LocalGap vs LOF scoring."""
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
from contextlib import closing
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

from experiments.exp479_missing_score_recovery_antigravity.evaluation_antigravity import (
    evaluate_score_vector_antigravity,
    load_retained_test_labels_antigravity,
)
from experiments.exp479_missing_score_recovery_antigravity.preprocessing_antigravity import (
    prepare_dataset_values_antigravity,
)
from experiments.exp489_lof_score_head_antigravity.lof_score_head_antigravity import (
    score_heads_antigravity,
)
from experiments.exp491_gain_prediction_antigravity.runtime_antigravity import (
    DATABASE_PATH_ANTIGRAVITY,
    EXCLUDED_FIVEFOLD_FAMILIES_ANTIGRAVITY,
    EXPECTED_REPRESENTATIVE_FAMILIES_ANTIGRAVITY,
    FROZEN_PREPROCESS_PATHS_ANTIGRAVITY,
    FROZEN_SCORER_PATHS_ANTIGRAVITY,
    ROOT_ANTIGRAVITY,
    SEED_ANTIGRAVITY,
    assess_acceleration_antigravity,
    assert_unbound_wal_antigravity,
    cpu_worker_policy_antigravity,
    dataset_stem_antigravity,
    file_identity_antigravity,
    file_sha256_antigravity,
    jsonable_antigravity,
    load_frozen_sources_antigravity,
    readonly_sqlite_antigravity,
    refuse_official_overwrite_antigravity,
    representative_rows_antigravity,
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
    PACKAGE_ANTIGRAVITY / "lof_score_head_antigravity.py",
    ROOT_ANTIGRAVITY / "experiments/exp491_gain_prediction_antigravity/runtime_antigravity.py",
    *FROZEN_SCORER_PATHS_ANTIGRAVITY,
    *FROZEN_PREPROCESS_PATHS_ANTIGRAVITY,
)


def _binding_antigravity(database, source_sha):
    return source_binding_antigravity(
        code_paths=CODE_PATHS_ANTIGRAVITY,
        database=database,
        source_sha=source_sha,
        extra={"experiment": "exp489_lof_score_head"},
    )


def _score_paths_antigravity(score_dir: Path, dataset: str) -> tuple[Path, Path]:
    stem = dataset_stem_antigravity(dataset)
    return score_dir / f"{stem}_scores_antigravity.npz", score_dir / f"{stem}_meta_antigravity.json"


def persist_score_checkpoint_antigravity(score_dir: Path, dataset: str, heads: dict, extra: dict) -> dict:
    score_dir.mkdir(parents=True, exist_ok=True)
    npz_path, meta_path = _score_paths_antigravity(score_dir, dataset)
    temporary = npz_path.with_name(npz_path.stem + "_temporary_antigravity.npz")
    np.savez_compressed(
        temporary,
        control_train_oof=np.asarray(heads["control"]["train_oof"], dtype=float),
        control_test_scores=np.asarray(heads["control"]["test_scores"], dtype=float),
        lof_train_oof=np.asarray(heads["lof"]["train_oof"], dtype=float),
        lof_test_scores=np.asarray(heads["lof"]["test_scores"], dtype=float),
    )
    temporary.replace(npz_path)
    meta = {
        "dataset": dataset,
        "binding": extra["binding"],
        "feature_dimension": int(heads["feature_dimension"]),
        "representation_fits": int(heads["representation_fits"]),
        "score_sha256": file_sha256_antigravity(npz_path),
        "n_train": int(len(heads["control"]["train_oof"])),
        "n_test": int(len(heads["control"]["test_scores"])),
        **{key: value for key, value in extra.items() if key != "binding"},
    }
    write_json_atomic_antigravity(meta_path, jsonable_antigravity(meta))
    return meta


def load_score_checkpoint_antigravity(score_dir: Path, dataset: str, binding) -> dict | None:
    npz_path, meta_path = _score_paths_antigravity(score_dir, dataset)
    if not npz_path.exists() or not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("binding") != binding or meta.get("score_sha256") != file_sha256_antigravity(npz_path):
        raise ValueError(f"stale or mixed-version score checkpoint for {dataset}")
    with np.load(npz_path, allow_pickle=False) as payload:
        arrays = {key: np.asarray(payload[key], dtype=float) for key in (
            "control_train_oof", "control_test_scores", "lof_train_oof", "lof_test_scores")}
    if meta.get("dataset") != dataset:
        raise ValueError("score checkpoint dataset mismatch")
    for key, array in arrays.items():
        count = meta.get("n_train") if key.endswith("train_oof") else meta.get("n_test")
        if type(count) is not int or count < 1 or array.shape != (count,) or not np.isfinite(array).all():
            raise ValueError("invalid score checkpoint vector")
    return {
        "meta": meta,
        "heads": {
            "feature_dimension": meta["feature_dimension"],
            "representation_fits": meta["representation_fits"],
            "control": {
                "train_oof": arrays["control_train_oof"],
                "test_scores": arrays["control_test_scores"],
            },
            "lof": {
                "train_oof": arrays["lof_train_oof"],
                "test_scores": arrays["lof_test_scores"],
            },
        },
    }


def _evaluate_heads_antigravity(connection, dataset, test_indices, heads) -> dict:
    labels = load_retained_test_labels_antigravity(connection, dataset, test_indices)
    control = evaluate_score_vector_antigravity(heads["control"]["test_scores"], labels)
    lof = evaluate_score_vector_antigravity(heads["lof"]["test_scores"], labels)
    return {
        "control": {"auc": float(control["auc"]), "zf": int(control["zf"])},
        "lof": {"auc": float(lof["auc"]), "zf": int(lof["zf"])},
        "delta_auc": float(lof["auc"]) - float(control["auc"]),
        "delta_zf": int(lof["zf"]) - int(control["zf"]),
    }


def score_dataset_worker_antigravity(payload: dict) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    database = Path(payload["database"])
    dataset = payload["dataset"]
    score_dir = Path(payload["score_dir"])
    binding = payload["binding"]
    started = perf_counter()
    with closing(readonly_sqlite_antigravity(database)) as connection:
        prepared = prepare_dataset_values_antigravity(connection, dataset)
        cached = None if payload.get("recompute") else load_score_checkpoint_antigravity(score_dir, dataset, binding)
        if cached is None:
            heads = score_heads_antigravity(prepared.train_rows, prepared.test_rows, seed=SEED_ANTIGRAVITY)
            persist_score_checkpoint_antigravity(
                score_dir,
                dataset,
                heads,
                {
                    "binding": binding,
                    "source_family": payload["source_family"],
                    "test_indices": list(prepared.test_indices),
                    "target_length": int(prepared.target_length),
                },
            )
        else:
            heads = cached["heads"]
            if (cached["meta"].get("test_indices") != list(prepared.test_indices)
                    or cached["meta"].get("n_train") != len(prepared.train_rows)
                    or cached["meta"].get("n_test") != len(prepared.test_rows)):
                raise ValueError("score checkpoint row alignment changed")
        metrics = _evaluate_heads_antigravity(connection, dataset, prepared.test_indices, heads)
    npz_path, _meta_path = _score_paths_antigravity(score_dir, dataset)
    return jsonable_antigravity(
        {
            "dataset": dataset,
            "source_family": payload["source_family"],
            "n_train": int(len(prepared.train_rows)),
            "n_test": int(len(prepared.test_rows)),
            "feature_dimension": int(heads["feature_dimension"]),
            "score_sha256": file_sha256_antigravity(npz_path),
            "seconds": perf_counter() - started,
            **metrics,
        }
    )


def validate_score_records_antigravity(records, representatives, score_dir, binding, *, complete=False):
    expected = {row.dataset: row.source_family for row in representatives}
    seen = set()
    for record in records:
        dataset = record.get("dataset")
        if dataset in seen or dataset not in expected or record.get("source_family") != expected[dataset]:
            raise ValueError("duplicate or unexpected score checkpoint dataset")
        seen.add(dataset)
        cached = load_score_checkpoint_antigravity(score_dir, dataset, binding)
        if (cached is None or cached["meta"]["score_sha256"] != record.get("score_sha256")
                or cached["meta"].get("source_family") != expected[dataset]):
            raise ValueError("score checkpoint artifact missing or changed")
    if complete and seen != set(expected):
        raise ValueError("score checkpoint population is incomplete")


def summarize_paired_antigravity(records: list[dict]) -> dict:
    if not records:
        raise ValueError("paired records are required")
    auc_wins = sum(row["delta_auc"] > 0 for row in records)
    auc_losses = sum(row["delta_auc"] < 0 for row in records)
    zf_wins = sum(row["delta_zf"] > 0 for row in records)
    zf_losses = sum(row["delta_zf"] < 0 for row in records)
    return {
        "dataset_count": len(records),
        "mean_auc_control": float(np.mean([row["control"]["auc"] for row in records])),
        "mean_auc_lof": float(np.mean([row["lof"]["auc"] for row in records])),
        "total_zf_control": int(sum(row["control"]["zf"] for row in records)),
        "total_zf_lof": int(sum(row["lof"]["zf"] for row in records)),
        "mean_delta_auc": float(np.mean([row["delta_auc"] for row in records])),
        "mean_delta_zf": float(np.mean([row["delta_zf"] for row in records])),
        "auc_wins": int(auc_wins),
        "auc_losses": int(auc_losses),
        "auc_ties": int(len(records) - auc_wins - auc_losses),
        "zf_wins": int(zf_wins),
        "zf_losses": int(zf_losses),
        "zf_ties": int(len(records) - zf_wins - zf_losses),
        "auc_harm_families": [row["source_family"] for row in records if row["delta_auc"] < 0],
        "zf_harm_families": [row["source_family"] for row in records if row["delta_zf"] < 0],
        "baseline_policy": "freshly_computed_paired_localgap_control; historical_score_cache_never_substituted",
    }


def render_report_antigravity(result: dict, manifest: dict) -> str:
    summary = result["summary"]
    excluded = ", ".join(result.get("excluded_families") or ())
    rows = "".join(
        "<tr>"
        f"<td><code>{escape(str(row['dataset']))}</code></td>"
        f"<td><code>{escape(str(row['source_family']))}</code></td>"
        f"<td>{row['control']['auc']:.6f}</td><td>{row['lof']['auc']:.6f}</td>"
        f"<td>{row['delta_auc']:.6f}</td>"
        f"<td>{row['control']['zf']}</td><td>{row['lof']['zf']}</td>"
        f"<td>{row['delta_zf']}</td>"
        "</tr>"
        for row in result.get("records", [])
    )
    return f"""<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Exp489 결과</title>
<style>body{{font:16px/1.7 system-ui;max-width:1100px;margin:32px auto;padding:0 20px;color:#172033}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccd3dc;padding:8px;text-align:left}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f5f8;padding:16px}}</style>
<h1>Exp489: MiniRocket LocalGap 대 LOF</h1>
<p>고정 MiniRocket 인코더를 폴드마다 한 번만 맞추고, 같은 특징과 RobustScaler(10,90)로 LocalGap 대조와 LOF 머리를 쌍으로 계산했습니다. 점수를 저장한 뒤에만 TEST 라벨을 열었습니다. 대상은 n_train≥5인 family의 사전식 첫 데이터셋 {summary['dataset_count']}개입니다. 제외 family: {escape(excluded)}.</p>
<p>평균 AUC 대조 {summary['mean_auc_control']:.6f}, LOF {summary['mean_auc_lof']:.6f}. ZF 합 대조 {summary['total_zf_control']}, LOF {summary['total_zf_lof']}. AUC 승/패/무 {summary['auc_wins']}/{summary['auc_losses']}/{summary['auc_ties']}, ZF 승/패/무 {summary['zf_wins']}/{summary['zf_losses']}/{summary['zf_ties']}.</p>
<p>ZF는 평가 정상 최댓값을 엄격히 넘는 이상만 셉니다. 운영 영-오탐 보장이 아닙니다. 한 팔만 비교했고 튜닝은 없습니다. 과거 점수 캐시를 대조로 바꿔 넣지 않았습니다.</p>
<table><thead><tr><th>데이터셋</th><th>family</th><th>대조 AUC</th><th>LOF AUC</th><th>ΔAUC</th><th>대조 ZF</th><th>LOF ZF</th><th>ΔZF</th></tr></thead><tbody>{rows}</tbody></table>
<h2>제한</h2>
<p>이 값은 선택된 125개 대표 데이터셋의 회고 평가입니다. 새 family 일반화나 라우팅 승격을 의미하지 않습니다.</p>
<h2>재현</h2>
<pre>PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m experiments.exp489_lof_score_head_antigravity.run_antigravity smoke
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m experiments.exp489_lof_score_head_antigravity.run_antigravity official</pre>
<pre>{escape(json.dumps(manifest, ensure_ascii=False, indent=2))}</pre>
</html>"""


def _synthetic_rows_antigravity():
    rng = np.random.default_rng(SEED_ANTIGRAVITY)
    train = rng.normal(size=(12, 24)).astype(np.float32)
    test = rng.normal(size=(6, 24)).astype(np.float32)
    train += np.linspace(0.0, 1.0, 12, dtype=np.float32)[:, None]
    return train, test


def run(mode: str, database=DATABASE_PATH_ANTIGRAVITY, output=PACKAGE_ANTIGRAVITY, rows=None):
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
    def current_binding():
        current_sources = load_frozen_sources_antigravity()[2] if rows is None else source_sha
        return _binding_antigravity(database, current_sources)
    binding = _binding_antigravity(database, source_sha)
    if current_binding() != binding:
        raise RuntimeError("source changed while loading inputs")
    if mode == "smoke":
        acceleration = assess_acceleration_antigravity()
        train, test = _synthetic_rows_antigravity()
        first = score_heads_antigravity(train, test, seed=SEED_ANTIGRAVITY)
        second = score_heads_antigravity(train, test, seed=SEED_ANTIGRAVITY)
        if first["representation_fits"] != 6:
            raise RuntimeError("paired heads must share one encoder fit per fold and full reference")
        for head in ("control", "lof"):
            np.testing.assert_array_equal(first[head]["train_oof"], second[head]["train_oof"])
            np.testing.assert_array_equal(first[head]["test_scores"], second[head]["test_scores"])
        smoke_dir = output / f"smoke_scores_{sha256_json_antigravity(binding)[:16]}_antigravity"
        repeated = None
        if representatives:
            sample = representatives[0]
            payload = dict(
                database=str(database),
                dataset=sample.dataset,
                source_family=sample.source_family,
                score_dir=str(smoke_dir),
                binding=binding,
                recompute=True,
            )
            first_real = score_dataset_worker_antigravity(payload)
            repeat_dir = smoke_dir / "independent_repeat_antigravity"
            score_dataset_worker_antigravity({**payload, "score_dir": str(repeat_dir)})
            a = load_score_checkpoint_antigravity(smoke_dir, sample.dataset, binding)["heads"]
            b = load_score_checkpoint_antigravity(repeat_dir, sample.dataset, binding)["heads"]
            for head in ("control", "lof"):
                for kind in ("train_oof", "test_scores"):
                    np.testing.assert_array_equal(a[head][kind], b[head][kind])
            repeated = first_real
        if before is not None and before != file_identity_antigravity(database):
            raise RuntimeError("raw database changed during smoke")
        if current_binding() != binding:
            raise RuntimeError("source or implementation changed during smoke")
        result = {
            "state": "smoke_nonofficial",
            "binding": binding,
            "acceleration": acceleration,
            "synthetic_repeatable": True,
            "representation_fits": first["representation_fits"],
            "excluded_families": list(excluded),
            "representative_repeat": repeated,
            "cpu_policy": workers,
            "elapsed_seconds": perf_counter() - started,
        }
        write_json_atomic_antigravity(output / "smoke_results_antigravity.json", jsonable_antigravity(result))
        write_text_atomic_antigravity(
            output / "smoke_report_antigravity.html",
            render_report_antigravity(
                {
                    "summary": summarize_paired_antigravity([repeated]),
                    "excluded_families": list(excluded),
                    "records": [] if repeated is None else [repeated],
                },
                {"mode": "smoke", "binding": binding, "acceleration": acceleration},
            ),
        )
        print(json.dumps({"state": result["state"], "output": str(output)}), flush=True)
        return result

    require_matching_smoke_antigravity(output / "smoke_results_antigravity.json", binding)
    score_dir = output / "score_checkpoints_antigravity"
    checkpoint_path = output / "score_checkpoint_index_antigravity.json"
    records = []
    if checkpoint_path.exists():
        existing = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if existing.get("binding") != binding:
            raise ValueError("score checkpoint binding mismatch")
        records = list(existing.get("records") or [])
        if existing.get("records_sha256") != sha256_json_antigravity(records):
            raise ValueError("score checkpoint index checksum mismatch")
    validate_score_records_antigravity(records, representatives, score_dir, binding)
    done = {row["dataset"] for row in records}
    pending = [row for row in representatives if row.dataset not in done]
    errors = []
    with ProcessPoolExecutor(max_workers=workers["workers"]) as pool:
        futures = {
            pool.submit(
                score_dataset_worker_antigravity,
                dict(
                    database=str(database),
                    dataset=row.dataset,
                    source_family=row.source_family,
                    score_dir=str(score_dir),
                    binding=binding,
                ),
            ): row.dataset
            for row in pending
        }
        for future in as_completed(futures):
            dataset = futures[future]
            try:
                records.append(future.result())
            except Exception as error:
                errors.append({"dataset": dataset, "error": f"{type(error).__name__}: {error}"})
            write_json_atomic_antigravity(
                checkpoint_path,
                jsonable_antigravity(
                    {
                        "binding": binding,
                        "records": records,
                        "records_sha256": sha256_json_antigravity(records),
                        "errors": errors,
                        "complete": False,
                    }
                ),
            )
    records.sort(key=lambda row: row["dataset"])
    if errors or len(records) != len(representatives):
        write_json_atomic_antigravity(
            checkpoint_path,
            jsonable_antigravity({"binding": binding, "records": records, "records_sha256": sha256_json_antigravity(records), "errors": errors, "complete": False}),
        )
        raise RuntimeError(
            f"partial Exp489 run: completed {len(records)}/{len(representatives)}, failed {len(errors)}"
        )
    validate_score_records_antigravity(records, representatives, score_dir, binding, complete=True)
    summary = summarize_paired_antigravity(records)
    result = {
        "schema": "exp489_lof_score_head_v1",
        "state": "official_complete",
        "binding": binding,
        "excluded_families": list(excluded),
        "expected_families": EXPECTED_REPRESENTATIVE_FAMILIES_ANTIGRAVITY,
        "records": records,
        "summary": summary,
    }
    manifest = {
        "schema": "exp489_run_manifest_v1",
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": perf_counter() - started,
        "cpu_policy": workers,
        "completed_datasets": len(records),
        "expected_datasets": len(representatives),
        "failed_datasets": 0,
        "excluded_families": list(excluded),
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
    write_json_atomic_antigravity(
        checkpoint_path,
        jsonable_antigravity({"binding": binding, "records": records, "records_sha256": sha256_json_antigravity(records), "errors": [], "complete": True}),
    )
    write_json_atomic_antigravity(official_path, jsonable_antigravity(result))
    print(json.dumps({"state": result["state"], "output": str(output)}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "official"))
    parser.add_argument("--database", type=Path, default=DATABASE_PATH_ANTIGRAVITY)
    parser.add_argument("--output", type=Path, default=PACKAGE_ANTIGRAVITY)
    args = parser.parse_args()
    run(args.mode, args.database, args.output)
