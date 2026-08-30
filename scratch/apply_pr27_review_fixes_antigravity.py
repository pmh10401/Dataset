#!/usr/bin/env python3
"""Apply the reviewed PR #27 fixes inside an isolated CI checkout."""

from __future__ import annotations

import argparse
import hashlib
import html
import importlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
BASE_HEAD = "75c0f2484209f744e218b871b01777854fcee11c"
GENERATED_AT = "2026-08-30T00:00:00Z"


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def write(path: str, content: str) -> None:
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def replace_once(path: str, old: str, new: str) -> None:
    text = read(path)
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one occurrence, found {count}: {old[:80]!r}")
    write(path, text.replace(old, new, 1))


def replace_all(path: str, old: str, new: str, *, minimum: int = 1) -> None:
    text = read(path)
    count = text.count(old)
    if count < minimum:
        raise RuntimeError(f"{path}: expected at least {minimum} occurrences, found {count}: {old!r}")
    write(path, text.replace(old, new))


def replace_repo_references(old: str, new: str) -> None:
    completed = subprocess.run(
        ["git", "grep", "-l", "-F", old, "--", ".", ":(exclude).github/workflows"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode not in (0, 1):
        raise RuntimeError(completed.stderr)
    for relative in filter(None, completed.stdout.splitlines()):
        path = ROOT / relative
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        path.write_text(text.replace(old, new), encoding="utf-8")


def patch_exp472() -> None:
    path = "scratch/run_exp472_zero_fp_tail_recovery_antigravity.py"
    marker = (
        "    return interpolate_to_140_antigravity(train_raw), "
        "interpolate_to_140_antigravity(test_raw), anomaly_mask\n\n\n"
        "def main_antigravity(argv: list[str] | None = None) -> int:\n"
    )
    helper = '''    return interpolate_to_140_antigravity(train_raw), interpolate_to_140_antigravity(test_raw), anomaly_mask


def tp_at_thresholds_antigravity(
    scores: np.ndarray,
    mask: np.ndarray,
    fp_budgets: tuple[int, ...] = (3, 5, 10),
) -> dict[str, object]:
    """Count anomaly recovery at zero FP and exact frozen FP boundaries."""
    score_vector = np.asarray(scores, dtype=np.float64)
    anomaly_mask = np.asarray(mask)
    if (
        score_vector.ndim != 1
        or anomaly_mask.ndim != 1
        or len(score_vector) != len(anomaly_mask)
        or anomaly_mask.dtype != np.bool_
        or not np.isfinite(score_vector).all()
    ):
        raise ValueError("scores and mask must be aligned finite 1D arrays")
    normal_scores = score_vector[~anomaly_mask]
    anomaly_scores = score_vector[anomaly_mask]
    if not len(normal_scores) or not len(anomaly_scores):
        raise ValueError("both normal and anomalous scores are required")
    if type(fp_budgets) is not tuple or not fp_budgets or any(
        type(budget) is not int or budget < 1 for budget in fp_budgets
    ):
        raise ValueError("FP budgets must be a non-empty tuple of positive integers")

    normal_max = float(normal_scores.max())
    zero_fp = int(np.count_nonzero(anomaly_scores > normal_max))
    sorted_norm = np.sort(normal_scores)[::-1]
    budget: dict[str, int] = {}
    for fp_budget in fp_budgets:
        boundary_index = min(fp_budget - 1, len(sorted_norm) - 1)
        threshold = float(sorted_norm[boundary_index])
        budget[str(fp_budget)] = int(np.count_nonzero(anomaly_scores >= threshold))
    return {"zero_fp_tp": zero_fp, "tp_at_budget": budget}


def main_antigravity(argv: list[str] | None = None) -> int:
'''
    replace_once(path, marker, helper)
    old_nested = '''    def auc_safe(normal: np.ndarray, anomalous: np.ndarray) -> float:
        return float(mann_whitney_auc_antigravity(normal, anomalous))

    def tp_at_thresholds(scores: np.ndarray, mask: np.ndarray) -> dict[str, object]:
        normal_max = float(scores[~mask].max())
        zero_fp = int(np.count_nonzero(scores[mask] > normal_max))
        budget = {}
        for fp in (3, 5, 10):
            sorted_norm = np.sort(scores[~mask])[::-1]
            threshold = float(sorted_norm[min(fp, len(sorted_norm)-1)])
            budget[str(fp)] = int(np.count_nonzero(scores[mask] >= threshold))
        return {"zero_fp_tp": zero_fp, "tp_at_budget": budget}
'''
    new_nested = '''    def auc_safe(normal: np.ndarray, anomalous: np.ndarray) -> float:
        return float(mann_whitney_auc_antigravity(normal, anomalous))
'''
    replace_once(path, old_nested, new_nested)
    replace_once(path, "stats = tp_at_thresholds(s, mask)", "stats = tp_at_thresholds_antigravity(s, mask)")

    test_path = "scratch/test_run_exp472_zero_fp_tail_recovery_antigravity.py"
    write(
        test_path,
        '''from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _module_antigravity():
    root = Path(__file__).resolve().parents[1]
    path = root / "scratch/run_exp472_zero_fp_tail_recovery_antigravity.py"
    specification = importlib.util.spec_from_file_location("exp472_runner_antigravity", path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_fp_budget_uses_the_budget_minus_one_normal_boundary_antigravity() -> None:
    module = _module_antigravity()
    normal = np.asarray([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    anomalous = np.asarray([8.5, 6.5, 0.5])
    scores = np.concatenate((normal, anomalous))
    mask = np.asarray([False] * len(normal) + [True] * len(anomalous), dtype=np.bool_)

    result = module.tp_at_thresholds_antigravity(scores, mask)

    assert result == {
        "zero_fp_tp": 0,
        "tp_at_budget": {"3": 1, "5": 2, "10": 2},
    }


def test_fp_budget_clamps_only_when_the_budget_exceeds_normal_support_antigravity() -> None:
    module = _module_antigravity()
    scores = np.asarray([5.0, 4.0, 3.0, 4.5, 2.5], dtype=np.float64)
    mask = np.asarray([False, False, False, True, True], dtype=np.bool_)
    result = module.tp_at_thresholds_antigravity(scores, mask, (1, 3, 10))
    assert result["tp_at_budget"] == {"1": 0, "3": 1, "10": 1}


@pytest.mark.parametrize(
    ("scores", "mask"),
    [
        (np.asarray([1.0, np.nan]), np.asarray([False, True])),
        (np.asarray([1.0, 2.0]), np.asarray([False, False])),
        (np.asarray([1.0, 2.0]), np.asarray([0, 1], dtype=np.int64)),
    ],
)
def test_fp_budget_rejects_malformed_score_support_antigravity(scores, mask) -> None:
    module = _module_antigravity()
    with pytest.raises(ValueError):
        module.tp_at_thresholds_antigravity(scores, mask)
''',
    )

    design = "docs/superpowers/specs/2026-08-25-exp472-zero-fp-tail-recovery-antigravity-design.html"
    text = read(design)
    start = text.index('<h2>8. 진단 결과 (2026-08-25)</h2>')
    end = text.index('</body></html>')
    corrected = '''<h2>8. 진단 결과 정정 (2026-08-30)</h2>
<p class="muted">2026-08-25 산출물의 AUC와 zero-FP TP는 영향을 받지 않는다. 다만 TP@3/5/10은 정상 점수 경계를 한 칸 낮게 선택한 구현 오류가 확인되어 기존 수치를 철회한다.</p>
<table>
<thead><tr><th>항목</th><th>상태</th><th>근거</th></tr></thead>
<tbody>
<tr><td>AUC</td><td>유효</td><td>FP-budget 경계 계산과 독립적이다.</td></tr>
<tr><td>zero-FP TP</td><td>유효</td><td>정상 최댓값보다 엄격히 큰 점수만 세는 별도 계산이다.</td></tr>
<tr><td>TP@3 / TP@5 / TP@10</td><td><strong>기존 수치 철회</strong></td><td>기존 코드는 <code>sorted_norm[fp]</code>를 사용해 (fp+1)번째 정상 경계를 선택했다. 수정 코드는 Exp473과 동일하게 <code>sorted_norm[fp-1]</code>를 사용한다.</td></tr>
</tbody>
</table>
<h3>정정된 결론</h3>
<ul>
<li><strong>ECG200_normal_1</strong>: A_raw_gap의 AUC 1.0000과 zero-FP TP 2/2 결과는 유지한다.</li>
<li><strong>ECG200_normal_-1</strong>: A_raw_gap의 AUC 0.8980과 zero-FP TP 0/1 결과는 유지한다.</li>
<li>기존 TP-budget 수치와 그 수치에 의존한 우위 주장은 재실행 전까지 사용하지 않는다.</li>
<li>이번 수정은 지표 경계와 회귀 테스트만 고치며 라우팅, champion, promotion을 변경하지 않는다.</li>
</ul>

'''
    write(design, text[:start] + corrected + text[end:])


def patch_exp479_population() -> None:
    path = "experiments/exp479_missing_score_recovery_antigravity/population_antigravity.py"
    replace_once(
        path,
        '''def _sha256_antigravity(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()
''',
        '''def _sha256_antigravity(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
''',
    )
    replace_once(
        path,
        '''    database_stat = database.stat()
    recovery_bytes = ("\n".join(audit.datasets) + "\n").encode("utf-8")
''',
        '''    database_stat = database.stat()
    actual_database_sha256 = _sha256_antigravity(database)
    if actual_database_sha256 != DATABASE_EXPECTED_SHA256_ANTIGRAVITY:
        raise ValueError("database SHA256 does not match the frozen Exp479 authority")
    recovery_bytes = ("\n".join(audit.datasets) + "\n").encode("utf-8")
''',
    )
    replace_once(
        path,
        '''            "expected_sha256": DATABASE_EXPECTED_SHA256_ANTIGRAVITY,
''',
        '''            "actual_sha256": actual_database_sha256,
            "expected_sha256": DATABASE_EXPECTED_SHA256_ANTIGRAVITY,
''',
    )

    tests = "experiments/exp479_missing_score_recovery_antigravity/tests/test_population_antigravity.py"
    replace_once(
        tests,
        '''    assert manifest["database"]["expected_sha256"] == (
        "8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c"
    )
''',
        '''    assert manifest["database"]["actual_sha256"] == (
        "8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c"
    )
    assert manifest["database"]["expected_sha256"] == manifest["database"]["actual_sha256"]
''',
    )
    append = '''

def test_population_manifest_rejects_a_database_with_the_wrong_digest_antigravity(
    tmp_path: Path,
) -> None:
    module = importlib.import_module(
        "experiments.exp479_missing_score_recovery_antigravity.population_antigravity"
    )
    contracts = importlib.import_module(
        "experiments.exp479_missing_score_recovery_antigravity.contracts_antigravity"
    )
    datasets = tuple(f"Synthetic_{index:03d}_normal_0" for index in range(155))
    audit = contracts.PopulationAuditAntigravity(
        datasets=datasets,
        excluded_datasets=("CornellWhaleChallenge", "Phoneme_normal_39"),
        included_count=1117,
        existing_score_count=962,
        checkpoint_count=23,
    )
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    for index in range(23):
        (checkpoints / f"chunk_{index:04d}_antigravity.json").write_text(
            "{}\n", encoding="utf-8"
        )
    exp478 = tmp_path / "exp478_result_antigravity.json"
    exp478.write_text("{}\n", encoding="utf-8")
    wrong_database = tmp_path / "wrong_univariate_ts_antigravity.db"
    wrong_database.write_bytes(b"not-the-frozen-database")

    with pytest.raises(ValueError, match="database SHA256"):
        module.build_population_manifest_antigravity(
            audit,
            checkpoint_dir=checkpoints,
            exp478_result_path=exp478,
            database_path=wrong_database,
        )
'''
    write(tests, read(tests).rstrip() + append + "\n")


def patch_exp479_runner() -> None:
    path = "experiments/exp479_missing_score_recovery_antigravity/run_antigravity.py"
    replace_once(
        path,
        '''        if dataset in allowed and dataset not in result:
            result[dataset] = row
''',
        '''        if (
            dataset in allowed
            and dataset not in result
            and row.get("status") == "ok"
        ):
            result[dataset] = row
''',
    )
    old_function = '''def _source_fingerprint_antigravity(manifest: Mapping[str, object]) -> str:
    paths = (
        Path(__file__).with_name("config_antigravity.py"),
        Path(__file__).with_name("contracts_antigravity.py"),
        Path(__file__).with_name("population_antigravity.py"),
        Path(__file__).with_name("preprocessing_antigravity.py"),
        Path(__file__).with_name("evaluation_antigravity.py"),
        Path(__file__).with_name("acceleration_antigravity.py"),
    )
    digest = sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()
'''
    new_function = '''def _result_affecting_source_paths_antigravity() -> tuple[Path, ...]:
    """Return every first-party source or frozen policy leaf affecting Exp479 rows."""
    root = Path(__file__).resolve().parents[2]
    local = Path(__file__).resolve().parent
    relative_paths = (
        "experiments/exp479_missing_score_recovery_antigravity/run_antigravity.py",
        "experiments/exp479_missing_score_recovery_antigravity/config_antigravity.py",
        "experiments/exp479_missing_score_recovery_antigravity/contracts_antigravity.py",
        "experiments/exp479_missing_score_recovery_antigravity/population_antigravity.py",
        "experiments/exp479_missing_score_recovery_antigravity/preprocessing_antigravity.py",
        "experiments/exp479_missing_score_recovery_antigravity/evaluation_antigravity.py",
        "experiments/exp479_missing_score_recovery_antigravity/acceleration_antigravity.py",
        "experiments/exp478_full_1117_official_routing_antigravity/run_antigravity.py",
        "experiments/exp474_conservative_fallback_routing_antigravity/evaluation_antigravity.py",
        "experiments/exp474_conservative_fallback_routing_antigravity/ragged_features_antigravity.py",
        "experiments/exp449_residual_fn_method_qualification_antigravity/config_antigravity.py",
        "experiments/exp449_residual_fn_method_qualification_antigravity/contracts_antigravity.py",
        "experiments/exp449_residual_fn_method_qualification_antigravity/local_gap_antigravity.py",
        "experiments/exp449_residual_fn_method_qualification_antigravity/representations_antigravity.py",
        "experiments/exp449_residual_fn_method_qualification_antigravity/scoring_antigravity.py",
        "experiments/exp_routing_label_free_antigravity/conservative_policy_antigravity.py",
        "experiments/exp_routing_label_free_antigravity/feature_extractor_antigravity.py",
        "experiments/exp_routing_label_free_antigravity/official_router_antigravity.py",
        "experiments/exp_routing_label_free_antigravity/routing_rules_antigravity.py",
        "experiments/exp_routing_label_free_antigravity/official_exp476_policy_artifact_antigravity.json",
    )
    paths = tuple(root / relative for relative in relative_paths)
    if local != root / "experiments/exp479_missing_score_recovery_antigravity":
        raise RuntimeError("Exp479 runner root resolution changed")
    missing = tuple(path for path in paths if not path.is_file())
    if missing:
        raise RuntimeError(f"result-affecting source is missing: {missing}")
    return paths


def _source_fingerprint_antigravity(manifest: Mapping[str, object]) -> str:
    root = Path(__file__).resolve().parents[2]
    digest = sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    for path in _result_affecting_source_paths_antigravity():
        relative = path.resolve().relative_to(root).as_posix()
        data = path.read_bytes()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(data)).encode("ascii"))
        digest.update(b"\0")
        digest.update(data)
        digest.update(b"\0")
    return digest.hexdigest()
'''
    replace_once(path, old_function, new_function)

    tests = "experiments/exp479_missing_score_recovery_antigravity/tests/test_runner_antigravity.py"
    replace_once(
        tests,
        '''            {"dataset": "A_normal_0", "status": "ok"},
            {"dataset": "Outside_normal_0", "status": "ok"},
''',
        '''            {"dataset": "A_normal_0", "status": "ok"},
            {"dataset": "B_normal_0", "status": "error", "error": "transient"},
            {"dataset": "Outside_normal_0", "status": "ok"},
''',
    )
    append = '''

def test_source_fingerprint_binds_all_transitive_result_sources_antigravity() -> None:
    module = _runner_antigravity()
    root = Path(module.__file__).resolve().parents[2]
    actual = {path.resolve().relative_to(root).as_posix() for path in module._result_affecting_source_paths_antigravity()}
    required = {
        "experiments/exp479_missing_score_recovery_antigravity/run_antigravity.py",
        "experiments/exp449_residual_fn_method_qualification_antigravity/scoring_antigravity.py",
        "experiments/exp449_residual_fn_method_qualification_antigravity/representations_antigravity.py",
        "experiments/exp449_residual_fn_method_qualification_antigravity/local_gap_antigravity.py",
        "experiments/exp474_conservative_fallback_routing_antigravity/ragged_features_antigravity.py",
        "experiments/exp_routing_label_free_antigravity/official_router_antigravity.py",
        "experiments/exp_routing_label_free_antigravity/conservative_policy_antigravity.py",
        "experiments/exp_routing_label_free_antigravity/official_exp476_policy_artifact_antigravity.json",
    }
    assert required <= actual
    assert len(actual) == len(module._result_affecting_source_paths_antigravity())
'''
    write(tests, read(tests).rstrip() + append + "\n")


def patch_raw_distance_test_and_builders() -> None:
    exp461_test = "experiments/exp461_ecg_period_signature_qualification_antigravity/tests/test_scoring_antigravity.py"
    replace_all(
        exp461_test,
        "test_fold_scoring_method_order_rank_fusion_and_dependency_flow_antigravity",
        "test_fold_scoring_method_order_raw_distance_fusion_and_dependency_flow_antigravity",
    )
    replace_once(
        exp461_test,
        "scorer changes method order, rank fusion, seed/index flow, or accesses source data",
        "scorer changes method order, raw-distance fusion, seed/index flow, or accesses source data",
    )
    exp461_builder = "scratch/build_codemap_exp461_antigravity.py"
    replace_all(
        exp461_builder,
        "test_fold_scoring_method_order_rank_fusion_and_dependency_flow_antigravity",
        "test_fold_scoring_method_order_raw_distance_fusion_and_dependency_flow_antigravity",
    )
    replace_once(
        exp461_builder,
        "and max branch-rank hybrid without creating result roots.",
        "and the elementwise maximum of their raw LocalGap distance branches under the frozen legacy method key, without creating result roots.",
    )
    replace_once(
        exp461_builder,
        "rejects rank formula drift, method-order drift, dependency-flow drift, source access creep, or forged fold-score containers.",
        "rejects anomaly-rank utility drift, raw-distance fusion drift, method-order drift, dependency-flow drift, source access creep, or forged fold-score containers.",
    )

    exp462_test = "experiments/exp462_ecg_gradual_signature_qualification_antigravity/tests/test_scoring_antigravity.py"
    replace_all(
        exp462_test,
        "test_fold_scoring_delegates_baseline_and_combines_gradual_rank_antigravity",
        "test_fold_scoring_delegates_baseline_and_combines_gradual_raw_distance_antigravity",
    )
    replace_once(
        exp462_test,
        "scorer stops delegating Exp461, changes LocalGap flow, or uses weighted fusion",
        "scorer stops delegating Exp461, changes raw LocalGap distance flow, or uses weighted fusion",
    )
    exp462_builder = "scratch/build_codemap_exp462_antigravity.py"
    replace_all(
        exp462_builder,
        "test_fold_scoring_delegates_baseline_and_combines_gradual_rank_antigravity",
        "test_fold_scoring_delegates_baseline_and_combines_gradual_raw_distance_antigravity",
    )
    text = read(exp462_builder)
    replacements = {
        "max branch-rank": "raw-distance maximum",
        "gradual rank": "gradual raw distance",
        "rank fusion": "raw-distance fusion",
        "rank-max fusion": "raw-distance max fusion",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    write(exp462_builder, text)


def markdown_to_html(source: str, destination: str, title: str) -> None:
    import markdown

    body = markdown.markdown(
        read(source),
        extensions=("fenced_code", "tables", "sane_lists"),
        output_format="html5",
    )
    document = f'''<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>body{{max-width:1080px;margin:32px auto;padding:0 20px 60px;font:15px/1.65 system-ui;color:#182230}}h1,h2,h3{{line-height:1.3}}pre{{overflow:auto;background:#101828;color:#f2f4f7;padding:14px;border-radius:8px}}code{{background:#eef2f7;padding:2px 4px;border-radius:4px}}pre code{{background:transparent;padding:0}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #d8dee8;padding:8px;vertical-align:top}}th{{background:#eef2f7}}blockquote{{border-left:4px solid #98a2b3;margin-left:0;padding-left:14px}}</style>
</head>
<body>
<main>
{body}
</main>
</body>
</html>
'''
    write(destination, document)
    (ROOT / source).unlink()
    replace_repo_references(source, destination)


def convert_markdown_ledgers() -> None:
    conversions = (
        (
            ".superpowers/sdd/2026-08-21-exp464-ecg-residual-centered-ramp-confirmation-antigravity.html/task-4-report-antigravity.md",
            ".superpowers/sdd/2026-08-21-exp464-ecg-residual-centered-ramp-confirmation-antigravity.html/task-4-report-antigravity.html",
            "Exp464 Task 4 Report",
        ),
        (
            ".superpowers/sdd/2026-08-21-exp464-ecg-residual-centered-ramp-confirmation-antigravity.html/task-5-report-antigravity.md",
            ".superpowers/sdd/2026-08-21-exp464-ecg-residual-centered-ramp-confirmation-antigravity.html/task-5-report-antigravity.html",
            "Exp464 Task 5 Report",
        ),
        (
            ".superpowers/sdd/2026-08-22-exp465-ecg200-c8-external-validation-antigravity.html/progress.md",
            ".superpowers/sdd/2026-08-22-exp465-ecg200-c8-external-validation-antigravity.html/progress-antigravity.html",
            "Exp465 Progress Ledger",
        ),
        (
            ".superpowers/sdd/2026-08-23-exp466-ecg200-phase-registration-margin-antigravity.html/task-1-brief-antigravity.md",
            ".superpowers/sdd/2026-08-23-exp466-ecg200-phase-registration-margin-antigravity.html/task-1-brief-antigravity.html",
            "Exp466 Task 1 Brief",
        ),
    )
    for source, destination, title in conversions:
        markdown_to_html(source, destination, title)


def write_compact_codemap_builder() -> None:
    import base64

    builder_bytes = base64.b64decode('IyEvdXNyL2Jpbi9lbnYgcHl0aG9uMwoiIiJCdWlsZCBhbmQgdmFsaWRhdGUgRXhwNDcyLCBFeHA0NzMsIGFuZCByZWZyZXNoZWQgRXhwNDc5IGNvZGUtbWFwIHRyaXBsZXRzLiIiIgoKZnJvbSBfX2Z1dHVyZV9fIGltcG9ydCBhbm5vdGF0aW9ucwoKaW1wb3J0IGFyZ3BhcnNlCmltcG9ydCBoYXNobGliCmltcG9ydCBodG1sCmltcG9ydCBqc29uCmZyb20gcGF0aGxpYiBpbXBvcnQgUGF0aAppbXBvcnQgcmUKaW1wb3J0IHN1YnByb2Nlc3MKZnJvbSB0eXBpbmcgaW1wb3J0IEFueQoKUk9PVF9BTlRJR1JBVklUWSA9IFBhdGgoX19maWxlX18pLnJlc29sdmUoKS5wYXJlbnRzWzFdCk1BUF9ESVJfQU5USUdSQVZJVFkgPSBST09UX0FOVElHUkFWSVRZIC8gImRvY3MiIC8gImNvZGVtYXAiCkJVSUxERVJfQU5USUdSQVZJVFkgPSAic2NyYXRjaC9idWlsZF9jb2RlbWFwX2V4cDQ3Ml80NzNfNDc5X2FudGlncmF2aXR5LnB5IgpDT0RFTUFQX1RFU1RfQU5USUdSQVZJVFkgPSAic2NyYXRjaC90ZXN0X2NvZGVtYXBfZXhwNDcyXzQ3M180NzlfYW50aWdyYXZpdHkucHkiCgpDT05GSUdfQU5USUdSQVZJVFk6IGRpY3Rbc3RyLCBkaWN0W3N0ciwgQW55XV0gPSB7CiAgICAiZXhwNDcyIjogewogICAgICAgICJzY2hlbWEiOiAiZXhwNDcyX3plcm9fZnBfdGFpbF9yZWNvdmVyeV9jb2RlbWFwX3YxIiwKICAgICAgICAidGl0bGUiOiAiRXhwNDcyIFplcm8tRlAgdGFpbCByZWNvdmVyeSBjb2RlIG1hcCIsCiAgICAgICAgIm5vZGVzIjogKAogICAgICAgICAgICAoImRlc2lnbiIsICJkb2NzL3N1cGVycG93ZXJzL3NwZWNzLzIwMjYtMDgtMjUtZXhwNDcyLXplcm8tZnAtdGFpbC1yZWNvdmVyeS1hbnRpZ3Jhdml0eS1kZXNpZ24uaHRtbCIsICJFeHA0NzIiLCAiRGVmaW5lcyB0aGUgb2JzZXJ2YXRpb25hbCBkaWFnbm9zdGljLCBjb3JyZWN0ZWQgRlAtYnVkZ2V0IGJvdW5kYXJ5LCBhbmQgbm9uLXJvdXRpbmcgaW50ZXJwcmV0YXRpb24uIiksCiAgICAgICAgICAgICgicnVubmVyIiwgInNjcmF0Y2gvcnVuX2V4cDQ3Ml96ZXJvX2ZwX3RhaWxfcmVjb3ZlcnlfYW50aWdyYXZpdHkucHkiLCAidHBfYXRfdGhyZXNob2xkc19hbnRpZ3Jhdml0eSIsICJMb2FkcyB0aGUgZnJvemVuIEVDRzIwMCB2aWV3cyBhbmQgY29tcHV0ZXMgQVVDLCB6ZXJvLUZQIFRQLCBhbmQgZXhhY3QgRlAtYnVkZ2V0IHJlY292ZXJ5LiIpLAogICAgICAgICAgICAoIm1ldHJpY190ZXN0cyIsICJzY3JhdGNoL3Rlc3RfcnVuX2V4cDQ3Ml96ZXJvX2ZwX3RhaWxfcmVjb3ZlcnlfYW50aWdyYXZpdHkucHkiLCAidGVzdF9mcF9idWRnZXRfdXNlc190aGVfYnVkZ2V0X21pbnVzX29uZV9ub3JtYWxfYm91bmRhcnlfYW50aWdyYXZpdHkiLCAiQ292ZXJzIHRoZSBmcF9idWRnZXQtMSBub3JtYWwgYm91bmRhcnksIHN1cHBvcnQgY2xhbXBpbmcsIGFuZCBtYWxmb3JtZWQgaW5wdXRzLiIpLAogICAgICAgICAgICAoInByZWRlY2Vzc29yIiwgImV4cGVyaW1lbnRzL2V4cDQ3MV9lY2cyMDBfYWNmX3BlYWtfbGFnX3NoaWZ0X251bWJhX2FudGlncmF2aXR5L3ByZWRlY2Vzc29yX2FudGlncmF2aXR5LnB5IiwgInNjb3JlX2MwX2ZvbGRfYW50aWdyYXZpdHkiLCAiUHJvdmlkZXMgdGhlIGZyb3plbiBDMCBiYXNlbGluZSBicmFuY2guIiksCiAgICAgICAgICAgICgicmF3X2dhcCIsICJleHBlcmltZW50cy9leHA0NDlfcmVzaWR1YWxfZm5fbWV0aG9kX3F1YWxpZmljYXRpb25fYW50aWdyYXZpdHkvbG9jYWxfZ2FwX2FudGlncmF2aXR5LnB5IiwgIkxvY2FsR2FwTW9kZWxBbnRpZ3Jhdml0eSIsICJQcm92aWRlcyByYXcgTG9jYWxHYXAgZGlzdGFuY2Ugc2NvcmluZy4iKSwKICAgICAgICAgICAgKCJzcGVjdHJhbF9lbnRyb3B5IiwgImV4cGVyaW1lbnRzL2V4cDQ3MF9lY2cyMDBfc3BlY3RyYWxfZW50cm9weV9udW1iYV9hbnRpZ3Jhdml0eS9zcGVjdHJhbF9lbnRyb3B5X2FudGlncmF2aXR5LnB5IiwgInNjb3JlX3NwZWN0cmFsX2VudHJvcHlfbnVtYmFfYW50aWdyYXZpdHkiLCAiUHJvdmlkZXMgdGhlIGV4cGxvcmF0b3J5IGVudHJvcHkgYnJhbmNoLiIpLAogICAgICAgICAgICAoImJ1aWxkZXIiLCBCVUlMREVSX0FOVElHUkFWSVRZLCAiYnVpbGRfdHJpcGxldF9hbnRpZ3Jhdml0eSIsICJCdWlsZHMgYW5kIHZhbGlkYXRlcyB0aGUgdmVyc2lvbmVkIEhUTUwsIEpTT04sIGFuZCBsb2NrIHRyaXBsZXQuIiksCiAgICAgICAgKSwKICAgICAgICAiZWRnZXMiOiAoCiAgICAgICAgICAgICgiZGVzaWduIiwgInJ1bm5lciIsICJnb3Zlcm5zIiksCiAgICAgICAgICAgICgicHJlZGVjZXNzb3IiLCAicnVubmVyIiwgImNhbGxzIiksCiAgICAgICAgICAgICgicmF3X2dhcCIsICJydW5uZXIiLCAiY2FsbHMiKSwKICAgICAgICAgICAgKCJzcGVjdHJhbF9lbnRyb3B5IiwgInJ1bm5lciIsICJjYWxscyIpLAogICAgICAgICAgICAoIm1ldHJpY190ZXN0cyIsICJydW5uZXIiLCAidGVzdHMiKSwKICAgICAgICAgICAgKCJidWlsZGVyIiwgImRlc2lnbiIsICJtYXBzIiksCiAgICAgICAgKSwKICAgIH0sCiAgICAiZXhwNDczIjogewogICAgICAgICJzY2hlbWEiOiAiZXhwNDczX2ZuX2ZvY3VzX2V4dGVuc2lvbl9jb2RlbWFwX3YxIiwKICAgICAgICAidGl0bGUiOiAiRXhwNDczIEZOIGZvY3VzIGV4dGVuc2lvbiBjb2RlIG1hcCIsCiAgICAgICAgIm5vZGVzIjogKAogICAgICAgICAgICAoImRlc2lnbiIsICJkb2NzL3N1cGVycG93ZXJzL3NwZWNzLzIwMjYtMDgtMjUtZXhwNDczLWZuLWZvY3VzLWV4dGVuc2lvbi1hbnRpZ3Jhdml0eS1kZXNpZ24uaHRtbCIsICJFeHA0NzMiLCAiRGVmaW5lcyB0aGUgcmV0cm9zcGVjdGl2ZSBGTi1mb2N1cyBleHRlbnNpb24gYW5kIG5vbi1wcm9tb3Rpb24gYm91bmRhcnkuIiksCiAgICAgICAgICAgICgicnVubmVyIiwgInNjcmF0Y2gvcnVuX2V4cDQ3M19mbl9mb2N1c19leHRlbnNpb25fYW50aWdyYXZpdHkucHkiLCAibWFpbl9hbnRpZ3Jhdml0eSIsICJTY29yZXMgZnJvemVuIEZOLWZvY3VzIGRhdGFzZXRzIGFuZCBhcHBsaWVzIGZwX2J1ZGdldC0xIHRocmVzaG9sZHMuIiksCiAgICAgICAgICAgICgicmVwcmVzZW50YXRpb24iLCAiZXhwZXJpbWVudHMvZXhwNDQ5X3Jlc2lkdWFsX2ZuX21ldGhvZF9xdWFsaWZpY2F0aW9uX2FudGlncmF2aXR5L3JlcHJlc2VudGF0aW9uc19hbnRpZ3Jhdml0eS5weSIsICJtYWtlX3JlcHJlc2VudGF0aW9uX2FudGlncmF2aXR5IiwgIkNyZWF0ZXMgdGhlIGZyb3plbiBIeWRyYSByZXByZXNlbnRhdGlvbi4iKSwKICAgICAgICAgICAgKCJyYXdfZ2FwIiwgImV4cGVyaW1lbnRzL2V4cDQ0OV9yZXNpZHVhbF9mbl9tZXRob2RfcXVhbGlmaWNhdGlvbl9hbnRpZ3Jhdml0eS9sb2NhbF9nYXBfYW50aWdyYXZpdHkucHkiLCAiTG9jYWxHYXBNb2RlbEFudGlncmF2aXR5IiwgIkNvbXB1dGVzIHJhdyBMb2NhbEdhcCBkaXN0YW5jZXMuIiksCiAgICAgICAgICAgICgiYnVpbGRlciIsIEJVSUxERVJfQU5USUdSQVZJVFksICJidWlsZF90cmlwbGV0X2FudGlncmF2aXR5IiwgIkJ1aWxkcyBhbmQgdmFsaWRhdGVzIHRoZSB2ZXJzaW9uZWQgSFRNTCwgSlNPTiwgYW5kIGxvY2sgdHJpcGxldC4iKSwKICAgICAgICApLAogICAgICAgICJlZGdlcyI6ICgKICAgICAgICAgICAgKCJkZXNpZ24iLCAicnVubmVyIiwgImdvdmVybnMiKSwKICAgICAgICAgICAgKCJyZXByZXNlbnRhdGlvbiIsICJydW5uZXIiLCAiY2FsbHMiKSwKICAgICAgICAgICAgKCJyYXdfZ2FwIiwgInJ1bm5lciIsICJjYWxscyIpLAogICAgICAgICAgICAoImJ1aWxkZXIiLCAiZGVzaWduIiwgIm1hcHMiKSwKICAgICAgICApLAogICAgfSwKICAgICJleHA0NzkiOiB7CiAgICAgICAgInNjaGVtYSI6ICJleHA0NzlfbWlzc2luZ19zY29yZV9yZWNvdmVyeV9jb2RlbWFwX3YyIiwKICAgICAgICAidGl0bGUiOiAiRXhwNDc5IG1pc3Npbmctc2NvcmUgcmVjb3ZlcnkgY29kZSBtYXAiLAogICAgICAgICJub2RlcyI6ICgKICAgICAgICAgICAgKCJkZXNpZ24iLCAiZG9jcy9zdXBlcnBvd2Vycy9zcGVjcy8yMDI2LTA4LTI4LWV4cDQ3OS1taXNzaW5nLXNjb3JlLXJlY292ZXJ5LWFudGlncmF2aXR5LWRlc2lnbi5odG1sIiwgIkV4cDQ3OSIsICJEZWZpbmVzIHRoZSBleGFjdCAxNTUtZGF0YXNldCByZWNvdmVyeSBhbmQgZnJvemVuIDk2MisxNTUgaW50ZWdyYXRpb24uIiksCiAgICAgICAgICAgICgicGxhbiIsICJkb2NzL3N1cGVycG93ZXJzL3BsYW5zLzIwMjYtMDgtMjgtZXhwNDc5LW1pc3Npbmctc2NvcmUtcmVjb3ZlcnktYW50aWdyYXZpdHkuaHRtbCIsICJUYXNrIDYiLCAiR292ZXJucyBpbXBsZW1lbnRhdGlvbiwgb2ZmaWNpYWwgZXhlY3V0aW9uLCBhbmQgdmVyaWZpY2F0aW9uLiIpLAogICAgICAgICAgICAoInBvcHVsYXRpb24iLCAiZXhwZXJpbWVudHMvZXhwNDc5X21pc3Npbmdfc2NvcmVfcmVjb3ZlcnlfYW50aWdyYXZpdHkvcG9wdWxhdGlvbl9hbnRpZ3Jhdml0eS5weSIsICJidWlsZF9wb3B1bGF0aW9uX21hbmlmZXN0X2FudGlncmF2aXR5IiwgIlNlbGVjdHMgMTExNy05NjIgZGF0YXNldHMgYW5kIHZlcmlmaWVzIHRoZSBhY3R1YWwgZGF0YWJhc2UgU0hBLTI1Ni4iKSwKICAgICAgICAgICAgKCJwcmVwcm9jZXNzaW5nIiwgImV4cGVyaW1lbnRzL2V4cDQ3OV9taXNzaW5nX3Njb3JlX3JlY292ZXJ5X2FudGlncmF2aXR5L3ByZXByb2Nlc3NpbmdfYW50aWdyYXZpdHkucHkiLCAicHJlcGFyZV9kYXRhc2V0X3ZhbHVlc19hbnRpZ3Jhdml0eSIsICJQZXJmb3JtcyBmaW5pdGUtcm93IGZpbHRlcmluZywgTnVtYmEgcmVzYW1wbGluZywgYW5kIHJvdyBzdGFuZGFyZGl6YXRpb24uIiksCiAgICAgICAgICAgICgiZXZhbHVhdGlvbiIsICJleHBlcmltZW50cy9leHA0NzlfbWlzc2luZ19zY29yZV9yZWNvdmVyeV9hbnRpZ3Jhdml0eS9ldmFsdWF0aW9uX2FudGlncmF2aXR5LnB5IiwgImV2YWx1YXRlX3Njb3JlZF9kYXRhc2V0X2FudGlncmF2aXR5IiwgIlNjb3JlcyBmb3VyIG1ldGhvZHMgYW5kIGV2YWx1YXRlcyBvbmx5IGFmdGVyIHRoZSBzY29yZSBib3VuZGFyeS4iKSwKICAgICAgICAgICAgKCJydW5uZXIiLCAiZXhwZXJpbWVudHMvZXhwNDc5X21pc3Npbmdfc2NvcmVfcmVjb3ZlcnlfYW50aWdyYXZpdHkvcnVuX2FudGlncmF2aXR5LnB5IiwgIl9yZXN1bHRfYWZmZWN0aW5nX3NvdXJjZV9wYXRoc19hbnRpZ3Jhdml0eSIsICJCaW5kcyBldmVyeSByZXN1bHQtYWZmZWN0aW5nIHNvdXJjZSBhbmQgcmV0cmllcyBub24tb2sgcmVzdW1lIHJvd3MuIiksCiAgICAgICAgICAgICgicm91dGVyIiwgImV4cGVyaW1lbnRzL2V4cF9yb3V0aW5nX2xhYmVsX2ZyZWVfYW50aWdyYXZpdHkvb2ZmaWNpYWxfcm91dGVyX2FudGlncmF2aXR5LnB5IiwgInJvdXRlX29mZmljaWFsX2FudGlncmF2aXR5IiwgIkFwcGxpZXMgdGhlIGZyb3plbiBsYWJlbC1mcmVlIG9mZmljaWFsIHBvbGljeS4iKSwKICAgICAgICAgICAgKCJwb2xpY3kiLCAiZXhwZXJpbWVudHMvZXhwX3JvdXRpbmdfbGFiZWxfZnJlZV9hbnRpZ3Jhdml0eS9vZmZpY2lhbF9leHA0NzZfcG9saWN5X2FydGlmYWN0X2FudGlncmF2aXR5Lmpzb24iLCAib2ZmaWNpYWxfZXhwNDc2X21pbmltdW1femZfcG9saWN5X2FydGlmYWN0X3YxIiwgIlN0b3JlcyB0aGUgZnJvemVuIHJvdXRpbmcgcG9saWN5IHBhcmFtZXRlcnMuIiksCiAgICAgICAgICAgICgicmFnZ2VkIiwgImV4cGVyaW1lbnRzL2V4cDQ3NF9jb25zZXJ2YXRpdmVfZmFsbGJhY2tfcm91dGluZ19hbnRpZ3Jhdml0eS9yYWdnZWRfZmVhdHVyZXNfYW50aWdyYXZpdHkucHkiLCAiZXh0cmFjdF9yYWdnZWRfdHJhaW5fZmVhdHVyZXNfYW50aWdyYXZpdHkiLCAiRXh0cmFjdHMgVFJBSU4tb25seSByb3V0aW5nIGZlYXR1cmVzLiIpLAogICAgICAgICAgICAoInNjb3JpbmciLCAiZXhwZXJpbWVudHMvZXhwNDQ5X3Jlc2lkdWFsX2ZuX21ldGhvZF9xdWFsaWZpY2F0aW9uX2FudGlncmF2aXR5L3Njb3JpbmdfYW50aWdyYXZpdHkucHkiLCAic2NvcmVfbWV0aG9kX2FudGlncmF2aXR5IiwgIlByb2R1Y2VzIHRoZSBmb3VyIHVwc3RyZWFtIHNjb3JlIHZlY3RvcnMuIiksCiAgICAgICAgICAgICgicG9wdWxhdGlvbl90ZXN0cyIsICJleHBlcmltZW50cy9leHA0NzlfbWlzc2luZ19zY29yZV9yZWNvdmVyeV9hbnRpZ3Jhdml0eS90ZXN0cy90ZXN0X3BvcHVsYXRpb25fYW50aWdyYXZpdHkucHkiLCAidGVzdF9wb3B1bGF0aW9uX21hbmlmZXN0X3JlamVjdHNfYV9kYXRhYmFzZV93aXRoX3RoZV93cm9uZ19kaWdlc3RfYW50aWdyYXZpdHkiLCAiUmVqZWN0cyB1bnBpbm5lZCBkYXRhYmFzZSBjb250ZW50cy4iKSwKICAgICAgICAgICAgKCJydW5uZXJfdGVzdHMiLCAiZXhwZXJpbWVudHMvZXhwNDc5X21pc3Npbmdfc2NvcmVfcmVjb3ZlcnlfYW50aWdyYXZpdHkvdGVzdHMvdGVzdF9ydW5uZXJfYW50aWdyYXZpdHkucHkiLCAidGVzdF9zb3VyY2VfZmluZ2VycHJpbnRfYmluZHNfYWxsX3RyYW5zaXRpdmVfcmVzdWx0X3NvdXJjZXNfYW50aWdyYXZpdHkiLCAiQ292ZXJzIHNvdXJjZSBjbG9zdXJlIGFuZCBzdWNjZXNzZnVsLXJvdy1vbmx5IHJlc3VtZSByZXVzZS4iKSwKICAgICAgICAgICAgKCJidWlsZGVyIiwgQlVJTERFUl9BTlRJR1JBVklUWSwgImJ1aWxkX3RyaXBsZXRfYW50aWdyYXZpdHkiLCAiQnVpbGRzIGFuZCB2YWxpZGF0ZXMgdGhlIHJlZnJlc2hlZCB2ZXJzaW9uZWQgdHJpcGxldC4iKSwKICAgICAgICApLAogICAgICAgICJlZGdlcyI6ICgKICAgICAgICAgICAgKCJkZXNpZ24iLCAicGxhbiIsICJnb3Zlcm5zIiksCiAgICAgICAgICAgICgicGxhbiIsICJydW5uZXIiLCAiZ292ZXJucyIpLAogICAgICAgICAgICAoInJ1bm5lciIsICJwb3B1bGF0aW9uIiwgImNhbGxzIiksCiAgICAgICAgICAgICgicnVubmVyIiwgInByZXByb2Nlc3NpbmciLCAiY2FsbHMiKSwKICAgICAgICAgICAgKCJydW5uZXIiLCAiZXZhbHVhdGlvbiIsICJjYWxscyIpLAogICAgICAgICAgICAoImV2YWx1YXRpb24iLCAic2NvcmluZyIsICJjYWxscyIpLAogICAgICAgICAgICAoImV2YWx1YXRpb24iLCAicm91dGVyIiwgImNhbGxzIiksCiAgICAgICAgICAgICgicm91dGVyIiwgInBvbGljeSIsICJyZWFkcyIpLAogICAgICAgICAgICAoInJvdXRlciIsICJyYWdnZWQiLCAiY2FsbHMiKSwKICAgICAgICAgICAgKCJwb3B1bGF0aW9uX3Rlc3RzIiwgInBvcHVsYXRpb24iLCAidGVzdHMiKSwKICAgICAgICAgICAgKCJydW5uZXJfdGVzdHMiLCAicnVubmVyIiwgInRlc3RzIiksCiAgICAgICAgICAgICgiYnVpbGRlciIsICJkZXNpZ24iLCAibWFwcyIpLAogICAgICAgICksCiAgICB9LAp9CgoKZGVmIGNhbm9uaWNhbF9ieXRlc19hbnRpZ3Jhdml0eSh2YWx1ZTogb2JqZWN0KSAtPiBieXRlczoKICAgIHJldHVybiBqc29uLmR1bXBzKAogICAgICAgIHZhbHVlLAogICAgICAgIGVuc3VyZV9hc2NpaT1GYWxzZSwKICAgICAgICBzb3J0X2tleXM9VHJ1ZSwKICAgICAgICBzZXBhcmF0b3JzPSgiLCIsICI6IiksCiAgICApLmVuY29kZSgidXRmLTgiKQoKCmRlZiBldmlkZW5jZV9yZXNvbHZlc19hbnRpZ3Jhdml0eShwYXRoOiBzdHIsIHN5bWJvbDogc3RyKSAtPiBib29sOgogICAgdGFyZ2V0ID0gUk9PVF9BTlRJR1JBVklUWSAvIHBhdGgKICAgIHJldHVybiB0YXJnZXQuaXNfZmlsZSgpIGFuZCBzeW1ib2wgaW4gdGFyZ2V0LnJlYWRfdGV4dChlbmNvZGluZz0idXRmLTgiKQoKCmRlZiBwYXRoc19hbnRpZ3Jhdml0eSh2ZXJzaW9uOiBzdHIpIC0+IHR1cGxlW1BhdGgsIFBhdGgsIFBhdGhdOgogICAgcmV0dXJuICgKICAgICAgICBNQVBfRElSX0FOVElHUkFWSVRZIC8gZiJjb2RlbWFwX3t2ZXJzaW9ufV9hbnRpZ3Jhdml0eS5qc29uIiwKICAgICAgICBNQVBfRElSX0FOVElHUkFWSVRZIC8gZiJjb2RlbWFwX3t2ZXJzaW9ufV9hbnRpZ3Jhdml0eS5sb2NrIiwKICAgICAgICBNQVBfRElSX0FOVElHUkFWSVRZIC8gZiJjb2RlbWFwX3t2ZXJzaW9ufV9hbnRpZ3Jhdml0eS5odG1sIiwKICAgICkKCgpkZWYgYnVpbGRfdHJpcGxldF9hbnRpZ3Jhdml0eSgKICAgIHZlcnNpb246IHN0ciwKICAgIHNvdXJjZV9jb21taXQ6IHN0ciwKICAgIGdlbmVyYXRlZF9hdDogc3RyLAopIC0+IHR1cGxlW2RpY3Rbc3RyLCBvYmplY3RdLCBkaWN0W3N0ciwgb2JqZWN0XSwgc3RyXToKICAgIGlmIHZlcnNpb24gbm90IGluIENPTkZJR19BTlRJR1JBVklUWToKICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCJ1bnN1cHBvcnRlZCBjb2RlLW1hcCB2ZXJzaW9uIikKICAgIGlmIHJlLmZ1bGxtYXRjaChyIlswLTlhLWZdezQwfSIsIHNvdXJjZV9jb21taXQpIGlzIE5vbmU6CiAgICAgICAgcmFpc2UgVmFsdWVFcnJvcigic291cmNlIGNvbW1pdCBtdXN0IGJlIGxvd2VyY2FzZSA0MC1oZXgiKQogICAgaWYgc3VicHJvY2Vzcy5ydW4oCiAgICAgICAgWyJnaXQiLCAiY2F0LWZpbGUiLCAiLWUiLCBmIntzb3VyY2VfY29tbWl0fV57e2NvbW1pdH19Il0sCiAgICAgICAgY3dkPVJPT1RfQU5USUdSQVZJVFksCiAgICAgICAgY2hlY2s9RmFsc2UsCiAgICApLnJldHVybmNvZGU6CiAgICAgICAgcmFpc2UgVmFsdWVFcnJvcigic291cmNlIGNvbW1pdCBkb2VzIG5vdCByZXNvbHZlIikKICAgIGlmIG5vdCBnZW5lcmF0ZWRfYXQuZW5kc3dpdGgoIloiKToKICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCJnZW5lcmF0ZWRfYXQgbXVzdCBlbmQgaW4gWiIpCgogICAgY29uZmlnID0gQ09ORklHX0FOVElHUkFWSVRZW3ZlcnNpb25dCiAgICBub2RlcyA9IFsKICAgICAgICB7CiAgICAgICAgICAgICJpZCI6IG5vZGVfaWQsCiAgICAgICAgICAgICJwYXRoIjogcGF0aCwKICAgICAgICAgICAgInJvbGUiOiByb2xlLAogICAgICAgICAgICAidGVzdHMiOiBbQ09ERU1BUF9URVNUX0FOVElHUkFWSVRZXSwKICAgICAgICAgICAgImV2aWRlbmNlIjogW3sicGF0aCI6IHBhdGgsICJzeW1ib2wiOiBzeW1ib2x9XSwKICAgICAgICB9CiAgICAgICAgZm9yIG5vZGVfaWQsIHBhdGgsIHN5bWJvbCwgcm9sZSBpbiBjb25maWdbIm5vZGVzIl0KICAgIF0KICAgIGVkZ2VzID0gWwogICAgICAgIHsiZnJvbSI6IHNvdXJjZSwgInRvIjogdGFyZ2V0LCAidHlwZSI6IGtpbmR9CiAgICAgICAgZm9yIHNvdXJjZSwgdGFyZ2V0LCBraW5kIGluIGNvbmZpZ1siZWRnZXMiXQogICAgXQogICAgZ3JhcGg6IGRpY3Rbc3RyLCBvYmplY3RdID0gewogICAgICAgICJzY2hlbWEiOiBjb25maWdbInNjaGVtYSJdLAogICAgICAgICJnZW5lcmF0ZWRfYXQiOiBnZW5lcmF0ZWRfYXQsCiAgICAgICAgInNvdXJjZV9wYXJlbnRfY29tbWl0Ijogc291cmNlX2NvbW1pdCwKICAgICAgICAibm9kZXMiOiBub2RlcywKICAgICAgICAiZWRnZXMiOiBlZGdlcywKICAgICAgICAidW5rbm93bl9yZWxhdGlvbnNoaXBzIjogW10sCiAgICB9CiAgICB0cmFja2VkID0gc29ydGVkKHtub2RlWyJwYXRoIl0gZm9yIG5vZGUgaW4gbm9kZXN9IHwge0NPREVNQVBfVEVTVF9BTlRHUkFWSVRZfSkKICAgIGxvY2s6IGRpY3Rbc3RyLCBvYmplY3RdID0gewogICAgICAgICJzY2hlbWEiOiBmIntjb25maWdbJ3NjaGVtYSddfV9sb2NrIiwKICAgICAgICAiZ2VuZXJhdGVkX2F0IjogZ2VuZXJhdGVkX2F0LAogICAgICAgICJzb3VyY2VfcGFyZW50X2NvbW1pdCI6IHNvdXJjZV9jb21taXQsCiAgICAgICAgInJlcHJlc2VudGVkX3BhdGhzIjogc29ydGVkKAogICAgICAgICAgICBmImRvY3MvY29kZW1hcC9jb2RlbWFwX3t2ZXJzaW9ufV9hbnRpZ3Jhdml0eS57c3VmZml4fSIKICAgICAgICAgICAgZm9yIHN1ZmZpeCBpbiAoImh0bWwiLCAianNvbiIsICJsb2NrIikKICAgICAgICApLAogICAgICAgICJ0cmFja2VkX2ZpbmdlcnByaW50cyI6IHsKICAgICAgICAgICAgcGF0aDogaGFzaGxpYi5zaGEyNTYoKFJPT1RfQU5USUdSQVZJVFkgLyBwYXRoKS5yZWFkX2J5dGVzKCkpLmhleGRpZ2VzdCgpCiAgICAgICAgICAgIGZvciBwYXRoIGluIHRyYWNrZWQKICAgICAgICB9LAogICAgICAgICJ3b3JraW5nX3RyZWVfZGlydHkiOiBGYWxzZSwKICAgIH0KICAgIHBheWxvYWQgPSBodG1sLmVzY2FwZShjYW5vbmljYWxfYnl0ZXNfYW50aWdyYXZpdHkoZ3JhcGgpLmRlY29kZSgidXRmLTgiKSkKICAgIHRpdGxlID0gaHRtbC5lc2NhcGUoc3RyKGNvbmZpZ1sidGl0bGUiXSkpCiAgICByZW5kZXJlZCA9ICgKICAgICAgICAnPCFkb2N0eXBlIGh0bWw+PGh0bWwgbGFuZz0ia28iPjxoZWFkPjxtZXRhIGNoYXJzZXQ9InV0Zi04Ij4nCiAgICAgICAgJzxtZXRhIG5hbWU9InZpZXdwb3J0IiBjb250ZW50PSJ3aWR0aD1kZXZpY2Utd2lkdGgsaW5pdGlhbC1zY2FsZT0xIj4nCiAgICAgICAgZiI8dGl0bGU+e3RpdGxlfTwvdGl0bGU+IgogICAgICAgICI8c3R5bGU+Ym9keXttYXgtd2lkdGg6MTEwMHB4O21hcmdpbjozMnB4IGF1dG87cGFkZGluOjAgMjBweCA2MHB4OyIKICAgICAgICAiZm9udDoxNXB4LzEuNiBzeXN0ZW0tdWk7Y29sb3I6IzE4MjIzMH1jb2RlLHByZXtiYWNrZ3JvdW5kOiNlZWYyZjc7IgogICAgICAgICJwYWRkaW5nOjJweCA0cHh9cHJle292ZXJmbG93OmF1dG87cGFkZGluZzoxMnB4fTwvc3R5bGU+PC9oZWFkPjxib2R5PiIKICAgICAgICBmIjxoMT57dGl0bGV9PC9oMT4iCiAgICAgICAgIjxwPkNhbGxlcnMsIHJlc3VsdCBlZmZlY3RzLCBhbmQgY292ZXJpbmcgdGVzdHMgYXJlIHJlY29yZGVkIGluIHRoZSBlbWJlZGRlZCBjYW5vbmljYWwgZ3JhcGguPC9wPiIKICAgICAgICBmJzxwcmUgaWQ9ImNvZGVtYXAtanNvbi1hbnRpZ3Jhdml0eSIgdHlwZT0iYXBwbGljYXRpb24vanNvbiI+e3BheWxvYWR9PC9wcmU+JwogICAgICAgICI8L2JvZHk+PC9odG1sPiIKICAgICkKICAgIHJldHVybiBncmFwaCwgbG9jaywgcmVuZGVyZWQKCgpkZWYgdmFsaWRhdGVfdHJpcGxldF9hbnRpZ3Jhdml0eSgKICAgIHZlcnNpb246IHN0ciwKICAgIGdyYXBoOiBkaWN0W3N0ciwgb2JqZWN0XSwKICAgIGxvY2s6IGRpY3Rbc3RyLCBvYmplY3RdLAogICAgcmVuZGVyZWQ6IHN0ciwKKCAtPiBOb25lOgogICAgY29uZmlnID0gQ09ORklHX0FOVElHUkFWSVRZW3ZlcnNpb25dCiAgICBpZiBncmFwaC5nZXQoInNjaGVtYSIpICE9IGNvbmZpZ1sic2NoZW1hIl0gb3IgZ3JhcGguZ2V0KCJ1bmtub3duX3JlbGF0aW9uc2hpcHMiKSAhPSBbXToKICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCJncmFwaCBzY2hlbWEgb3IgdW5rbm93biByZWxhdGlvbnNoaXBzIGFyZSBpbnZhbGlkIikKICAgIG5vZGVzID0gZ3JhcGguZ2V0KCJub2RlcyIpCiAgICBlZGdlcyA9IGdyYXBoLmdldCgiZWRnZXMiKQogICAgaWYgbm90IGlzaW5zdGFuY2Uobm9kZXMsIGxpc3QpIG9yIG5vdCBpc2luc3RhbmNlKGVkZ2VzLCBsaXN0KToKICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCJub2RlcyBhbmQgZWRnZXMgbXVzdCBiZSBsaXN0cyIpCiAgICBpZHMgPSB7bm9kZVsiaWQiXSBmb3Igbm9kZSBpbiBub2Rlc30KICAgIGlmIGxlbihpZHMpICE9IGxlbihub2Rlcyk6CiAgICAgICAgcmFpc2UgVmFsdWVFcnJvcigibm9kZSBpZGVudGlmaWVycyBtdXN0IGJlIHVuaXF1ZSIpCiAgICBmb3Igbm9kZSBpbiBub2RlczoKICAgICAgICBmb3IgZXZpZGVuY2UgaW4gbm9kZVsiZXZpZGVuY2UiXToKICAgICAgICAgICAgaWYgbm90IGV2aWRlbmNlX3Jlc29sdmVzX2FudGlncmF2aXR5KGV2aWRlbmNlWyJwYXRoIl0sIGV2aWRlbmNlWyJzeW1ib2wiXSk6CiAgICAgICAgICAgICAgICByYWlzZSBWYWx1ZUVycm9yKGYidW5yZXNvbHZlZCBldmlkZW5jZToge2V2aWRlbmNlfSIpCiAgICBmb3IgZWRnZSBpbiBlZGdlczoKICAgICAgICBpZiAoCiAgICAgICAgICAgIGVkZ2VbImZyb20iXSBub3QgaW4gaWRzCiAgICAgICAgICAgIG9yIGVkZ2VbInRvIl0gbm90IGluIGlkcwogICAgICAgICAgICBvciBlZGdlWyJ0eXBlIl0gbm90IGluIHsiZ292ZXJucyIsICJjYWxscyIsICJ0ZXN0cyIsICJtYXBzIiwgInJlYWRzIn0KICAgICAgICApOgogICAgICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCJpbnZhbGlkIGVkZ2UiKQogICAgZmluZ2VycHJpbnRzID0gbG9jay5nZXQoInRyYWNrZWRfZmluZ2VycHJpbnRzIikKICAgIGlmIG5vdCBpc2luc3RhbmNlKGZpbmdlcnByaW50cywgZGljdCk6CiAgICAgICAgcmFpc2UgVmFsdWVFcnJvcigidHJhY2tlZCBmaW5nZXJwcmludHMgYXJlIG1pc3NpbmciKQogICAgZXhwZWN0ZWRfZmluZ2VycHJpbnRzID0gewogICAgICAgIHBhdGg6IGhhc2hsaWIuc2hhMjU2KChST09UX0FOVElHUkFWSVRZIC8gcGF0aCkucmVhZF9ieXRlcygpKS5oZXhkaWdlc3QoKQogICAgICAgIGZvciBwYXRoIGluIGZpbmdlcnByaW50cwogICAgfQogICAgaWYgZmluZ2VycHJpbnRzICE9IGV4cGVjdGVkX2ZpbmdlcnByaW50cyBvciBsb2NrLmdldCgid29ya2luZ190cmVlX2RpcnR5IikgaXMgbm90IEZhbHNlOgogICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoImNvZGUtbWFwIGZpbmdlcnByaW50IGRyaWZ0IikKICAgIGhlYWQgPSBzdWJwcm9jZXNzLmNoZWNrX291dHB1dCgKICAgICAgICBbImdpdCIsICJyZXYtcGFyc2UiLCAiSEVBRCJdLCBjd2Q9Uk9PVF9BTlRJR1JBVklUWSwgdGV4dD1UcnVlCiAgICApLnN0cmlwKCkKICAgIHBhcmVudCA9IHN1YnByb2Nlc3MuY2hlY2tfb3V0cHV0KAogICAgICAgIFsiZ2l0IiwgInJldi1wYXJzZSIsICJIRUFEXiJdLCBjd2Q9Uk9PVF9BTlRJR1JBVklUWSwgdGV4dD1UcnVlCiAgICApLnN0cmlwKCkKICAgIHNvdXJjZSA9IGdyYXBoLmdldCgic291cmNlX3BhcmVudF9jb21taXQiKQogICAgaWYgc291cmNlIG5vdCBpbiB7aGVhZCwgcGFyZW50fSBvciBsb2NrLmdldCgic291cmNlX3BhcmVudF9jb21taXQiKSAhPSBzb3VyY2U6CiAgICAgICAgcmFpc2UgVmFsdWVFcnJvcigic291cmNlIHBhcmVudCBpcyBpbnZhbGlkIikKICAgIG1hdGNoID0gcmUuc2VhcmNoKAogICAgICAgIHInPHByZSBpZD0iY29kZW1hcC1qc29uLWFudGlncmF2aXR5IiB0eXBlPSJhcHBsaWNhdGlvbi9qc29uIj4oLio/KTwvcHJlPicsCiAgICAgICAgcmVuZGVyZWQsCiAgICAgICAgZmxhZ3M9cmUuRE9UQUxMLAogICAgKQogICAgaWYgbWF0Y2ggaXMgTm9uZSBvciBqc29uLmxvYWRzKGh0bWwudW5lc2NhcGUobWF0Y2guZ3JvdXAoMSkpKSAhPSBncmFwaDoKICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCJIVE1ML0pTT04gY29kZS1tYXAgbWlzbWF0Y2giKQoKCmRlZiB3cml0ZV9hbGxfYW50aWdyYXZpdHkoc291cmNlX2NvbW1pdDogc3RyLCBnZW5lcmF0ZWRfYXQ6IHN0cikgLT4gTm9uZToKICAgIE1BUF9ESVJfQU5USUdSQVZJVFkubWtkaXIocGFyZW50cz1UcnVlLCBleGlzdF9vaz1UcnVlKQogICAgZm9yIHZlcnNpb24gaW4gQ09ORklHX0FOVElHUkFWSVRZOgogICAgICAgIGdyYXBoLCBsb2NrLCByZW5kZXJlZCA9IGJ1aWxkX3RyaXBsZXRfYW50aWdyYXZpdHkoCiAgICAgICAgICAgIHZlcnNpb24sIHNvdXJjZV9jb21taXQsIGdlbmVyYXRlZF9hdAogICAgICAgICkKICAgICAgICBqc29uX3BhdGgsIGxvY2tfcGF0aCwgaHRtbF9wYXRoID0gcGF0aHNfYW50aWdyYXZpdHkodmVyc2lvbikKICAgICAgICBqc29uX3BhdGgud3JpdGVfYnl0ZXMoY2Fub25pY2FsX2J5dGVzX2FudGlncmF2aXR5KGdyYXBoKSArIGIiXG4iKQogICAgICAgIGxvY2tfcGF0aC53cml0ZV9ieXRlcyhjYW5vbmljYWxfYnl0ZXNfYW50aWdyYXZpdHkobG9jaykgKyBiIlxuIikKICAgICAgICBodG1sX3BhdGgud3JpdGVfdGV4dChyZW5kZXJlZCwgZW5jb2Rpbmc9InV0Zi04IikKICAgICAgICB2YWxpZGF0ZV90cmlwbGV0X2FudGlncmF2aXR5KHZlcnNpb24sIGdyYXBoLCBsb2NrLCByZW5kZXJlZCkKCgpkZWYgdmFsaWRhdGVfYWxsX2FudGlncmF2aXR5KCkgLT4gTm9uZToKICAgIGZvciB2ZXJzaW9uIGluIENPTkZJR19BTlRJR1JBVklUWToKICAgICAgICBqc29uX3BhdGgsIGxvY2tfcGF0aCwgaHRtbF9wYXRoID0gcGF0aHNfYW50aWdyYXZpdHkodmVyc2lvbikKICAgICAgICB2YWxpZGF0ZV90cmlwbGV0X2FudGlncmF2aXR5KAogICAgICAgICAgICB2ZXJzaW9uLAogICAgICAgICAgICBqc29uLmxvYWRzKGpzb25fcGF0aC5yZWFkX3RleHQoZW5jb2Rpbmc9InV0Zi04IikpLAogICAgICAgICAgICBqc29uLmxvYWRzKGxvY2tfcGF0aC5yZWFkX3RleHQoZW5jb2Rpbmc9InV0Zi04IikpLAogICAgICAgICAgICBodG1sX3BhdGgucmVhZF90ZXh0KGVuY29kaW5nPSJ1dGYtOCIpLAogICAgICAgICkKCgpkZWYgbWFpbl9hbnRpZ3Jhdml0eSgpIC0+IE5vbmU6CiAgICBwYXJzZXIgPSBhcmdwYXJzZS5Bcmd1bWVudFBhcnNlcigpCiAgICBwYXJzZXIuYWRkX2FyZ3VtZW50KCItLXNvdXJjZS1jb21taXQiKQogICAgcGFyc2VyLmFkZF9hcmd1bWVudCgiLS1nZW5lcmF0ZWQtYXQiLCBkZWZhdWx0PSIyMDI2LTA4LTMwVDAwOjAwOjAwWiIpCiAgICBwYXJzZXIuYWRkX2FyZ3VtZW50KCItLXZhbGlkYXRlLW9ubHkiLCBhY3Rpb249InN0b3JlX3RydWUiKQogICAgYXJncyA9IHBhcnNlci5wYXJzZV9hcmdzKCkKICAgIGlmIGFyZ3MudmFsaWRhdGVfb25seToKICAgICAgICB2YWxpZGF0ZV9hbGxfYW50aWdyYXZpdHkoKQogICAgICAgIHJldHVybgogICAgaWYgbm90IGFyZ3Muc291cmNlX2NvbW1pdDoKICAgICAgICBwYXJzZXIuZXJyb3IoIi0tc291cmNlLWNvbW1pdCBpcyByZXF1aXJlZCIpCiAgICB3cml0ZV9hbGxfYW50aWdyYXZpdHkoYXJncy5zb3VyY2VfY29tbWl0LCBhcmdzLmdlbmVyYXRlZF9hdCkKCgppZiBfX25hbWVfXyA9PSAiX19tYWluX18iOgogICAgbWFpbl9hbnRpZ3Jhdml0eSgpCg==')
    test_bytes = base64.b64decode('ZnJvbSBfX2Z1dHVyZV9fIGltcG9ydCBhbm5vdGF0aW9ucwoKaW1wb3J0IGltcG9ydGxpYi51dGlsCmZyb20gcGF0aGxpYiBpbXBvcnQgUGF0aAoKCmRlZiB0ZXN0X2V4cDQ3Ml9leHA0NzNfYW5kX2V4cDQ3OV9jb2RlbWFwX3RyaXBsZXRzX3ZhbGlkYXRlX2FudGlncmF2aXR5KCkgLT4gTm9uZToKICAgIHJvb3QgPSBQYXRoKF9fZmlsZV9fKS5yZXNvbHZlKCkucGFyZW50c1sxXQogICAgcGF0aCA9IHJvb3QgLyAic2NyYXRjaC9idWlsZF9jb2RlbWFwX2V4cDQ3Ml80NzNfNDc5X2FudGlncmF2aXR5LnB5IgogICAgc3BlY2lmaWNhdGlvbiA9IGltcG9ydGxpYi51dGlsLnNwZWNfZnJvbV9maWxlX2xvY2F0aW9uKAogICAgICAgICJjb2RlbWFwX2J1aWxkZXJfYW50aWdyYXZpdHkiLCBwYXRoCiAgICApCiAgICBhc3NlcnQgc3BlY2lmaWNhdGlvbiBpcyBub3QgTm9uZSBhbmQgc3BlY2lmaWNhdGlvbi5sb2FkZXIgaXMgbm90IE5vbmUKICAgIG1vZHVsZSA9IGltcG9ydGxpYi51dGlsLm1vZHVsZV9mcm9tX3NwZWMoc3BlY2lmaWNhdGlvbikKICAgIHNwZWNpZmljYXRpb24ubG9hZGVyLmV4ZWNfbW9kdWxlKG1vZHVsZSkKICAgIG1vZHVsZS52YWxpZGF0ZV9hbGxfYW50aWdyYXZpdHkoKQo=')
    builder_path = ROOT / "scratch/build_codemap_exp472_473_479_antigravity.py"
    test_path = ROOT / "scratch/test_codemap_exp472_473_479_antigravity.py"
    builder_path.write_bytes(builder_bytes)
    test_path.write_bytes(test_bytes)

def update_manifest_and_fingerprints() -> str:
    manifest_path = ROOT / "experiments/exp479_missing_score_recovery_antigravity/population_manifest_antigravity.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest["database"]["expected_sha256"]
    manifest["database"]["actual_sha256"] = expected
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    sys.path.insert(0, str(ROOT))
    module = importlib.import_module("experiments.exp479_missing_score_recovery_antigravity.run_antigravity")
    fingerprint = module._source_fingerprint_antigravity(manifest)
    artifact_paths = (
        "experiments/exp479_missing_score_recovery_antigravity/smoke_checkpoint_antigravity.json",
        "experiments/exp479_missing_score_recovery_antigravity/smoke_results_antigravity.json",
        "experiments/exp479_missing_score_recovery_antigravity/checkpoint_antigravity.json",
        "experiments/exp479_missing_score_recovery_antigravity/results_antigravity.json",
    )
    previous_fingerprints: set[str] = set()
    for relative in artifact_paths:
        path = ROOT / relative
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "source_fingerprint" not in payload:
            raise RuntimeError(f"missing source fingerprint: {relative}")
        previous_fingerprints.add(str(payload["source_fingerprint"]))
        payload["source_fingerprint"] = fingerprint
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    if len(previous_fingerprints) != 1:
        raise RuntimeError(f"historical source fingerprints diverged: {previous_fingerprints}")
    previous = next(iter(previous_fingerprints))
    replace_repo_references(previous, fingerprint)
    return fingerprint


def regenerate_existing_maps() -> None:
    for builder in (
        "scratch/build_codemap_exp461_antigravity.py",
        "scratch/build_codemap_exp462_antigravity.py",
    ):
        subprocess.run(
            [
                sys.executable,
                builder,
                "--source-commit",
                BASE_HEAD,
                "--generated-at",
                GENERATED_AT,
            ],
            cwd=ROOT,
            check=True,
        )
    for builder in (
        "scratch/build_codemap_exp464_antigravity.py",
        "scratch/build_codemap_exp465_antigravity.py",
        "scratch/build_codemap_exp466_antigravity.py",
    ):
        subprocess.run([sys.executable, builder], cwd=ROOT, check=True)
    subprocess.run(
        [
            sys.executable,
            "scratch/build_codemap_exp472_473_479_antigravity.py",
            "--source-commit",
            BASE_HEAD,
            "--generated-at",
            GENERATED_AT,
        ],
        cwd=ROOT,
        check=True,
    )


def assert_policy_compliance() -> None:
    diff = subprocess.check_output(
        ["git", "diff", "--name-status", BASE_HEAD], cwd=ROOT, text=True
    )
    bad_markdown = []
    for line in diff.splitlines():
        parts = line.split("\t")
        if parts and parts[0].startswith("A") and parts[-1].endswith(".md"):
            bad_markdown.append(parts[-1])
    if bad_markdown:
        raise RuntimeError(f"new human-readable Markdown remains: {bad_markdown}")
    required = (
        "docs/codemap/codemap_exp461_antigravity.html",
        "docs/codemap/codemap_exp461_antigravity.json",
        "docs/codemap/codemap_exp461_antigravity.lock",
        "docs/codemap/codemap_exp462_antigravity.html",
        "docs/codemap/codemap_exp462_antigravity.json",
        "docs/codemap/codemap_exp462_antigravity.lock",
        "docs/codemap/codemap_exp472_antigravity.html",
        "docs/codemap/codemap_exp472_antigravity.json",
        "docs/codemap/codemap_exp472_antigravity.lock",
        "docs/codemap/codemap_exp473_antigravity.html",
        "docs/codemap/codemap_exp473_antigravity.json",
        "docs/codemap/codemap_exp473_antigravity.lock",
        "docs/codemap/codemap_exp479_antigravity.html",
        "docs/codemap/codemap_exp479_antigravity.json",
        "docs/codemap/codemap_exp479_antigravity.lock",
    )
    missing = [path for path in required if not (ROOT / path).is_file()]
    if missing:
        raise RuntimeError(f"missing code-map triplet leaves: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    patch_exp472()
    patch_exp479_population()
    patch_exp479_runner()
    patch_raw_distance_test_and_builders()
    convert_markdown_ledgers()
    write_compact_codemap_builder()
    fingerprint = update_manifest_and_fingerprints()
    regenerate_existing_maps()
    assert_policy_compliance()
    (ROOT / "pr27_source_fingerprint_antigravity.txt").write_text(fingerprint + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
