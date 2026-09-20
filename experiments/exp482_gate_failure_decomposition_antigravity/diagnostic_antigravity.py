"""Exp482 train-side diagnostics for the frozen Exp481 strict gate."""

from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
from html import escape
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from experiments.exp480_family_balanced_evidence_routing_antigravity.contracts_antigravity import (
    EvidenceRowAntigravity,
)
from experiments.exp480_family_balanced_evidence_routing_antigravity.evaluation_antigravity import (
    iter_family_splits_antigravity,
)
from experiments.exp480_family_balanced_evidence_routing_antigravity.evidence_antigravity import (
    load_evidence_database_antigravity,
    validate_evidence_database_antigravity,
)
from experiments.exp481_strict_evidence_residual_correction_antigravity.config_antigravity import (
    BASELINE_METHOD_ANTIGRAVITY,
    EXP480_EVIDENCE_DATABASE_PATH_ANTIGRAVITY,
    K_ANTIGRAVITY,
    OFFICIAL_RESULT_PATH_ANTIGRAVITY,
)
from experiments.exp481_strict_evidence_residual_correction_antigravity.contracts_antigravity import (
    StrictCandidateAssessmentAntigravity,
)
from experiments.exp481_strict_evidence_residual_correction_antigravity.evaluation_antigravity import (
    _feature_mapping_antigravity,
    _row_policy_antigravity,
    evaluate_nested_out_of_family_antigravity,
)
from experiments.exp481_strict_evidence_residual_correction_antigravity.policy_antigravity import (
    StrictResidualPolicyAntigravity,
)
from experiments.exp481_strict_evidence_residual_correction_antigravity.run_antigravity import (
    json_bytes_antigravity,
    write_json_atomic_antigravity,
)


ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR_ANTIGRAVITY = Path(__file__).resolve().parent
RESULT_PATH_ANTIGRAVITY = EXPERIMENT_DIR_ANTIGRAVITY / "results_antigravity.json"
DASHBOARD_PATH_ANTIGRAVITY = ROOT_ANTIGRAVITY / "docs/EXP482_DASHBOARD_antigravity.html"
REPORT_PATH_ANTIGRAVITY = (
    ROOT_ANTIGRAVITY / "docs/EXP482_IMPLEMENTATION_REPORT_antigravity.html"
)
EXPECTED_EXP481_RESULT_SHA256_ANTIGRAVITY = (
    "7038b994e422dc93a18927c810ce5e3bdb7601039ba01d2b388895444e09956f"
)


def diagnose_assessments_antigravity(
    assessments: Sequence[StrictCandidateAssessmentAntigravity],
    *,
    outer_family: str,
    inner_family: str,
    dataset: str,
) -> tuple[dict[str, object], ...]:
    items = tuple(assessments)
    ranks = {
        item.method: (
            item.zf_wins - item.zf_losses,
            item.min_auc_gain,
            item.q20_auc_gain,
        )
        for item in items
        if item.passed
    }
    best_methods: set[str] = set()
    if ranks:
        best_rank = max(ranks.values())
        best_methods = {method for method, rank in ranks.items() if rank == best_rank}

    records = []
    for item in items:
        failures = []
        if item.zf_wins <= item.zf_losses:
            failures.append("wins_not_greater")
        if item.min_auc_gain < 0.0:
            failures.append("negative_min_auc")
        if item.q20_auc_gain <= 0.0:
            failures.append("nonpositive_q20_auc")
        if not item.passed:
            rank_status = "not_reached"
        elif item.method not in best_methods:
            rank_status = "not_best"
        elif len(best_methods) > 1:
            rank_status = "best_tie"
            failures.append("candidate_rank_tie")
        else:
            rank_status = "unique_best"
        records.append(
            {
                "outer_family": outer_family,
                "inner_family": inner_family,
                "dataset": dataset,
                "method": item.method,
                "support": item.support,
                "zf_wins": item.zf_wins,
                "zf_losses": item.zf_losses,
                "zf_ties": item.zf_ties,
                "zf_margin": item.zf_wins - item.zf_losses,
                "min_auc_margin": item.min_auc_gain,
                "q20_auc_margin": item.q20_auc_gain,
                "statistics_passed": item.passed,
                "rank_status": rank_status,
                "failure_clauses": tuple(failures),
                "accepted": rank_status == "unique_best",
            }
        )
    return tuple(records)


def collect_train_census_antigravity(
    rows: Sequence[EvidenceRowAntigravity],
) -> dict[str, object]:
    ordered = tuple(sorted(rows, key=lambda row: row.dataset))
    routing = {
        "inner_rows": 0,
        "strict_local_gate_failed": 0,
        "strict_candidate_accept": 0,
        "row_knn_alternative_preserved": 0,
    }
    records: list[dict[str, object]] = []
    for outer_family, outer_train, _outer_test in iter_family_splits_antigravity(
        ordered
    ):
        for inner_family, inner_train, inner_test in iter_family_splits_antigravity(
            outer_train
        ):
            row_policy = _row_policy_antigravity(inner_train, None)
            residual_policy = StrictResidualPolicyAntigravity(
                k=K_ANTIGRAVITY
            ).fit(inner_train)
            for row in inner_test:
                routing["inner_rows"] += 1
                features = _feature_mapping_antigravity(row)
                row_decision = row_policy.route(features, row.proposal)
                if row_decision.method != BASELINE_METHOD_ANTIGRAVITY:
                    routing["row_knn_alternative_preserved"] += 1
                    continue
                residual = residual_policy.route(
                    features,
                    row_knn_method=row_decision.method,
                    enabled_methods=None,
                )
                routing[
                    "strict_candidate_accept"
                    if residual.changed
                    else "strict_local_gate_failed"
                ] += 1
                records.extend(
                    diagnose_assessments_antigravity(
                        residual.assessments,
                        outer_family=outer_family,
                        inner_family=inner_family,
                        dataset=row.dataset,
                    )
                )
    return {"routing": routing, "records": tuple(records)}


def summarize_census_antigravity(census: dict[str, object]) -> dict[str, object]:
    records = tuple(census["records"])
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        groups[(str(record["outer_family"]), str(record["method"]))].append(record)

    clauses = (
        "wins_not_greater",
        "negative_min_auc",
        "nonpositive_q20_auc",
        "candidate_rank_tie",
    )
    quantile_labels = ("min", "p20", "median", "p80", "max")
    quantiles = (0.0, 0.2, 0.5, 0.8, 1.0)
    margin_key_by_clause = {
        "wins_not_greater": "zf_margin",
        "negative_min_auc": "min_auc_margin",
        "nonpositive_q20_auc": "q20_auc_margin",
        "candidate_rank_tie": None,
    }
    clause_failures = []
    combinations = []
    margin_rows = []
    single_failures = []
    for (outer_family, method), items in sorted(groups.items()):
        candidate_count = len(items)
        clause_row: dict[str, object] = {
            "outer_family": outer_family,
            "method": method,
            "candidate_count": candidate_count,
        }
        for clause in clauses:
            count = sum(clause in item["failure_clauses"] for item in items)
            clause_row[f"{clause}_count"] = count
            clause_row[f"{clause}_rate"] = count / candidate_count
        clause_failures.append(clause_row)

        counts = Counter(
            "+".join(item["failure_clauses"]) or "pass" for item in items
        )
        combinations.extend(
            {
                "outer_family": outer_family,
                "method": method,
                "combination": combination,
                "count": count,
            }
            for combination, count in sorted(counts.items())
        )

        margin_row: dict[str, object] = {
            "outer_family": outer_family,
            "method": method,
            "candidate_count": candidate_count,
            "rank_tie_count": sum(
                item["rank_status"] == "best_tie" for item in items
            ),
        }
        for key in ("zf_margin", "min_auc_margin", "q20_auc_margin"):
            values = np.asarray([float(item[key]) for item in items], dtype=np.float64)
            for label, value in zip(
                quantile_labels, np.quantile(values, quantiles), strict=True
            ):
                margin_row[f"{key}_{label}"] = float(value)
        margin_rows.append(margin_row)

        by_clause: dict[str, list[dict[str, object]]] = defaultdict(list)
        for item in items:
            failures = tuple(item["failure_clauses"])
            if len(failures) == 1:
                by_clause[failures[0]].append(item)
        for clause, failed_items in sorted(by_clause.items()):
            margin_key = margin_key_by_clause[clause]
            values = np.asarray(
                [0.0 if margin_key is None else float(item[margin_key]) for item in failed_items],
                dtype=np.float64,
            )
            row: dict[str, object] = {
                "outer_family": outer_family,
                "method": method,
                "clause": clause,
                "count": len(failed_items),
            }
            for label, value in zip(
                quantile_labels, np.quantile(values, quantiles), strict=True
            ):
                row[f"margin_{label}"] = float(value)
            single_failures.append(row)

    routing = dict(census["routing"])
    return {
        "routing_composition": [
            {"reason": reason, "count": int(count)}
            for reason, count in routing.items()
        ],
        "clause_failures": clause_failures,
        "failure_combinations": combinations,
        "margin_quantiles": margin_rows,
        "single_clause_failures": single_failures,
    }


def aggregate_findings_antigravity(tables: dict[str, object]) -> dict[str, int]:
    clause_rows = tuple(tables["clause_failures"])
    combination_rows = tuple(tables["failure_combinations"])
    return {
        "candidate_count": sum(int(row["candidate_count"]) for row in clause_rows),
        "passed_count": sum(
            int(row["count"])
            for row in combination_rows
            if row["combination"] == "pass"
        ),
        "single_clause_failure_count": sum(
            int(row["count"])
            for row in combination_rows
            if row["combination"] != "pass" and "+" not in row["combination"]
        ),
        "multiple_clause_failure_count": sum(
            int(row["count"])
            for row in combination_rows
            if "+" in row["combination"]
        ),
        "wins_not_greater_count": sum(
            int(row["wins_not_greater_count"]) for row in clause_rows
        ),
        "negative_min_auc_count": sum(
            int(row["negative_min_auc_count"]) for row in clause_rows
        ),
        "nonpositive_q20_auc_count": sum(
            int(row["nonpositive_q20_auc_count"]) for row in clause_rows
        ),
        "candidate_rank_tie_count": sum(
            int(row["candidate_rank_tie_count"]) for row in clause_rows
        ),
    }


def render_dashboard_antigravity(result: dict[str, object]) -> str:
    identity = dict(result["exp481_identity"])
    findings = aggregate_findings_antigravity(dict(result["tables"]))
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Exp482 게이트 실패 진단 대시보드</title><style>
body{{font-family:system-ui,-apple-system,sans-serif;max-width:900px;margin:30px auto;padding:0 20px 60px;line-height:1.65;color:#182230}}
.status{{border-left:5px solid #175cd3;background:#eff8ff;padding:12px 16px}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}}.card{{border:1px solid #d8dee8;border-radius:9px;padding:14px}}.big{{font-size:1.5rem;font-weight:750}}label{{display:block;margin:8px 0}}
</style></head><body><h1>Exp482 게이트 실패 진단</h1>
<p class="status"><strong>진단 완료</strong> — 라우팅 규칙과 임계값은 바꾸지 않았습니다.</p>
<div class="cards"><div class="card"><div>데이터셋</div><div class="big">{int(result['dataset_count'])}</div></div><div class="card"><div>family</div><div class="big">{int(result['family_count'])}</div></div><div class="card"><div>후보 평가 기록</div><div class="big">{int(result['candidate_record_count'])}</div></div></div>
<p>Exp481 결정 동일: <strong>{'예' if identity['evaluation_equal'] else '아니요'}</strong></p>
<p>후보 통과: <strong>{findings['passed_count']}</strong>, 단일 조건 실패: <strong>{findings['single_clause_failure_count']}</strong>, 다중 조건 실패: <strong>{findings['multiple_clause_failure_count']}</strong></p>
<p><strong>다음 판단:</strong> 한 조건만 낮추지 않고, TRAIN 데이터에서 안전하게 좋아질 후보가 실제로 충분한지 먼저 확인합니다.</p>
<p><strong>Grok·Codex 검토:</strong> 단일 조건 완화는 기각하고 K=31은 유지합니다. 안전한 개선 기회가 20개 family 이상일 때만 결과-오라클 이웃 검사를 진행합니다.</p>
<h2>진행 체크</h2>
<label><input type="checkbox" checked disabled> Task 1 — 코드맵과 기준 테스트 확인</label>
<label><input type="checkbox" checked disabled> Task 2 — 실패 조건 센서스 구현</label>
<label><input type="checkbox" checked disabled> Task 3 — 1,117개 TRAIN-side 전수 계측</label>
<label><input type="checkbox" checked disabled> Task 4 — 결과 검증과 보고서 작성</label>
<p><a href="EXP482_IMPLEMENTATION_REPORT_antigravity.html">상세 진단 보고서 보기</a></p>
</body></html>"""


def _table_antigravity(rows: Sequence[dict[str, object]]) -> str:
    if not rows:
        return "<table><tbody><tr><td>해당 기록 없음</td></tr></tbody></table>"
    keys = tuple(rows[0])
    head = "".join(f"<th>{escape(key)}</th>" for key in keys)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{escape(str(row.get(key, '')))}</td>" for key in keys)
        + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def render_report_antigravity(result: dict[str, object]) -> str:
    tables = dict(result["tables"])
    findings = aggregate_findings_antigravity(tables)
    sections = (
        (
            "1. 내부 라우팅 구성",
            "후보가 실제로 채택됐는지, 엄격 조건에서 탈락했는지, 기존 대안을 그대로 썼는지 보여줍니다.",
            tables["routing_composition"],
        ),
        (
            "2. 조건별 실패",
            "각 method와 바깥 family에서 어떤 조건이 후보를 막았는지 보여줍니다.",
            tables["clause_failures"],
        ),
        (
            "3. 함께 실패한 조건",
            "한 조건만 실패했는지 여러 조건이 동시에 실패했는지 보여줍니다.",
            tables["failure_combinations"],
        ),
        (
            "4. 기준까지의 거리",
            "0에 가까우면 해당 기준을 아슬아슬하게 넘지 못한 것이고, 큰 음수면 기준에서 멀다는 뜻입니다.",
            tables["margin_quantiles"],
        ),
        (
            "5. 한 조건만 실패한 후보",
            "다른 조건은 모두 통과하고 한 조건만 실패한 후보와 그 거리를 보여줍니다.",
            tables["single_clause_failures"],
        ),
    )
    content = "".join(
        f"<h2>{title}</h2><p>{description}</p>{_table_antigravity(rows)}"
        for title, description, rows in sections
    )
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Exp482 게이트 실패 진단 보고서</title><style>
body{{font-family:system-ui,-apple-system,sans-serif;max-width:1180px;margin:30px auto;padding:0 20px 60px;line-height:1.62;color:#182230}}table{{border-collapse:collapse;width:100%;font-size:.84rem;display:block;overflow:auto;margin-bottom:26px}}th,td{{border:1px solid #d8dee8;padding:7px;white-space:nowrap;text-align:left}}th{{background:#eef2f7;position:sticky;top:0}}.status{{border-left:5px solid #175cd3;background:#eff8ff;padding:12px 16px}}
</style></head><body><h1>Exp482 게이트 실패 진단 보고서</h1>
<p class="status"><strong>목적:</strong> 임계값을 바꾸기 전에 Exp481 후보가 정확히 어느 조건에서 막혔는지 확인합니다.</p>
<p>쉽게 말해, 후보마다 시험 문제 네 개를 따로 채점했습니다. 이 보고서는 어떤 문제를 가장 자주 틀렸고 합격선에서 얼마나 멀었는지 보여줍니다.</p>
<h2>먼저 보는 결론</h2>
<p>전체 후보 <strong>{findings['candidate_count']}</strong>개 중 통과는 <strong>{findings['passed_count']}</strong>개, 한 조건만 실패한 후보는 <strong>{findings['single_clause_failure_count']}</strong>개, 여러 조건을 함께 실패한 후보는 <strong>{findings['multiple_clause_failure_count']}</strong>개입니다.</p>
<p>따라서 한 조건만 낮추기 전에, TRAIN 데이터 안에 ZF가 좋아지면서 AUC가 나빠지지 않는 후보가 충분히 존재하는지 확인해야 합니다. 그 기회가 충분한데도 현재 이웃이 찾지 못하면 7개 특징이나 거리 계산을 바꾸고, 기회 자체가 부족하면 현재 잔여 보정 후보군을 중단합니다.</p>
<h2>Grok 독립 검토와 최종 결정 규칙</h2>
<p>Grok의 독립 검토를 Exp482 표와 대조한 결과, 단일 조건 완화는 기각하고 K=31과 현재 세 조건은 다음 진단까지 유지합니다.</p>
<ol><li>방법별로 <code>zf_gain &gt; 0</code>이면서 <code>auc_gain ≥ 0</code>인 TRAIN 행과 그 family 수 <code>F(m)</code>을 셉니다.</li><li>세 방법 모두 <code>F(m) &lt; 20</code>이면 현재 세 대안으로 하는 잔여 보정을 종료합니다.</li><li>어떤 방법이 20개 family 이상이면 결과를 아는 진단용 오라클 K=31을 적용하고 엄격 조건을 다시 계산합니다.</li><li>오라클 통과 family 수 <code>A_oracle(m)</code>도 20보다 작으면 7개 특징을 바꾸지 않고 후보·게이트·K 구조를 다시 설계합니다.</li><li><code>A_oracle(m) ≥ 20</code>이면 K=31과 게이트를 유지한 채 7개 특징이나 거리 계산만 교체합니다.</li></ol>
<p>오라클은 가능성의 상한을 재는 TRAIN 전용 도구이며 실제 라우터에는 사용하지 않습니다.</p>
{content}</body></html>"""


def build_result_antigravity(
    census: dict[str, object],
    *,
    stored_evaluation: dict[str, object],
    recomputed_evaluation: dict[str, object],
    dataset_count: int,
    family_count: int,
    exp481_result_sha256: str,
) -> dict[str, object]:
    evaluation_equal = json_bytes_antigravity(
        stored_evaluation
    ) == json_bytes_antigravity(recomputed_evaluation)
    if not evaluation_equal:
        raise RuntimeError("Exp481 evaluation changed during diagnostic replay")
    records = tuple(census["records"])
    return {
        "schema": "exp482_gate_failure_decomposition_v1",
        "state": "diagnostic_complete",
        "dataset_count": int(dataset_count),
        "family_count": int(family_count),
        "candidate_record_count": len(records),
        "exp481_identity": {
            "evaluation_equal": True,
            "result_sha256": exp481_result_sha256,
            "summaries": stored_evaluation.get("summaries", {}),
        },
        "acceleration_assessment": {
            "selected_backend": "exp481_numpy_reference",
            "numba_assessed": True,
            "numba_selected": False,
            "numba_fallback_reason": (
                "no new numeric kernel; object and string grouping wraps the frozen "
                "Exp481 NumPy policy"
            ),
            "mlx_assessed": True,
            "mlx_selected": False,
            "mlx_fallback_reason": "diagnostic control flow is not a dense tensor stage",
        },
        "tables": summarize_census_antigravity(census),
    }


if __name__ == "__main__":
    validate_evidence_database_antigravity(EXP480_EVIDENCE_DATABASE_PATH_ANTIGRAVITY)
    result_bytes = OFFICIAL_RESULT_PATH_ANTIGRAVITY.read_bytes()
    result_sha = sha256(result_bytes).hexdigest()
    if result_sha != EXPECTED_EXP481_RESULT_SHA256_ANTIGRAVITY:
        raise RuntimeError("Exp481 official result SHA changed")
    stored_result = json.loads(result_bytes)
    rows = load_evidence_database_antigravity(EXP480_EVIDENCE_DATABASE_PATH_ANTIGRAVITY)
    print("stage=exp481_identity_replay", flush=True)
    recomputed = evaluate_nested_out_of_family_antigravity(rows)
    print("stage=train_side_gate_census", flush=True)
    census = collect_train_census_antigravity(rows)
    result = build_result_antigravity(
        census,
        stored_evaluation=stored_result["evaluation"],
        recomputed_evaluation=recomputed,
        dataset_count=len(rows),
        family_count=len({row.source_family for row in rows}),
        exp481_result_sha256=result_sha,
    )
    write_json_atomic_antigravity(RESULT_PATH_ANTIGRAVITY, result)
    DASHBOARD_PATH_ANTIGRAVITY.write_text(
        render_dashboard_antigravity(result), encoding="utf-8"
    )
    REPORT_PATH_ANTIGRAVITY.write_text(
        render_report_antigravity(result), encoding="utf-8"
    )
    print("stage=complete", flush=True)
