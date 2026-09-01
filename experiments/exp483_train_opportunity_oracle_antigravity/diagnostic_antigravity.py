"""Exp483 TRAIN-only residual opportunity audit."""

from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
from html import escape
import json
from pathlib import Path
from typing import Iterable, Sequence

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
    ALTERNATIVE_METHODS_ANTIGRAVITY,
    BASELINE_METHOD_ANTIGRAVITY,
    INNER_MIN_CHANGED_FAMILIES_ANTIGRAVITY,
    K_ANTIGRAVITY,
    METHODS_ANTIGRAVITY,
    EXP480_EVIDENCE_DATABASE_PATH_ANTIGRAVITY,
    OFFICIAL_RESULT_PATH_ANTIGRAVITY,
)
from experiments.exp481_strict_evidence_residual_correction_antigravity.evaluation_antigravity import (
    _feature_mapping_antigravity,
    _row_policy_antigravity,
)
from experiments.exp481_strict_evidence_residual_correction_antigravity.policy_antigravity import (
    strict_statistics_reference_antigravity,
)
from experiments.exp481_strict_evidence_residual_correction_antigravity.run_antigravity import (
    write_json_atomic_antigravity,
)


ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR_ANTIGRAVITY = Path(__file__).resolve().parent
RESULT_PATH_ANTIGRAVITY = EXPERIMENT_DIR_ANTIGRAVITY / "results_antigravity.json"
EXP482_RESULT_PATH_ANTIGRAVITY = (
    ROOT_ANTIGRAVITY
    / "experiments/exp482_gate_failure_decomposition_antigravity/results_antigravity.json"
)
DASHBOARD_PATH_ANTIGRAVITY = ROOT_ANTIGRAVITY / "docs/EXP483_DASHBOARD_antigravity.html"
REPORT_PATH_ANTIGRAVITY = (
    ROOT_ANTIGRAVITY / "docs/EXP483_IMPLEMENTATION_REPORT_antigravity.html"
)
EXPECTED_EXP481_RESULT_SHA256_ANTIGRAVITY = (
    "7038b994e422dc93a18927c810ce5e3bdb7601039ba01d2b388895444e09956f"
)
EXPECTED_EXP482_RESULT_SHA256_ANTIGRAVITY = (
    "d35f28ebb5d441f5a654b21f8de8f4c93a771aabe39067c00449a582c68bfab1"
)


def opportunity_records_antigravity(
    row: EvidenceRowAntigravity,
    *,
    inner_family: str,
) -> tuple[dict[str, object], ...]:
    """Compare each frozen residual candidate with MiniRocket on one TRAIN row."""
    baseline = row.scores[0]
    records = []
    for metric in row.scores[1:]:
        zf_gain = metric.zf - baseline.zf
        auc_gain = metric.auc - baseline.auc
        zf_positive = zf_gain > 0
        auc_nonnegative = auc_gain >= 0.0
        auc_positive = auc_gain > 0.0
        records.append(
            {
                "inner_family": inner_family,
                "dataset": row.dataset,
                "method": metric.method,
                "zf_positive": zf_positive,
                "auc_nonnegative": auc_nonnegative,
                "auc_positive": auc_positive,
                "opportunity": zf_positive and auc_nonnegative,
                "strict_positive": zf_positive and auc_positive,
            }
        )
    return tuple(records)


def summarize_opportunities_antigravity(
    records: Iterable[dict[str, object]],
) -> list[dict[str, object]]:
    """Aggregate row outcomes without collapsing repeated nested folds."""
    groups: dict[str, dict[str, object]] = defaultdict(
        lambda: {
            "evaluated_rows": 0,
            "zf_positive_count": 0,
            "auc_nonnegative_count": 0,
            "auc_positive_count": 0,
            "opportunity_count": 0,
            "strict_positive_count": 0,
            "opportunity_datasets": set(),
            "opportunity_families": set(),
        }
    )
    for record in records:
        group = groups[str(record["method"])]
        group["evaluated_rows"] = int(group["evaluated_rows"]) + 1
        for key in (
            "zf_positive",
            "auc_nonnegative",
            "auc_positive",
            "opportunity",
            "strict_positive",
        ):
            count_key = f"{key}_count"
            group[count_key] = int(group[count_key]) + int(bool(record[key]))
        if record["opportunity"]:
            group["opportunity_datasets"].add(str(record["dataset"]))
            group["opportunity_families"].add(str(record["inner_family"]))

    rows = []
    for method, group in sorted(groups.items()):
        evaluated = int(group["evaluated_rows"])
        rows.append(
            {
                "method": method,
                "evaluated_rows": evaluated,
                "zf_positive_count": int(group["zf_positive_count"]),
                "auc_nonnegative_count": int(group["auc_nonnegative_count"]),
                "auc_positive_count": int(group["auc_positive_count"]),
                "opportunity_count": int(group["opportunity_count"]),
                "strict_positive_count": int(group["strict_positive_count"]),
                "opportunity_dataset_count": len(group["opportunity_datasets"]),
                "opportunity_family_count": len(group["opportunity_families"]),
                "opportunity_rate": int(group["opportunity_count"]) / evaluated,
                "strict_positive_rate": int(group["strict_positive_count"]) / evaluated,
            }
        )
    return rows


def collect_opportunity_census_antigravity(
    rows: Sequence[EvidenceRowAntigravity], *, k: int = K_ANTIGRAVITY
) -> dict[str, object]:
    """Measure outcomes only where nested row-KNN keeps MiniRocket."""
    ordered = tuple(sorted(rows, key=lambda row: row.dataset))
    routing = {
        "inner_rows": 0,
        "minirocket_inner_rows": 0,
        "row_knn_alternative_preserved": 0,
    }

    def records_antigravity() -> Iterable[dict[str, object]]:
        for _outer_family, outer_train, _outer_test in iter_family_splits_antigravity(
            ordered
        ):
            for inner_family, inner_train, inner_test in iter_family_splits_antigravity(
                outer_train
            ):
                row_policy = _row_policy_antigravity(inner_train, None)
                oracle_by_method = {
                    method: oracle_neighbour_assessment_antigravity(
                        inner_train, method=method, k=k
                    )
                    for method in ALTERNATIVE_METHODS_ANTIGRAVITY
                }
                for row in inner_test:
                    routing["inner_rows"] += 1
                    decision = row_policy.route(
                        _feature_mapping_antigravity(row), row.proposal
                    )
                    if decision.method != BASELINE_METHOD_ANTIGRAVITY:
                        routing["row_knn_alternative_preserved"] += 1
                        continue
                    routing["minirocket_inner_rows"] += 1
                    for record in opportunity_records_antigravity(
                        row, inner_family=inner_family
                    ):
                        oracle = oracle_by_method[str(record["method"])]
                        yield {
                            **record,
                            "opportunity_neighbour_count": oracle[
                                "opportunity_family_count"
                            ],
                            "oracle_support_ge_k": oracle["support_ge_k"],
                            "oracle_accepted": bool(record["opportunity"])
                            and bool(oracle["passed"]),
                        }

    records = tuple(records_antigravity())
    return {
        "routing": routing,
        "opportunity_summary": summarize_opportunities_antigravity(records),
        **summarize_oracle_records_antigravity(records),
    }


def oracle_neighbour_assessment_antigravity(
    rows: Sequence[EvidenceRowAntigravity], *, method: str, k: int
) -> dict[str, object]:
    """Build the most favourable deterministic K-family TRAIN neighbourhood."""
    if method not in ALTERNATIVE_METHODS_ANTIGRAVITY:
        raise ValueError("method must be a frozen residual alternative")
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")
    method_index = METHODS_ANTIGRAVITY.index(method)
    opportunities = []
    for row in rows:
        baseline = row.scores[0]
        metric = row.scores[method_index]
        zf_gain = metric.zf - baseline.zf
        auc_gain = metric.auc - baseline.auc
        if zf_gain > 0 and auc_gain >= 0.0:
            opportunities.append(
                (row.source_family, row.dataset, zf_gain, auc_gain)
            )

    representatives = []
    seen = set()
    for item in sorted(
        opportunities,
        key=lambda value: (value[0], -value[3], -value[2], value[1]),
    ):
        if item[0] not in seen:
            seen.add(item[0])
            representatives.append(item)
    selected = tuple(
        sorted(
            representatives,
            key=lambda value: (-value[3], -value[2], value[0], value[1]),
        )[:k]
    )
    result: dict[str, object] = {
        "method": method,
        "k": k,
        "opportunity_family_count": len(representatives),
        "support": len(selected),
        "support_ge_k": len(selected) == k,
        "passed": False,
        "reason": (
            "strict_oracle_gate_failed"
            if len(selected) == k
            else "insufficient_distinct_families"
        ),
        "zf_wins": None,
        "zf_losses": None,
        "zf_ties": None,
        "min_auc_gain": None,
        "q20_auc_gain": None,
        "neighbour_families": tuple(item[0] for item in selected),
        "neighbour_datasets": tuple(item[1] for item in selected),
    }
    if len(selected) == k:
        stats = strict_statistics_reference_antigravity(
            [item[2] for item in selected], [item[3] for item in selected]
        )
        result.update(
            {
                "passed": stats.passed,
                "reason": (
                    "strict_oracle_accept"
                    if stats.passed
                    else "strict_oracle_gate_failed"
                ),
                "zf_wins": stats.zf_wins,
                "zf_losses": stats.zf_losses,
                "zf_ties": stats.zf_ties,
                "min_auc_gain": stats.min_auc_gain,
                "q20_auc_gain": stats.q20_auc_gain,
            }
        )
    return result


def summarize_oracle_records_antigravity(
    records: Iterable[dict[str, object]],
) -> dict[str, object]:
    """Summarize the optimistic TRAIN-only upper bound for opportunity rows."""
    groups: dict[str, dict[str, object]] = defaultdict(
        lambda: {
            "opportunity_count": 0,
            "support_count": 0,
            "accept_count": 0,
            "opportunity_families": set(),
            "accept_datasets": set(),
            "accept_families": set(),
            "neighbour_counts": Counter(),
        }
    )
    for record in records:
        if not record["opportunity"]:
            continue
        group = groups[str(record["method"])]
        group["opportunity_count"] = int(group["opportunity_count"]) + 1
        group["support_count"] = int(group["support_count"]) + int(
            bool(record["oracle_support_ge_k"])
        )
        group["accept_count"] = int(group["accept_count"]) + int(
            bool(record["oracle_accepted"])
        )
        group["opportunity_families"].add(str(record["inner_family"]))
        group["neighbour_counts"][int(record["opportunity_neighbour_count"])] += 1
        if record["oracle_accepted"]:
            group["accept_datasets"].add(str(record["dataset"]))
            group["accept_families"].add(str(record["inner_family"]))

    summary = []
    distributions = []
    for method, group in sorted(groups.items()):
        count = int(group["opportunity_count"])
        summary.append(
            {
                "method": method,
                "opportunity_count": count,
                "opportunity_family_count": len(group["opportunity_families"]),
                "opportunity_neighbour_ge_k_count": int(group["support_count"]),
                "opportunity_neighbour_ge_k_rate": int(group["support_count"])
                / count,
                "oracle_accept_count": int(group["accept_count"]),
                "oracle_accept_dataset_count": len(group["accept_datasets"]),
                "oracle_accept_family_count": len(group["accept_families"]),
                "oracle_accept_rate": int(group["accept_count"]) / count,
            }
        )
        distributions.extend(
            {
                "method": method,
                "opportunity_neighbour_count": neighbour_count,
                "target_opportunity_rows": row_count,
            }
            for neighbour_count, row_count in sorted(
                group["neighbour_counts"].items()
            )
        )
    return {
        "oracle_summary": summary,
        "opportunity_neighbour_counts": distributions,
    }


def decide_next_step_antigravity(
    opportunity_summary: Sequence[dict[str, object]],
    oracle_summary: Sequence[dict[str, object]],
    *,
    min_families: int,
) -> dict[str, object]:
    """Apply the frozen Exp483 family-count decision tree."""
    oracle_by_method = {str(row["method"]): row for row in oracle_summary}
    relevant = tuple(
        str(row["method"])
        for row in opportunity_summary
        if int(row["opportunity_family_count"]) >= min_families
    )
    qualifying = tuple(
        method
        for method in relevant
        if int(oracle_by_method.get(method, {}).get("oracle_accept_family_count", 0))
        >= min_families
    )
    if not relevant:
        decision = "abandon_current_residual_candidates"
    elif qualifying:
        decision = "replace_descriptors_and_distance"
    else:
        decision = "redesign_candidates_gate_or_k"
    return {
        "decision": decision,
        "min_families": min_families,
        "relevant_methods": relevant,
        "qualifying_methods": qualifying,
    }


def build_result_antigravity(
    census: dict[str, object],
    *,
    exp482_result: dict[str, object],
    dataset_count: int,
    family_count: int,
    exp481_result_sha256: str,
    exp482_result_sha256: str,
) -> dict[str, object]:
    """Freeze the diagnostic only when its nested population matches Exp482."""
    stored_routing = {
        str(row["reason"]): int(row["count"])
        for row in exp482_result["tables"]["routing_composition"]
    }
    expected_routing = {
        "inner_rows": stored_routing["inner_rows"],
        "minirocket_inner_rows": stored_routing["strict_local_gate_failed"]
        + stored_routing["strict_candidate_accept"],
        "row_knn_alternative_preserved": stored_routing[
            "row_knn_alternative_preserved"
        ],
    }
    if census["routing"] != expected_routing:
        raise RuntimeError("Exp482 routing changed during Exp483 replay")
    opportunity_summary = list(census["opportunity_summary"])
    oracle_summary = list(census["oracle_summary"])
    return {
        "schema": "exp483_train_opportunity_oracle_v1",
        "state": "diagnostic_complete",
        "dataset_count": int(dataset_count),
        "family_count": int(family_count),
        "candidate_record_count": expected_routing["minirocket_inner_rows"] * 3,
        "source_identity": {
            "exp481_result_sha256": exp481_result_sha256,
            "exp482_result_sha256": exp482_result_sha256,
            "exp482_routing_equal": True,
        },
        "acceleration_assessment": {
            "selected_backend": "frozen_numpy_policy_plus_python_grouping",
            "numba_assessed": True,
            "numba_selected": False,
            "numba_parity_result": "not_applicable",
            "numba_benchmark_seconds": None,
            "numba_fallback_reason": (
                "new work is string-keyed family grouping; frozen numeric policies are reused"
            ),
            "mlx_assessed": True,
            "mlx_selected": False,
            "mlx_parity_result": "not_applicable",
            "mlx_benchmark_seconds": None,
            "mlx_fallback_reason": "no new dense tensor operation",
        },
        "decision": decide_next_step_antigravity(
            opportunity_summary,
            oracle_summary,
            min_families=INNER_MIN_CHANGED_FAMILIES_ANTIGRAVITY,
        ),
        "tables": {
            "routing_composition": [
                {"reason": reason, "count": int(count)}
                for reason, count in census["routing"].items()
            ],
            "opportunity_summary": opportunity_summary,
            "oracle_summary": oracle_summary,
            "opportunity_neighbour_counts": list(
                census["opportunity_neighbour_counts"]
            ),
        },
    }


def _decision_text_antigravity(decision: str) -> str:
    return {
        "abandon_current_residual_candidates": "현재 세 후보를 이용한 잔여 보정을 중단합니다.",
        "redesign_candidates_gate_or_k": "후보 방법·통과 규칙·K 구조를 다시 설계합니다.",
        "replace_descriptors_and_distance": "7개 특징과 거리 계산을 개선합니다.",
    }[decision]


def _method_summary_antigravity(rows: Sequence[dict[str, object]]) -> str:
    labels = {
        "hydra_local_gap": "Hydra",
        "hydra_multirocket_local_gap": "Hydra+MultiRocket",
        "multirocket_local_gap": "MultiRocket",
    }
    return "<ul>" + "".join(
        f"<li>{labels[str(row['method'])]}: 개선 기회 "
        f"<strong>{int(row['opportunity_family_count'])}개 family</strong> → "
        f"상한선 통과 <strong>{int(row['oracle_accept_family_count'])}개 family</strong></li>"
        for row in rows
    ) + "</ul>"


def _table_antigravity(rows: Sequence[dict[str, object]]) -> str:
    if not rows:
        return "<table><tbody><tr><td>해당 기록 없음</td></tr></tbody></table>"
    preferred = ("reason", "method")
    keys = tuple(
        sorted(
            rows[0],
            key=lambda key: (
                preferred.index(key) if key in preferred else len(preferred),
                key,
            ),
        )
    )
    head = "".join(f"<th>{escape(key)}</th>" for key in keys)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{escape(str(row.get(key, '')))}</td>" for key in keys)
        + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def render_dashboard_antigravity(result: dict[str, object]) -> str:
    tables = result["tables"]
    decision_text = _decision_text_antigravity(str(result["decision"]["decision"]))
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Exp483 TRAIN 개선 가능성 대시보드</title><style>
body{{font-family:system-ui,-apple-system,sans-serif;max-width:940px;margin:30px auto;padding:0 20px 60px;line-height:1.65;color:#182230}}.status{{border-left:5px solid #175cd3;background:#eff8ff;padding:12px 16px}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}}.card{{border:1px solid #d8dee8;border-radius:9px;padding:14px}}.big{{font-size:1.5rem;font-weight:750}}table{{border-collapse:collapse;width:100%;display:block;overflow:auto}}th,td{{border:1px solid #d8dee8;padding:7px;white-space:nowrap;text-align:left}}th{{background:#eef2f7}}label{{display:block;margin:8px 0}}
</style></head><body><h1>Exp483 TRAIN 개선 가능성 검사</h1>
<p class="status"><strong>진단 완료</strong> — {decision_text}</p>
<div class="cards"><div class="card"><div>데이터셋</div><div class="big">{int(result['dataset_count'])}</div></div><div class="card"><div>family</div><div class="big">{int(result['family_count'])}</div></div><div class="card"><div>MiniRocket 검사 행</div><div class="big">{int(tables['routing_composition'][1]['count'])}</div></div></div>
<h2>방법별 핵심 결과</h2>{_method_summary_antigravity(tables['oracle_summary'])}{_table_antigravity(tables['oracle_summary'])}
<p><strong>결론:</strong> {decision_text}</p>
<p>오라클은 결과를 미리 아는 낙관적인 TRAIN 전용 검사이며 실제 라우터에는 넣지 않습니다.</p>
<h2>진행 체크</h2>
<label><input type="checkbox" checked disabled> Task 1 — 기준 결과와 코드맵 확인</label>
<label><input type="checkbox" checked disabled> Task 2 — 개선 기회 F(m) 전수 검사</label>
<label><input type="checkbox" checked disabled> Task 3 — K=31 오라클 상한선 검사</label>
<label><input type="checkbox" checked disabled> Task 4 — 결과 검증과 HTML 보고서 작성</label>
<p><a href="EXP483_IMPLEMENTATION_REPORT_antigravity.html">상세 결과 보고서 보기</a></p>
</body></html>"""


def render_report_antigravity(result: dict[str, object]) -> str:
    tables = result["tables"]
    decision = str(result["decision"]["decision"])
    decision_text = _decision_text_antigravity(decision)
    if decision == "replace_descriptors_and_distance":
        explanation_title = "왜 특징과 거리가 문제인가"
        explanation = (
            "결과를 미리 알고 올바른 TRAIN family를 골랐을 때 세 방법 모두 "
            "20개가 넘는 family에서 현재 K=31과 세 통과 조건을 만족했습니다. "
            "즉 후보 방법에 개선 능력은 있지만, 현재 7개 특징과 거리 계산이 "
            "그 family들을 가까운 이웃으로 찾지 못하고 있습니다."
        )
    else:
        explanation_title = "왜 이 방향을 선택했는가"
        explanation = "사전 등록한 family 수 기준 20에 따라 다음 단계를 선택했습니다."
    sections = (
        (
            "1. 검사 대상 구성",
            "중첩 TRAIN에서 MiniRocket을 유지한 행만 후보 방법과 비교했습니다.",
            tables["routing_composition"],
        ),
        (
            "2. 실제 개선 기회",
            "ZF가 늘고 AUC가 줄지 않은 행과 서로 다른 family 수 F(m)를 셌습니다.",
            tables["opportunity_summary"],
        ),
        (
            "3. 결과를 아는 K=31 상한선",
            "각 TRAIN family에서 가장 좋은 행 하나만 사용해 현재 세 통과 조건을 그대로 계산했습니다.",
            tables["oracle_summary"],
        ),
        (
            "4. 확보 가능한 개선 이웃 수",
            "개선 가능성이 있는 목표 행마다 서로 다른 TRAIN family 이웃을 몇 개 확보할 수 있는지 보여줍니다.",
            tables["opportunity_neighbour_counts"],
        ),
    )
    content = "".join(
        f"<h2>{title}</h2><p>{description}</p>{_table_antigravity(rows)}"
        for title, description, rows in sections
    )
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Exp483 TRAIN 개선 가능성 보고서</title><style>
body{{font-family:system-ui,-apple-system,sans-serif;max-width:1180px;margin:30px auto;padding:0 20px 60px;line-height:1.62;color:#182230}}table{{border-collapse:collapse;width:100%;font-size:.84rem;display:block;overflow:auto;margin-bottom:26px}}th,td{{border:1px solid #d8dee8;padding:7px;white-space:nowrap;text-align:left}}th{{background:#eef2f7;position:sticky;top:0}}.status{{border-left:5px solid #175cd3;background:#eff8ff;padding:12px 16px}}
</style></head><body><h1>Exp483 TRAIN 개선 가능성 보고서</h1>
<p class="status"><strong>최종 방향:</strong> {decision_text}</p>
<p>먼저 후보 방법이 실제로 좋아지는 경우가 충분한지 확인했습니다. 그다음 7개 특징과 거리 계산을 완전히 빼고, 결과를 미리 아는 가장 유리한 이웃을 골랐을 때 현재 규칙이 통과할 수 있는지 확인했습니다.</p>
<p><strong>주의:</strong> 이 오라클은 가능성의 상한만 재는 TRAIN 전용 도구입니다. 실제 라우터에는 사용하지 않습니다.</p>
<h2>핵심 숫자</h2>{_method_summary_antigravity(tables['oracle_summary'])}
<h2>{explanation_title}</h2><p>{explanation}</p>
{content}</body></html>"""


__all__ = [
    "build_result_antigravity",
    "collect_opportunity_census_antigravity",
    "decide_next_step_antigravity",
    "oracle_neighbour_assessment_antigravity",
    "opportunity_records_antigravity",
    "render_dashboard_antigravity",
    "render_report_antigravity",
    "summarize_opportunities_antigravity",
    "summarize_oracle_records_antigravity",
]


if __name__ == "__main__":
    validate_evidence_database_antigravity(EXP480_EVIDENCE_DATABASE_PATH_ANTIGRAVITY)
    exp481_bytes = OFFICIAL_RESULT_PATH_ANTIGRAVITY.read_bytes()
    exp482_bytes = EXP482_RESULT_PATH_ANTIGRAVITY.read_bytes()
    exp481_sha = sha256(exp481_bytes).hexdigest()
    exp482_sha = sha256(exp482_bytes).hexdigest()
    if exp481_sha != EXPECTED_EXP481_RESULT_SHA256_ANTIGRAVITY:
        raise RuntimeError("Exp481 official result SHA changed")
    if exp482_sha != EXPECTED_EXP482_RESULT_SHA256_ANTIGRAVITY:
        raise RuntimeError("Exp482 diagnostic result SHA changed")
    rows = load_evidence_database_antigravity(EXP480_EVIDENCE_DATABASE_PATH_ANTIGRAVITY)
    print("stage=train_opportunity_and_oracle_census", flush=True)
    census = collect_opportunity_census_antigravity(rows)
    result = build_result_antigravity(
        census,
        exp482_result=json.loads(exp482_bytes),
        dataset_count=len(rows),
        family_count=len({row.source_family for row in rows}),
        exp481_result_sha256=exp481_sha,
        exp482_result_sha256=exp482_sha,
    )
    write_json_atomic_antigravity(RESULT_PATH_ANTIGRAVITY, result)
    DASHBOARD_PATH_ANTIGRAVITY.write_text(
        render_dashboard_antigravity(result), encoding="utf-8"
    )
    REPORT_PATH_ANTIGRAVITY.write_text(
        render_report_antigravity(result), encoding="utf-8"
    )
    print("stage=complete", flush=True)
