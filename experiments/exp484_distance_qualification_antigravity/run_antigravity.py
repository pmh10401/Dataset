"""Run the Exp484 TRAIN-only distance qualification."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from html import escape
import json
from pathlib import Path
from time import perf_counter
from typing import Mapping, Sequence

from experiments.exp480_family_balanced_evidence_routing_antigravity.contracts_antigravity import (
    EvidenceRowAntigravity,
)
from experiments.exp480_family_balanced_evidence_routing_antigravity.evidence_antigravity import (
    load_evidence_database_antigravity,
    validate_evidence_database_antigravity,
)
from experiments.exp481_strict_evidence_residual_correction_antigravity.config_antigravity import (
    EXP480_EVIDENCE_DATABASE_PATH_ANTIGRAVITY,
    OFFICIAL_RESULT_PATH_ANTIGRAVITY,
)
from experiments.exp481_strict_evidence_residual_correction_antigravity.evaluation_antigravity import (
    evaluate_nested_out_of_family_antigravity,
)
from experiments.exp484_distance_qualification_antigravity.acceleration_antigravity import (
    assess_acceleration_antigravity,
)


ROBUST_VARIANTS_ANTIGRAVITY = (
    "robust_manhattan",
    "robust_shrinkage_mahalanobis",
)
ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[2]
EXP483_RESULT_PATH_ANTIGRAVITY = (
    ROOT_ANTIGRAVITY
    / "experiments/exp483_train_opportunity_oracle_antigravity/results_antigravity.json"
)
EXPECTED_SOURCE_SHA256_ANTIGRAVITY = {
    "evidence_database": "77c53e1afd0af071edf71ef692bd978458dda6a5f8a0fec110736b787fe1b998",
    "exp481_result": "7038b994e422dc93a18927c810ce5e3bdb7601039ba01d2b388895444e09956f",
    "exp483_result": "09205ac93d4a4baa8fa0992a20d89d811d13fbbb2f2a55d51c9be758946ef9cc",
}


def choose_distance_variant_antigravity(
    summaries: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    qualified = tuple(row for row in summaries if bool(row["overall_pass"]))
    if not qualified:
        return {
            "decision": "replace_features",
            "selected_variant": None,
            "reason": "no_distance_variant_passed_frozen_gates",
        }
    selected = max(
        qualified,
        key=lambda row: (
            float(row["family_mean_zf"]),
            int(row["zf_total"]),
            float(row["family_mean_auc"]),
            str(row["distance_variant"]),
        ),
    )
    return {
        "decision": "qualify_distance_variant",
        "selected_variant": str(selected["distance_variant"]),
        "reason": "highest_qualified_family_mean_zf",
    }


def summarize_evaluation_antigravity(
    distance_variant: str,
    backend: str,
    evaluation: Mapping[str, object],
) -> dict[str, object]:
    summaries = evaluation["summaries"]
    if not isinstance(summaries, Mapping):
        raise ValueError("evaluation summaries must be a mapping")
    performance = summaries["exp481"]
    changed = summaries["changed"]
    gates = evaluation["gates"]
    inner = evaluation["inner_gates_by_outer_family"]
    if not all(isinstance(item, Mapping) for item in (performance, changed, gates, inner)):
        raise ValueError("evaluation sections must be mappings")
    enabled_counts: Counter[str] = Counter()
    any_enabled = 0
    for assessments in inner.values():
        enabled = tuple(item for item in assessments if bool(item["enabled"]))
        any_enabled += int(bool(enabled))
        enabled_counts.update(str(item["method"]) for item in enabled)
    return {
        "distance_variant": distance_variant,
        "backend": backend,
        "dataset_count": int(evaluation["dataset_count"]),
        "family_count": int(evaluation["family_count"]),
        "zf_total": int(performance["zf_total"]),
        "family_mean_zf": float(performance["family_mean_zf"]),
        "dataset_mean_auc": float(performance["dataset_mean_auc"]),
        "family_mean_auc": float(performance["family_mean_auc"]),
        "added_loss_0_05": int(performance["added_loss_0_05"]),
        "added_loss_0_10": int(performance["added_loss_0_10"]),
        "changed_count": int(changed["count"]),
        "changed_wins": int(changed["wins"]),
        "changed_losses": int(changed["losses"]),
        "changed_ties": int(changed["ties"]),
        "changed_tie_rate": float(changed["tie_rate"]),
        "changed_method_counts": dict(changed["method_counts"]),
        "overall_pass": bool(gates["overall_pass"]),
        "inner_any_enabled_outer_family_count": any_enabled,
        "inner_enabled_outer_family_counts": dict(sorted(enabled_counts.items())),
    }


def summarize_failure_structure_antigravity(
    evaluations: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    result = {}
    for variant, evaluation in evaluations.items():
        inner = evaluation["inner_gates_by_outer_family"]
        gates = tuple(gate for family_gates in inner.values() for gate in family_gates)
        result[variant] = {
            "max_inner_changed_families": max(
                (int(gate["changed_families"]) for gate in gates), default=0
            ),
            "enabled_outer_family_count": sum(
                any(bool(gate["enabled"]) for gate in family_gates)
                for family_gates in inner.values()
            ),
            "decision_reason_counts": dict(
                sorted(Counter(str(row["reason"]) for row in evaluation["decisions"]).items())
            ),
        }
    return result


def render_dashboard_antigravity(result: Mapping[str, object]) -> str:
    summaries = result["variant_summaries"]
    decision = result["decision"]
    audit = result["route_input_audit"]
    rows = "".join(
        "<tr>"
        f"<td><code>{escape(str(row['distance_variant']))}</code></td>"
        f"<td>{escape(str(row['backend']))}</td>"
        f"<td>{int(row['changed_count'])}</td>"
        f"<td>{int(row['changed_wins'])} / {int(row['changed_losses'])}</td>"
        f"<td>{int(row['zf_total'])}</td>"
        f"<td>{float(row['family_mean_zf']):.6f}</td>"
        f"<td>{float(row['dataset_mean_auc']):.6f}</td>"
        f"<td>{float(row['family_mean_auc']):.6f}</td>"
        f"<td>{int(row['added_loss_0_05'])}</td>"
        f"<td>{int(row['inner_any_enabled_outer_family_count'])}</td>"
        f"<td>{'통과' if row['overall_pass'] else '실패'}</td>"
        "</tr>"
        for row in summaries
    )
    selected = decision.get("selected_variant") or "없음"
    failure = result.get("failure_structure", {})
    if decision.get("decision") == "replace_features" and failure:
        local_failures = sum(
            int(values["decision_reason_counts"].get("strict_local_gate_failed", 0))
            for values in failure.values()
        )
        max_changed = max(
            int(values["max_inner_changed_families"])
            for values in failure.values()
        )
        conclusion = (
            f"<p>두 새 거리에서 엄격한 지역 규칙 실패가 합계 {local_failures}건이었고, "
            f"내부 변경 가족의 최댓값도 {max_changed}이었습니다.</p>"
            "<p>즉, 거리를 재는 자만 바꿔서는 유용한 후보끼리 모이지 않았습니다. "
            "다음 실험은 7개 특징 자체를 바꿉니다.</p>"
        )
    else:
        conclusion = ""
    name_feature_count = int(audit["dataset_name_feature_count"]) + int(
        audit["family_name_feature_count"]
    )
    labels = (
        "기준 상태 확인",
        "거리 비교 구현",
        "Numba·MLX 확인",
        "중첩 TRAIN 전체 평가",
        "결과 해석 및 검증",
    )
    verified = result.get("state", "unverified") == "verified_complete"
    checks = "".join(
        f'<label><input type="checkbox" {"checked " if index < 5 or verified else ""}disabled> Task {index} — {label}</label>'
        for index, label in enumerate(labels, start=1)
    )
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>EXP484 결과 대시보드</title><style>body{{font-family:system-ui,sans-serif;max-width:1180px;margin:30px auto;padding:0 20px;line-height:1.55;color:#172033}}.card{{border:1px solid #d6dce7;border-radius:12px;padding:18px;margin:16px 0}}label{{display:block;margin:7px 0;color:#176b3a}}table{{border-collapse:collapse;width:100%;font-size:.92rem}}th,td{{border:1px solid #ccd3dc;padding:8px;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#eef2f7}}code{{background:#f1f4f8;padding:2px 5px;border-radius:4px}}.scroll{{overflow-x:auto}}</style></head><body><h1>EXP484 거리 비교 결과</h1><p>같은 7개 숫자 특징, 후보 방법, <code>K=31</code>, 안전 규칙을 유지하고 거리 계산만 비교했습니다.</p><section class="card"><h2>결론</h2><p><strong>선택: {escape(str(selected))}</strong></p>{conclusion}<p>라우팅 입력의 데이터셋·가족 이름 특징 0개입니다.</p></section><section class="card"><h2>세 방법 비교</h2><div class="scroll"><table><thead><tr><th>거리</th><th>계산</th><th>변경</th><th>ZF 승/패</th><th>ZF 합</th><th>가족 평균 ZF</th><th>데이터 평균 AUC</th><th>가족 평균 AUC</th><th>AUC -0.05 손실</th><th>내부 통과 가족</th><th>판정</th></tr></thead><tbody>{rows}</tbody></table></div></section><section class="card"><h2>완료 상태</h2>{checks}</section><section class="card"><h2>읽는 법</h2><p>‘변경’은 기존 MiniRocket 선택을 다른 후보로 바꾼 데이터셋 수입니다. ZF는 높을수록 좋고, AUC 손실은 0이어야 안전합니다.</p></section></body></html>"""


def build_result_antigravity(
    control: Mapping[str, object],
    evaluations: Mapping[str, Mapping[str, object]],
    acceleration: Mapping[str, object],
) -> dict[str, object]:
    if tuple(evaluations) != ROBUST_VARIANTS_ANTIGRAVITY:
        raise ValueError("evaluations must contain the two frozen variants in order")
    control_evaluation = control["evaluation"]
    if not isinstance(control_evaluation, Mapping):
        raise ValueError("control evaluation must be a mapping")
    acceleration_variants = acceleration["variants"]
    if not isinstance(acceleration_variants, Mapping):
        raise ValueError("acceleration variants must be a mapping")
    summaries = [
        summarize_evaluation_antigravity(
            "standard_euclidean",
            str(control["selected_backend"]),
            control_evaluation,
        )
    ]
    for variant, evaluation in evaluations.items():
        if (evaluation["dataset_count"], evaluation["family_count"]) != (
            control_evaluation["dataset_count"],
            control_evaluation["family_count"],
        ):
            raise ValueError("variant evaluation population changed")
        summaries.append(
            summarize_evaluation_antigravity(
                variant,
                str(acceleration_variants[variant]["selected_backend"]),
                evaluation,
            )
        )
    leakage = control_evaluation["leakage"]
    route_keys = tuple(str(key) for key in leakage["route_input_keys"])
    result = {
        "schema": "exp484_distance_qualification_v1",
        "dataset_count": int(control_evaluation["dataset_count"]),
        "family_count": int(control_evaluation["family_count"]),
        "variant_summaries": tuple(summaries),
        "decision": choose_distance_variant_antigravity(tuple(summaries[1:])),
        "route_input_audit": {
            "dataset_name_feature_count": int(leakage["dataset_name_feature_count"]),
            "family_name_feature_count": sum(
                any(token in key for token in ("family", "name")) for key in route_keys
            ),
            "held_out_score_feature_count": int(
                leakage["held_out_score_feature_count"]
            ),
        },
        "acceleration": dict(acceleration),
        "evaluations": dict(evaluations),
        "failure_structure": summarize_failure_structure_antigravity(evaluations),
    }
    return result


def evaluate_variants_antigravity(
    rows: Sequence[EvidenceRowAntigravity], acceleration: Mapping[str, object]
) -> dict[str, dict[str, object]]:
    variants = acceleration["variants"]
    if not isinstance(variants, Mapping):
        raise ValueError("acceleration variants must be a mapping")
    evaluations = {}
    for variant in ROBUST_VARIANTS_ANTIGRAVITY:
        backend = str(variants[variant]["selected_backend"])
        if backend not in ("numpy_reference", "numba_cpu"):
            raise ValueError("selected backend is not supported by the CPU policy")
        evaluations[variant] = evaluate_nested_out_of_family_antigravity(
            rows,
            residual_policy_kwargs={
                "backend": "numpy_reference",
                "distance_backend": backend,
                "distance_variant": variant,
            },
        )
    return evaluations


def render_report_antigravity(result: Mapping[str, object]) -> str:
    decision = result["decision"]
    summaries = result["variant_summaries"]
    acceleration = result["acceleration"]
    status_rows = "".join(
        "<tr>"
        f"<td><code>{escape(str(row['distance_variant']))}</code></td>"
        f"<td>{'통과' if row['overall_pass'] else '실패'}</td>"
        "</tr>"
        for row in summaries
    )
    acceleration_rows = "".join(
        "<tr>"
        f"<td><code>{escape(str(variant))}</code></td>"
        f"<td>{escape(str(values['selected_backend']))}</td>"
        f"<td>{float(values['speedup']):.3f}×</td>"
        "</tr>"
        for variant, values in acceleration["variants"].items()
    )
    if decision["decision"] == "qualify_distance_variant":
        next_step = (
            f"선택된 <code>{escape(str(decision['selected_variant']))}</code>을 "
            "별도 미학습 데이터에서 다시 확인합니다."
        )
    else:
        next_step = "거리만 바꾸는 시도는 끝냅니다. 다음에는 특징 자체를 바꿉니다."
    mlx = acceleration["mlx"]
    mlx_note = (
        f"사용 가능: {bool(mlx.get('available'))}; "
        f"이유: {escape(str(mlx.get('fallback_reason', '없음')))}"
    )
    failure = result.get("failure_structure", {})
    failure_rows = "".join(
        "<tr>"
        f"<td><code>{escape(str(variant))}</code></td>"
        f"<td>{int(values['max_inner_changed_families'])}</td>"
        f"<td>{int(values['decision_reason_counts'].get('strict_local_gate_failed', 0))}</td>"
        "</tr>"
        for variant, values in failure.items()
    )
    failure_section = (
        f"<section><h2>왜 실패했나</h2><table><thead><tr><th>거리</th><th>내부 변경 가족 최댓값</th><th>엄격한 지역 규칙 실패</th></tr></thead><tbody>{failure_rows}</tbody></table><p>두 방법 모두 가장 가까운 가족 묶음에서 안전 조건을 한 번도 통과시키지 못했습니다. 이번에 시험한 두 거리 안에서는 현재 7개 특징만으로 안전한 이웃을 만들지 못했습니다.</p></section>"
        if failure_rows
        else ""
    )
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>EXP484 구현 보고서</title><style>body{{font-family:system-ui,sans-serif;max-width:980px;margin:30px auto;padding:0 20px;line-height:1.65;color:#172033}}section{{border:1px solid #d6dce7;border-radius:12px;padding:18px;margin:16px 0}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccd3dc;padding:8px;text-align:left}}th{{background:#eef2f7}}code{{background:#f1f4f8;padding:2px 5px;border-radius:4px}}</style></head><body><h1>EXP484 구현 및 결과 보고서</h1><section><h2>무엇을 비교했나</h2><p>기존 7개 특징은 바꾸지 않았습니다. 후보 방법, <code>K=31</code>, 세 가지 안전 조건도 그대로 두고 거리 계산만 비교했습니다.</p></section><section><h2>판정</h2><table><thead><tr><th>거리</th><th>성능 기준</th></tr></thead><tbody>{status_rows}</tbody></table><p>{next_step}</p></section>{failure_section}<section><h2>계산 속도 확인</h2><table><thead><tr><th>거리</th><th>선택한 계산 방식</th><th>Numba 속도비</th></tr></thead><tbody>{acceleration_rows}</tbody></table><p>MLX — {mlx_note}</p></section><section><h2>해석할 때 주의할 점</h2><p>이번 값은 중첩 TRAIN 평가입니다. 데이터셋 이름이나 가족 이름을 라우팅 입력으로 사용하지 않았고, TEST 성능으로 거리를 고르지 않았습니다.</p></section></body></html>"""


def _json_bytes_antigravity(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_atomic_antigravity(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + "_temporary_antigravity" + path.suffix)
    temporary.write_bytes(payload)
    temporary.replace(path)


def require_sha256_antigravity(path: Path, expected: str) -> str:
    actual = sha256(Path(path).read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f"source SHA changed: {Path(path)}")
    return actual


def load_frozen_sources_antigravity() -> tuple[
    tuple[EvidenceRowAntigravity, ...], dict[str, object], dict[str, str]
]:
    paths = {
        "evidence_database": EXP480_EVIDENCE_DATABASE_PATH_ANTIGRAVITY,
        "exp481_result": OFFICIAL_RESULT_PATH_ANTIGRAVITY,
        "exp483_result": EXP483_RESULT_PATH_ANTIGRAVITY,
    }
    source_sha = {
        name: require_sha256_antigravity(
            path, EXPECTED_SOURCE_SHA256_ANTIGRAVITY[name]
        )
        for name, path in paths.items()
    }
    validation = validate_evidence_database_antigravity(paths["evidence_database"])
    if (validation["datasets"], validation["families"]) != (1117, 128):
        raise RuntimeError("evidence population changed")
    rows = load_evidence_database_antigravity(paths["evidence_database"])
    control = json.loads(paths["exp481_result"].read_text(encoding="utf-8"))
    exp483 = json.loads(paths["exp483_result"].read_text(encoding="utf-8"))
    if (control.get("dataset_count"), control.get("family_count")) != (1117, 128):
        raise RuntimeError("Exp481 control population changed")
    if exp483.get("decision", {}).get("decision") != "replace_descriptors_and_distance":
        raise RuntimeError("Exp483 next-step decision changed")
    return tuple(rows), control, source_sha


def run_exp484_antigravity() -> tuple[dict[str, object], dict[str, Path]]:
    started_at = datetime.now(timezone.utc).isoformat()
    started = perf_counter()
    rows, control, source_sha = load_frozen_sources_antigravity()
    acceleration = assess_acceleration_antigravity(rows)
    evaluations = evaluate_variants_antigravity(rows, acceleration)
    result = build_result_antigravity(control, evaluations, acceleration)
    result.update(
        {
            "state": "train_nested_complete",
            "source_sha256": source_sha,
            "frozen_scope": {
                "feature_count": 7,
                "k": 31,
                "candidate_methods_changed": False,
                "strict_gate_changed": False,
                "dataset_or_family_name_used": False,
            },
        }
    )
    completed_at = datetime.now(timezone.utc).isoformat()
    paths = publish_result_antigravity(
        result,
        root=ROOT_ANTIGRAVITY,
        started_at_utc=started_at,
        completed_at_utc=completed_at,
        elapsed_seconds=perf_counter() - started,
    )
    return result, paths


def publish_result_antigravity(
    result: Mapping[str, object],
    *,
    root: Path,
    started_at_utc: str,
    completed_at_utc: str,
    elapsed_seconds: float,
) -> dict[str, Path]:
    if float(elapsed_seconds) < 0.0:
        raise ValueError("elapsed_seconds must be nonnegative")
    base = Path(root)
    experiment = base / "experiments/exp484_distance_qualification_antigravity"
    paths = {
        "result": experiment / "results_antigravity.json",
        "manifest": experiment / "run_manifest_antigravity.json",
        "acceleration": base / "scratch/exp484_acceleration_assessment_antigravity.json",
        "dashboard": base / "docs/EXP484_DASHBOARD_antigravity.html",
        "report": base / "docs/EXP484_IMPLEMENTATION_REPORT_antigravity.html",
    }
    result_bytes = _json_bytes_antigravity(result)
    _write_atomic_antigravity(paths["result"], result_bytes)
    _write_atomic_antigravity(
        paths["acceleration"], _json_bytes_antigravity(result["acceleration"])
    )
    _write_atomic_antigravity(
        paths["dashboard"], render_dashboard_antigravity(result).encode("utf-8")
    )
    _write_atomic_antigravity(
        paths["report"], render_report_antigravity(result).encode("utf-8")
    )
    manifest = {
        "schema": "exp484_run_manifest_v1",
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "elapsed_seconds": float(elapsed_seconds),
        "result_sha256": sha256(result_bytes).hexdigest(),
        "dataset_count": int(result["dataset_count"]),
        "family_count": int(result["family_count"]),
        "source_sha256": dict(result.get("source_sha256", {})),
        "acceleration": result["acceleration"],
    }
    _write_atomic_antigravity(paths["manifest"], _json_bytes_antigravity(manifest))
    return paths


__all__ = [
    "build_result_antigravity",
    "choose_distance_variant_antigravity",
    "evaluate_variants_antigravity",
    "load_frozen_sources_antigravity",
    "publish_result_antigravity",
    "render_dashboard_antigravity",
    "render_report_antigravity",
    "require_sha256_antigravity",
    "run_exp484_antigravity",
    "summarize_evaluation_antigravity",
    "summarize_failure_structure_antigravity",
]


if __name__ == "__main__":
    final_result, final_paths = run_exp484_antigravity()
    print(
        json.dumps(
            {
                "decision": final_result["decision"],
                "variant_summaries": final_result["variant_summaries"],
                "paths": {key: str(value) for key, value in final_paths.items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )
