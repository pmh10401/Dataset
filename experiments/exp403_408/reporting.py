"""Korean retrospective report and hash-rich manifest for Exp403-408."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
from typing import Any

from .contracts import SEED
from .independent_reaggregate import IndependentReaggregation, independent_reaggregate


REPORT_NAME = "EXP403_408_RESULT_REPORT_KO.md"
MANIFEST_JSON = "MANIFEST.json"
MANIFEST_MARKDOWN = "MANIFEST.md"
_MUTABLE_OR_SELF_REFERENTIAL = frozenset({"status.json", MANIFEST_JSON, MANIFEST_MARKDOWN})


@dataclass(frozen=True)
class ReportArtifacts:
    report: Path
    manifest_json: Path
    manifest_markdown: Path
    reaggregation: IndependentReaggregation


def build_report(output_dir: Path) -> str:
    """Build a Korean, retrospective-only report from verified publications."""
    reaggregation = independent_reaggregate(Path(output_dir))
    return _report_text(reaggregation.metrics)


def write_reports(output_dir: Path, *, command: str) -> ReportArtifacts:
    """Write report and machine/human manifests after independent verification."""
    output = Path(output_dir)
    reaggregation = independent_reaggregate(output)
    report_path = output / REPORT_NAME
    report_path.write_text(_report_text(reaggregation.metrics), encoding="utf-8")
    manifest_path = output / MANIFEST_JSON
    markdown_path = output / MANIFEST_MARKDOWN
    manifest = _manifest(output, command=command, reaggregation=reaggregation)
    markdown = _manifest_markdown(manifest)
    manifest["manifest_markdown_sha256"] = hashlib.sha256(markdown.encode("utf-8")).hexdigest()
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown, encoding="utf-8")
    return ReportArtifacts(report_path, manifest_path, markdown_path, reaggregation)


def validate_final_reports(output_dir: Path) -> None:
    """Reject a missing or tampered final publication, excluding mutable status."""
    output = Path(output_dir)
    manifest = _mapping(json.loads((output / MANIFEST_JSON).read_text(encoding="utf-8")))
    files = _mapping(manifest.get("artifact_sha256"))
    for name, digest in files.items():
        path = output / name
        if name in _MUTABLE_OR_SELF_REFERENTIAL or not path.is_file() or _sha256(path) != digest:
            raise ValueError(f"final report artifact SHA256 mismatch: {name}")
    markdown = output / MANIFEST_MARKDOWN
    markdown_hash = manifest.get("manifest_markdown_sha256")
    if (
        REPORT_NAME not in files
        or not isinstance(markdown_hash, str)
        or not markdown.is_file()
        or _sha256(markdown) != markdown_hash
    ):
        raise ValueError("final report artifacts are incomplete")


def _report_text(metrics: Mapping[str, object]) -> str:
    exp404 = _mapping(metrics["exp404"])
    exp405 = _mapping(metrics["exp405"])
    exp406 = _mapping(metrics["exp406"])
    exp407 = _mapping(metrics["exp407"])
    variants = _mapping(metrics["exp408"])
    coverage = _mapping(metrics["train_size_coverage"])
    workload = _mapping(metrics["workload"])
    lines = [
        "# Exp403-408 독립 재집계 결과",
        "",
        "## 범위와 주장 경계",
        "",
        "이 문서는 동결된 prediction, evaluation CSV, consensus JSONL, manifest만 독립적으로 재집계한 retrospective 결과다.",
        "자동 경보는 변경하지 않음. Exp405는 production 변경이 아닌 반사실적(counterfactual) 하향 가능성 진단이다.",
        "Exp406-408의 신규 후보는 human-assisted review-only Shadow이며 automatic Hard로 승격되지 않는다.",
        "신규 설비 데이터에 대한 prospective 검증 또는 deployment 검증을 주장하지 않는다.",
        "",
        "## Autonomous Hard",
        "",
        "이 표는 immutable automatic Hard만 다룬다. human-assisted review 결과를 Hard 성과에 합치지 않는다.",
        "",
        "| 항목 | 값 |",
        "|---|---:|",
        f"| 동결 Hard 후보 | {exp404['hard_candidate_count']} |",
        f"| Hard TP / FP / FN | {exp404['hard_tp']} / {exp404['hard_fp']} / {exp404['hard_fn']} |",
        f"| 재확인 필요 (모든 alpha) | {exp404['recheck_count']} |",
        f"| Hard lane parity | {str(exp404['frozen_lane_parity']).lower()} |",
        "",
        "| Alpha | high | medium | recheck | abstain |",
        "|---|---:|---:|---:|---:|",
    ]
    for alpha, counts in sorted(_mapping(exp404["grade_counts"]).items()):
        grades = _mapping(counts)
        lines.append(
            f"| {alpha} | {grades.get('high', 0)} | {grades.get('medium', 0)} | "
            f"{grades.get('recheck', 0)} | {grades.get('abstain', 0)} |"
        )
    lines.extend((
        "",
        "## Exp405 반사실적 진단",
        "",
        "| 항목 | 값 |",
        "|---|---:|",
        f"| 재확인 필요 | {exp405['recheck_count']} |",
        f"| 실제 FP | {exp405['false_positive_count']} |",
        f"| 실제 TP | {exp405['true_positive_count']} |",
        f"| FP 비율 | {float(exp405['false_positive_fraction']):.12f} |",
        f"| Hard TP 보존율 | {float(exp405['baseline_true_positive_retention']):.12f} |",
        f"| TRAIN 정상성 안정성 | {float(exp405['train_stability']):.12f} |",
        f"| 연구 gate 통과 | {str(exp405.get('passes_research_gate', False)).lower()} |",
        f"| FP Wilson 95% | {_interval_text(exp405['false_positive_fraction_wilson_interval'])} |",
        f"| FP paired bootstrap 95% | {_interval_text(exp405['false_positive_fraction_bootstrap_interval'])} |",
        f"| TP 보존 paired bootstrap 95% | {_interval_text(exp405['baseline_true_positive_retention_bootstrap_interval'])} |",
        f"| Dataset concentration (HHI) | {float(exp405['dataset_concentration']):.12f} |",
        f"| Lineage concentration (HHI) | {float(exp405['lineage_concentration']):.12f} |",
        "",
        "Exp405 결과는 운영 lane을 바꾸지 않으며, 재사전등록과 신규 데이터 검증 전에는 정책 변경 근거가 아니다.",
        "",
        "## Human-Assisted Shadow",
        "",
        "아래 후보는 review workload를 위한 보조 지표다. autonomous Hard와 결합하거나 성능을 대체하지 않는다.",
        "",
        "| Source | Alpha | 후보 | TP | FP | 정밀도 | FN 회수 | Standard overlap | Priority overlap | 기존 외 후보 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ))
    for source, values in (("Exp406 self-supervised Shadow", exp406), ("Exp407 synthetic-anomaly Shadow", exp407)):
        for alpha, value in sorted(values.items()):
            row = _mapping(value)
            lines.append(
                f"| {source} | {alpha} | {row['candidate_count']} | {row['tp']} | {row['fp']} | "
                f"{float(row['precision']):.12f} | {row['fn_recovered']} | {row['standard_overlap']} | "
                f"{row['priority_overlap']} | {row['review_outside_existing']} |"
            )
    lines.extend(("", "## Exp408 동결 Variant", "", "| Variant | Alpha | 후보 | TP | FP | 정밀도 | FN 회수 | Zero-F1 전 | Zero-F1 후 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"))
    for variant, alpha_values in sorted(variants.items()):
        for alpha, value in sorted(_mapping(alpha_values).items()):
            row = _mapping(value)
            lines.append(
                f"| {variant} | {alpha} | {row['candidate_count']} | {row['tp']} | {row['fp']} | "
                f"{float(row['precision']):.12f} | {row['fn_recovered']} | {row['zero_f1_before']} | {row['zero_f1_after']} |"
            )
    lines.extend((
        "",
        "## TRAIN Coverage 및 Workload",
        "",
        "| TRAIN 범주 | 데이터셋 수 |",
        "|---|---:|",
    ))
    for name in ("abstain", "robust_distance", "classical_committee", "learned_methods", "candidate_datasets"):
        lines.append(f"| {name} | {coverage.get(name, 0)} |")
    lines.extend(("", "| Variant | Alpha | review workload |", "|---|---:|---:|"))
    for variant, alpha_values in sorted(workload.items()):
        for alpha, value in sorted(_mapping(alpha_values).items()):
            lines.append(f"| {variant} | {alpha} | {value} |")
    lines.extend(("", "| Variant | Alpha | dataset HHI | lineage HHI |", "|---|---:|---:|---:|"))
    for variant, alpha_values in sorted(_mapping(metrics["concentration"]).items()):
        for alpha, value in sorted(_mapping(alpha_values).items()):
            row = _mapping(value)
            lines.append(f"| {variant} | {alpha} | {float(row['dataset_hhi']):.12f} | {float(row['lineage_hhi']):.12f} |")
    lines.extend((
        "",
        "## 알려진 한계",
        "",
        "- 동일 TEST benchmark를 재사용한 retrospective 평가다.",
        "- Exp405는 counterfactual이며 production automatic Hard를 변경하지 않는다.",
        "- Exp406-408은 human-assisted review-only Shadow다. prospective 검증은 수행하지 않았다.",
        "- 모든 integer는 정확히, float는 1e-12 이내에서 독립 재집계와 일치해야 한다.",
        "",
    ))
    return "\n".join(lines)


def _manifest(output: Path, *, command: str, reaggregation: IndependentReaggregation) -> dict[str, object]:
    files = {
        path.name: _sha256(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name not in _MUTABLE_OR_SELF_REFERENTIAL
    }
    code_paths = tuple(
        Path(__file__).with_name(name)
        for name in (
            "reporting.py", "independent_reaggregate.py", "pipeline.py", "evaluation.py",
            "prediction.py", "consensus.py", "foundation.py", "contracts.py", "artifact_io.py",
        )
    )
    provenance_hashes = _provenance_hashes(output)
    publication_hashes = _publication_hashes(output)
    return {
        "schema": "exp403_408_reporting_manifest/v1",
        "command": command,
        "commands": [
            "python3 scratch/run_exp403_408_feature_ml_validation.py --stage all --output-dir <output-dir>",
            command,
        ],
        "seed": SEED,
        "retrospective_only": True,
        "deployment_eligible": False,
        "prospective_validated": False,
        "operational_policy_changed": False,
        "autonomous_scope": "Immutable automatic Hard only; automatic alerts remain unchanged.",
        "human_assisted_scope": "Exp406-408 Shadow candidates are review-only and are not added to automatic Hard.",
        "claims_forbidden": [
            "Exp405 changed production automatic Hard",
            "Exp406-408 were prospectively validated",
            "deployment validation completed",
        ],
        "known_limitations": [
            "Retrospective reuse of frozen TEST evaluation artifacts",
            "No prospective or deployment validation",
            "Human-assisted review metrics are separate from autonomous Hard metrics",
        ],
        "independent_reaggregation": {
            "mismatch_count": reaggregation.mismatch_count,
            "float_tolerance": 1e-12,
            "comparison_count": len(reaggregation.comparisons),
        },
        "code_sha256": {str(path.resolve()): _sha256(path) for path in code_paths},
        "prediction_sha256": _prediction_sha256(output),
        "input_sha256": provenance_hashes,
        "database_sha256": provenance_hashes.get("dataset_db", "not-published"),
        "publication_sha256": publication_hashes,
        "artifact_sha256": files,
        "runtime": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "packages": _package_versions(),
        },
        "acceleration": {"device": "cpu", "mlx_used": False, "parity": "not applicable", "fallback": "frozen CPU reference path"},
    }


def _manifest_markdown(manifest: Mapping[str, object]) -> str:
    files = _mapping(manifest["artifact_sha256"])
    code = _mapping(manifest["code_sha256"])
    lines = [
        "# Exp403-408 실행 매니페스트",
        "",
        f"- Command: `{manifest['command']}`",
        f"- Seed: `{manifest['seed']}`",
        "- Scope: retrospective only; deployment/prospective validation 아님",
        "- Autonomous: automatic Hard 변경 없음",
        "- Human-assisted: Exp406-408 Shadow review-only",
        f"- Python: `{_mapping(manifest['runtime'])['python_version']}`",
        f"- Device: `{_mapping(manifest['acceleration'])['device']}`",
        f"- Float tolerance: `{_mapping(manifest['independent_reaggregation'])['float_tolerance']}`",
        f"- Prediction SHA-256: `{manifest['prediction_sha256']}`",
        f"- Database SHA-256: `{manifest['database_sha256']}`",
        "",
        "## Code SHA-256",
        "",
    ]
    lines.extend(f"- `{path}`: `{digest}`" for path, digest in sorted(code.items()))
    lines.extend(("", "## 핵심 입력/발행 SHA-256", ""))
    lines.extend(f"- `{name}`: `{digest}`" for name, digest in sorted(_mapping(manifest["input_sha256"]).items()))
    lines.extend(f"- `{name}`: `{digest}`" for name, digest in sorted(_mapping(manifest["publication_sha256"]).items()))
    lines.extend(("", "## Artifact SHA-256", "", "| File | SHA-256 |", "|---|---|"))
    lines.extend(f"| {path} | `{digest}` |" for path, digest in sorted(files.items()))
    lines.extend(("", "## Package versions", ""))
    lines.extend(f"- `{name}`: `{version}`" for name, version in sorted(_mapping(_mapping(manifest['runtime'])['packages']).items()))
    lines.extend(("", "## Limitations", ""))
    lines.extend(f"- {item}" for item in manifest["known_limitations"])
    return "\n".join(lines) + "\n"


def _package_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for name in ("numpy", "scipy", "scikit-learn", "torch", "mlx", "mlx-metal"):
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = "not-installed"
    return result


def _prediction_sha256(output: Path) -> str:
    path = output / "exp404_prediction_manifest.json"
    if not path.is_file():
        return "not-published"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        hashes = _mapping(manifest.get("artifact_hashes", {}))
    except (OSError, json.JSONDecodeError, ValueError):
        return "not-published"
    value = hashes.get("prediction")
    return str(value) if isinstance(value, str) else "not-published"


def _publication_hashes(output: Path) -> dict[str, str]:
    names = (
        "exp404_prediction_manifest.json", "exp404_hard_confidence.jsonl",
        "exp405_demotion_evaluation.json", "exp408_consensus_results.jsonl",
        "exp403_408_evaluation_input_manifest.json",
        "exp403_408_evaluation_publication_manifest.json", REPORT_NAME,
    )
    return {name: _sha256(output / name) for name in names if (output / name).is_file()}


def _interval_text(value: object) -> str:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("verified interval is invalid")
    return f"[{float(value[0]):.12f}, {float(value[1]):.12f}]"


def _provenance_hashes(output: Path) -> dict[str, str]:
    path = output / "exp403_404_provenance_contract.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        provenance = _mapping(payload.get("provenance", {}))
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    result: dict[str, str] = {}
    for category in ("input_artifacts", "baseline_artifacts", "feature_artifacts", "config_artifacts"):
        artifacts = provenance.get(category, {})
        if not isinstance(artifacts, Mapping):
            continue
        for name, detail in artifacts.items():
            if isinstance(detail, Mapping) and isinstance(detail.get("sha256"), str):
                result[str(name)] = str(detail["sha256"])
    return result


def _mapping(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("verified reaggregation payload is invalid")
    return {str(key): child for key, child in value.items()}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
