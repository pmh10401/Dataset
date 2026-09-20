"""Korean retrospective report and hash-rich manifest for Exp403-408."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path
import platform
from typing import Any

from .contracts import SEED
from .code_provenance import local_code_closure
from .independent_reaggregate import IndependentReaggregation, independent_reaggregate


REPORT_NAME = "EXP403_408_RESULT_REPORT_KO.md"
MANIFEST_JSON = "MANIFEST.json"
MANIFEST_MARKDOWN = "MANIFEST.md"
MPS_BENCHMARK_NAME = "task13_exp406_mps_benchmark.json"
_MUTABLE_OR_SELF_REFERENTIAL = frozenset({"status.json", MANIFEST_JSON, MANIFEST_MARKDOWN})
_IMMUTABLE_ARTIFACT_ALLOWLIST = frozenset({
    "exp403_feature_audit.jsonl", "exp403_feature_manifest.csv",
    "exp403_404_frozen_config.json", "exp403_404_provenance_contract.json",
    "exp404_prediction_manifest.json", "exp404_hard_confidence.jsonl",
    "exp405_demotion_evaluation.json", "exp405_demotion_dataset_metrics.csv",
    "exp405_demotion_candidate_evaluation.csv", "exp408_consensus_results.jsonl",
    "exp408_consensus_summary.csv", "exp408_variant_metrics.csv",
    "exp408_variant_candidate_evaluation.csv", "exp403_408_frozen_test_labels.csv",
    "exp403_408_evaluation_input_manifest.json",
    "exp403_408_evaluation_publication_manifest.json", "exp403_408_results.csv",
    "exp403_408_summary.csv", MPS_BENCHMARK_NAME, REPORT_NAME,
})
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
    lines.extend((
        "", "## Exp408 동결 Variant", "",
        "| Variant | Alpha | 후보 | TP | FP | 정밀도 | FN 회수 | Zero-F1 전 | Zero-F1 후 | Standard overlap | Priority overlap | 기존 외 후보 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ))
    for variant, alpha_values in sorted(variants.items()):
        for alpha, value in sorted(_mapping(alpha_values).items()):
            row = _mapping(value)
            lines.append(
                f"| {variant} | {alpha} | {row['candidate_count']} | {row['tp']} | {row['fp']} | "
                f"{float(row['precision']):.12f} | {row['fn_recovered']} | {row['zero_f1_before']} | {row['zero_f1_after']} | "
                f"{row['standard_overlap']} | {row['priority_overlap']} | {row['review_outside_existing']} |"
            )
    lines.extend((
        "", "## Exp408 paired bootstrap (dataset)", "",
        "| Variant | Alpha | TP 95% | FP 95% | 정밀도 95% | FN 회수 95% | Zero-F1 차이 95% |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ))
    for variant, alpha_values in sorted(variants.items()):
        for alpha, value in sorted(_mapping(alpha_values).items()):
            bootstrap = _mapping(_mapping(value)["paired_dataset_bootstrap"])
            lines.append(
                f"| {variant} | {alpha} | {_interval_text(bootstrap['tp'])} | {_interval_text(bootstrap['fp'])} | "
                f"{_interval_text(bootstrap['precision'])} | {_interval_text(bootstrap['fn_recovered'])} | "
                f"{_interval_text(bootstrap['zero_f1_difference'])} |"
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
        name: _sha256(output / name)
        for name in sorted(_IMMUTABLE_ARTIFACT_ALLOWLIST)
        if (output / name).is_file()
    }
    code_provenance = execution_code_provenance()
    provenance_hashes = _provenance_hashes(output)
    publication_hashes = _publication_hashes(output)
    external = _evaluation_external_hashes(output)
    task13 = _task13_execution_metadata(output)
    commands = [
        str(task13["full_command"]), str(task13["report_command"]),
    ] if task13 is not None else [
        f"python3 scratch/run_exp403_408_feature_ml_validation.py --stage all --output-dir {output}",
        command,
    ]
    acceleration = {
        "device": "cpu",
        "mlx_used": False,
        "parity": "not applicable",
        "fallback": "frozen CPU reference path",
    }
    if task13 is not None:
        decision = _mapping(task13["decision"])
        acceleration = {
            "device": str(decision["selected_device"]),
            "mlx_used": False,
            "parity": "passed",
            "fallback": "CPU retained because parity passed but MPS slower",
            "decision": "CPU retained because parity passed but MPS slower",
            "benchmark_artifact": MPS_BENCHMARK_NAME,
            "benchmark_sha256": files[MPS_BENCHMARK_NAME],
        }
    return {
        "schema": "exp403_408_reporting_manifest/v1",
        "command": commands[-1],
        "commands": commands,
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
        "code_provenance": code_provenance,
        "code_sha256": code_provenance["files"],
        "prediction_sha256": _prediction_sha256(output),
        "input_sha256": provenance_hashes,
        "external_sha256": external["external_sha256"],
        "database_snapshot": external["database_snapshot"],
        "database_sha256": _mapping(external["database_snapshot"]).get("composite_sha256", "not-published"),
        "publication_sha256": publication_hashes,
        "artifact_sha256": files,
        "runtime": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "packages": _package_versions(),
        },
        "task13_execution": task13,
        "acceleration": acceleration,
    }


def _manifest_markdown(manifest: Mapping[str, object]) -> str:
    files = _mapping(manifest["artifact_sha256"])
    code = _mapping(manifest["code_sha256"])
    code_provenance = _mapping(manifest["code_provenance"])
    lines = [
        "# Exp403-408 실행 매니페스트",
        "",
        f"- Command: `{manifest['command']}`",
        f"- Seed: `{manifest['seed']}`",
        "- Scope: retrospective only; deployment/prospective validation 아님",
        "- Autonomous: automatic Hard 변경 없음",
        "- Human-assisted: Exp406-408 Shadow review-only",
        f"- Python: `{_mapping(manifest['runtime'])['python_version']}`",
        f"- Platform: `{_mapping(manifest['runtime'])['platform']}`",
        f"- Device: `{_mapping(manifest['acceleration'])['device']}`",
        f"- Acceleration parity/fallback: `{_mapping(manifest['acceleration'])['parity']}` / `{_mapping(manifest['acceleration'])['fallback']}`",
        f"- Float tolerance: `{_mapping(manifest['independent_reaggregation'])['float_tolerance']}`",
        f"- Prediction SHA-256: `{manifest['prediction_sha256']}`",
        f"- Database SHA-256: `{manifest['database_sha256']}`",
        "",
        "## Commands",
        "",
    ]
    lines.extend(f"- `{command}`" for command in manifest["commands"])
    task13 = manifest.get("task13_execution")
    if isinstance(task13, Mapping):
        benchmark = _mapping(task13)
        lines.extend(("", "## Task 13 MPS Benchmark", ""))
        lines.append(f"- Artifact: `{MPS_BENCHMARK_NAME}`")
        lines.append(f"- Benchmark command: `{benchmark['benchmark_command']}`")
        lines.append(f"- Decision: `CPU retained because parity passed but MPS slower`")
        for run in benchmark["runs"]:
            detail = _mapping(run)
            lines.append(
                f"- {detail['run']}: `{detail['device']}` `{detail['runtime_seconds']}`s "
                f"hash `{detail['prediction_sha256']}` rank `{detail['rank_match']}` "
                f"max_delta `{detail['max_raw_score_delta']}`"
            )
    lines.extend(("", "## Code provenance closure", ""))
    lines.append(f"- Method: `{code_provenance['method']}`")
    lines.append(f"- Closure SHA-256: `{code_provenance['closure_sha256']}`")
    lines.extend(f"- Root: `{path}`" for path in code_provenance["roots"])
    lines.extend(("", "## Code SHA-256", ""))
    lines.extend(f"- `{path}`: `{digest}`" for path, digest in sorted(code.items()))
    lines.extend(("", "## 핵심 입력/발행 SHA-256", ""))
    lines.extend(f"- `{name}`: `{digest}`" for name, digest in sorted(_mapping(manifest["input_sha256"]).items()))
    lines.extend(f"- external `{name}`: `{digest}`" for name, digest in sorted(_mapping(manifest["external_sha256"]).items()))
    snapshot = _mapping(manifest["database_snapshot"])
    lines.append(f"- database snapshot composite: `{snapshot.get('composite_sha256', 'not-published')}`")
    for item in snapshot.get("artifacts", []):
        detail = _mapping(item)
        lines.append(f"- database snapshot `{detail['path']}`: `{detail['sha256']}`")
    lines.extend(f"- `{name}`: `{digest}`" for name, digest in sorted(_mapping(manifest["publication_sha256"]).items()))
    lines.extend(("", "## Artifact SHA-256", "", "| File | SHA-256 |", "|---|---|"))
    lines.extend(f"| {path} | `{digest}` |" for path, digest in sorted(files.items()))
    lines.extend(("", "## Package versions", ""))
    lines.extend(f"- `{name}`: `{version}`" for name, version in sorted(_mapping(_mapping(manifest['runtime'])['packages']).items()))
    lines.extend(("", "## Limitations", ""))
    lines.extend(f"- {item}" for item in manifest["known_limitations"])
    lines.extend(("", "## Forbidden claims", ""))
    lines.extend(f"- {item}" for item in manifest["claims_forbidden"])
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


def execution_code_provenance() -> dict[str, object]:
    """Return the deterministic local-import closure for all execution roots."""
    repository = Path(__file__).resolve().parents[2]
    roots = sorted(Path(__file__).parent.glob("*.py"), key=str)
    roots.extend((
        repository / "experiments" / "exp393_396" / "artifact_io.py",
        repository / "scratch" / "run_exp403_408_feature_ml_validation.py",
        repository / "run_rank_experiments_sequential.py",
        repository / "serve_rank_dashboard.py",
    ))
    missing = [str(path) for path in roots if not path.is_file()]
    if missing:
        raise ValueError("execution module hash source is missing: " + ", ".join(missing))
    return local_code_closure(repository, roots)


def _evaluation_external_hashes(output: Path) -> dict[str, object]:
    path = output / "exp403_408_evaluation_input_manifest.json"
    if not path.is_file():
        return {
            "external_sha256": {},
            "database_snapshot": {"composite_sha256": "not-published", "artifacts": []},
        }
    payload = _mapping(json.loads(path.read_text(encoding="utf-8")))
    external = _mapping(payload.get("external_sha256", {}))
    snapshot = _mapping(payload.get("database_snapshot", {}))
    return {
        "external_sha256": dict(external),
        "database_snapshot": {
            "composite_sha256": str(snapshot.get("composite_sha256", "not-published")),
            "artifacts": list(snapshot.get("artifacts", [])),
        },
    }


def _task13_execution_metadata(output: Path) -> dict[str, object] | None:
    """Load the immutable Task 13 benchmark only when it satisfies its contract."""
    path = output / MPS_BENCHMARK_NAME
    if not path.is_file():
        return None
    payload = _mapping(json.loads(path.read_text(encoding="utf-8")))
    if payload.get("schema") != "exp403_408_task13_mps_benchmark/v1":
        raise ValueError("Task 13 MPS benchmark schema is invalid")
    for key in ("benchmark_command", "full_command", "report_command"):
        if not isinstance(payload.get(key), str) or not str(payload[key]).strip():
            raise ValueError(f"Task 13 MPS benchmark {key} is invalid")
    runs = payload.get("runs")
    if not isinstance(runs, list) or len(runs) != 4:
        raise ValueError("Task 13 MPS benchmark requires four runs")
    expected_devices = ("cpu", "cpu", "mps", "mps")
    prediction_hashes: set[str] = set()
    runtimes: dict[str, list[float]] = {"cpu": [], "mps": []}
    for detail, device in zip(runs, expected_devices, strict=True):
        row = _mapping(detail)
        if row.get("device") != device:
            raise ValueError("Task 13 MPS benchmark device order is invalid")
        for key in ("run", "prediction_sha256", "rank_match", "max_raw_score_delta", "runtime_seconds"):
            if key not in row:
                raise ValueError(f"Task 13 MPS benchmark run field is missing: {key}")
        if row.get("rank_match") is not True:
            raise ValueError("Task 13 MPS benchmark rank parity is invalid")
        digest = row.get("prediction_sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("Task 13 MPS benchmark prediction hash is invalid")
        prediction_hashes.add(digest)
        runtime = _finite_positive(row.get("runtime_seconds"), "runtime")
        delta = _finite_nonnegative(row.get("max_raw_score_delta"), "max delta")
        if delta > 1e-5:
            raise ValueError("Task 13 MPS benchmark raw-score delta is invalid")
        runtimes[device].append(runtime)
    if len(prediction_hashes) != 1:
        raise ValueError("Task 13 MPS benchmark prediction hashes differ")
    cpu_mean = sum(runtimes["cpu"]) / 2.0
    mps_mean = sum(runtimes["mps"]) / 2.0
    decision = _mapping(payload.get("decision"))
    if decision.get("selected_device") != "cpu" or decision.get("reason") != "parity passed but MPS slower" or not mps_mean > cpu_mean:
        raise ValueError("Task 13 MPS benchmark decision is invalid")
    if _finite_positive(decision.get("cpu_mean_runtime_seconds"), "CPU mean") != cpu_mean or _finite_positive(decision.get("mps_mean_runtime_seconds"), "MPS mean") != mps_mean:
        raise ValueError("Task 13 MPS benchmark decision means are invalid")
    return dict(payload)


def _finite_positive(value: object, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Task 13 MPS benchmark {name} is invalid") from error
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"Task 13 MPS benchmark {name} is invalid")
    return number


def _finite_nonnegative(value: object, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Task 13 MPS benchmark {name} is invalid") from error
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"Task 13 MPS benchmark {name} is invalid")
    return number


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
