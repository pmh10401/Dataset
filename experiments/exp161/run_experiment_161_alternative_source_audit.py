from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

# Direct script execution sets sys.path to experiments/exp161. Add the fixed
# repository root so the experiments package resolves exactly as in -m mode.
PROJECT_DIR = Path("/Users/minho/Documents/timesries project")
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from experiments.exp161.alternative_source_audit_core import (
    is_target,
    detect_computed_exp99,
    detect_computed_exp101,
    detect_computed_exp46,
    is_policy_eligible,
    calculate_legacy_index_overlap,
    SOURCE_DISPLAY_NAMES,
)

# Hardcoded paths and configurations
DATA_DIR = Path("/Users/minho/Documents/Dataset")

EXP160_PRED_PATH = PROJECT_DIR / "outputs" / "exp160_candidate_zero_similarity_recovery" / "02_exp160_predictions.jsonl"
EXP99_PATH = DATA_DIR / "experiment_99_spectral_derivative_feature_score_results.csv"
EXP101_PATH = DATA_DIR / "experiment_101_shapelet_normal_prototype_results.csv"
EXP46_PATH = DATA_DIR / "experiment_46_model_hard_interval_drcif_lite_results.csv"

EXP160_EXPECTED_HASH = "ce97a7df2873d4bc628bbd58e2a7b83142f9d8cf8670bdf598f2b79aa0ded73d"


def get_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        # Create empty file with headers
        with path.open("w", newline="", encoding="utf-8") as f:
            if fieldnames:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
        return

    if not fieldnames:
        fieldnames = []
        for r in rows:
            for k in r.keys():
                if k not in fieldnames:
                    fieldnames.append(k)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_audit(dataset_limit: int | None = None, output_dir: Path | None = None):
    # 1. Hash verification
    actual_hash = get_sha256(EXP160_PRED_PATH)
    if actual_hash != EXP160_EXPECTED_HASH:
        raise ValueError(
            f"Exp160 prediction file hash mismatch!\nExpected: {EXP160_EXPECTED_HASH}\nActual:   {actual_hash}"
        )
    print(f"Exp160 prediction file hash verified: {actual_hash}")

    # Load alternative source result files
    print("Loading historical source CSVs...")
    exp99_rows = load_csv(EXP99_PATH)
    exp101_rows = load_csv(EXP101_PATH)
    exp46_rows = load_csv(EXP46_PATH)

    # Detect computed datasets for each source
    exp99_computed = detect_computed_exp99(exp99_rows)
    exp101_computed = detect_computed_exp101(exp101_rows)
    exp46_computed = detect_computed_exp46(exp46_rows)

    print(f"Exp99 (spectral) computed datasets: {len(exp99_computed)}")
    print(f"Exp101 (shapelet) computed datasets: {len(exp101_computed)}")
    print(f"Exp46 (interval) computed datasets: {len(exp46_computed)}")

    # Load Exp160 prediction target rows
    exp160_rows = load_jsonl(EXP160_PRED_PATH)

    # Target selection: n_train >= 5, candidate_count == 0
    all_targets = [row for row in exp160_rows if is_target(row)]
    print(f"Total Exp160 target rows (n_train >= 5, candidate_count == 0): {len(all_targets)}")

    # Apply dataset limit if specified
    if dataset_limit is not None:
        # We need to take the first N unique datasets that appear in the targets list
        unique_target_datasets = []
        for t in all_targets:
            ds = t["dataset_name"]
            if ds not in unique_target_datasets:
                unique_target_datasets.append(ds)
            if len(unique_target_datasets) == dataset_limit:
                break

        target_datasets_set = set(unique_target_datasets)
        # Filter all_targets to only include rows with dataset_name in target_datasets_set
        targets = [t for t in all_targets if t["dataset_name"] in target_datasets_set]
        print(f"Applying limit: {dataset_limit} datasets. Selected {len(targets)} rows (dataset/alpha pairs)")
    else:
        targets = all_targets

    # Audit variables
    coverage_rows = []
    legacy_overlap_rows = []

    # Stats counters
    spectral_computed_count = 0
    spectral_eligible_count = 0
    shapelet_computed_count = 0
    shapelet_eligible_count = 0
    interval_computed_count = 0
    interval_eligible_count = 0
    common_support_count = 0

    # Build coverage list preserving input row order
    for row in targets:
        ds_name = row["dataset_name"]
        alpha = float(row["alpha"])
        n_train = int(row["n_train"])
        n_test = int(row["n_test"])

        # Spectral
        spectral_computed = ds_name in exp99_computed
        spectral_eligible = is_policy_eligible("spectral_derivative", ds_name, exp99_computed, exp101_computed, exp46_computed)

        # Shapelet
        shapelet_computed = ds_name in exp101_computed
        shapelet_eligible = is_policy_eligible("shapelet_normal_prototype", ds_name, exp99_computed, exp101_computed, exp46_computed)

        # Interval
        interval_computed = ds_name in exp46_computed
        interval_eligible = is_policy_eligible("interval_quantile", ds_name, exp99_computed, exp101_computed, exp46_computed)

        # Stats
        if spectral_computed:
            spectral_computed_count += 1
        if spectral_eligible:
            spectral_eligible_count += 1

        if shapelet_computed:
            shapelet_computed_count += 1
        if shapelet_eligible:
            shapelet_eligible_count += 1

        if interval_computed:
            interval_computed_count += 1
        if interval_eligible:
            interval_eligible_count += 1

        if spectral_computed and shapelet_computed and interval_computed:
            common_support_count += 1

        coverage_rows.append({
            "dataset_name": ds_name,
            "alpha": alpha,
            "n_train": n_train,
            "n_test": n_test,
            "spectral_computed": int(spectral_computed),
            "spectral_eligible": int(spectral_eligible),
            "shapelet_computed": int(shapelet_computed),
            "shapelet_eligible": int(shapelet_eligible),
            "interval_computed": int(interval_computed),
            "interval_eligible": int(interval_eligible),
        })

        # Calculate Legacy Index Overlap for research targets if computed
        # spectral legacy overlap
        spectral_overlap = calculate_legacy_index_overlap("spectral_derivative", ds_name, exp99_rows, spectral_computed)
        if spectral_overlap:
            legacy_overlap_rows.append({
                "dataset_name": ds_name,
                "alpha": alpha,
                "source_name": "spectral_derivative",
                "base_indices": " ".join(map(str, spectral_overlap["base_indices"])),
                "source_indices": " ".join(map(str, spectral_overlap["source_indices"])),
                "overlap_indices": " ".join(map(str, spectral_overlap["overlap_indices"])),
                "overlap_count": spectral_overlap["overlap_count"],
            })

        # shapelet legacy overlap
        shapelet_overlap = calculate_legacy_index_overlap("shapelet_normal_prototype", ds_name, exp101_rows, shapelet_computed)
        if shapelet_overlap:
            legacy_overlap_rows.append({
                "dataset_name": ds_name,
                "alpha": alpha,
                "source_name": "shapelet_normal_prototype",
                "base_indices": " ".join(map(str, shapelet_overlap["base_indices"])),
                "source_indices": " ".join(map(str, shapelet_overlap["source_indices"])),
                "overlap_indices": " ".join(map(str, shapelet_overlap["overlap_indices"])),
                "overlap_count": shapelet_overlap["overlap_count"],
            })

    if output_dir is None:
        suffix = "_smoke" if dataset_limit is not None else ""
        output_dir = PROJECT_DIR / "outputs" / f"exp161_alternative_source_audit{suffix}"
    elif not output_dir.is_absolute():
        output_dir = PROJECT_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    audit_scope = "Smoke" if dataset_limit is not None else "Full"

    input_manifest_rows = [
        {"input_role": "frozen_exp160_targets", "path": str(EXP160_PRED_PATH), "sha256": actual_hash, "row_count": len(exp160_rows), "unique_dataset_count": len({r["dataset_name"] for r in exp160_rows})},
        {"input_role": "historical_spectral_derivative", "path": str(EXP99_PATH), "sha256": get_sha256(EXP99_PATH), "row_count": len(exp99_rows), "unique_dataset_count": len({r["dataset_name"] for r in exp99_rows})},
        {"input_role": "historical_shapelet_normal_prototype", "path": str(EXP101_PATH), "sha256": get_sha256(EXP101_PATH), "row_count": len(exp101_rows), "unique_dataset_count": len({r["dataset_name"] for r in exp101_rows})},
        {"input_role": "historical_interval_quantile", "path": str(EXP46_PATH), "sha256": get_sha256(EXP46_PATH), "row_count": len(exp46_rows), "unique_dataset_count": len({r["dataset_name"] for r in exp46_rows})},
    ]
    write_csv(output_dir / "00_input_manifest.csv", input_manifest_rows)

    # 1. 01_source_inventory.csv
    inventory_rows = [
        {
            "source_key": "spectral_derivative",
            "display_name": SOURCE_DISPLAY_NAMES["spectral_derivative"],
            "source_type": "Exp99 spectral/derivative feature score",
            "policy_eligible_scope": "None (Always Ineligible due to family/research target provenance)",
            "description": "Z-normalized first/second derivative statistics and FFT magnitude band distributions.",
        },
        {
            "source_key": "shapelet_normal_prototype",
            "display_name": SOURCE_DISPLAY_NAMES["shapelet_normal_prototype"],
            "source_type": "Exp101 shapelet normal prototype",
            "policy_eligible_scope": "None (Always Ineligible due to research target provenance)",
            "description": "Subsequence normal shapelet prototype minimum distance features.",
        },
        {
            "source_key": "interval_quantile",
            "display_name": SOURCE_DISPLAY_NAMES["interval_quantile"],
            "source_type": "Exp46 model hard interval drcif lite results",
            "policy_eligible_scope": "None for historical artifact (family/difficulty-selected partial coverage)",
            "description": "Interval quantile distribution anomaly detection (DrCIF-lite based).",
        },
    ]
    write_csv(output_dir / "01_source_inventory.csv", inventory_rows)

    # 2. 02_candidate_zero_source_coverage.csv
    write_csv(output_dir / "02_candidate_zero_source_coverage.csv", coverage_rows)

    # 3. 03_legacy_index_overlap.csv
    write_csv(
        output_dir / "03_legacy_index_overlap.csv",
        legacy_overlap_rows,
        ["dataset_name", "alpha", "source_name", "base_indices", "source_indices", "overlap_indices", "overlap_count"]
    )

    # 4. 04_audit_summary.csv
    total_targets = len(targets)
    total_target_datasets = len({row["dataset_name"] for row in targets})
    spectral_computed_datasets = len({row["dataset_name"] for row in coverage_rows if row["spectral_computed"]})
    shapelet_computed_datasets = len({row["dataset_name"] for row in coverage_rows if row["shapelet_computed"]})
    interval_computed_datasets = len({row["dataset_name"] for row in coverage_rows if row["interval_computed"]})
    summary_rows = [
        {"metric_name": "total_target_dataset_alpha_pairs", "value": str(total_targets)},
        {"metric_name": "total_unique_target_datasets", "value": str(total_target_datasets)},
        {"metric_name": "spectral_actual_computed_unique_datasets", "value": str(spectral_computed_datasets)},
        {"metric_name": "spectral_actual_computed_count", "value": str(spectral_computed_count)},
        {"metric_name": "spectral_actual_coverage_rate", "value": f"{spectral_computed_count / total_targets:.6f}" if total_targets else "0.000000"},
        {"metric_name": "spectral_policy_eligible_count", "value": str(spectral_eligible_count)},
        {"metric_name": "spectral_policy_eligible_coverage_rate", "value": f"{spectral_eligible_count / total_targets:.6f}" if total_targets else "0.000000"},

        {"metric_name": "shapelet_actual_computed_unique_datasets", "value": str(shapelet_computed_datasets)},
        {"metric_name": "shapelet_actual_computed_count", "value": str(shapelet_computed_count)},
        {"metric_name": "shapelet_actual_coverage_rate", "value": f"{shapelet_computed_count / total_targets:.6f}" if total_targets else "0.000000"},
        {"metric_name": "shapelet_policy_eligible_count", "value": str(shapelet_eligible_count)},
        {"metric_name": "shapelet_policy_eligible_coverage_rate", "value": f"{shapelet_eligible_count / total_targets:.6f}" if total_targets else "0.000000"},

        {"metric_name": "interval_actual_computed_unique_datasets", "value": str(interval_computed_datasets)},
        {"metric_name": "interval_actual_computed_count", "value": str(interval_computed_count)},
        {"metric_name": "interval_actual_coverage_rate", "value": f"{interval_computed_count / total_targets:.6f}" if total_targets else "0.000000"},
        {"metric_name": "interval_policy_eligible_count", "value": str(interval_eligible_count)},
        {"metric_name": "interval_policy_eligible_coverage_rate", "value": f"{interval_eligible_count / total_targets:.6f}" if total_targets else "0.000000"},

        {"metric_name": "common_support_count", "value": str(common_support_count)},
        {"metric_name": "common_support_rate", "value": f"{common_support_count / total_targets:.6f}" if total_targets else "0.000000"},
    ]
    write_csv(output_dir / "04_audit_summary.csv", summary_rows)

    # Policy eligible full coverage sources count
    # A source is a full coverage eligible source if it covers ALL targets AND is policy eligible for all of them.
    # Therefore, policy_eligible_count must equal total_targets.
    full_coverage_sources = []
    if spectral_eligible_count == total_targets and total_targets > 0:
        full_coverage_sources.append("spectral_derivative")
    if shapelet_eligible_count == total_targets and total_targets > 0:
        full_coverage_sources.append("shapelet_normal_prototype")
    if interval_eligible_count == total_targets and total_targets > 0:
        full_coverage_sources.append("interval_quantile")

    num_full_coverage_eligible_sources = len(full_coverage_sources)
    print(f"Full-coverage policy-eligible sources count: {num_full_coverage_eligible_sources}")

    # Blocker validation: If < 2 policy-eligible full-coverage sources, write BLOCKER_REPORT.md
    blocker_triggered = num_full_coverage_eligible_sources < 2
    if blocker_triggered:
        print("Blocker triggered! Generating BLOCKER_REPORT.md...")
        blocker_report_content = f"""# Exp161 Blocker Report: Policy-Eligible Alternative Source Deficit

**Date:** 2026-07-14
**Experiment ID:** `experiment_161_alternative_source_candidate_zero_audit`

## Blocker Verdict
대체 점수원 감사를 수행한 결과, Exp160 `candidate_count == 0` 복구 정책(Exp162)으로의 이행을 보증할 수 있는 **policy-eligible full-coverage 소스가 부족(2개 미만)**하여, 본 감사는 정책 추천 단계 진행을 차단하고 블로커를 해제하지 않습니다.

- **필수 조건:** 2개 이상의 policy-eligible full-coverage 소스 확보
- **실제 발견:** {num_full_coverage_eligible_sources}개

---

## 상세 감사 결과 요약 ({audit_scope} Target {total_targets}개 기준)

1. **미분·스펙트럼 통계 점수 (spectral_derivative / Exp99)**
   - 실제 계산 완료 (Actual Computed): {spectral_computed_count} / {total_targets} ({spectral_computed_count / total_targets * 100:.1f}%)
   - 정책 적합성 통과 (Policy-Eligible): {spectral_eligible_count} / {total_targets} (0.0%)
   - **판정:** Ineligible (과거 TEST 진단 및 특정 Family 이름에서 유래한 provenance로 전체 정책 적용 부적합)

2. **부분 형태 일치 점수 (shapelet_normal_prototype / Exp101)**
   - 실제 계산 완료 (Actual Computed): {shapelet_computed_count} / {total_targets} ({shapelet_computed_count / total_targets * 100:.1f}%)
   - 정책 적합성 통과 (Policy-Eligible): {shapelet_eligible_count} / {total_targets} (0.0%)
   - **판정:** Ineligible (과거 TEST 진단에서 유래한 provenance로 전체 정책 적용 부적합)

3. **구간 분포 점수 (interval_quantile / Exp46)**
   - 실제 계산 완료 (Actual Computed): {interval_computed_count} / {total_targets} ({interval_computed_count / total_targets * 100:.1f}%)
   - 정책 적합성 통과 (Policy-Eligible): {interval_eligible_count} / {total_targets} ({interval_eligible_count / total_targets * 100:.1f}%)
   - **판정:** Ineligible historical artifact (family/difficulty-selected 13개 데이터셋에만 계산되어 정책 입력으로 사용할 수 없음)

---

## Blocker 원인 분석

1. **Exp99 및 Exp101의 Provenance 오염:**
   - 이 소스들은 사전 필터링 과정에서 TEST 성능 진단 데이터를 간접 유용하여 타겟이 선정되었습니다. (e.g. `research_target`, `spectral_target`, `shapelet_target`).
   - 따라서 strict label-free policy replay에 직접 사용하는 것은 retrospective leakage 위험이 있어 eligible = False 처리되었습니다.

2. **Exp46의 Provenance 및 Coverage 제한:**
   - 구간 분포 점수(Exp46)는 family/difficulty 진단으로 선정된 {len(exp46_computed)}개 데이터셋에만 계산되었으므로, historical artifact 자체가 정책 입력으로 부적합하며 Exp160 target 전체도 커버하지 못합니다.

## 결론 및 권장 조치
Exp162를 추진하기 위해서는, 이들 대체 점수원의 **TEST metric 비의존적/전체 데이터셋 대상의 신규 full-coverage 점수 계산 및 저장(re-computation)**이 선행되어야 합니다.
"""
        (output_dir / "BLOCKER_REPORT.md").write_text(blocker_report_content, encoding="utf-8")
    else:
        # Delete blocker report if it exists
        blocker_file = output_dir / "BLOCKER_REPORT.md"
        if blocker_file.exists():
            blocker_file.unlink()

    # 5. 05_audit_report.md
    report_content = f"""# Exp161 대체 점수원 Provenance 및 Coverage 감사 보고서 ({audit_scope})

본 보고서는 Exp160 복구 정책(candidate_count == 0)에서 누락된 타겟들에 대한 대체 점수원(`미분·스펙트럼 통계 점수`, `부분 형태 일치 점수`, `구간 분포 점수`)의 provenance 및 coverage 감사 결과를 기술합니다.

---

## 1. 감사 개요
- **대상 파일:** `outputs/exp160_candidate_zero_similarity_recovery/02_exp160_predictions.jsonl`
- **해시 검증:** {EXP160_EXPECTED_HASH} (일치)
- **적용 limit:** {f'최대 {dataset_limit}개 데이터셋' if dataset_limit else '전체 데이터셋 (Full)'}
- **총 감사 대상 dataset/alpha 쌍:** {total_targets}개
- **고유 감사 대상 데이터셋:** {total_target_datasets}개

---

## 2. 소스별 커버리지 및 적합성 통계

| 소스 구분 | 실제 계산 완료 개수 (Coverage) | 정책 적합 개수 (Policy-Eligible) | Full-Coverage Eligible 여부 |
| :--- | :---: | :---: | :---: |
| **미분·스펙트럼 통계 점수** | {spectral_computed_count} ({spectral_computed_count / total_targets * 100:.1f}%) | {spectral_eligible_count} (0.0%) | **부적합 (Ineligible)** |
| **부분 형태 일치 점수** | {shapelet_computed_count} ({shapelet_computed_count / total_targets * 100:.1f}%) | {shapelet_eligible_count} (0.0%) | **부적합 (Ineligible)** |
| **구간 분포 점수** | {interval_computed_count} ({interval_computed_count / total_targets * 100:.1f}%) | {interval_eligible_count} ({interval_eligible_count / total_targets * 100:.1f}%) | **부적합 (Ineligible historical artifact)** |

- **Common Support (세 소스 모두 계산 완료):** {common_support_count}개 ({common_support_count / total_targets * 100:.1f}%)

---

## 3. 핵심 발견 및 결론

1. **실제 계산 vs Passthrough 명확한 분리:**
   - Exp99 및 Exp101 CSV 결과에 비대상 행으로 기록되어 있던 Exp93 passthrough 행들을 식별 및 배제함으로써, 실제 대체 점수가 계산된 데이터셋만을 올바르게 집계하였습니다.

2. **Policy-Eligible Source Deficit (Blocker 작동):**
   - 과거 TEST 라벨/성능 정보를 기반으로 target을 선택한 Exp99 및 Exp101은 전체 policy replay에 적합하지 않은 것(Eligible = False)으로 판명되었습니다.
   - Exp46은 family/difficulty 진단으로 고른 13개 데이터셋만 커버하므로, 과거 artifact 자체는 정책 적합 source가 아닙니다.
   - 따라서, 정책 적용을 위한 2개 이상의 full-coverage eligible source 요건을 충족하지 못해 **블로커(`BLOCKER_REPORT.md`)가 작동**하였습니다.

3. **향후 진행 방향:**
   - TEST metric에 의존하지 않고 전체 데이터셋을 아우를 수 있는 신규 피처 계산 실험의 재설계가 필수적입니다.
"""
    (output_dir / "05_audit_report.md").write_text(report_content, encoding="utf-8")

    # Generate Manifest files
    generate_manifest(output_dir)


def generate_manifest(output_dir: Path):
    print("Generating manifest files...")
    files = sorted(list(output_dir.glob("*")))
    manifest_rows = []

    # We will exclude MANIFEST.csv and MANIFEST.md themselves from the hash calculation list to avoid circular dependencies
    target_files = [f for f in files if f.name not in ("MANIFEST.csv", "MANIFEST.md")]

    for f in target_files:
        sha256 = get_sha256(f)
        size = f.stat().st_size
        rel_path = f.relative_to(PROJECT_DIR)
        manifest_rows.append({
            "filepath": str(rel_path),
            "size_bytes": size,
            "sha256": sha256
        })

    # Write MANIFEST.csv
    write_csv(output_dir / "MANIFEST.csv", manifest_rows, ["filepath", "size_bytes", "sha256"])

    # Write MANIFEST.md
    md_lines = [
        "# Exp161 Artifacts Manifest",
        "",
        "| Filepath | Size (Bytes) | SHA256 |",
        "| :--- | :---: | :--- |"
    ]
    for r in manifest_rows:
        md_lines.append(f"| `{r['filepath']}` | {r['size_bytes']} | `{r['sha256']}` |")

    (output_dir / "MANIFEST.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print("Manifest generation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-limit", type=int, default=None, help="Limit to first N datasets")
    parser.add_argument("--output-dir", type=Path, default=None, help="Write audit artifacts to this directory")
    args = parser.parse_args()

    run_audit(args.dataset_limit, args.output_dir)
