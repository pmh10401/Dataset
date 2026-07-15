# Exp161 대체 점수원 Provenance/Coverage 감사 사전 등록 (Preregistration)

- **실험 ID:** `experiment_161_alternative_source_candidate_zero_audit`
- **작성일:** 2026-07-14
- **담당자:** Antigravity (Gemini 3.5 Flash High)

---

## 1. 배경 및 목적
Exp160 실험에서 `n_train >= 5` 이지만 유사도 기반 reference pool 부족 등의 사유로 `candidate_count == 0` (후보 0개) 상태가 된 데이터셋들에 대해, 기존에 계산된 대체 점수원들을 이용하여 복구 조치를 취할 수 있는지 검증한다.
이를 위해 과거 실험인 **Exp99 (미분·스펙트럼 통계 점수)**, **Exp101 (부분 형태 일치 점수)**, **Exp46 (구간 분포 점수)**의 데이터 provenance 및 coverage에 대한 read-only 감사를 수행한다.
본 감사는 복구 알고리즘의 성능 평가(F1, TP/FP 등)가 아니며, 순수하게 데이터 수집 이력과 정책 적합성(Policy-Eligibility)만을 평가하는 안전장치로 설계된다.

---

## 2. 핵심 가설 및 감사 규칙

### 가설 1. 대체 점수원들의 Provenance 오염
- **Exp99 (`spectral_derivative`) 및 Exp101 (`shapelet_normal_prototype`):** 해당 실험들의 `research_target` 및 `spectral_target`/`shapelet_target`은 과거 TEST 데이터 성능 진단 이력에서 유래했다. 따라서 strict label-free policy replay에 직접 적용하는 것은 retrospective leakage(사후 정보 누설) 위험이 존재하며, **Policy-Eligible = False**로 판정될 것이다.
- **Exp46 (`interval_quantile`):** 특정 family 및 난이도가 높은 13개 데이터셋에 대해서만 한정적으로 계산되었으므로, 과거 artifact 자체는 policy-eligible하지 않으며 전체 타겟 인구를 커버하는 **Full-Coverage Source**도 되지 못할 것이다.

### 가설 2. Blocker 작동
- 두 개 이상의 policy-eligible full-coverage 소스가 확보되지 않을 경우, 복구 정책 실행을 차단하고 `BLOCKER_REPORT.md`를 생성한다. 가설 1에 의해 모든 소스가 full-coverage policy-eligible 요건을 충족하지 못하므로, 본 감사에서는 **Blocker가 반드시 작동할 것**으로 예측한다.

---

## 3. 감사 대상 및 범위

### A. 감사 대상 인구 (Target Population)
- Frozen Exp160 prediction (`outputs/exp160_candidate_zero_similarity_recovery/02_exp160_predictions.jsonl`)에서 다음 조건을 만족하는 dataset/alpha 행:
  - `n_train >= 5`
  - `candidate_count == 0`
- 모든 alpha (`0.005`, `0.01`, `0.02`, `0.05`)를 동일하게 감사 대상에 포함한다.

### B. 배제 조건 (Forbidden Inputs & Rules)
- `tp`, `fp`, `fn`, `f1`, `auc_roc`, `auc_pr`, `oracle_f1`, `anomaly_count` 및 TEST label 등 성능 관련 필드는 소스 coverage 집계, eligibility 판정, 소스 추천에 **절대 사용하지 않는다.**
- CSV에 행이 존재한다는 사실만으로 대체 점수가 실제로 계산되었다고 오인하지 않는다 (Exp93 passthrough 행을 엄격히 발라내고 실제 계산 완료된 행만 computed = True로 인정).
- `selected_indices` overlap 분석은 `research_target == 1`로 명시되고 실제로 계산이 완료된 경우에만 순수 진단 목적으로 계산하며, 정책 승격 근거로 사용하지 않는다.

---

## 4. 기대 산출물
- `01_source_inventory.csv`: 대체 점수원 목록 및 메타데이터
- `02_candidate_zero_source_coverage.csv`: 타겟 dataset/alpha 쌍별 3개 소스의 computed 및 policy-eligible 여부
- `03_legacy_index_overlap.csv`: 진단용 selected_indices overlap 분석
- `04_audit_summary.csv`: 총합 coverage 및 통계 요약
- `05_audit_report.md`: 한글 요약 보고서
- `BLOCKER_REPORT.md`: 블로커 발생 시 경고서
- `MANIFEST.csv`, `MANIFEST.md`: 산출물 무결성 관리 정보
