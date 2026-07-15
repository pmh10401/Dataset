# Exp161 Blocker Report: Policy-Eligible Alternative Source Deficit

**Date:** 2026-07-14
**Experiment ID:** `experiment_161_alternative_source_candidate_zero_audit`

## Blocker Verdict
대체 점수원 감사를 수행한 결과, Exp160 `candidate_count == 0` 복구 정책(Exp162)으로의 이행을 보증할 수 있는 **policy-eligible full-coverage 소스가 부족(2개 미만)**하여, 본 감사는 정책 추천 단계 진행을 차단하고 블로커를 해제하지 않습니다.

- **필수 조건:** 2개 이상의 policy-eligible full-coverage 소스 확보
- **실제 발견:** 0개

---

## 상세 감사 결과 요약 (Full Target 1412개 기준)

1. **미분·스펙트럼 통계 점수 (spectral_derivative / Exp99)**
   - 실제 계산 완료 (Actual Computed): 487 / 1412 (34.5%)
   - 정책 적합성 통과 (Policy-Eligible): 0 / 1412 (0.0%)
   - **판정:** Ineligible (과거 TEST 진단 및 특정 Family 이름에서 유래한 provenance로 전체 정책 적용 부적합)

2. **부분 형태 일치 점수 (shapelet_normal_prototype / Exp101)**
   - 실제 계산 완료 (Actual Computed): 439 / 1412 (31.1%)
   - 정책 적합성 통과 (Policy-Eligible): 0 / 1412 (0.0%)
   - **판정:** Ineligible (과거 TEST 진단에서 유래한 provenance로 전체 정책 적용 부적합)

3. **구간 분포 점수 (interval_quantile / Exp46)**
   - 실제 계산 완료 (Actual Computed): 14 / 1412 (1.0%)
   - 정책 적합성 통과 (Policy-Eligible): 0 / 1412 (0.0%)
   - **판정:** Ineligible historical artifact (family/difficulty-selected 13개 데이터셋에만 계산되어 정책 입력으로 사용할 수 없음)

---

## Blocker 원인 분석

1. **Exp99 및 Exp101의 Provenance 오염:**
   - 이 소스들은 사전 필터링 과정에서 TEST 성능 진단 데이터를 간접 유용하여 타겟이 선정되었습니다. (e.g. `research_target`, `spectral_target`, `shapelet_target`).
   - 따라서 strict label-free policy replay에 직접 사용하는 것은 retrospective leakage 위험이 있어 eligible = False 처리되었습니다.

2. **Exp46의 Provenance 및 Coverage 제한:**
   - 구간 분포 점수(Exp46)는 family/difficulty 진단으로 선정된 13개 데이터셋에만 계산되었으므로, historical artifact 자체가 정책 입력으로 부적합하며 Exp160 target 전체도 커버하지 못합니다.

## 결론 및 권장 조치
Exp162를 추진하기 위해서는, 이들 대체 점수원의 **TEST metric 비의존적/전체 데이터셋 대상의 신규 full-coverage 점수 계산 및 저장(re-computation)**이 선행되어야 합니다.
