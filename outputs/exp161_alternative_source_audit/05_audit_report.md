# Exp161 대체 점수원 Provenance 및 Coverage 감사 보고서 (Full)

본 보고서는 Exp160 복구 정책(candidate_count == 0)에서 누락된 타겟들에 대한 대체 점수원(`미분·스펙트럼 통계 점수`, `부분 형태 일치 점수`, `구간 분포 점수`)의 provenance 및 coverage 감사 결과를 기술합니다.

---

## 1. 감사 개요
- **대상 파일:** `outputs/exp160_candidate_zero_similarity_recovery/02_exp160_predictions.jsonl`
- **해시 검증:** ce97a7df2873d4bc628bbd58e2a7b83142f9d8cf8670bdf598f2b79aa0ded73d (일치)
- **적용 limit:** 전체 데이터셋 (Full)
- **총 감사 대상 dataset/alpha 쌍:** 1412개
- **고유 감사 대상 데이터셋:** 537개

---

## 2. 소스별 커버리지 및 적합성 통계

| 소스 구분 | 실제 계산 완료 개수 (Coverage) | 정책 적합 개수 (Policy-Eligible) | Full-Coverage Eligible 여부 |
| :--- | :---: | :---: | :---: |
| **미분·스펙트럼 통계 점수** | 487 (34.5%) | 0 (0.0%) | **부적합 (Ineligible)** |
| **부분 형태 일치 점수** | 439 (31.1%) | 0 (0.0%) | **부적합 (Ineligible)** |
| **구간 분포 점수** | 14 (1.0%) | 0 (0.0%) | **부적합 (Ineligible historical artifact)** |

- **Common Support (세 소스 모두 계산 완료):** 0개 (0.0%)

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
