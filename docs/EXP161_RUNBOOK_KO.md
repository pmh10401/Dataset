# Exp161 대체 점수원 감사 실행 가이드 (Runbook)

본 문서는 Exp161 대체 점수원 provenance 및 coverage 감사 실험의 구현 검증, 단위 테스트 실행, 그리고 smoke audit 실행을 위한 절차를 설명합니다.

---

## 1. 전제 조건 및 환경 정보
- **저장소 위치:** `/Users/minho/Documents/timesries project`
- **데이터셋 폴더:** `/Users/minho/Documents/Dataset`
- **개발 환경:** Python 3 standard library, `unittest`
- **핵심 입력 파일:**
  - Frozen Exp160 prediction: `outputs/exp160_candidate_zero_similarity_recovery/02_exp160_predictions.jsonl`
  - Exp99 CSV: `experiment_99_spectral_derivative_feature_score_results.csv`
  - Exp101 CSV: `experiment_101_shapelet_normal_prototype_results.csv`
  - Exp46 CSV: `experiment_46_model_hard_interval_drcif_lite_results.csv`

---

## 2. 코드 구조 검증 및 단위 테스트 실행

### A. 구문 오류 및 컴파일 테스트
Python 파일들에 대한 기본적인 컴파일 상태를 확인하기 위해 다음 명령어를 실행합니다:
```bash
python3 -m py_compile experiments/exp161/*.py
```

### B. 단위 테스트 및 구조 테스트 실행
감사 기능 핵심 로직의 정확성 및 디렉토리 구조 검증을 위한 테스트를 수행합니다:
```bash
python3 -m unittest \
  experiments/exp161/test_exp161_alternative_source_audit.py \
  experiments/exp161/test_exp161_structure.py
```

---

## 3. Smoke Audit 실행 (Dataset Limit = 10)
Full audit 실행 전, 구현체 검증 및 10개 데이터셋 제한 smoke 테스트를 진행합니다.
```bash
python3 experiments/exp161/run_experiment_161_alternative_source_audit.py \
  --dataset-limit 10 \
  --output-dir outputs/exp161_alternative_source_audit_smoke
```

- **산출물 디렉토리:** `outputs/exp161_alternative_source_audit_smoke/`
- **생성 파일 목록:**
  - `01_source_inventory.csv`
  - `02_candidate_zero_source_coverage.csv`
  - `03_legacy_index_overlap.csv`
  - `04_audit_summary.csv`
  - `05_audit_report.md`
  - `BLOCKER_REPORT.md` (블로커 조건 충족 시 자동 생성됨)
  - `MANIFEST.csv` / `MANIFEST.md`

### Full Audit
```bash
python3 experiments/exp161/run_experiment_161_alternative_source_audit.py \
  --output-dir outputs/exp161_alternative_source_audit
```

---

## 4. Blocker 발생 시 조치 및 해석 가이드
감사 결과 policy-eligible full-coverage 소스가 2개 미만일 때 `BLOCKER_REPORT.md` 파일이 자동 생성됩니다.
- 본 블로커는 사후 정보 누설(Retrospective leakage) 및 특정 데이터셋 편향 문제를 안고 있는 기존 대체 점수원을 그대로 복구 정책(Exp162)에 유용하는 것을 방지하는 안전장치입니다.
- **해결 방안:** 향후 Exp162 정책 제안을 위해서는, TEST metric 및 사전 필터링 진단 이력에 의존하지 않는 **전체 데이터셋 대상의 독립적 피처 재계산(re-computation)** 작업 계획을 수립해야 합니다.
