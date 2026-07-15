# Exp164 실행 지침서 (Runbook)

본 문서는 Exp164 TRAIN-Normal Agreement Risk Audit 실험을 실행하고 검증하기 위한 가이드라인이다.

## 1. 실행 전 확인 사항

실행 전 동결 데이터 파일들이 존재하고 올바른 SHA256 해시를 가졌는지 확인하십시오.
- **데이터베이스**: `/Users/minho/Documents/Dataset/univariate_ts.db` (SHA256: `8458e678...`)
- **Exp162 Manifest**: `outputs/exp162_full_coverage_alternative_score_cache/02_cache_manifest.csv` (SHA256: `ec8e1a1c...`)
- **Exp162 Contract**: `outputs/exp162_full_coverage_alternative_score_cache/01_source_contract.json` (SHA256: `0a80d8a2...`)

## 2. 테스트 실행 방법

### 단위 테스트 및 구조 테스트 실행

TDD 준수 여부 및 TEST split 접근 차단 검증을 위해 아래 명령어를 수행합니다.

```bash
# 전체 테스트 실행
PYTHONPATH=. pytest experiments/exp164/

# 개별 테스트 실행
PYTHONPATH=. pytest experiments/exp164/test_exp164_train_normal_risk_audit.py
PYTHONPATH=. pytest experiments/exp164/test_exp164_structure.py
```

## 3. Smoke Test 실행 방법

Smoke Test는 10개 데이터셋을 대상으로 2개의 worker를 사용하고 체크포인트를 리셋하여 실행합니다.

```bash
python3 experiments/exp164/run_experiment_164_train_normal_agreement_risk_audit.py \
    --dataset-limit 10 \
    --workers 2 \
    --reset-checkpoint \
    --output-dir outputs/exp164_train_normal_agreement_risk_audit_smoke
```

### Smoke Test 독립 재집계 및 검증

```bash
python3 experiments/exp164/independent_reaggregate_exp164.py \
    outputs/exp164_train_normal_agreement_risk_audit_smoke
```

## 4. Full Run 실행 방법 (운영용)

> [!WARNING]
> 1차 구현 및 단위/구조 테스트, 10-dataset smoke 단계에서는 Full Run을 시작하지 마십시오.

1,117개 데이터셋 전체에 대해 7개의 worker를 이용하여 실행하는 명령어입니다.

```bash
python3 experiments/exp164/run_experiment_164_train_normal_agreement_risk_audit.py \
    --workers 7 \
    --output-dir outputs/exp164_train_normal_agreement_risk_audit
```

### Full Run 독립 재집계 및 검증

```bash
python3 experiments/exp164/independent_reaggregate_exp164.py \
    outputs/exp164_train_normal_agreement_risk_audit
```
