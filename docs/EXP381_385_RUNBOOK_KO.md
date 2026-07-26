# Exp381-385 실행 안내

## 전체 순서

```bash
python3 experiments/exp381_385/run_experiment_381_coverage_provenance_audit.py
python3 experiments/exp381_385/run_experiment_382_alternative_score_lattice.py
python3 experiments/exp381_385/run_experiment_383_source_dependence_audit.py
python3 experiments/exp381_385/run_experiment_384_exhaustive_lattice_risk.py
python3 experiments/exp381_385/run_experiment_385_pipeline_structures.py
```

## 사전 검증

```bash
python3 -m py_compile experiments/exp381_385/*.py
python3 -m unittest experiments.exp381_385.test_coverage_first_lattice
```

## 결과 위치

`/Users/minho/Documents/Dataset/exp381_385_coverage_first_feature_lattice`

## 중단 조건

- Exp381 캐시 유효 coverage가 1,117개 미만
- Exp357 또는 Exp382 prediction SHA256 불일치
- TEST 라벨 또는 평가 필드가 prediction에 포함됨
- 데이터셋 누락 또는 중복
- 결과 확인 후 상수나 규칙을 바꾸려는 경우

## 해석

- Exp382/384 단독 점수 성능은 기존 자동 경보의 대체 근거가 아니다.
- Exp385 `baseline_plus_review`는 사람 검토 지표다.
- Exp385 `train_risk_hard_expansion`은 사전 고정 risk target별 sensitivity다.
- `replacement_research`는 운영 후보가 아니다.
- TEST 성능이 좋은 행을 운영 기본값으로 자동 선택하지 않는다.
