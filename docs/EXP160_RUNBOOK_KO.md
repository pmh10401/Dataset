# Exp160 실행 가이드 (RUNBOOK)

## 1. 구현 파일 구성

- `experiments/exp160/__init__.py`: 패키지 초기화 파일
- `experiments/exp160/candidate_zero_core.py`: DB 및 label 접근이 배제된 순수 수학/논리 헬퍼 함수
- `experiments/exp160/run_experiment_160_candidate_zero_similarity_recovery.py`: audit/predict/evaluate/manifest 파이프라인
- `experiments/exp160/test_exp160_candidate_zero_recovery.py`: 계약 기반 요구사항에 대한 단위 테스트
- `experiments/exp160/test_exp160_structure.py`: 구조적 제약 및 AST 검사 테스트

## 2. 단위 및 구조 테스트 실행

저장소 루트 디렉토리(`/Users/minho/Documents/timesries project`)에서 다음 명령어를 실행하여 코드 구문 오류 검사와 단위 테스트를 통과하는지 확인합니다.

```bash
# 구문 및 컴파일 오류 검사
python3 -m py_compile experiments/exp160/__init__.py \
  experiments/exp160/candidate_zero_core.py \
  experiments/exp160/run_experiment_160_candidate_zero_similarity_recovery.py \
  experiments/exp160/test_exp160_candidate_zero_recovery.py \
  experiments/exp160/test_exp160_structure.py

# 단위 테스트 및 구조 테스트 실행
python3 -m unittest \
  experiments/exp160/test_exp160_candidate_zero_recovery.py \
  experiments/exp160/test_exp160_structure.py
```

## 3. 10개 데이터셋 Smoke 테스트 실행

테스트가 통과하면 다음 명령어로 10개 target 데이터셋에 대한 파이프라인(audit -> predict -> evaluate -> manifest)을 실행합니다. `--dataset-limit 10`은 prediction/evaluation target만 제한하며, donor descriptor와 pooled reference는 전체 1,117개 데이터셋의 TRAIN OOF score로 구성합니다. 따라서 smoke에서도 실제 similarity-pooling 경로가 실행됩니다.

```bash
# Smoke 테스트 실행
python3 experiments/exp160/run_experiment_160_candidate_zero_similarity_recovery.py \
  --stage audit \
  --stage predict \
  --stage evaluate \
  --stage manifest \
  --dataset-limit 10 \
  --output-dir outputs/exp160_candidate_zero_similarity_recovery_smoke
```

## 4. 산출물 검증

- **평가 필드 누락 검증**: `outputs/exp160_candidate_zero_similarity_recovery_smoke/02_exp160_predictions.jsonl`에 `y_test`, `TP`, `FP`, `F1` 관련 키가 전혀 들어있지 않아야 합니다. (contains_test_labels = False, contains_evaluation_metrics = False)
- **candidate-zero 정책 명시**: 모든 prediction 행에 `uses_test_candidate_empty_gate=1`을 기록하고, 실제 gate가 적용된 dataset/alpha 행만 `candidate_empty_gate_applied=1`과 `candidate_zero_target=1`로 표시합니다.
- **사용자용 점수 명칭**: `source_display_names`에 시간 패턴 점수, 주파수 변화 점수, 반복 구조 점수, 다중 커널 점수가 기록되어야 합니다.
- **상호 배타성 및 Union 일치**: evaluation 단계에서 다음 사항을 엄격히 검사합니다.
  - `hard_indices`, `standard_review_indices`, `priority_review_indices` 간의 중복(overlap)이 0인지 확인합니다.
  - 세 lane의 합집합(union)이 `candidate_indices`와 정확히 일치하는지 확인합니다.
- **해시 검증**: prediction JSONL 생성 이후 manifest가 정상 동결(freeze)되었으며 evaluation이 해당 prediction의 SHA256 해시를 검증하며 실행되는지 확인합니다.

## 5. 전체 retrospective 실험 (참고)

본 1차 구현 단계에서는 실행하지 않으나, 추후 진행할 전체 retrospective 실행 명령어는 다음과 같습니다.

```bash
python3 experiments/exp160/run_experiment_160_candidate_zero_similarity_recovery.py \
  --stage audit \
  --stage predict \
  --stage evaluate \
  --stage manifest \
  --output-dir outputs/exp160_candidate_zero_similarity_recovery
```

전체 prediction은 `1,117 datasets x 4 alpha = 4,468 rows`여야 합니다. target이 아닌 dataset/alpha의 Standard, Hard, Priority review는 모두 frozen Exp152와 exact match여야 합니다.

## 6. 해석 제한

- Exp160은 기존 autonomous Hard alert를 변경하는 실험이 아닙니다. Hard 결과를 보존하면서 baseline candidate count가 0인 작은 데이터셋의 Standard review coverage를 추가(recovery)한 실험입니다.
- 실제 equipment/recipe/sensor/step 메타데이터가 없어 score-shape similarity가 물리적 동질성을 뜻하는지는 확인되지 않았습니다.
- 동일 benchmark TEST를 반복 사용한 retrospective 결과입니다.
- end-to-end strict TRAIN-only 또는 실제 설비 배포 성능을 주장할 수 없습니다.
