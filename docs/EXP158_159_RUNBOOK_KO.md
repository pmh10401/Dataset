# Exp158–159 실행 가이드

## 1. 구현 파일 구성

- `experiments/exp158_159/__init__.py`
- `experiments/exp158_159/similarity_pool_core.py`: DB 및 label 접근이 배제된 순수 수학/논리 헬퍼 함수
- `experiments/exp158_159/run_experiment_158_159_similarity_pooling.py`: audit/predict/evaluate/manifest 파이프라인
- `experiments/exp158_159/test_exp158_159_similarity_pooling.py`: 계약 기반 17개 요구사항 테스트
- `experiments/exp158_159/test_exp158_159_structure.py`: 구조적 제약 및 AST 검사 테스트

## 2. 단위 및 구조 테스트 실행

저장소 루트 디렉토리(`/Users/minho/Documents/timesries project`)에서 다음 명령어를 실행하여 코드 구문 오류 검사와 단위 테스트를 통과하는지 확인합니다.

```bash
# 구문 및 컴파일 오류 검사
python3 -m py_compile experiments/exp158_159/*.py

# 단위 테스트 및 구조 테스트 실행
python3 -m unittest \
  experiments/exp158_159/test_exp158_159_similarity_pooling.py \
  experiments/exp158_159/test_exp158_159_structure.py
```

## 3. 10개 데이터셋 Smoke 테스트 실행

테스트가 통과하면 다음 명령어로 10개 target 데이터셋에 대한 파이프라인(audit -> predict -> evaluate -> manifest)을 실행합니다. `--dataset-limit 10`은 prediction/evaluation target만 제한하며, donor descriptor와 pooled reference는 전체 1,117개 데이터셋의 TRAIN OOF score로 구성합니다. 따라서 smoke에서도 실제 similarity-pooling 경로가 실행됩니다.

```bash
# Smoke 테스트 실행
python3 experiments/exp158_159/run_experiment_158_159_similarity_pooling.py \
  --stage audit \
  --stage predict \
  --stage evaluate \
  --stage manifest \
  --dataset-limit 10 \
  --output-dir outputs/exp158_159_small_train_similarity_pooling_smoke
```

## 4. 산출물 검증

- **평가 필드 누락 검증**: `outputs/exp158_159_small_train_similarity_pooling_smoke/02_exp158_predictions.jsonl` 및 `04_exp159_predictions.jsonl`에 `y_test`, `TP`, `FP`, `F1` 관련 키가 전혀 들어있지 않아야 합니다. (contains_test_labels = False, contains_evaluation_metrics = False)
- **상호 배타성 및 Union 일치**: evaluation 단계에서 다음 사항을 엄격히 검사합니다.
  - `hard_indices`, `standard_review_indices`, `priority_review_indices` 간의 중복(overlap)이 0인지 확인합니다.
  - 세 lane의 합집합(union)이 `candidate_indices`와 정확히 일치하는지 확인합니다.
- **해시 검증**: prediction JSONL 생성 이후 manifest가 정상 동결(freeze)되었으며 evaluation이 해당 prediction의 SHA256 해시를 검증하며 실행되는지 확인합니다.

## 5. 전체 retrospective 실험

```bash
python3 experiments/exp158_159/run_experiment_158_159_similarity_pooling.py \
  --stage audit \
  --stage predict \
  --stage evaluate \
  --stage manifest \
  --output-dir outputs/exp158_159_small_train_similarity_pooling
```

전체 prediction은 실험별로 `1,117 datasets x 4 alpha = 4,468 rows`여야 합니다. Exp159의 Hard/Priority index는 frozen Exp152와 dataset/alpha별 exact match여야 하며, `5 <= n_train < 20` 범위 밖 데이터셋의 Standard review도 baseline과 같아야 합니다.
