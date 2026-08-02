# Exp422-TEST-EVAL 실행 안내

## 사전 점검

작업 트리에서 테스트를 먼저 실행한다.

```bash
python3 -m unittest discover -s experiments/exp422_test_eval/tests -p 'test_*.py'
```

DB quick check와 SHA256은 읽기 전용으로 확인한다. 현재 Exp422-IR/IR-E
출력은 혼동행렬 입력으로 사용하지 않는다.

## 실행

실행은 항상 새 출력 디렉터리를 사용한다.

```bash
python3 scratch/run_exp422_test_eval.py \
  --database /Users/mac/Documents/Dataset/univariate_ts.db \
  --prediction /Users/mac/Documents/Dataset/<frozen-exp422-prediction>.jsonl \
  --prediction-manifest /Users/mac/Documents/Dataset/<frozen-exp422-prediction-manifest>.json \
  --frozen-test-labels /Users/mac/Documents/Dataset/exp403_408_feature_ml_validation/exp403_408_frozen_test_labels.csv \
  --output-root /Users/mac/Documents/Dataset/exp422_test_eval_attempt_YYYYMMDD-HHMMSS
```

예측 파일이 아직 없으면 동일 명령을 실행해도 label 파일을 열지 않고
`blocked_no_prediction` 상태로 종료한다. 이 상태는 실패한 혼동행렬이
아니라 필요한 입력이 아직 없다는 기록이다.

## 로그 해석

`progress.jsonl`의 `labels_read_started` 이벤트 이전에는 TEST label을
열면 안 된다. `prediction_frozen`이 없으면 라벨을 읽은 것으로 간주하지
않는다. 성공 시 `matrix_written`과 `completed` 이벤트가 마지막에 남는다.

## 결과 해석

혼동행렬은 연구용 평가이며 운영 자동화에 직접 연결하지 않는다. 특히 F1
평균 하나만 보고 판단하지 않고 FP(오경보)와 FN(놓친 이상)을 함께 본다.
