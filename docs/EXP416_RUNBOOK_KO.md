# Exp416 Exp404 Hard TEST 혼동행렬 감사 실행서

## 실행 경계

Exp416은 이미 동결된 Exp404 Hard prediction 2,218개 좌표를 frozen TEST
label 195,900개와 한 번만 대조하는 retrospective audit이다. 아래 CLI와
이 실행서를 먼저 커밋하고, worktree가 정확히 clean인 것을 확인한 뒤에만
실제 frozen TEST label을 연다. 즉, 사전등록 및 실행 commit은 실제 TEST
label load보다 반드시 앞선다.

Exp415 prediction은 생성하지 않는다. Exp404 prediction, manifest, frozen
TEST label 및 source DB는 수정하지 않는다. 이 결과를 보고 alpha,
threshold, source, candidate budget, lane, family 또는 dataset subset을
재선택하지 않으며 운영 경보나 운영 정책으로 승격하지 않는다.

## 고정 SHA 확인

다음 네 SHA256이 모두 사전등록 값과 정확히 같아야 한다.

```bash
shasum -a 256 \
  "/Users/mac/Documents/Dataset/univariate_ts.db" \
  "/Users/mac/Documents/Dataset/exp403_408_feature_ml_validation/exp404_hard_confidence.jsonl" \
  "/Users/mac/Documents/Dataset/exp403_408_feature_ml_validation/exp404_prediction_manifest.json" \
  "/Users/mac/Documents/Dataset/exp403_408_feature_ml_validation/exp403_408_frozen_test_labels.csv"
```

```text
8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c  /Users/mac/Documents/Dataset/univariate_ts.db
437dc63754a29d0d65c0b30bfee54102abdd9fb56d45d2c5b1c5a511b0448e70  /Users/mac/Documents/Dataset/exp403_408_feature_ml_validation/exp404_hard_confidence.jsonl
41a913415d22bb6cedb692a1c1cce85eafccae49f8a3fd108ed53e1bc3d2a7b8  /Users/mac/Documents/Dataset/exp403_408_feature_ml_validation/exp404_prediction_manifest.json
b483ba00cee6f74dc6e2db1e4c9bafea4237a3be7ba60d7f45dc32aa29e4f3f4  /Users/mac/Documents/Dataset/exp403_408_feature_ml_validation/exp403_408_frozen_test_labels.csv
```

## 합성 테스트와 clean commit

실제 TEST label을 열기 전에 다음 검증을 실행한다.

```bash
.venv/bin/python -m unittest discover \
  -s experiments/exp416_exp404_hard_test_confusion_audit/tests \
  -p "test_*.py"
.venv/bin/python -m unittest discover \
  -s experiments/exp414_missing_aware_near_repeat/tests \
  -p "test_*.py"
.venv/bin/python -m compileall -q \
  experiments/exp416_exp404_hard_test_confusion_audit \
  scratch/run_exp416_exp404_hard_test_confusion_audit.py
git diff --check
```

CLI와 이 실행서를 커밋한 뒤 다음 상태를 확인한다.

```bash
git status --short --branch
git rev-parse HEAD
```

`git status --short --branch`에는 branch header만 있어야 한다. tracked 또는
untracked entry가 하나라도 있으면 실행하지 않는다. `git rev-parse HEAD`의
40자리 commit은 실제 TEST label load 전에 생성된 clean commit이어야 한다.

출력 경로가 존재하지 않는 것도 실행 전에 확인한다.

```bash
test ! -e "/Users/mac/Documents/Dataset/exp416_exp404_hard_test_confusion_audit"
```

## 단 한 번의 실행

다음 명령 전체를 동기식으로 정확히 한 번만 실행한다.

```bash
.venv/bin/python scratch/run_exp416_exp404_hard_test_confusion_audit.py \
  --database-reference "/Users/mac/Documents/Dataset/univariate_ts.db" \
  --prediction "/Users/mac/Documents/Dataset/exp403_408_feature_ml_validation/exp404_hard_confidence.jsonl" \
  --prediction-manifest "/Users/mac/Documents/Dataset/exp403_408_feature_ml_validation/exp404_prediction_manifest.json" \
  --frozen-test-labels "/Users/mac/Documents/Dataset/exp403_408_feature_ml_validation/exp403_408_frozen_test_labels.csv" \
  --output-dir "/Users/mac/Documents/Dataset/exp416_exp404_hard_test_confusion_audit"
```

출력 디렉터리가 이미 존재하면 중단한다. 기존 디렉터리 또는 산출물을
삭제하거나 덮어쓰지 않으며, 성공 또는 실패 후에도 같은 명령을
재실행하지 않는다.

## 게시 결과 독립 검증

정확히 네 파일이고 dataset CSV가 header 포함 1,118줄인지 확인한다.

```bash
output="/Users/mac/Documents/Dataset/exp416_exp404_hard_test_confusion_audit"
test "$(find "$output" -maxdepth 1 -type f | wc -l | tr -d ' ')" = "4"
cd "$output"
jq -r '.artifact_sha256 | to_entries[] | "\(.value)  \(.key)"' \
  MANIFEST.json | shasum -a 256 -c -
test "$(wc -l < exp416_dataset_metrics.csv | tr -d ' ')" = "1118"
jq '.micro.counts | .tp + .fp + .fn + .tn' \
  exp416_confusion_matrix.json
```

Manifest가 기록한 세 non-manifest payload SHA는 모두 `OK`여야 하고 micro
count 합계는 `195900`이어야 한다. Manifest 자체 SHA도 별도로 기록한다.

```bash
shasum -a 256 MANIFEST.json
```

이 감사는 TEST-informed 재선택, 새 prediction 생성 또는 운영 승격을
허가하지 않는다. 향후 변경은 새 experiment 번호와 새 사전등록이
필요하다.
