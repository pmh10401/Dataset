# Exp422-TEST-EVAL 사전등록

## 목적

Exp422의 TRAIN-only MPS/CPU 수치 진단이 실제 TEST 이상을 얼마나 잘
구분하는지, 운영 결정을 바꾸지 않는 별도 사후 평가로 측정한다. 이 단계는
새 모델이나 새 threshold를 고르는 실험이 아니다.

현재 Exp422-IR/IR-E 매니페스트에는 전체 TEST 예측 좌표가 없으므로, 예측
파일이 생기기 전에는 혼동행렬을 계산하지 않고 `blocked_no_prediction`으로
기록한다.

## 동결 입력과 확인값

- DB: `/Users/mac/Documents/Dataset/univariate_ts.db`
  - SHA256: `8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c`
- TEST label snapshot:
  `/Users/mac/Documents/Dataset/exp403_408_feature_ml_validation/exp403_408_frozen_test_labels.csv`
  - SHA256: `b483ba00cee6f74dc6e2db1e4c9bafea4237a3be7ba60d7f45dc32aa29e4f3f4`
  - 1,117개 dataset, 195,900개 TEST instance
- 현재 Exp422 진단 참고 매니페스트(성능 입력 아님):
  - IR SHA256: `cdf6d4adab4f0533a56445955afae96d778f5627e97e981b0c8ece30e2a46c87`
  - IR-E SHA256: `cb8e01e24c5e4f5c95bf05429149b1cbf43b2267ed0b1edc82e2858dd093b6cc`

예측 파일과 예측 매니페스트의 SHA256은 파일이 제공되는 시점에 먼저
동결하여 결과 매니페스트에 기록한다.

## 평가 순서

1. 새 출력 디렉터리·PID·진행 로그를 만든다.
2. prediction JSONL과 매니페스트의 파일 해시를 계산한다.
3. prediction 행의 중복·범위·라벨/평가 필드와 매니페스트의 governance를 검증한다.
4. 위 검증이 끝난 뒤에만 frozen TEST label CSV를 연다.
5. 모든 TEST instance를 대상으로 predicted positive는 prediction 좌표,
   predicted negative는 그 밖의 좌표로 정의한다.
6. `[[TN, FP], [FN, TP]]`와 dataset별·전체 지표를 계산한다.
7. 입력을 다시 해시하고 새 디렉터리에 결과를 게시한다.

## 혼동행렬 정의

- TN: 정상 label을 정상으로 둔 instance
- FP: 정상 label을 이상으로 둔 instance
- FN: 이상 label을 정상으로 둔 instance
- TP: 이상 label을 이상으로 둔 instance
- 행렬 표기: `[[TN, FP], [FN, TP]]`
- precision, recall, specificity, FPR, FNR, F1을 계산하며 분모가 0이면
  `null`로 남긴다.

## 금지사항

- TEST label로 alpha, threshold, source, batch size, budget 또는 lane을
  재선택하지 않는다.
- TEST 결과로 모델·운영 정책을 수정하거나 Priority/Shadow를 자동 경보로
  승격하지 않는다.
- Exp422-IR/IR-E의 숫자 SHA나 CBF TRAIN sample을 TEST 예측으로 간주하지 않는다.
- 예측 파일이 없거나 TEST 좌표 범위를 덮지 못하면 숫자를 만들어내지 않는다.
- 기존 Exp416 결과를 Exp422 결과로 복사하지 않는다.

## 산출물

성공 시 새 디렉터리에 다음을 쓴다.

- `exp422_test_eval_confusion_matrix.json`
- `exp422_test_eval_dataset_metrics.csv`
- `EXP422_TEST_EVAL_RESULT_REPORT_KO.md`
- `MANIFEST.json`
- `progress.jsonl`
- `pid.txt`

예측 입력이 없으면 혼동행렬 파일 대신 `status.json`과 `MANIFEST.json`에
`blocked_no_prediction`, `test_labels_opened=false`를 기록한다.
