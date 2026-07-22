# Exp309-316 신규 점수원 기반 FN 복구 설계

## 1. 목적

현재 Exp291 자동 경보는 전체 1,117개 데이터셋에서 실제 이상 3,918건 중
1,915건을 탐지하고 303건을 오탐했다. 자동 경보 정밀도는 86.339%, 재현율은
48.877%, 데이터셋 평균 F1은 0.608983이다.

자동 경보가 놓친 이상은 2,003건이며, 기존 자동 경보·일반 검토·연구 복구 후보
어디에도 포함되지 않은 이상은 1,529건이다. Exp309-316의 목적은 기존 후보의
순서를 다시 정하는 데 그치지 않고, 이 후보 밖 이상을 찾을 수 있는 독립적인
점수원을 추가하는 것이다.

## 2. 운영 원칙

- 기존 Exp291 자동 경보 2,218건은 변경하지 않는다.
- 신규 후보는 모두 retrospective research Shadow로 시작한다.
- Shadow는 사용자 자동 경보로 전송하지 않는다.
- TRAIN 정상 데이터만 feature fitting, score calibration, threshold 및 Gate 결정에 사용한다.
- TEST 라벨과 실제 이상 위치는 prediction hash 동결 후 평가에만 사용한다.
- family 또는 dataset 이름, 과거 family별 TEST 성능으로 source나 threshold를 선택하지 않는다.
- TEST 길이 기반 top-N budget과 tail replacement를 사용하지 않는다.
- Priority 또는 review 후보를 이번 실험에서 자동 경보로 승격하지 않는다.
- TEST 결과가 나쁘다는 이유로 동결된 window, alpha, architecture 또는 threshold를 변경하지 않는다.
- 자동 경보 승격은 결과 확인 후 별도의 새 실험 번호와 사전등록으로만 검토한다.

## 3. 전처리 계약

모든 신규 점수원은 다음 순서를 사용한다.

1. 원본 시간축에서 결측값과 NaN/Inf를 보정한다.
2. TRAIN 정상 데이터로 결정한 목표 길이에 정렬한다.
3. 길이 정렬 후 TRAIN 정상 통계로 정규화한다.

TEST 전체 평균, 표준편차 또는 TEST 길이는 calibration과 budget 결정에 사용하지 않는다.
각 prediction manifest에는 실제 적용한 전처리 계약과 코드 hash를 기록한다.

이 평가 DB에서 하나의 `candidate_index`는 시계열 내부의 시간점이 아니라 TEST 배열의
한 instance, 즉 하나의 가상 run trace 위치를 뜻한다. Exp309-315는 각 run 내부의
부분구간 진단을 점수 근거로 저장할 수 있지만, 최종 후보와 TP/FP 평가는 run instance
단위로 수행한다.

## 4. 실험 구성

### Exp309: 다중 구간 Discord

전체 시계열을 하나의 벡터로만 비교하지 않고 여러 길이의 부분구간을 TRAIN 정상
부분구간 memory와 비교한다. 정상 TRAIN에서 보기 어려운 국소 형태의 최대 거리와
상위 구간 거리 분포를 점수로 사용한다.

- 목표: 짧은 spike, 국소 진동, 짧은 형태 붕괴 탐지
- 실행: CPU 7 workers
- 출력: dataset별 점수, TRAIN calibration, 후보 index, 이상 구간 시작·끝
- 금지: TEST 후보 수를 이용한 top-N 제한

### Exp310: 다중 윈도 변화점

여러 시간 범위에서 후보 지점 앞뒤의 위치, 분산 및 분포 변화를 계산한다. 각 변화량은
TRAIN 정상 시계열에서 얻은 변화량 분포로 보정한다.

- 목표: 급격한 level shift와 느린 drift를 동시에 탐지
- 실행: CPU 7 workers
- 출력: window별 score와 통합 score
- 경계 구간은 최대 window의 receptive-field 영향을 별도로 기록

### Exp311: Wavelet 위상·일관성

Wavelet 계수의 에너지뿐 아니라 scale별 위상 변화, 주기 일관성 및 entropy를 TRAIN
정상 기준과 비교한다.

- 목표: 진동, 주기 변화, 주파수 크기는 비슷하지만 위상이 어긋난 이상 탐지
- 실행: CPU 7 workers
- 출력: scale별 score, 위상 score, 통합 score

### Exp312: 비학습 점수원 융합

Exp309-311의 후보와 기존 독립 방법군을 TRAIN-normal empirical p-value 또는 e-value로
변환한다. 상관된 점수의 단순 투표를 피하기 위해 방법군별 최대 하나의 대표 근거만
융합한다.

- 목표: 비학습 점수원으로 넓은 후보를 만들되 오탐 증가를 제한
- 출력: source별 p/e-value, 독립 방법군 수, 융합 강도
- 결과: 자동 경보가 아닌 research Shadow

### Exp313: 예측 잔차 모델

정상 TRAIN 앞 구간으로 뒤 구간을 예측하는 경량 temporal convolution 모델을 학습한다.
TEST에서 예측 오차가 TRAIN 정상 residual 분포를 초과하면 새로운 근거로 기록한다.

- 목표: 정상 순서 관계와 다음 구간 패턴 붕괴 탐지
- 실행: Apple MPS, 불가능할 때만 CPU fallback을 명시
- 입력: TRAIN 정상 데이터만
- 출력: point/segment residual score와 TRAIN calibration

### Exp314: 자기지도 표현 Memory Bank

TRAIN 정상 시계열의 서로 다른 augmentation을 같은 표현으로 모으는 contrastive 또는
masked-context encoder를 학습한다. TEST embedding과 TRAIN 정상 memory의 거리를 점수로
사용한다.

- 목표: ROCKET과 수작업 통계가 놓치는 복잡한 형태 탐지
- 실행: Apple MPS
- 출력: embedding distance, 이웃 근거, calibration score

### Exp315: Deep SVDD 정상 경계

TRAIN 정상 embedding을 작은 hypersphere 안에 모으고 중심으로부터의 거리를 점수로
사용한다. Exp314와 architecture를 공유할 경우에도 calibration과 결과 파일은 분리한다.

- 목표: KNN의 국소 거리만으로 표현하기 어려운 정상 영역의 전체 경계 학습
- 실행: Apple MPS
- 출력: center distance, TRAIN radius distribution, candidate score

### Exp316: 전체 점수원 융합

Exp309-315의 동결 결과를 결합한다. 비학습·학습 source를 방법군으로 나누고 각 그룹의
TRAIN false-alert risk, calibration resolution 및 provenance가 유효한 경우에만 사용한다.

- 후보 생성: 신규 source 중 독립 방법군 2개 이상이 지지
- 강한 연구 Shadow: 독립 방법군 3개 이상과 강한 정상 Guard 통과
- 근거 부족: 기존 일반 검토 또는 No Alert 유지
- 자동 경보: Exp291 그대로 유지

## 5. 강한 정상 증명 Guard

탐지 범위를 넓히면서 FP를 억제하기 위해 신규 후보가 다음 정상 근거에 해당하는지
별도로 확인한다.

- 원본 TRAIN 정상과 byte-exact 또는 수치 tolerance 내 동일
- TRAIN 정상 memory의 매우 가까운 이웃
- 여러 정상 표현 공간에서 동시에 낮은 이상도
- TRAIN bootstrap에서 후보가 불안정

정상 근거가 강하면 자동 경보 연구 Shadow에서 제외하고 검토 또는 정상 품질 기록으로
내린다. Guard의 threshold도 TRAIN 정상 데이터만으로 결정한다.

## 6. Prediction과 Evaluation 분리

Prediction 함수는 `x_train_normal`, `x_test`, 동결 config만 입력받는다. `y_test`, 실제
이상 위치, family별 성능은 입력받지 않는다. Prediction JSONL에는 TP, FP, F1 또는
정답 label을 저장하지 않는다.

Prediction 파일과 config hash를 동결한 뒤 별도 evaluation 단계에서만 TEST 라벨을
읽는다. 평가 결과가 prediction 파일을 변경해서는 안 된다.

## 7. 평가 지표

각 실험은 다음을 독립적으로 보고한다.

- 전체 candidate, TP, FP, precision
- 기존 자동 경보 밖에서 새로 발견한 TP
- 기존 모든 검토·복구 후보 밖에서 새로 발견한 TP
- 1,529개 outside-candidate FN 중 회수한 수와 비율
- zero-Hard-TP 데이터셋 356개 중 신규 TP가 생긴 데이터셋 수
- 기존 점수원과 exact 및 tolerance 후보 중복률
- test 크기 구간별 신규 TP/FP
- dataset별 candidate workload mean, median, p90, p95, max
- CPU/MPS 실행시간과 peak memory
- source별 TRAIN normal false-alert rate와 calibration resolution

정밀도 80%는 사용자가 제시한 운영 참고선으로 보고하지만, 동일 TEST에서 threshold나
variant를 선택하는 최적화 기준으로 사용하지 않는다.

## 8. 불변성 및 무결성 테스트

- TEST label을 변경해도 prediction hash가 동일하다.
- 이상 위치를 이동하거나 제거해도 prediction이 동일하다.
- dataset/family 이름을 바꿔도 데이터 내용이 같으면 정책 결과가 동일하다.
- 입력 행 순서가 달라도 dataset별 prediction이 동일하다.
- 동일 seed와 입력의 반복 실행 hash가 동일하다.
- TEST 뒤에 데이터를 추가했을 때 receptive-field 밖 기존 prefix score가 동일하다.
- prediction 단계가 label 파일을 열면 테스트가 실패한다.
- 모든 후보 index가 TEST 범위 안에 있다.
- 기존 Exp291 Hard index가 모든 실험에서 정확히 유지된다.
- 신규 Shadow, review, Hard lane이 상호 배타적이며 후보 합계가 일치한다.

## 9. 실행 순서와 중단 조건

실행은 Exp309부터 Exp316까지 한 번에 하나씩 순차 진행한다. 각 실험은 unit test,
10-dataset smoke, full prediction, prediction hash freeze, full evaluation 순서를 따른다.

다음 상황에서는 해당 실험과 의존 실험을 중단한다.

- TRAIN-only calibration을 구성할 수 없음
- prediction 단계에서 TEST label 접근 발견
- 기존 Exp291 Hard 후보가 변경됨
- 누락 dataset 또는 index 범위 오류 발생
- 동일 seed 결과가 재현되지 않음
- MPS 결과와 CPU reference smoke가 허용 오차 밖에서 다름
- 결과를 본 뒤 동결 상수를 변경해야만 목표 지표가 개선됨

실패한 source는 빈 후보로 간주하지 않고 blocker와 원인을 기록한다.

## 10. 대시보드와 산출물

각 실험은 `status.json`, log, prediction manifest, prediction JSONL, dataset result CSV,
summary CSV, `RESULT_REPORT.md`를 독립 폴더에 저장한다. 대시보드는 현재 dataset,
진행률, 처리 속도, ETA, 오류, 장치 CPU/MPS와 완료 결과를 표시한다.

Exp316 화면에서는 기존 자동 경보와 신규 research Shadow를 명확히 분리한다. Shadow를
자동 경보처럼 보이게 표시하지 않는다.

## 11. 해석 제한

Exp309-316은 이미 반복 사용한 1,117개 TEST 데이터에 대한 retrospective 연구다.
결과가 좋아도 end-to-end strict TRAIN-only 개발 완료, prospective validation 또는 실제
반도체 설비 성능을 증명하지 않는다. 운영 자동 경보 승격에는 동결된 새 데이터 또는
shadow-mode prospective 검증이 추가로 필요하다.
