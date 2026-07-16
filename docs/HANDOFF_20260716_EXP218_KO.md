# 다른 PC Codex 인수인계: Exp218까지

## 목적과 현재 상태

이 프로젝트는 정상 TRAIN 시계열만으로 이상 후보를 생성하고, **오탐(False Positive)을 낮춰 사용자가 신뢰할 수 있는 경보를 제공하는 것**이 목적이다. 단일 mean F1 최대화보다 자동 경보의 precision, 검토량, 그리고 놓친 이상을 분리해서 본다.

현재 작업 브랜치는 `experiment/exp177-185-batch`이다. 새 PC에서는 아래 명령으로 현재 커밋을 확인한다.

```bash
git rev-parse HEAD
git status --short --branch
```

최근 완료 실험은 `experiment_218_true_bootstrap_consensus_normal_prototypes`이며, 2026-07-16에 251개 대상에서 완료됐다. 이는 **운영 후보가 아니라 연구용 사후 후보 평가**다. Exp137의 자동 경보 기준을 바꾸거나 승격하지 않는다.

## 반드시 유지할 원칙

- feature, score, calibration, threshold, 후보 규칙은 TRAIN 정상 데이터만으로 결정한다.
- prediction 단계에서는 TEST label, 실제 이상 위치, TEST 길이, 과거 family별 TEST 성능을 사용하지 않는다.
- `mean_combined_f1`은 사람이 review lane을 확인했다고 가정한 human-assisted 지표다. autonomous Hard alert와 섞지 않는다.
- Priority review는 자동 경보가 아니라 검토 전용이다.
- tail replacement는 재도입하지 않는다.
- 새 결과가 좋지 않아도 TEST 지표를 보고 alpha, threshold, source, budget을 재선택하지 않는다. 새 가설은 새 실험 번호와 사전등록으로 분리한다.

## 데이터베이스 복사

현재 코드의 DB 경로는 아래 절대 경로다. 새 PC에도 **동일한 경로**로 DB를 복사하는 것이 가장 안전하다.

```bash
mkdir -p "/Users/minho/Documents/Dataset"
cp /source/location/univariate_ts.db "/Users/minho/Documents/Dataset/univariate_ts.db"
shasum -a 256 "/Users/minho/Documents/Dataset/univariate_ts.db"
```

기대 SHA256:

```text
8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c
```

DB 크기는 약 1.6 GB다. DB는 GitHub에 올리지 않는다.

또한 기존 결과를 대시보드에서 그대로 보거나 cache를 재사용하려면 아래 결과 디렉터리도 함께 복사한다. 없더라도 실험을 새로 실행할 수는 있으나, 과거 결과와 cache는 즉시 보이지 않는다.

```text
/Users/minho/Documents/Dataset/exp211_214_calibration_transport/
/Users/minho/Documents/Dataset/exp215_217_shapelet_stability/
```

## 새 PC 초기화

```bash
git clone https://github.com/pmh10401/Dataset.git "~/Documents/timesries project"
cd "~/Documents/timesries project"
git checkout experiment/exp177-185-batch

python3 -m unittest \
  experiments/exp211_214/test_exp211_214_core.py \
  experiments/exp211_214/test_exp211_214_structure.py \
  experiments/exp215_217/test_exp215_217_core.py \
  experiments/exp215_217/test_exp215_217_structure.py
```

대시보드는 다음처럼 실행한다.

```bash
python3 serve_rank_dashboard.py
```

브라우저 주소는 `http://127.0.0.1:8765/`이다. 큐 상태는 아래 명령으로도 확인한다.

```bash
python3 run_rank_experiments_sequential.py status
```

## 최근 연구 결과

### Exp215: 고정 분석 사전(dictionary) 대조

원래 Shapelet 점수원의 fold별 불안정성은 랜덤 dictionary가 매 fold마다 달라지는 영향이 컸다. 고정 분석 사전을 쓰면 251개 중 241개가 transport-stable이고 median rank correlation은 `0.996314`였다. 이 실험은 score 재현성 문제를 진단한 것이며, 운영 승격 결론은 아니다.

### Exp216: 중앙 정상 prototype

정상 TRAIN에서 중앙 medoid prototype을 구성해 보았지만, bootstrap consensus는 아니었다. 후보 union이 F1과 false alarm에 불리해 운영 적용하지 않았다.

### Exp217: 정상 pseudo-run 위험 점검

정상 TRAIN pseudo-run 기반 false-alert 위험을 별도 진단했다. 후보 자체의 성능과 혼동하지 않는다.

### Exp218: 진짜 bootstrap consensus normal prototype

16회 bootstrap dictionary에서 60% 이상 반복된 prototype을 합의로 남기고, signature distance `2.5`로 중복을 제거했다. transport 안정은 `48/251`, median rank correlation은 `0.305991`이었다. 연구 후보 precision은 alpha `0.005 / 0.01 / 0.02 / 0.05`에서 각각 약 `18.8% / 10.9% / 6.4% / 3.5%`로, 현재 운영 경보에 채택할 수준이 아니다.

최근 결과 원본은 다음에 있다.

```text
/Users/minho/Documents/Dataset/exp215_217_shapelet_stability/
  experiment_218_true_bootstrap_consensus_normal_prototypes_summary.csv
  experiment_218_true_bootstrap_consensus_normal_prototypes_transport_summary.csv
  experiment_218_true_bootstrap_consensus_normal_prototypes_train_normal_risk_summary.csv
  exp218_completion.json
```

## 다음 작업을 시작하기 전 점검

1. DB hash가 위 값과 일치하는지 확인한다.
2. `git status --short --branch`로 코드가 어떤 커밋인지 기록한다.
3. 기존 Exp218은 후보 연구 결과일 뿐이라는 점을 유지한다.
4. 정상 TRAIN-only 입력 계약을 단위 테스트와 코드 경로로 확인한 뒤에만 새 실험을 설계한다.
5. 결과를 보고 정책을 바꾸지 말고, 가설과 상수를 먼저 문서화한다.

## 참고 문서와 코드

- `docs/superpowers/plans/2026-07-16-exp211-214-calibration-transport.md`
- `docs/superpowers/plans/2026-07-16-exp215-217-shapelet-stability.md`
- `outputs/exp211_214_validation/RESULT_ANALYSIS.md`
- `experiments/exp211_214/`
- `experiments/exp215_217/`
- `run_rank_experiments_sequential.py`
- `serve_rank_dashboard.py`
