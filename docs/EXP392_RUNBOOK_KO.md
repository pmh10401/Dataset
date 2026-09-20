# Exp392 실행 안내

## 출력 위치

`/Users/minho/Documents/Dataset/exp392_label_guided_score_discovery_audit`

기존 Exp391 및 다른 결과를 덮어쓰지 않는다.

## 단계별 실행

라벨 없는 후보 행렬 생성:

```bash
python3 experiments/exp392_label_guided_discovery/run_experiment_392_label_guided_score_discovery_audit.py \
  --phase prepare
```

동결 해시 검증 후 후향 평가:

```bash
python3 experiments/exp392_label_guided_discovery/run_experiment_392_label_guided_score_discovery_audit.py \
  --phase evaluate
```

두 단계를 연속 실행:

```bash
python3 experiments/exp392_label_guided_discovery/run_experiment_392_label_guided_score_discovery_audit.py
```

순차 큐 실행:

```bash
python3 run_rank_experiments_sequential.py queue \
  experiment_392_label_guided_score_discovery_audit
python3 run_rank_experiments_sequential.py run-next
```

## 검증

```bash
python3 -m py_compile experiments/exp392_label_guided_discovery/*.py
python3 -m unittest \
  experiments.exp392_label_guided_discovery.test_exp392_label_guided_discovery \
  experiments.exp391_fn_reachability.test_exp391_fn_reachability
git diff --check
```

## 완료 조건

- prepare 행 1,117개, 라벨·평가 지표 없음
- evaluate 데이터셋 1,117개
- 단일 근거 54개
- 조합 1,944개
- 상보성 216개
- 후보 인덱스 범위 오류 0
- 기존 lane 변경 0
- 선택된 운영 규칙 0
- status `completed`, errors 0

## 해석 주의

결과 보고서의 `preregistration_candidate`도 자동 경보 후보가 아니다.
별도 동결 실험을 만들 가치가 있는 연구 신호라는 뜻이다.
