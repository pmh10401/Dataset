# 다른 PC Codex 재개 프롬프트

아래 내용을 새 PC의 Codex에 전달한다.

```text
저장소: ~/Documents/timesries project
DB: /Users/minho/Documents/Dataset/univariate_ts.db

먼저 docs/HANDOFF_20260716_EXP218_KO.md를 읽고, 아래 검증부터 실행해줘.

1. git status --short --branch
2. git rev-parse HEAD
3. shasum -a 256 /Users/minho/Documents/Dataset/univariate_ts.db
4. python3 -m unittest \
     experiments/exp211_214/test_exp211_214_core.py \
     experiments/exp211_214/test_exp211_214_structure.py \
     experiments/exp215_217/test_exp215_217_core.py \
     experiments/exp215_217/test_exp215_217_structure.py
5. python3 run_rank_experiments_sequential.py status

목표는 정상 TRAIN 데이터만으로 후보를 만들고 false alarm을 낮춰 사용자 신뢰를 높이는 것이다. 단일 mean F1 최대화가 목표가 아니다.

절대 지켜야 할 원칙:
- prediction에는 TEST label, 실제 이상 위치, TEST 길이, 과거 family별 TEST 성능을 사용하지 않는다.
- mean_combined_f1은 human-assisted 지표로만 다룬다.
- Priority review는 자동 경보로 승격하지 않는다.
- tail replacement를 사용하지 않는다.
- 결과를 본 뒤 alpha, threshold, source, budget을 선택하거나 바꾸지 않는다.

Exp218은 완료됐지만 운영 채택 실험이 아니라 bootstrap consensus prototype의 연구 후보 검증이다. 가장 최신 결과는 /Users/minho/Documents/Dataset/exp215_217_shapelet_stability/에 있다. 먼저 결과 파일과 completion JSON을 읽고, 그 다음 새 가설을 제안하되 사전등록 및 TRAIN-only 코드 감사를 먼저 해줘.
```
