# Exp169/170 Antigravity 1차 구현 요청

작업 저장소:

`/tmp/timeseries-exp167-168-parity`

현재 브랜치:

`experiment/exp169-170-oob-pooled-calibration`

당신의 역할은 Exp169/170의 1차 구현자다. 다음 두 문서를 먼저 전부 읽고 그 계약을 그대로 구현한다.

- `docs/superpowers/specs/2026-07-15-exp169-170-oob-pooled-calibration-design.md`
- `docs/superpowers/plans/2026-07-15-exp169-170-oob-pooled-calibration.md`

## 이번 작업 범위

구현 계획의 Task 1~4만 수행한다.

1. deterministic OOB aggregation core
2. target-excluded TRAIN-only donor pool core
3. OOB cache builder와 Exp169 label-free audit
4. Exp170 label-free prediction, frozen hash evaluation, independent reaggregation

다음 파일을 구현한다.

- `experiments/exp169_170/__init__.py`
- `experiments/exp169_170/oob_core.py`
- `experiments/exp169_170/pool_core.py`
- `experiments/exp169_170/run_experiment_169_170_oob_pooled_calibration.py`
- `experiments/exp169_170/independent_reaggregate_exp169_170.py`
- `experiments/exp169_170/test_exp169_170_oob_core.py`
- `experiments/exp169_170/test_exp169_170_pool_core.py`
- `experiments/exp169_170/test_exp169_170_structure.py`

## 반드시 지킬 계약

- DB: `/Users/minho/Documents/Dataset/univariate_ts.db`
- DATA_DIR: `/Users/minho/Documents/Dataset`
- 대상: `5 <= n_train < 20`, 정확히 316개
- OOB models: 32
- holdout fraction: 0.35
- seed: 20260715
- minimum pooled reference: 200
- donor contribution cap: 20
- alpha: `(0.005, 0.01, 0.02, 0.05)`
- pool policies: `similarity_pool`, `coarse_strata_pool`, `global_balanced_pool`
- sources: `spectral`, `shapelet`, `interval`
- source configurations: 세 source pair, `2of3`, `3of3`
- prediction rows: 316 x 3 x 4 x 5 = 18,960

Prediction 단계는 TEST values만 읽을 수 있고 다음을 읽으면 안 된다.

- TEST labels
- actual anomaly positions
- family name 기반 정책
- dataset name 기반 정책
- TEST length 기반 threshold 또는 budget
- 과거 TEST 성능

기존 Exp165~168 코드와 결과는 수정하지 않는다. tail replacement, top-N budget, Priority review, Hard alert를 추가하지 않는다.

## 구현 방식

- 기존 `experiments.exp164.crossfit_sources.score_fold`와 Exp167/168의 검증 패턴을 재사용한다.
- prediction 함수와 evaluation 함수를 분리한다.
- evaluation은 frozen prediction SHA256을 확인한 뒤에만 TEST label을 읽는다.
- target dataset은 모든 donor pool에서 제외한다.
- pool이 source별 200 reference를 충족하지 못하면 빈 후보로 처리하지 말고 실패한다.
- 결과를 보고 alpha나 pool을 선택하는 코드를 만들지 않는다.
- `selected_alpha`, `selected_pool_policy`, `selected_source_config`는 모두 `None` 또는 false 계약으로 유지한다.

## TDD와 검증

각 core는 실패 테스트를 먼저 작성한 뒤 최소 구현한다. 구현 후 다음을 실행한다.

```bash
python3 -m py_compile experiments/exp169_170/*.py
python3 -m unittest \
  experiments/exp169_170/test_exp169_170_oob_core.py \
  experiments/exp169_170/test_exp169_170_pool_core.py \
  experiments/exp169_170/test_exp169_170_structure.py
python3 -m unittest \
  experiments/exp167_168/test_exp167_168_parity_core.py \
  experiments/exp167_168/test_exp167_168_structure.py
git diff --check
```

10-dataset smoke는 코드와 테스트가 통과한 뒤 가능하면 실행한다. smoke에서 reference 200을 임의로 낮추려면 반드시 `--minimum-reference 20`처럼 명시적 인자를 사용하고 full config와 hash가 달라야 한다. 전체 316개 full experiment는 실행하지 않는다.

## 금지 사항

- git commit 금지
- git push 금지
- branch 변경 금지
- 기존 파일 삭제 금지
- TEST 결과를 보고 상수 변경 금지
- 관련 없는 refactor 금지

완료 보고에는 다음을 포함한다.

1. 생성·수정 파일 목록
2. 구현한 함수와 데이터 흐름
3. 실행한 테스트와 정확한 결과
4. smoke 실행 여부와 결과
5. 발견한 blocker 또는 불확실성
6. 아직 구현하지 않은 계획 항목
