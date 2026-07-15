# Exp153–157 TRAIN-only Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Exp152의 높은 precision을 유지하면서 TRAIN-normal calibration validity, 소표본 pooling, source fusion, confirmation calibration, risk-controlled alpha 정책을 원인별로 검증한다.

**Architecture:** Exp151/152의 label-free source scoring을 공유 cache로 한 번 계산한다. Exp153은 TRAIN pseudo-normal audit을 수행하고, Exp154–156은 각각 한 요인만 변경한 prediction을 동결한 뒤 별도 평가한다. Exp157은 Exp153 TRAIN audit만으로 alpha를 선택한다.

**Tech Stack:** Python 3, NumPy, scikit-learn, SQLite, 기존 Exp151/152 연구 모듈

## Global Constraints

- DB path: `/Users/minho/Documents/Dataset/univariate_ts.db`
- Alpha grid: `0.005, 0.01, 0.02, 0.05`
- TEST label은 frozen prediction 이후 평가에만 사용
- Hard autonomous / Review human-assisted 분리
- TEST 성능으로 variant 또는 정책 선택 금지
- 모든 결과는 retrospective counterfactual

---

### Task 1: Shared label-free score cache

**Files:**
- `experiments/exp153_157/exp153_157_bridge.py`
- `experiments/exp153_157/build_exp153_157_score_cache.py`

- [ ] Exp152 B2 contract 1,117 coverage를 검증한다.
- [ ] 네 source OOF/TEST score를 저장한다.
- [ ] Block-B/C OOF/TEST score와 legacy index를 저장한다.
- [ ] TEST label이 cache에 없는지 검사한다.
- [ ] cache SHA256 manifest를 생성한다.

### Task 2: Exp153 calibration validity audit

**File:** `run_experiment_153_train_normal_calibration_audit.py`

- [ ] OOF score 자기 자신을 reference에서 제거한다.
- [ ] pseudo-normal Hard false alerts를 집계한다.
- [ ] normal run 100개당 alert/review를 보고한다.
- [ ] TEST values/labels를 읽지 않았음을 기록한다.

### Task 3: Exp154 pooled calibration

**File:** `run_experiment_154_pooled_conformal.py`

- [ ] local/global/group/hybrid 네 variant를 prediction한다.
- [ ] target dataset을 pool reference에서 제외한다.
- [ ] grouping이 TRAIN count와 TRAIN length만 사용하는지 검사한다.
- [ ] prediction을 동결한 뒤 별도 평가한다.

### Task 4: Exp155 source fusion

**File:** `run_experiment_155_source_fusion.py`

- [ ] TRAIN OOF Spearman correlation audit을 생성한다.
- [ ] discrete/top2/correlation-weighted variant를 모두 prediction한다.
- [ ] TEST 결과로 fusion method를 선택하지 않는다.

### Task 5: Exp156 confirmation calibration

**File:** `run_experiment_156_confirmation_calibration.py`

- [ ] frozen Exp152 candidate universe를 검증한다.
- [ ] legacy/strict/balanced confirmation을 적용한다.
- [ ] Block-C는 review-only로 유지한다.

### Task 6: Exp157 TRAIN-only risk policy

**File:** `run_experiment_157_risk_controlled_alpha_policy.py`

- [ ] Exp153 audit만으로 Wilson upper bound를 계산한다.
- [ ] 세 risk target을 모두 sensitivity로 보고한다.
- [ ] 만족 alpha가 없으면 abstain한다.
- [ ] TEST label로 alpha를 선택하지 않는다.

### Task 7: Verification and handoff

**Files:**
- 두 test 파일
- `docs/EXP153_157_RUNBOOK_KO.md`
- `prompts/CODEX_EXP153_157_EXECUTION_PROMPT_KO.md`

- [ ] py_compile 실행
- [ ] unit/structure test 실행
- [ ] 10-dataset smoke 실행
- [ ] full coverage와 독립 재집계 확인
- [ ] MANIFEST와 presentation claims 작성
