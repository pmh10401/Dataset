# Project Research Context Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 현재 데이터와 실험 근거, 동결 운영 기준, 주장 한계, 신규 실데이터가 없는 조건에서의 후속 연구 방향을 한 문서에서 정확하게 설명하는 공식 프로젝트 입구를 만든다.

**Architecture:** 기존 문서는 각 시점의 역사적 근거로 보존하고 새 문서 `docs/PROJECT_RESEARCH_CONTEXT_KO.md`만 추가한다. 문서의 앞부분은 문제·데이터·연구 연혁·동결 기준선을 설명하고, 뒷부분은 Exp403-409의 의미·확정된 교훈·주장 경계·후속 연구 규칙·근거 지도를 제공한다. 수치와 해석은 동결 결과/감사 보고서, 실험 계약, handoff, 과거 종합 문서 순으로 교차 확인한다.

**Tech Stack:** Markdown, Git, `rg`, POSIX shell 검증 명령

## Global Constraints

- 기준일은 `2026-07-28`이다.
- 필수 산출물은 `docs/PROJECT_RESEARCH_CONTEXT_KO.md` 하나다.
- 기존 코드, 실험 설정, 동결 결과, `README.md`는 수정하지 않는다.
- 현재 SQLite 데이터는 반도체 설비 원본이 아니라 UCR 기반 연구용 벤치마크임을 명시한다.
- 신규 반도체 실데이터의 추가 확보 및 외부 반출은 불가능하다는 확정 조건을 반영한다.
- 프로젝트 목표는 단일 mean F1 최대화가 아니라 정상 TRAIN만으로 후보를 만들고 false alarm을 억제하여 사용자 신뢰를 높이는 것이다.
- prediction에서 TEST label, 실제 이상 위치, 과거 family별 TEST 성능을 사용하지 않는다.
- `mean_combined_f1`은 human-assisted 지표로만 다룬다.
- Priority review와 Shadow 후보를 자동 경보로 승격하지 않는다.
- tail replacement를 사용하지 않는다.
- TEST 결과를 보고 alpha, threshold, source, budget을 재선택하지 않는다.
- 새 개선은 새 실험 번호와 사전등록으로 분리한다.
- 현재 동결 lane은 `Hard 2,218 / Standard 806 / Priority 31`이다.
- Exp406-408 새 근거와 Exp408 엄격 변형 후보는 review-only로만 해석한다.
- Exp409는 준비성·거버넌스 검증이며 prospective 성능 근거가 아니다.
- 누락된 `docs/EXP403_408_PREREGISTRATION_KO.md`를 생성하거나 존재하는 문서처럼 인용하지 않는다.
- 승인된 설계는 `docs/superpowers/specs/2026-07-28-project-research-context-design.md`다.

---

## File Map

### 생성 파일

- `docs/PROJECT_RESEARCH_CONTEXT_KO.md`: 현재 프로젝트의 공식 연구 맥락, 동결 기준선, 주장 경계, 후속 연구 방향, 근거 문서 지도를 제공한다.

### 수정하지 않는 핵심 근거

- `README.md`: 초기 947개 데이터셋·VAE 중심 단계의 역사적 기록
- `docs/HANDOFF_20260728_EXP403_408_KO.md`: 복원 상태와 Exp403-408 핵심 동결 사실
- `docs/task.md`: 프로젝트 목표와 평가 원칙
- `docs/anomaly_validation_difficulty_report_20260707.md`: 데이터셋별 평가 난이도와 취약 모집단
- `docs/experiment_consistency_audit_20260707.md`: TRAIN/TEST 중복, family 반복, 메타데이터 일관성 감사
- `docs/experiment_39_50_results_analysis_20260707.md`: 초기 실험 결과의 한계와 false alarm 문제
- `docs/model_hard_response_research_20260707.md`: model-hard 대응 연구와 표현 개선 필요성
- `docs/EXP403_408_RUNBOOK_KO.md`: Exp403-408 실행 계약
- `docs/EXP403_408_CHECKPOINT_PROVENANCE_AUDIT_KO.md`: checkpoint 직접 재사용 불가 근거
- `docs/superpowers/specs/2026-07-27-exp403-408-feature-ml-validation-design.md`: Exp403-408 설계 목적
- `.superpowers/sdd/2026-07-27-exp403-408-feature-ml-validation/task-9-report.md`: Exp405 결과
- `.superpowers/sdd/2026-07-27-exp403-408-feature-ml-validation/task-10-report.md`: Exp406 결과
- `.superpowers/sdd/2026-07-27-exp403-408-feature-ml-validation/task-12-report.md`: Exp408 결과
- `.superpowers/sdd/2026-07-27-exp403-408-feature-ml-validation/task-13-report.md`: Exp403-408 최종 검증
- `docs/EXP409_PREREGISTRATION_KO.md`: Exp409 사전등록 계약
- `docs/EXP409_RUNBOOK_KO.md`: Exp409 실행·검증 절차
- `docs/superpowers/specs/2026-07-28-exp409-prospective-readiness-design.md`: Exp409 claim boundary와 fail-closed 설계

### 현재 Mac에만 있는 외부 동결 근거

- `/Users/mac/Documents/Dataset/exp403_408_feature_ml_validation/EXP403_408_RESULT_REPORT_KO.md`: Exp403-408 전체 결과 및 동결 lane 수치
- `/Users/mac/Documents/Dataset/exp409_prospective_readiness_smoke/EXP409_READINESS_REPORT_KO.md`: Exp409 representative smoke 준비성 결과

---

### Task 1: 프로젝트 현실과 현재 기준선 작성

**Files:**
- Create: `docs/PROJECT_RESEARCH_CONTEXT_KO.md`
- Reference: `docs/superpowers/specs/2026-07-28-project-research-context-design.md`
- Reference: `README.md`
- Reference: `docs/task.md`
- Reference: `docs/anomaly_validation_difficulty_report_20260707.md`
- Reference: `docs/experiment_consistency_audit_20260707.md`
- Reference: `docs/experiment_39_50_results_analysis_20260707.md`
- Reference: `docs/model_hard_response_research_20260707.md`
- Reference: `/Users/mac/Documents/Dataset/exp403_408_feature_ml_validation/EXP403_408_RESULT_REPORT_KO.md`

**Interfaces:**
- Consumes: 승인된 설계 명세와 위 감사·결과 문서에 기록된 사실
- Produces: 기준일, 문서 지위, 문제 정의, 데이터 현실, 데이터 위험, 연구 연혁, 동결 기준선을 담은 섹션 1-5

- [ ] **Step 1: 작성 전 작업 범위를 고정한다**

Run:

```bash
git status --short --branch
test ! -e docs/PROJECT_RESEARCH_CONTEXT_KO.md
test -f docs/superpowers/specs/2026-07-28-project-research-context-design.md
```

Expected:

- 브랜치는 `codex/exp409-prospective-readiness`다.
- 계획 문서 커밋 이후 worktree에는 미커밋 변경이 없다.
- 대상 통합 문서는 아직 없고 승인된 설계 문서는 존재한다.

- [ ] **Step 2: 핵심 수치와 데이터 제약을 원문에서 다시 확인한다**

Run:

```bash
rg -n "1,117|2,218|806|31|1,915|303|86\\.339|0\\.608983" \
  docs/HANDOFF_20260728_EXP403_408_KO.md \
  /Users/mac/Documents/Dataset/exp403_408_feature_ml_validation/EXP403_408_RESULT_REPORT_KO.md
rg -n "788|478|460|81\\.2|218|789|110|153" \
  docs/anomaly_validation_difficulty_report_20260707.md \
  docs/experiment_consistency_audit_20260707.md
```

Expected:

- 현재 모집단 `1,117`과 lane `2,218 / 806 / 31`의 근거가 확인된다.
- Hard 결과 `TP 1,915 / FP 303 / precision 86.339% / mean F1 0.608983`이 확인된다.
- 작은 평가 표본, 작은 TRAIN, 정상 중복, family 중복, 가변 길이 관련 위험 수치의 원문 위치가 확인된다.
- 어떤 수치가 검색되지 않으면 기억으로 보충하지 않고 해당 원문을 다시 읽어 실제 표기를 사용한다.

- [ ] **Step 3: 문서 머리말과 섹션 1-2를 작성한다**

Create `docs/PROJECT_RESEARCH_CONTEXT_KO.md` with:

```markdown
# 현재 프로젝트 연구 맥락 및 의사결정 기준

기준일: 2026-07-28

> 이 문서는 현재 프로젝트를 이해하기 위한 공식 입구다. 과거 실험 문서는
> 각 시점의 계획과 결과를 보존하는 역사적 근거이며, 이 문서는 이를 소급해
> 변경하지 않는다.

## 1. 프로젝트가 해결하려는 문제

이 프로젝트의 목표는 하나의 평균 F1 수치를 최대화하는 것이 아니다. 정상
TRAIN 데이터만으로 정상 동작의 기준과 후보 생성 규칙을 만들고, 정답을
모르는 TEST 입력에 그 규칙을 적용하면서 false alarm을 억제하는 것이
핵심이다. 사용자가 반복되는 오경보 때문에 경보를 무시하기 시작하면 실제
이상을 찾아도 시스템의 목적을 달성했다고 보기 어렵기 때문이다.

예를 들어 새 방법이 실제 이상 한 건을 더 찾으면서 정상 샘플 열 건을
추가로 경보한다면 평균 점수 일부는 좋아질 수 있어도 현장 사용자의 신뢰는
낮아질 수 있다. 따라서 후보를 넓히는 것뿐 아니라 자동 경보, 사람 검토
후보, 연구용 Shadow 근거를 분리하고 각각의 false alarm 비용을 관리해야
한다.

## 2. 데이터의 실제 성격과 영구 제약

현재 SQLite 데이터는 반도체 설비에서 반출한 원본 신호가 아니다. 주된
구성은 UCR Time Series Archive를 class별 one-vs-all 문제로 변환한 연구용
벤치마크이며 CornellWhaleChallenge 계열도 포함되어 왔다. 현재 연구
모집단은 1,117개 데이터셋으로 동결되어 있다. 이 데이터는 알고리즘을
비교하고 실패 양상을 연구하는 데 사용할 수 있지만, 데이터 출처 자체가
반도체 설비 현장을 대표하지는 않는다.

반도체 설비는 이 연구의 최종 적용 목표다. 그러나 보안과 외부 반출 제한
때문에 신규 반도체 실데이터를 추가로 확보할 수 없으며, 앞으로도 새
실데이터가 들어오지 않는다는 조건을 연구 계획의 고정 전제로 둔다.

따라서 현재 데이터에서 얻은 결과는 retrospective benchmark 근거로만
해석한다. 이를 반도체 현장 성능, 실제 배포 검증 또는 새 데이터에 대한
prospective 성능이라고 부르지 않는다. 후속 연구는 현재 데이터 안에서
family 중복, TRAIN/TEST 중복, 설정 선택과 최종 평가의 혼입을 최대한 줄이는
방향으로 설계한다.
```

위 문안을 사용하고, 원문 교차 확인에서 고유명사의 공식 표기만 다를 경우
그 표기만 근거 문서와 일치시킨다. `SOTA`, `배포 검증 완료`,
`prospective 성능`이라는 근거 없는 주장은 사용하지 않는다.

- [ ] **Step 4: 섹션 3의 데이터 위험을 독자가 이해할 수 있게 작성한다**

`## 3. 데이터셋과 평가의 주요 위험` 아래에 다음 표를 작성한다.

| 위험 | 확인된 현상 | 해석 시 주의점 |
|---|---|---|
| 작은 TEST와 단일 이상 | 788개 데이터셋은 이상 1개이고 TEST 크기 50 이하 | 한 샘플의 성공·실패가 F1을 크게 바꿀 수 있음 |
| 작은 TRAIN | 478개 데이터셋은 TRAIN 20 미만 | calibration과 conformal 해상도가 제한됨 |
| 정상 TRAIN/TEST 중복 | 460개 데이터셋은 정상 TEST의 90% 이상이 TRAIN 정상과 정확히 겹침 | 일반화 성능이 실제보다 좋아 보일 수 있음 |
| 전체 정상 중복 | 감사 기준 정상 TEST 인스턴스 약 81.2%가 TRAIN 정상과 정확히 겹침 | 무작위 dataset 평균만으로 독립성을 주장할 수 없음 |
| family 반복 | one-vs-all 구성으로 동일 원자료 계열이 여러 데이터셋으로 반복됨 | family를 가로지르는 누수와 과대대표를 막아야 함 |
| label 의미 | `labels_blob`은 전체 series의 instance label | 실제 이상 시점 위치로 해석하면 안 됨 |
| 가변 길이 | 153개 데이터셋에서 길이 관련 메타데이터 불일치가 보고됨 | 고정 길이 가정을 적용하기 전 원자료 확인 필요 |
| 난이도 구분 | model-hard 218, evaluation-hard 789, 상대적으로 신뢰 가능한 집합 110 | 모델 실패와 평가 불안정을 같은 문제로 취급하지 않음 |

표 아래에서 각 수치의 출처 문서를 저장소 상대 경로로 연결한다. 동일 위험을
과도하게 반복하지 말고, “데이터를 사용할 수 없다”가 아니라 “주장 범위를
좁혀야 한다”는 결론으로 마무리한다.

- [ ] **Step 5: 섹션 4의 연구 발전 과정을 작성한다**

`## 4. 연구가 현재 구조에 도달한 과정`에 다음 여섯 단계를 순서대로
설명한다.

1. 초기 947개 데이터셋·VAE 중심 탐색
2. 데이터·평가 일관성 감사로 기존 평균 성능 해석 범위 축소
3. 전역 replacement와 후보 union에서 false positive 급증 확인
4. family별 표현 신호 발견과 동시에 TEST 성능 기반 routing 누수 금지
5. 자동 Hard와 사람 검토 Standard/Priority를 분리한 보수적 구조 동결
6. Exp403-408 feature/ML 검증과 Exp409 준비성·거버넌스 검증

초기 결과는 “틀렸다”가 아니라 당시 모집단과 평가 계약에서 얻은 역사적
결과라고 표현한다. 현재 기준선보다 오래된 결과를 현재 운영 성능처럼
병렬 비교하지 않는다.

- [ ] **Step 6: 섹션 5의 현재 동결 기준선을 작성한다**

`## 5. 현재 동결 기준선`에 다음 표를 포함한다.

| Lane | 개수 | 현재 의미 | 자동 승격 여부 |
|---|---:|---|---|
| Hard | 2,218 | 현재 동결된 자동 경보 기준선 | 기준선 자체 |
| Standard | 806 | 사람이 검토하는 보조 후보 | 자동 승격 금지 |
| Priority | 31 | 더 우선적으로 사람이 검토하는 후보 | 자동 승격 금지 |

표 아래에 Hard의 retrospective benchmark 결과를 별도 상자로 기록한다.

```text
TP 1,915 / FP 303 / precision 86.339% / mean F1 0.608983
```

이 수치를 반도체 현장 성능, prospective 성능, human-assisted 결과로
확장하지 않는다는 문장을 바로 뒤에 둔다.

- [ ] **Step 7: 섹션 1-5의 구조와 필수 문구를 검증한다**

Run:

```bash
rg -n "^## [1-5]\\." docs/PROJECT_RESEARCH_CONTEXT_KO.md
rg -n "UCR|반도체|신규 실데이터|false alarm|Hard|Standard|Priority|2,218|806|31|1,915|303|86\\.339|0\\.608983|788|478|460|81\\.2|218|789|110|153" \
  docs/PROJECT_RESEARCH_CONTEXT_KO.md
rg -n "SOTA|배포 검증 완료|prospective 성능을 검증" \
  docs/PROJECT_RESEARCH_CONTEXT_KO.md
```

Expected:

- 섹션 1-5가 순서대로 한 번씩 존재한다.
- 필수 개념과 수치가 모두 검색된다.
- 마지막 금지 표현 검색은 결과가 없거나, 해당 표현을 부정·경계하는
  문장에서만 나타난다.

- [ ] **Step 8: 첫 번째 문서 단위를 커밋한다**

Run:

```bash
git diff --check
git add docs/PROJECT_RESEARCH_CONTEXT_KO.md
git diff --cached --check
git commit -m "docs: establish current project research context"
```

Expected:

- whitespace 오류가 없다.
- 새 통합 문서의 섹션 1-5만 커밋된다.

---

### Task 2: 실험 해석·주장 경계·후속 방향·근거 지도 완성

**Files:**
- Modify: `docs/PROJECT_RESEARCH_CONTEXT_KO.md`
- Reference: `docs/HANDOFF_20260728_EXP403_408_KO.md`
- Reference: `docs/EXP403_408_RUNBOOK_KO.md`
- Reference: `docs/EXP403_408_CHECKPOINT_PROVENANCE_AUDIT_KO.md`
- Reference: `docs/superpowers/specs/2026-07-27-exp403-408-feature-ml-validation-design.md`
- Reference: `.superpowers/sdd/2026-07-27-exp403-408-feature-ml-validation/task-9-report.md`
- Reference: `.superpowers/sdd/2026-07-27-exp403-408-feature-ml-validation/task-10-report.md`
- Reference: `.superpowers/sdd/2026-07-27-exp403-408-feature-ml-validation/task-12-report.md`
- Reference: `.superpowers/sdd/2026-07-27-exp403-408-feature-ml-validation/task-13-report.md`
- Reference: `docs/EXP409_PREREGISTRATION_KO.md`
- Reference: `docs/EXP409_RUNBOOK_KO.md`
- Reference: `docs/superpowers/specs/2026-07-28-exp409-prospective-readiness-design.md`
- Reference: `/Users/mac/Documents/Dataset/exp403_408_feature_ml_validation/EXP403_408_RESULT_REPORT_KO.md`
- Reference: `/Users/mac/Documents/Dataset/exp409_prospective_readiness_smoke/EXP409_READINESS_REPORT_KO.md`

**Interfaces:**
- Consumes: Task 1의 섹션 1-5와 Exp403-409 설계·실행·결과 근거
- Produces: Exp403-409 해석, 확정 교훈, metric claim boundary, 현 데이터 기반 연구 방향, 실험 원칙, 문서 지도를 포함한 완성 문서

- [ ] **Step 1: Exp403-409 핵심 결과를 원문에서 다시 확인한다**

Run:

```bash
rg -n "473|395|78|62|29|33|46\\.77|two_new_plus_existing_independent|0\\.005|review-only" \
  .superpowers/sdd/2026-07-27-exp403-408-feature-ml-validation/task-9-report.md \
  .superpowers/sdd/2026-07-27-exp403-408-feature-ml-validation/task-12-report.md \
  /Users/mac/Documents/Dataset/exp403_408_feature_ml_validation/EXP403_408_RESULT_REPORT_KO.md
rg -n "not_evaluated|prospective_validated|deployment_eligible|operational_policy_changed|budget|lane" \
  docs/superpowers/specs/2026-07-28-exp409-prospective-readiness-design.md \
  /Users/mac/Documents/Dataset/exp409_prospective_readiness_smoke/EXP409_READINESS_REPORT_KO.md
```

Expected:

- Exp405 recheck `473`, TP `395`, FP `78`과 demotion gate 실패가 확인된다.
- Exp408 엄격 변형의 후보 `62`, TP `29`, FP `33`, precision `46.77%`,
  variant, alpha가 확인된다.
- Exp409가 `not_evaluated`, `prospective_validated: false`,
  `deployment_eligible: false`, `operational_policy_changed: false`임이
  확인된다.

- [ ] **Step 2: 섹션 6에 Exp403-409의 의미를 작성한다**

`## 6. Exp403-409가 제공한 증거`에 다음 표를 추가한다.

| 실험 | 질문 | 현재 결론 |
|---|---|---|
| Exp403 | 9개 FeatureBlock의 안정성 | TRAIN-only feature 근거의 안정성과 한계를 확인 |
| Exp404 | confidence committee | 보조 confidence 근거를 검토하되 자동 승격 근거로 사용하지 않음 |
| Exp405 | 기존 Hard의 demotion 가능성 | 473개 재검토 중 TP 395, FP 78로 gate 실패; 안전한 자동 demotion 근거가 아님 |
| Exp406 | tri-view self-supervised shadow | 새 표현 근거이나 review-only |
| Exp407 | synthetic anomaly shadow | 합성 기반 보조 근거이나 review-only |
| Exp408 | 독립 근거 consensus | 엄격 변형도 62개 중 TP 29, FP 33; 자동 경보 승격에 부족 |
| Exp409 | label-blind 준비성·거버넌스 | smoke에서 실행 계약과 provenance를 확인했지만 성능은 평가하지 않음 |

Exp408 행 아래에 variant `two_new_plus_existing_independent`, alpha `0.005`,
retrospective precision `46.77%`를 보충한다. 후보의 사후 정답 평가는
연구 분석일 뿐 prediction 입력이 아니며, 그 결과를 보고 설정을 다시
선택하지 않는다는 경계를 명시한다.

- [ ] **Step 3: 섹션 7에 확정된 교훈을 작성한다**

`## 7. 반복 실험에서 확정된 교훈`에 다음 항목을 각각 원인과 운영적 의미를
포함한 짧은 문단으로 작성한다.

- 넓은 후보 확장은 TP보다 FP를 빠르게 늘리는 경향
- model-hard는 단순 threshold보다 새로운 score representation이 필요
- 작은 TRAIN에서 conformal resolution이 제한됨
- review workload도 제한된 운영 자원
- historical family TEST performance를 사용한 routing은 leakage
- tail replacement는 합성 TEST 배열 구조를 이용하므로 금지
- provenance 미확인 checkpoint는 직접 재사용하지 않음

`docs/EXP403_408_CHECKPOINT_PROVENANCE_AUDIT_KO.md`의 결론을 사용해
“checkpoint 파일이 존재한다”와 “입력 계보가 재현되어 안전하게 재개할 수
있다”를 구분한다.

- [ ] **Step 4: 섹션 8에 주장 경계를 작성한다**

`## 8. 성능 주장과 해석의 경계`에 다음 표를 포함한다.

| 구분 | 의미 | 현재 허용되는 표현 |
|---|---|---|
| Autonomous | 정답을 본 사람의 개입 없이 생성·실행되는 경보 | 동결 Hard의 retrospective 자동 결과 |
| Retrospective | 현재 동결 TEST의 정답으로 사후 평가 | 벤치마크 내 비교·진단 |
| Human-assisted | review 후보를 사람이 완벽히 판별한다고 가정한 상한·보조 지표 | `mean_combined_f1` 등 명시적으로 사람 개입을 가정한 결과 |

표 아래에 다음 네 문장을 의미 변경 없이 포함한다.

1. `mean_combined_f1`은 autonomous 성능이 아니다.
2. Priority와 Shadow 후보는 자동 경보로 승격하지 않는다.
3. 현재 벤치마크 결과는 반도체 설비 현장 성능이 아니다.
4. Exp409 smoke는 prospective 성능 검증이 아니다.

- [ ] **Step 5: 섹션 9에 현재 데이터만을 사용하는 후속 연구 방향을 작성한다**

`## 9. 앞으로의 연구 방향`에 아래 표를 작성한다.

| 방법 | 쉬운 예 | 줄이려는 위험 |
|---|---|---|
| family-grouped validation | ShapesAll 계열을 쪼개 학습·평가 양쪽에 넣지 않고 family 전체를 한쪽 fold에 둠 | family 유사성 누수와 과대대표 |
| nested validation | 안쪽 fold에서 alpha/threshold를 고르고 바깥 fold는 최종 평가에만 사용 | TEST를 본 뒤 설정을 재선택하는 과적합 |
| overlap-aware reporting | 정상 중복이 높은 집합과 낮은 집합의 결과를 따로 보고 전체 평균과 함께 제시 | TRAIN/TEST 정확 중복이 만드는 낙관적 해석 |
| uncertainty-aware reporting | 이상 1개인 데이터셋과 충분한 이상이 있는 데이터셋을 분리하고 신뢰구간·민감도를 제시 | 작은 분모에 의한 F1 급변 |

이 접근은 신규 데이터 기반 prospective validation의 대체물이 아니라,
추가 데이터가 없는 조건에서 누수와 과장을 줄이는 엄격한 retrospective
개발 방식이라고 결론내린다.

- [ ] **Step 6: 섹션 10에 실험 불변 원칙과 문서 지도를 작성한다**

`## 10. 후속 작업의 불변 원칙`에 Global Constraints의 일곱 가지 실험
금지·분리 규칙을 체크리스트로 다시 적는다.

그 뒤 `## 11. 근거 문서 지도`를 만들고 다음 범주별로 저장소 상대 링크를
제공한다.

- 복원과 현재 상태
- 데이터 품질과 평가 난이도
- 실험 일관성과 역사적 결과
- Exp403-408 설계·실행·결과
- Exp409 설계·실행·준비성
- 새 실험 작성 규칙

외부 결과 파일 두 개는 절대 경로가 현재 Mac에서만 유효할 수 있다고
표시한다.

마지막에 `## 12. 알려진 문서 결손과 충돌 처리`를 추가한다.

- `docs/EXP403_408_PREREGISTRATION_KO.md`가 handoff에서 참조되지만
  working tree와 확인 가능한 Git 이력에는 없다고 기록한다.
- 초기 `README.md`의 947개 모집단과 현재 1,117개 모집단을 각각 역사적
  수치와 현재 동결 수치로 구분한다.
- Exp409의 “새 데이터가 도착할 때”라는 설계 전제는 역사적으로 보존하되
  현재 실행 가능한 계획이 아니라고 설명한다.
- 근거 우선순위를 `동결 결과·감사 → 실험 사전등록/런북 → handoff →
  과거 종합 보고서 → 초기 README`로 제시한다.

- [ ] **Step 7: 설계 요구사항과 문서 구조를 완전 검사한다**

Run:

```bash
rg -n "^## ([1-9]|1[0-2])\\." docs/PROJECT_RESEARCH_CONTEXT_KO.md
rg -n "Exp403|Exp404|Exp405|Exp406|Exp407|Exp408|Exp409|473|395|78|62|29|33|46\\.77|two_new_plus_existing_independent|0\\.005" \
  docs/PROJECT_RESEARCH_CONTEXT_KO.md
rg -n "Autonomous|Retrospective|Human-assisted|mean_combined_f1|Priority|Shadow|family-grouped|nested validation|overlap-aware|uncertainty" \
  docs/PROJECT_RESEARCH_CONTEXT_KO.md
rg -n "TEST label|실제 이상 위치|family.*TEST|tail replacement|alpha|threshold|source|budget|새 실험 번호|사전등록" \
  docs/PROJECT_RESEARCH_CONTEXT_KO.md
rg -n "EXP403_408_PREREGISTRATION_KO.md|947|1,117|README|동결 결과.*감사" \
  docs/PROJECT_RESEARCH_CONTEXT_KO.md
```

Expected:

- 섹션 1-12가 번호 순서대로 존재한다.
- Exp403-409의 필수 수치와 해석이 모두 검색된다.
- 세 metric claim 종류와 네 후속 검증 방법이 모두 존재한다.
- 금지 원칙, 누락 문서, 역사적/현재 모집단 구분이 모두 존재한다.

- [ ] **Step 8: 인용된 저장소 내부 문서와 변경 범위를 검증한다**

Run:

```bash
for p in \
  README.md \
  docs/HANDOFF_20260728_EXP403_408_KO.md \
  docs/task.md \
  docs/anomaly_validation_difficulty_report_20260707.md \
  docs/experiment_consistency_audit_20260707.md \
  docs/experiment_39_50_results_analysis_20260707.md \
  docs/model_hard_response_research_20260707.md \
  docs/EXP403_408_RUNBOOK_KO.md \
  docs/EXP403_408_CHECKPOINT_PROVENANCE_AUDIT_KO.md \
  docs/EXP409_PREREGISTRATION_KO.md \
  docs/EXP409_RUNBOOK_KO.md; do
  test -f "$p" || exit 1
done
test ! -e docs/EXP403_408_PREREGISTRATION_KO.md
git diff --check
git diff --name-only HEAD
```

Expected:

- 인용한 내부 핵심 문서가 모두 존재한다.
- 누락되었다고 기록한 preregistration 문서는 실제로 존재하지 않는다.
- whitespace 오류가 없다.
- Task 2의 미커밋 변경은 `docs/PROJECT_RESEARCH_CONTEXT_KO.md` 하나뿐이다.

- [ ] **Step 9: 완성 문서를 자체 검토한다**

승인된 설계
`docs/superpowers/specs/2026-07-28-project-research-context-design.md`의
완료 기준 10개를 한 항목씩 문서에서 대조한다.

Run:

```bash
git diff -- docs/PROJECT_RESEARCH_CONTEXT_KO.md
git diff --check
```

Expected:

- 현재 데이터와 반도체 현장 데이터가 구분된다.
- 신규 실데이터 없음, lane 수치, review-only, Exp409 claim boundary,
  human-assisted 경계, 누락 preregistration, 금지 원칙이 모두 명시된다.
- 코드·설정·기존 동결 결과 변경은 없다.

- [ ] **Step 10: 두 번째 문서 단위를 커밋하고 최종 상태를 확인한다**

Run:

```bash
git add docs/PROJECT_RESEARCH_CONTEXT_KO.md
git diff --cached --check
git commit -m "docs: complete research evidence and decision guide"
git status --short --branch
git log -3 --oneline
```

Expected:

- 섹션 6-12와 Task 1 검토 중 필요한 문구 보정이 한 커밋에 담긴다.
- worktree에 미커밋 변경이 없다.
- 계획 커밋과 구현 커밋 두 개가 최근 로그에 나타난다.

---

## Final Verification

전체 구현 후 다음 명령을 새로 실행한다.

```bash
git diff HEAD~2..HEAD --check
git diff --stat HEAD~2..HEAD
rg -n "^## ([1-9]|1[0-2])\\." docs/PROJECT_RESEARCH_CONTEXT_KO.md
rg -n "Hard.*2,218|Standard.*806|Priority.*31|review-only|mean_combined_f1|prospective.*아니|신규 실데이터|EXP403_408_PREREGISTRATION_KO.md" \
  docs/PROJECT_RESEARCH_CONTEXT_KO.md
git status --short --branch
```

Expected:

- diff whitespace 오류가 없다.
- 구현 범위는 새 통합 문서 하나다.
- 열두 개의 본문 섹션이 모두 존재한다.
- 동결 lane, review-only, metric claim boundary, 신규 데이터 제약, 누락 문서가 확인된다.
- worktree가 clean이다.
