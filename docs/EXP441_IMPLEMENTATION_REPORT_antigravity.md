# Exp441 구현 검증 보고서

## 핵심 판단

Exp441 구현은 공식 실행 직전 단계까지 완료됐다. Exp441은 새 모델을 다시 학습하는
실험이 아니라, 이미 동결된 Exp440의 label-free 후보 중에서 기존 Hard 탐지가 한
건도 없던 데이터셋만 골라 데이터셋당 1개, 전체 최대 30개를 제안하는 좁은
복구 선택기다.

Task 12 검증에서 실제 1,117개 입력을 다음 두 경로로 각각 한 번 읽었다.

1. 공식 소스 검증과 같은 CPU 11-worker 경로
2. 최종 감사에만 허용된 단일-worker 기준 경로

두 경로의 입력 스냅샷과 선택 결과는 완전히 같았다. SQLite 연결은 예외가 나도록
막은 상태였고 TEST label 접근 수는 0이었다. 아직 readiness, `new`, `inspect`,
`evaluate` 명령은 실행하지 않았으며 고정 실행 루트도 존재하지 않는다.

## 실제 1,117개 label-free 재생

| 항목 | 결과 |
|---|---:|
| 전체 데이터셋 | 1,117 |
| Exp440 임시 후보 | 840 |
| 기존 Hard 좌표 밖 후보 | 394 |
| Hard 탐지가 전혀 없던 데이터셋의 후보 | 73 |
| 데이터셋별 대표 후보 | 49 |
| 최종 선택 | 30 |
| 공식 경로 worker | 11 |
| 공식 경로 시간 | 24.101초 |
| 단일-worker 기준 시간 | 24.027초 |
| 부모 진행 이벤트 | 1,117 |
| TEST label 접근 | 0 |

고정 신원은 다음과 같다.

```text
Hard coordinate SHA256
858e24e6aa182e78985f571778be02bcc87332639958fa887515c24074654322

Exp441 selected coordinate SHA256
db0b9ed00ed5b9313fef10306d04bb539136c5b28b97ebcd325662d9302a3048
```

모든 후보의 초과 비율과 경험 순위는 유한한 수였고, 11-worker 완료 순서가 아니라
동결된 데이터셋 순서대로 1,117개 이벤트가 확정됐다. AST 검사와 호출 감시도
production 모듈이 단일-worker 기준 함수를 참조하거나 호출하지 않음을 확인했다.

## 회귀·순서 검증

검증 환경은 macOS arm64, Python 3.14.6, NumPy 2.4.6, 논리 CPU 12개이며 공식
worker 정책은 `max CPU - 1`, 즉 11개다.

| 범위 | 테스트 | 시간 | 결과 |
|---|---:|---:|---|
| Exp441 전체 | 194 | 67.874초 | 성공 |
| Exp440 설정·계약·checkpoint·선행 입력 + Exp417 입력 | 39 | 0.103초 | 성공 |
| 임시 루트 기반 실행→출판→평가 순서 smoke | 9 | 3.104초 | 성공 |

임시 smoke는 다음을 확인했다.

- readiness는 없던 루트만 소유하고 기존 루트를 덮어쓰지 않는다.
- attempt는 새로 만들기만 하며 실패·완료 결과를 고치지 않는다.
- prediction 출판은 정확히 6개 파일을 고정하고 다시 읽어 검증한다.
- source·범위 사전검증이 실패하면 label-capable import와 DB 연결보다 먼저 멈춘다.
- 평가 경로는 quick check 뒤 1,117개 context 순서 쿼리만 사용한다.
- 평가는 정확히 8개 파일로 출판되고 기존 prediction/attempt 바이트를 바꾸지 않는다.

새 파일 이름은 모두 `antigravity`를 포함한다. label SQL과 label loader는 lazy
post-freeze 평가 모듈에만 있고 prediction, readiness, runtime, reporting에는 없다.

## 가속 판단

Exp441의 이번 단계는 JSON 검증과 정렬·선택이므로 MLX/MPS tensor 연산으로 바꿀
대상이 아니다. 알고리즘을 바꾸지 않는 가속 조건을 만족하지 않으므로 CPU가
정본이고 11개 dataset worker를 사용한다. MLX/MPS를 억지로 넣어 동일성이나
실험 정의를 바꾸지 않는다.

## 고정 연구 지위

다음 9개 필드는 구현 보고서와 Exp441 제어 레코드에서 같은 뜻으로 고정된다.

```json
{
  "official_run_purpose": "deterministic_retrospective_reproduction",
  "bootstrap_interpretation": "descriptive_post_selection",
  "adaptive_followup": true,
  "independent_validation": false,
  "external_generalization_supported": false,
  "confirmatory_statistical_claim": false,
  "scientific_winner_selected": false,
  "review_only": true,
  "operational_alert_eligible": false
}
```

따라서 Exp441은 champion이나 자동 경보 후보가 아니다. 사후 bootstrap은 설명용
재생일 뿐 새 독립 검증이나 확증 통계가 아니다.

## 계획 문구 정정

Task 12 Step 7에는 `run_identity`를 `null`로 바꾸라는 문구가 있지만, 동결 설계,
사전등록, runbook, 실제 frozen config는 모두
`exp441-hard-empty-single-shot-recovery-antigravity-20260808`을 고정한다. `null`로
바꾸면 동일 실행을 묶는 신원이 깨지고 readiness가 frozen config와 어긋난다.
따라서 Task 1의 정확한 스키마는 유지하되 값은 동결된 문자열을 보존했다. 이는
실험 규칙 변경이 아니라 서로 충돌한 계획 문구를 상위 동결 계약에 맞춘 것이다.

## 공식 실행 경계와 다음 단계

현재 고정 루트는 없다.

```text
/Users/mac/Documents/Dataset/exp441_hard_empty_single_shot_recovery_antigravity_20260808
```

새 최종 구현 커밋을 Sol/High가 독립 검토해 `proceed`를 반환하기 전에는 아래
명령을 실행하지 않는다. 검토가 통과한 깨끗한 HEAD에서 다음 readiness 명령을
정확히 한 번 실행한다.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 \
  scratch/run_exp441_hard_empty_single_shot_recovery_antigravity.py readiness
```

readiness를 다시 읽어 reviewed commit, 고정 소스 해시, CPU/11-worker, NumPy
2.4.6, TEST label 접근 0을 확인한 뒤에만 `new`를 고려한다.
